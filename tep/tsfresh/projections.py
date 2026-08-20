"""Projektionen (Dimensionsreduktion) als Registry aus einfachen dicts.

Jede Projektion kennt drei Dinge: wie sie heisst (der Name ist zugleich
Cache-Praefix), wie ihre Kanaele heissen und wie sie rechnet. Ein Notebook
waehlt ueber die `configs`-Liste aus, welche Specs es fahren will:

    ("raw",)                        52 Rohkanaele
    ("pca", n)                      PCA
    ("dyca", m, n)                  DyCA
    ("dpca", n)                     Dynamic PCA (Lag-Stacking + PCA)
    ("cva", n)                      Canonical Variate Analysis
    ("ica", n)                      FastICA
    ("dycvda", m, n, s, r)          DyCVDA (Wu et al. 2026)

Ein Eintrag in PROJECTORS hat die Schluessel:

    name      spec -> Kurzname (Cache-Praefix und Spaltenpraefix)
    channels  spec -> Liste der Kanalnamen
    apply     (X_scaled, spec, **params) -> (T-Strich, C)-Array
    flip      ob die Vorzeichenkonvention angewandt wird. Bei "raw" nicht,
              dort gibt es keine Vorzeichenwillkuer zu beheben. Default True.
    validate  optionaler Check der Spec vor dem Lauf

Eigene Verfahren kommen als weiterer PROJECTORS-Eintrag dazu, ohne die
Notebooks anzufassen. Jede `apply`-Funktion nimmt genau die Parameter
entgegen, die sie braucht, und schluckt den Rest mit `**_`.

WICHTIG - Vertraeglichkeit mit dem Cache: die Rechnungen hier sind
zeilengetreu aus den urspruenglichen Notebooks uebernommen, inklusive der
dtype-Fuehrung. Nur so passen neu berechnete Chunks zu den bereits im
Cache liegenden.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA, FastICA

from ..core import (PROC_COLS, cva_covs, dyca_amplitudes, flip_signs,
                    inv_sqrt_psd, lag_stack, scale)


# =========================================================================
# Die Verfahren
# =========================================================================

def _apply_pca(X, spec, **_):
    # svd_solver="full" wie im Eigenwert-Notebook; sklearn wendet intern
    # bereits svd_flip an, die Vorzeichenkonvention unten macht es explizit.
    # Bewusst KEIN astype: X kommt bei global_mean als float32 herein und
    # die bereits gecachten Chunks sind genau so gerechnet.
    return PCA(n_components=spec[1], svd_solver="full").fit_transform(X)


def _apply_dyca(X, spec, **_):
    return dyca_amplitudes(np.asarray(X, dtype=np.float64), spec[1], spec[2])


def _validate_dyca(spec):
    m, n = spec[1], spec[2]
    assert m >= n - m, f"dyca verlangt m >= n - m, verletzt von {(m, n)}"


def _apply_dpca(X, spec, dpca_lags=2, **_):
    # Lag-Stacking + PCA wie in DPCA_Eigenwerte.ipynb; svd_solver "full"
    # haelt die Komponenten numerisch identisch zum dortigen Vollspektrum.
    Z = lag_stack(np.asarray(X, dtype=np.float64), dpca_lags)
    return PCA(n_components=spec[1], svd_solver="full").fit_transform(Z)


def _apply_cva(X, spec, cva_past=1, cva_fut=1, cva_ridge_rel=1e-6, **_):
    """CVA pro Run: die ersten n kanonischen Variaten der Vergangenheit als
    Zeitreihe (T - cva_past - cva_fut + 1, n).

    Mathematik wie in CVA_Eigenwerte.ipynb: H = S_ff^(-1/2) S_fp S_pp^(-1/2);
    die rechten Singulaervektoren von H spannen (nach Ruecktransformation
    mit S_pp^(-1/2)) die Projektionsmatrix J auf, z_t = J^T (p_t - mean).
    Die Variaten haben Einheitsvarianz.
    """
    n = spec[1]
    Pc, _, Spp, Sff, Sfp = cva_covs(np.asarray(X, dtype=np.float64),
                                    cva_past, cva_fut)
    Wp = inv_sqrt_psd(Spp, cva_ridge_rel)
    Wf = inv_sqrt_psd(Sff, cva_ridge_rel)
    _, _, Vt = np.linalg.svd(Wf @ Sfp @ Wp)
    J = Wp @ Vt[:n].T                       # (dim_p, n)
    return Pc @ J


def _apply_ica(X, spec, ica_max_iter=1000, ica_tol=1e-3,
               ica_random_state=42, **_):
    # ConvergenceWarnings unterdruecken: Nicht-Konvergenz ist kein Ausfall
    # (die teil-konvergierte Rotation wird verwendet, wie in
    # ICA_Eigenwerte.ipynb).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica = FastICA(n_components=spec[1], whiten="unit-variance",
                      max_iter=ica_max_iter, tol=ica_tol,
                      random_state=ica_random_state)
        S = ica.fit_transform(np.asarray(X, dtype=np.float64))
    # Ordnungskonvention: staerkste Nicht-Gaussianitaet zuerst - sonst waere
    # "Kanal ic1" ueber die Runs hinweg willkuerlich.
    kurt = stats.kurtosis(S, axis=0, fisher=True, bias=True)
    return S[:, np.argsort(-np.abs(kurt))]


def _apply_dycvda(X, spec, cva_ridge_rel=1e-6, **_):
    """DyCVDA pro Run (Wu et al. 2026, IEEE TASE 23, S. 9560-9570):
    DyCA-Amplituden -> CVA zwischen Vergangenheits- und Zukunftsstapeln ->
    Dissimilaritaetskanaele.

    Stufe 1 - DyCA (Paper Gl. 5-11), identisch zur "dyca"-Projektion:
    n Amplituden y(t), Form (T, n).

    Stufe 2 - CVA-Dissimilaritaet (Paper Gl. 12-15) mit Horizont s:
        y_p(t) = [y(t-1); ...; y(t-s)],  y_f(t) = [y(t); ...; y(t+s-1)]
        H = S_ff^(-1/2) S_fp S_pp^(-1/2) = U Lam V^T
        J = V_r^T S_pp^(-1/2),  L = U_r^T S_ff^(-1/2)
        d(t) = J y_p(t) - Lam_r L y_f(t)
    Die Stapel werden vor der CVA zentriert (das Paper normalisiert die
    Daten vorab). Selbsttest: auf den Fit-Daten gilt cov(d) = I - Lam_r^2 -
    ohne Ridge auf Maschinengenauigkeit verifiziert.

    Rueckgabe: (T - 2s + 1, r).
    """
    m, n, s, r = spec[1:]
    Y = dyca_amplitudes(np.asarray(X, dtype=np.float64), m, n)   # (T, n)

    # Gueltige Zeitpunkte (0-basiert): t = s .. T-s  ->  T - 2s + 1 Zeilen.
    Pc, Fc, Spp, Sff, Sfp = cva_covs(Y, s, s)
    Wp = inv_sqrt_psd(Spp, cva_ridge_rel)
    Wf = inv_sqrt_psd(Sff, cva_ridge_rel)
    U, sv, Vt = np.linalg.svd(Wf @ Sfp @ Wp)
    J = Vt[:r] @ Wp                                # (r, n*s)
    L = U[:, :r].T @ Wf                            # (r, n*s)
    return Pc @ J.T - (Fc @ L.T) * sv[:r]


def _validate_dycvda(spec):
    m, n, s, r = spec[1:]
    assert m >= n - m, f"dyca verlangt m >= n - m, verletzt von {(m, n)}"
    assert r <= n * s, f"r <= n*s verletzt von {(m, n, s, r)}"


PROJECTORS = {
    "raw": {"name": lambda spec: "raw",
            "channels": lambda spec: list(PROC_COLS),
            "apply": lambda X, spec, **_: X,
            "flip": False},
    "pca": {"name": lambda spec: f"pca_{spec[1]}",
            "channels": lambda spec: [f"pc{i}"
                                      for i in range(1, spec[1] + 1)],
            "apply": _apply_pca},
    "dyca": {"name": lambda spec: f"dyca_m{spec[1]}_n{spec[2]}",
             "channels": lambda spec: [f"dy{i}"
                                       for i in range(1, spec[2] + 1)],
             "apply": _apply_dyca, "validate": _validate_dyca},
    "dpca": {"name": lambda spec: f"dpca_{spec[1]}",
             "channels": lambda spec: [f"dpc{i}"
                                       for i in range(1, spec[1] + 1)],
             "apply": _apply_dpca},
    "cva": {"name": lambda spec: f"cva_{spec[1]}",
            "channels": lambda spec: [f"cv{i}"
                                      for i in range(1, spec[1] + 1)],
            "apply": _apply_cva},
    "ica": {"name": lambda spec: f"ica_{spec[1]}",
            "channels": lambda spec: [f"ic{i}"
                                      for i in range(1, spec[1] + 1)],
            "apply": _apply_ica},
    # Kanalordnung: kanonische Korrelation absteigend (cvd1 = staerkste) -
    # deterministisch, keine Ordnungswillkuer wie bei ICA.
    "dycvda": {"name": lambda spec: (f"dycvda_m{spec[1]}n{spec[2]}"
                                     f"_s{spec[3]}_r{spec[4]}"),
               "channels": lambda spec: [f"cvd{i}"
                                         for i in range(1, spec[4] + 1)],
               "apply": _apply_dycvda, "validate": _validate_dycvda},
}


def get(spec: tuple) -> dict:
    """Der PROJECTORS-Eintrag zu einer Spec, mit klarer Fehlermeldung."""
    if spec[0] not in PROJECTORS:
        raise ValueError(f"Unbekannte Konfiguration: {spec!r} "
                         f"(bekannt: {sorted(PROJECTORS)})")
    return PROJECTORS[spec[0]]


def config_name(spec: tuple) -> str:
    """Kurzname einer Spec - dient als Cache-Praefix und Spaltenpraefix."""
    return get(spec)["name"](spec)


def channel_names(spec: tuple) -> list:
    return get(spec)["channels"](spec)


def n_channels(spec: tuple) -> int:
    return len(channel_names(spec))


def validate(configs) -> list:
    """Prueft alle Specs (Rangbedingungen, doppelte Namen) und liefert die
    Kurznamen in Reihenfolge."""
    names = []
    for spec in configs:
        proj = get(spec)                       # wirft bei unbekannter Art
        if "validate" in proj:
            proj["validate"](spec)
        names.append(config_name(spec))
    if len(set(names)) != len(names):
        raise ValueError(f"Doppelte Konfigurationsnamen: {names}")
    return names


# =========================================================================
# Aufruf
# =========================================================================

def project(X: np.ndarray, spec: tuple, scaling_mode: str = "global_mean",
            scaler=None, fix_signs: bool = True, **params):
    """Projiziert einen Run.

    params : Verfahrensparameter (dpca_lags, cva_past, cva_fut,
             cva_ridge_rel, ica_max_iter, ica_tol, ica_random_state) -
             jede apply-Funktion nimmt sich daraus, was sie braucht.

    Rueckgabe: (Y, names) mit Y der Form (T-Strich, C). Die Laenge ist
    projektionsabhaengig (raw/pca/dyca/ica: T; cva: T - past - fut + 1;
    dpca: T - lags; dycvda: T - 2s + 1). Wirft eine Exception, wenn die
    Projektion fuer diesen Run scheitert - der Run wird dann fuer DIESE
    Konfiguration uebersprungen.
    """
    proj = get(spec)
    Y = proj["apply"](scale(X, scaling_mode, scaler), spec, **params)

    # float64, NICHT float32: bei scaling_mode="global_mean" liegt die
    # Dynamik der DyCA-Amplituden 3-5 Groessenordnungen unter dem
    # Gleichanteil. float32 (~7 signifikante Stellen) wuerde genau den
    # informativen Anteil wegquantisieren. Der Speicherpreis ist klein -
    # gecacht wird spaeter ohnehin nur die Feature-Matrix als float32.
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    if fix_signs and proj.get("flip", True):
        Y = flip_signs(Y)
    return Y, proj["channels"](spec)
