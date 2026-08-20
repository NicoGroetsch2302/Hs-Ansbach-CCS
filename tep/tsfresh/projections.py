"""Projektionen (Dimensionsreduktion) als Registry.

Jede Projektion ist ein Projector und kennt drei Dinge: wie sie heisst
(der Name ist zugleich Cache-Praefix), wie ihre Kanaele heissen und wie
sie rechnet. Ein Notebook waehlt ueber PipelineConfig.configs aus, welche
Specs es fahren will:

    ("raw",)                        52 Rohkanaele
    ("pca", n)                      PCA
    ("dyca", m, n)                  DyCA
    ("dpca", n)                     Dynamic PCA (Lag-Stacking + PCA)
    ("cva", n)                      Canonical Variate Analysis
    ("ica", n)                      FastICA
    ("dycvda", m, n, s, r)          DyCVDA (Wu et al. 2026)

Eigene Verfahren kommen ueber register() dazu, ohne die Notebooks
anzufassen.

WICHTIG - Vertraeglichkeit mit dem Cache: die Rechnungen hier sind
zeilengetreu aus den urspruenglichen Notebooks uebernommen, inklusive der
dtype-Fuehrung. Nur so passen neu berechnete Chunks zu den bereits im
Cache liegenden.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA, FastICA

from ..core import (PROC_COLS, dyca_amplitudes, flip_signs, inv_sqrt_psd,
                    lag_stack, scale)
from .config import PipelineConfig


# =========================================================================
# Registry
# =========================================================================

@dataclass(frozen=True)
class Projector:
    """Ein Projektionsverfahren.

    name      : spec -> Kurzname (Cache-Praefix und Spaltenpraefix)
    channels  : spec -> Liste der Kanalnamen
    apply     : (X_scaled, spec, cfg) -> (T-Strich, C)-Array
    flip      : ob die Vorzeichenkonvention angewandt wird. Bei "raw" nicht,
                dort gibt es keine Vorzeichenwillkuer zu beheben.
    validate  : optionaler Check der Spec beim Anlegen der Pipeline
    """
    name: Callable[[tuple], str]
    channels: Callable[[tuple], list]
    apply: Callable[[np.ndarray, tuple, PipelineConfig], np.ndarray]
    flip: bool = True
    validate: Callable[[tuple], None] | None = None


PROJECTORS: dict = {}


def register(kind: str, projector: Projector) -> None:
    """Traegt ein Verfahren unter kind ein (ueberschreibt vorhandene)."""
    PROJECTORS[kind] = projector


def get(spec: tuple) -> Projector:
    kind = spec[0]
    if kind not in PROJECTORS:
        raise ValueError(f"Unbekannte Konfiguration: {spec!r} "
                         f"(bekannt: {sorted(PROJECTORS)})")
    return PROJECTORS[kind]


def config_name(spec: tuple) -> str:
    """Kurzname einer Spec - dient als Cache-Praefix und Spaltenpraefix."""
    return get(spec).name(spec)


def channel_names(spec: tuple) -> list:
    return get(spec).channels(spec)


def n_channels(spec: tuple) -> int:
    return len(channel_names(spec))


# =========================================================================
# Bausteine
# =========================================================================

# flip_signs, lag_stack, inv_sqrt_psd und dyca_amplitudes stehen in
# tep/core.py - tep.eigen braucht dieselben Bausteine.


# =========================================================================
# Die Verfahren
# =========================================================================

register("raw", Projector(
    name=lambda spec: "raw",
    channels=lambda spec: list(PROC_COLS),
    apply=lambda X, spec, cfg: X,
    flip=False,
))


def _apply_pca(X, spec, cfg):
    # svd_solver="full" wie im Eigenwert-Notebook; sklearn wendet intern
    # bereits svd_flip an, die Vorzeichenkonvention unten macht es explizit.
    # Bewusst KEIN astype: X kommt bei global_mean als float32 herein und
    # die bereits gecachten Chunks sind genau so gerechnet.
    return PCA(n_components=spec[1], svd_solver="full").fit_transform(X)


register("pca", Projector(
    name=lambda spec: f"pca_{spec[1]}",
    channels=lambda spec: [f"pc{i}" for i in range(1, spec[1] + 1)],
    apply=_apply_pca,
))


def _apply_dyca(X, spec, cfg):
    return dyca_amplitudes(np.asarray(X, dtype=np.float64), spec[1], spec[2])


def _validate_dyca(spec):
    m, n = spec[1], spec[2]
    assert m >= n - m, f"dyca verlangt m >= n - m, verletzt von {(m, n)}"


register("dyca", Projector(
    name=lambda spec: f"dyca_m{spec[1]}_n{spec[2]}",
    channels=lambda spec: [f"dy{i}" for i in range(1, spec[2] + 1)],
    apply=_apply_dyca,
    validate=_validate_dyca,
))


def _apply_dpca(X, spec, cfg):
    # Lag-Stacking + PCA wie in DPCA_Eigenwerte.ipynb; svd_solver "full"
    # haelt die Komponenten numerisch identisch zum dortigen Vollspektrum.
    Z = lag_stack(np.asarray(X, dtype=np.float64), cfg.dpca_lags)
    return PCA(n_components=spec[1], svd_solver="full").fit_transform(Z)


register("dpca", Projector(
    name=lambda spec: f"dpca_{spec[1]}",
    channels=lambda spec: [f"dpc{i}" for i in range(1, spec[1] + 1)],
    apply=_apply_dpca,
))


def _apply_cva(X, spec, cfg):
    """CVA pro Run: die ersten n kanonischen Variaten der Vergangenheit als
    Zeitreihe (T - cva_past - cva_fut + 1, n).

    Mathematik wie in CVA_Eigenwerte.ipynb: H = S_ff^(-1/2) S_fp S_pp^(-1/2);
    die rechten Singulaervektoren von H spannen (nach Ruecktransformation
    mit S_pp^(-1/2)) die Projektionsmatrix J auf, z_t = J^T (p_t - mean).
    Die Variaten haben Einheitsvarianz.
    """
    X = np.asarray(X, dtype=np.float64)
    n = spec[1]
    T = X.shape[0]
    past, fut = cfg.cva_past, cfg.cva_fut
    P = np.hstack([X[past - j: T - fut + 1 - j] for j in range(1, past + 1)])
    F = np.hstack([X[past + j: T - fut + 1 + j] for j in range(fut)])
    N = P.shape[0]
    Pc = P - P.mean(axis=0)
    Fc = F - F.mean(axis=0)
    Spp = Pc.T @ Pc / (N - 1)
    Sff = Fc.T @ Fc / (N - 1)
    Sfp = Fc.T @ Pc / (N - 1)
    Wp = inv_sqrt_psd(Spp, cfg.cva_ridge_rel)
    Wf = inv_sqrt_psd(Sff, cfg.cva_ridge_rel)
    _, _, Vt = np.linalg.svd(Wf @ Sfp @ Wp)
    J = Wp @ Vt[:n].T                       # (dim_p, n)
    return Pc @ J


register("cva", Projector(
    name=lambda spec: f"cva_{spec[1]}",
    channels=lambda spec: [f"cv{i}" for i in range(1, spec[1] + 1)],
    apply=_apply_cva,
))


def _apply_ica(X, spec, cfg):
    n = spec[1]
    # ConvergenceWarnings unterdruecken: Nicht-Konvergenz ist kein Ausfall
    # (die teil-konvergierte Rotation wird verwendet, wie in
    # ICA_Eigenwerte.ipynb).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica = FastICA(n_components=n, whiten="unit-variance",
                      max_iter=cfg.ica_max_iter, tol=cfg.ica_tol,
                      random_state=cfg.ica_random_state)
        S = ica.fit_transform(np.asarray(X, dtype=np.float64))
    # Ordnungskonvention: staerkste Nicht-Gaussianitaet zuerst - sonst waere
    # "Kanal ic1" ueber die Runs hinweg willkuerlich.
    kurt = stats.kurtosis(S, axis=0, fisher=True, bias=True)
    return S[:, np.argsort(-np.abs(kurt))]


register("ica", Projector(
    name=lambda spec: f"ica_{spec[1]}",
    channels=lambda spec: [f"ic{i}" for i in range(1, spec[1] + 1)],
    apply=_apply_ica,
))


def _apply_dycvda(X, spec, cfg):
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

    T = Y.shape[0]
    # Gueltige Zeitpunkte (0-basiert): t = s .. T-s  ->  T - 2s + 1 Zeilen.
    P = np.hstack([Y[s - j: T - s - j + 1] for j in range(1, s + 1)])
    F = np.hstack([Y[s + j: T - s + j + 1] for j in range(s)])
    N = P.shape[0]
    Pc = P - P.mean(axis=0)
    Fc = F - F.mean(axis=0)
    Spp = Pc.T @ Pc / (N - 1)
    Sff = Fc.T @ Fc / (N - 1)
    Sfp = Fc.T @ Pc / (N - 1)
    Wp = inv_sqrt_psd(Spp, cfg.cva_ridge_rel)
    Wf = inv_sqrt_psd(Sff, cfg.cva_ridge_rel)
    U, sv, Vt = np.linalg.svd(Wf @ Sfp @ Wp)
    J = Vt[:r] @ Wp                                # (r, n*s)
    L = U[:, :r].T @ Wf                            # (r, n*s)
    return Pc @ J.T - (Fc @ L.T) * sv[:r]


def _validate_dycvda(spec):
    m, n, s, r = spec[1:]
    assert m >= n - m, f"dyca verlangt m >= n - m, verletzt von {(m, n)}"
    assert r <= n * s, f"r <= n*s verletzt von {(m, n, s, r)}"


register("dycvda", Projector(
    name=lambda spec: f"dycvda_m{spec[1]}n{spec[2]}_s{spec[3]}_r{spec[4]}",
    # Kanalordnung: kanonische Korrelation absteigend (cvd1 = staerkste) -
    # deterministisch, keine Ordnungswillkuer wie bei ICA.
    channels=lambda spec: [f"cvd{i}" for i in range(1, spec[4] + 1)],
    apply=_apply_dycvda,
    validate=_validate_dycvda,
))


# =========================================================================
# Aufruf
# =========================================================================

def project(X: np.ndarray, spec: tuple, cfg: PipelineConfig, scaler=None):
    """Projiziert einen Run.

    Rueckgabe: (Y, names) mit Y der Form (T-Strich, C). Die Laenge ist
    projektionsabhaengig (raw/pca/dyca/ica: T; cva: T - past - fut + 1;
    dpca: T - lags; dycvda: T - 2s + 1). Wirft eine Exception, wenn die
    Projektion fuer diesen Run scheitert - der Run wird dann fuer DIESE
    Konfiguration uebersprungen.
    """
    proj = get(spec)
    Y = proj.apply(scale(X, cfg.scaling_mode, scaler), spec, cfg)

    # float64, NICHT float32: bei scaling_mode="global_mean" liegt die
    # Dynamik der DyCA-Amplituden 3-5 Groessenordnungen unter dem
    # Gleichanteil. float32 (~7 signifikante Stellen) wuerde genau den
    # informativen Anteil wegquantisieren. Der Speicherpreis ist klein -
    # gecacht wird spaeter ohnehin nur die Feature-Matrix als float32.
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    if cfg.fix_signs and proj.flip:
        Y = flip_signs(Y)
    return Y, proj.channels(spec)
