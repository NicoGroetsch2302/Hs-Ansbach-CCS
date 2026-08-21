"""Die Spektren-Verfahren als Registry aus einfachen dicts.

Jedes Verfahren liefert pro (faultNumber, simulationRun) einen Vektor von
Kennzahlen - Eigenwerte, kanonische Korrelationen oder Kurtosis-Werte. Der
Ablauf drumherum (gruppieren, Pre-Fault verwerfen, zu kurze Runs
ueberspringen, Fehler zaehlen, DataFrame bauen) ist fuer alle gleich und
steht in `run_spectra`.

    "pca"    Varianzanteile der run-eigenen Kovarianz      -> lambda_1..52
    "dyca"   generalisierte Eigenwerte der DyCA            -> dyca_1..52
    "dpca"   Varianzanteile nach Lag-Stacking              -> dpca_1..156
    "cva"    kanonische Korrelationen Vergangenheit/Zukunft-> cva_1..52
    "ica"    |Kurtosis| der FastICA-Quellen, absteigend    -> ica_1..12
    "lda"    Fisher-Kennzahl gegen einen Normalbetriebslauf-> lda_eigenvalue

Ein Eintrag in SPECTRA hat die Schluessel:

    prefix          Spaltenpraefix der Werte ("lambda_", "dyca_", ...)
    label           Klartextname fuer Plottitel ("PCA", "DyCA", ...)
    csv_stem        Dateiname-Stamm des Exports ("pca_eigenvalues")
    apply           (X_scaled, **params) -> Wertevektor, oder
                    (Wertevektor, extras-dict)
    scalar          True, wenn EINE Zahl je Run herauskommt (nur LDA)
    forced_scaling  erzwingt einen Skalierungsmodus (LDA: "scaler")
    extra_cols      zusaetzliche Spalten aus dem extras-dict (ICA)

Die letzten drei sind optional; wer sie nicht setzt, bekommt den Default
ueber SPECTRA[method].get(...). Eigene Verfahren kommen als weiterer
SPECTRA-Eintrag dazu.

Jede `apply`-Funktion nimmt genau die Parameter entgegen, die sie
wirklich braucht, und schluckt den Rest mit `**_`. Was ein Verfahren
steuert, steht damit in seiner eigenen Signatur.

WICHTIG - die Spaltennamen und die CSV-Namen sind eingefroren:
`LazyClassifier_PCA_DyCA` liest die exportierten Dateien.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA, FastICA
from sklearn.exceptions import ConvergenceWarning
from tqdm.auto import tqdm

from ..core import (PRE_FAULT_CUTOFF, PROC_COLS, cva_covs, inv_sqrt_psd,
                    lag_stack, scale)


# =========================================================================
# Die Verfahren
# =========================================================================

def _apply_pca(X, **_):
    # n_components = Anzahl Prozessvariablen -> ALLE Eigenwerte, auch die
    # kleinen aus dem Noise-Subspace. svd_solver="full" garantiert das
    # vollstaendige Spektrum, absteigend sortiert und reell.
    pca = PCA(n_components=len(PROC_COLS), svd_solver="full")
    pca.fit(X)
    # explained_variance_ratio_ = auf die Gesamtvarianz NORMIERTE
    # Eigenwerte (Summe je Run = 1). Fuer die unnormierten waere
    # explained_variance_ zu nehmen.
    return pca.explained_variance_ratio_


def _apply_dyca(X, dyca_m=2, dyca_n=4, **_):
    from dyca import dyca

    res = dyca(X, m=dyca_m, n=dyca_n)
    return np.asarray(res["generalized_eigenvalues"], dtype=float)


def _apply_dpca(X, dpca_lags=2, **_):
    Z = lag_stack(X, dpca_lags)
    pca = PCA(n_components=None, svd_solver="full")
    pca.fit(Z)
    return pca.explained_variance_ratio_


def _apply_cva(X, cva_past=1, cva_fut=1, ridge_rel=1e-6, **_):
    """Kanonische Korrelationen zwischen Vergangenheits- und
    Zukunftsstapel: H = S_ff^(-1/2) S_fp S_pp^(-1/2), die Singulaerwerte
    von H sind die Korrelationen und liegen in [0, 1]."""
    _, _, Spp, Sff, Sfp = cva_covs(np.asarray(X, dtype=np.float64),
                                   cva_past, cva_fut)
    H = inv_sqrt_psd(Sff, ridge_rel) @ Sfp @ inv_sqrt_psd(Spp, ridge_rel)
    corrs = np.linalg.svd(H, compute_uv=False)
    # Floating-Point-Rauschen kann winzige negative Werte bzw. Werte
    # knapp ueber 1 erzeugen - beides ist keine echte Korrelation.
    return np.clip(corrs, 0.0, 1.0)


def _apply_ica(X, ica_n=12, ica_max_iter=1000, ica_tol=1e-3,
               ica_random_state=42, **_):
    """Nicht-Gaussianitaets-Spektrum: Kurtosis der FastICA-Quellen, nach
    Betrag absteigend sortiert. Ob FastICA konvergiert ist, wird als
    eigene Spalte mitgefuehrt."""
    X = np.asarray(X, dtype=np.float64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ica = FastICA(n_components=ica_n, whiten="unit-variance",
                      max_iter=ica_max_iter, tol=ica_tol,
                      random_state=ica_random_state)
        S = ica.fit_transform(X)
    converged = not any(issubclass(w.category, ConvergenceWarning)
                        for w in caught)
    kurt = stats.kurtosis(S, axis=0, fisher=True, bias=True)
    order = np.argsort(-np.abs(kurt))
    return kurt[order], {"converged": int(converged)}


def lda_eigenvalue(X0: np.ndarray, X1: np.ndarray, ridge_rel: float) -> float:
    """Fisher-Kennzahl zweier Punktwolken (Zwei-Klassen-LDA).

    Sw = Summe der klassenzentrierten Streumatrizen, d = Mittelwerts-
    differenz. lambda = n0*n1/(n0+n1) * d^T Sw^-1 d - der einzige von Null
    verschiedene Eigenwert des Zwei-Klassen-Problems und damit ein Mass
    fuer die Separierbarkeit.
    """
    n0, n1 = X0.shape[0], X1.shape[0]
    m0, m1 = X0.mean(axis=0), X1.mean(axis=0)
    X0c, X1c = X0 - m0, X1 - m1
    Sw = X0c.T @ X0c + X1c.T @ X1c
    p = Sw.shape[0]
    Sw = Sw + ridge_rel * (np.trace(Sw) / p) * np.eye(p)
    d = m1 - m0
    return n0 * n1 / (n0 + n1) * float(d @ np.linalg.solve(Sw, d))


def _apply_lda(X, ff_by_run=None, fault=0, run=0, ridge_rel=1e-6, **_):
    """Jeder Lauf wird gegen einen Normalbetriebslauf gestellt.

    Fault != 0 : Fehlerlauf gegen den FaultFree-Lauf gleicher Run-Nummer.
    Fault == 0 : FaultFree-Lauf gegen seinen Nachbarlauf - sonst waere die
                 Referenzklasse mit sich selbst verglichen und lambda
                 trivialerweise null.
    """
    run_ids = sorted(ff_by_run)
    if fault == 0:
        X1 = ff_by_run[run]
        X0 = ff_by_run[run_ids[(run_ids.index(run) + 1) % len(run_ids)]]
    else:
        X1 = X
        X0 = ff_by_run[run]
    return np.array([lda_eigenvalue(X0, X1, ridge_rel)])


SPECTRA = {
    "pca": {"prefix": "lambda_", "label": "PCA",
            "csv_stem": "pca_eigenvalues", "apply": _apply_pca},
    "dyca": {"prefix": "dyca_", "label": "DyCA",
             "csv_stem": "dyca_eigenvalues", "apply": _apply_dyca},
    "dpca": {"prefix": "dpca_", "label": "DPCA",
             "csv_stem": "dpca_eigenvalues", "apply": _apply_dpca},
    "cva": {"prefix": "cva_", "label": "CVA",
            "csv_stem": "cva_eigenvalues", "apply": _apply_cva},
    "ica": {"prefix": "ica_", "label": "ICA",
            "csv_stem": "ica_eigenvalues", "apply": _apply_ica,
            "extra_cols": ("converged",)},
    # LDA vergleicht ZWEI Laeufe. Bei "global_mean" zoege jeder Lauf seinen
    # eigenen skalaren Mittelwert ab - die Mittelwertsdifferenz d, also
    # genau das Signal, waere dann verfaelscht.
    "lda": {"prefix": "lda_eigenvalue", "label": "LDA",
            "csv_stem": "lda_eigenvalues", "apply": _apply_lda,
            "scalar": True, "forced_scaling": "scaler"},
}


def get(method: str) -> dict:
    """Der SPECTRA-Eintrag zu `method`, mit klarer Fehlermeldung."""
    if method not in SPECTRA:
        raise ValueError(f"Unbekanntes Verfahren: {method!r} "
                         f"(bekannt: {sorted(SPECTRA)})")
    return SPECTRA[method]


def label(method: str) -> str:
    """Klartextname fuer Plottitel, z.B. "DyCA"."""
    return get(method)["label"]


def prefix(method: str) -> str:
    """Spaltenpraefix der Spektrumswerte, z.B. "lambda_" oder "dyca_"."""
    return get(method)["prefix"]


def needs_scaler(method: str, scaling_mode: str = "global_mean") -> bool:
    """Ob ein StandardScaler gefittet werden muss: bei
    scaling_mode="scaler" und immer bei LDA (das Verfahren erzwingt ihn)."""
    return (scaling_mode == "scaler"
            or get(method).get("forced_scaling") == "scaler")


def csv_name(method: str, scaling_mode: str = "global_mean",
             split: str = "train") -> str:
    """Dateiname der Export-CSV.

    Die Namen sind eingefroren - `LazyClassifier_PCA_DyCA` liest sie. Der
    Skalierungsmodus steckt im Namen, ein Umschalten ueberschreibt also
    keine alten Ergebnisse.

    Massgeblich ist der TATSAECHLICH verwendete Modus: LDA erzwingt
    "scaler" (siehe forced_scaling), seine Datei traegt deshalb immer das
    Suffix - sonst hiesse eine Datei mit Scaler-Daten wie eine ohne.
    """
    mode = get(method).get("forced_scaling") or scaling_mode
    suffix = "" if mode == "global_mean" else "_scaler"
    return f"{get(method)['csv_stem']}_{split}{suffix}.csv"


def min_samples(method: str, dyca_m: int = 2, dyca_n: int = 4,
                cva_past: int = 1, cva_fut: int = 1) -> int | None:
    """Mindestlaenge eines Runs; kuerzere werden uebersprungen.
    None = keine Pruefung noetig."""
    if method == "dyca":
        return max(dyca_m, dyca_n) + 5
    if method == "cva":
        return len(PROC_COLS) + cva_past + cva_fut + 5
    return None


# =========================================================================
# Die gemeinsame Schleife
# =========================================================================

def run_spectra(df_all, method: str, *, scaling_mode: str = "global_mean",
                scaler=None, ff_by_run=None,
                pre_fault_cutoff: int = PRE_FAULT_CUTOFF["train"],
                head_fault0: int | None = None,
                head_faulty: int | None = None,
                dyca_m: int = 2, dyca_n: int = 4, dpca_lags: int = 2,
                cva_past: int = 1, cva_fut: int = 1, ridge_rel: float = 1e-6,
                ica_n: int = 12, ica_max_iter: int = 1000,
                ica_tol: float = 1e-3, ica_random_state: int = 42,
                verbose: bool = True) -> pd.DataFrame:
    """Berechnet das Spektrum je (faultNumber, simulationRun).

    Die Verfahrensparameter (dyca_*, dpca_*, cva_*, ica_*, ridge_rel)
    stehen alle hier, damit ein Notebook nur setzt, was sein Verfahren
    braucht; weitergereicht wird an die `apply`-Funktion des Verfahrens.

    head_fault0 / head_faulty kuerzen jeden Run auf so viele Samples.
    Gebraucht wird das nur im Testsplit, um die Fenster an das Training
    anzugleichen (siehe classify.test_spectra).

    Rueckgabe: DataFrame mit 'faultNumber', 'simulationRun', den
    Spektrumsspalten und ggf. Extraspalten (ICA: 'converged').
    """
    spec = get(method)
    name = spec["label"]
    mode = spec.get("forced_scaling") or scaling_mode
    if spec.get("forced_scaling") and scaling_mode != spec["forced_scaling"]:
        print(f"Hinweis: {name} arbeitet immer auf "
              f"scaling_mode='{spec['forced_scaling']}' "
              f"(uebergeben wurde '{scaling_mode}').")

    params = dict(dyca_m=dyca_m, dyca_n=dyca_n, dpca_lags=dpca_lags,
                  cva_past=cva_past, cva_fut=cva_fut, ridge_rel=ridge_rel,
                  ica_n=ica_n, ica_max_iter=ica_max_iter, ica_tol=ica_tol,
                  ica_random_state=ica_random_state,
                  ff_by_run=ff_by_run or {})
    limit = min_samples(method, dyca_m, dyca_n, cva_past, cva_fut)

    iterator = tqdm(df_all.groupby(["faultNumber", "simulationRun"],
                                   sort=True),
                    desc=f"Berechne {name}-Spektrum pro Run")

    records, first_errors = [], []
    n_err = n_skip = 0

    for (fault, run), group in iterator:
        # Pre-Fault verwerfen - nur bei echten Faults. Der Fehler wird
        # 1 h nach Simulationsstart injiziert (20 Samples bei 3-min-
        # Sampling); die ersten Samples sind effektiv Normalbetrieb und
        # wuerden die Fault-Statistik verwaessern.
        if fault != 0:
            group = group[group["sample"] >= pre_fault_cutoff]
        if limit is not None and len(group) < limit:
            n_skip += 1
            continue

        try:
            # Defensive Sortierung: df_all ist zwar global nach
            # (fault, run, sample) sortiert, aber DyCA und CVA brauchen die
            # zeitliche Ordnung zwingend - der Sort ist auf <= 500 Zeilen
            # billig und macht die Funktion unabhaengig vom Aufrufer.
            g = group.sort_values("sample")
            head = head_fault0 if fault == 0 else head_faulty
            if head is not None:
                g = g.head(head)
            X = scale(g[PROC_COLS].values, mode, scaler)
            out = spec["apply"](X, fault=int(fault), run=int(run), **params)
        except Exception as exc:
            # Ein einzelner numerisch gescheiterter Run darf eine Schleife
            # ueber 10 500 Laeufe nicht abbrechen - gezaehlt und weiter.
            n_err += 1
            if len(first_errors) < 5:
                first_errors.append(f"  fault={fault}, run={run}: {exc}")
            continue

        values, extras = out if isinstance(out, tuple) else (out, {})
        row = {"faultNumber": fault, "simulationRun": run}
        if spec.get("scalar"):
            row[spec["prefix"]] = float(values[0])
        else:
            for k, val in enumerate(values, start=1):
                row[f"{spec['prefix']}{k}"] = float(val)
        row.update(extras)
        records.append(row)

    if not records:
        # Ein Lauf ohne ein einziges Ergebnis ist nie beabsichtigt - frueher
        # lief das Notebook stumm weiter und exportierte eine leere CSV.
        raise RuntimeError(
            f"{name}: KEIN einziger Run erfolgreich "
            f"({n_err} Fehler, {n_skip} uebersprungen). "
            + ("Erste Fehler:\n" + "\n".join(first_errors)
               if first_errors else
               "Alle Runs waren kuerzer als das Minimum."))

    df = pd.DataFrame.from_records(records)
    if not df.empty and not spec.get("scalar"):
        # Spalten nach Komponentenindex sortieren, damit die Reihenfolge
        # nicht von der Einfuegereihenfolge abhaengt.
        pre = spec["prefix"]
        cols = sorted([c for c in df.columns if c.startswith(pre)],
                      key=lambda c: int(c[len(pre):]))
        df = df[["faultNumber", "simulationRun"] + cols
                + list(spec.get("extra_cols", ()))]

    if verbose:
        print(f"Erfolgreiche Runs: {len(df)}")
        if limit is not None:
            print(f"Uebersprungen   : {n_skip} (weniger als {limit} Samples)")
        print(f"Fehler          : {n_err}")
        if "converged" in df.columns:
            n_conv = int(df["converged"].sum())
            print(f"FastICA konvergiert: {n_conv} / {len(df)} "
                  f"({100 * n_conv / max(len(df), 1):.1f} %)")
        if first_errors:
            print("Erste Fehler:")
            for line in first_errors:
                print(line)
    return df
