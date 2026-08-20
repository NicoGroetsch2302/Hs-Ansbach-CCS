"""Die Spektren-Verfahren als Registry.

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

Eigene Verfahren kommen ueber `register()` dazu.

WICHTIG - die Spaltennamen und die CSV-Namen sind eingefroren:
`LazyClassifier_PCA_DyCA` liest die exportierten Dateien.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA, FastICA
from sklearn.exceptions import ConvergenceWarning

from ..core import PROC_COLS, inv_sqrt_psd, lag_stack, scale

try:
    from tqdm.auto import tqdm
    TQDM = True
except ImportError:                                    # pragma: no cover
    TQDM = False


# =========================================================================
# Registry
# =========================================================================

@dataclass(frozen=True)
class Spectrum:
    """Ein Spektrum-Verfahren.

    prefix      Spaltenpraefix der Werte ("lambda_", "dyca_", ...)
    label       Klartextname fuer Plottitel ("PCA", "DyCA", ...)
    csv_stem    Dateiname-Stamm des Exports ("pca_eigenvalues")
    apply       (X_scaled, ctx) -> Wertevektor, oder (Vektor, extras-dict)
    scalar      True, wenn das Verfahren EINE Zahl je Run liefert (LDA)
    min_samples cfg -> Mindestlaenge eines Runs; kuerzere werden
                uebersprungen. None = keine Pruefung.
    forced_scaling  erzwingt einen Skalierungsmodus (LDA: "scaler")
    extra_cols  zusaetzliche Spalten, die apply() im extras-dict liefert
    """
    prefix: str
    label: str
    csv_stem: str
    apply: Callable
    scalar: bool = False
    min_samples: Callable | None = None
    forced_scaling: str | None = None
    extra_cols: tuple = ()


SPECTRA: dict = {}


def register(method: str, spectrum: Spectrum) -> None:
    """Traegt ein Verfahren unter `method` ein (ueberschreibt vorhandene)."""
    SPECTRA[method] = spectrum


def get(method: str) -> Spectrum:
    if method not in SPECTRA:
        raise ValueError(f"Unbekanntes Verfahren: {method!r} "
                         f"(bekannt: {sorted(SPECTRA)})")
    return SPECTRA[method]


@dataclass
class Context:
    """Was ein Verfahren ausser den Messdaten noch braucht."""
    cfg: object
    scaler: object = None
    ff_by_run: dict = field(default_factory=dict)
    fault: int = 0
    run: int = 0


# =========================================================================
# Die Verfahren
# =========================================================================

def _apply_pca(X, ctx):
    # n_components = Anzahl Prozessvariablen -> ALLE Eigenwerte, auch die
    # kleinen aus dem Noise-Subspace. svd_solver="full" garantiert das
    # vollstaendige Spektrum, absteigend sortiert und reell.
    pca = PCA(n_components=len(PROC_COLS), svd_solver="full")
    pca.fit(X)
    # explained_variance_ratio_ = auf die Gesamtvarianz NORMIERTE
    # Eigenwerte (Summe je Run = 1). Fuer die unnormierten waere
    # explained_variance_ zu nehmen.
    return pca.explained_variance_ratio_


register("pca", Spectrum(
    prefix="lambda_", label="PCA", csv_stem="pca_eigenvalues",
    apply=_apply_pca,
))


def _apply_dyca(X, ctx):
    from dyca import dyca
    res = dyca(X, m=ctx.cfg.dyca_m, n=ctx.cfg.dyca_n)
    return np.asarray(res["generalized_eigenvalues"], dtype=float)


register("dyca", Spectrum(
    prefix="dyca_", label="DyCA", csv_stem="dyca_eigenvalues",
    apply=_apply_dyca,
    min_samples=lambda cfg: max(cfg.dyca_m, cfg.dyca_n) + 5,
))


def _apply_dpca(X, ctx):
    Z = lag_stack(X, ctx.cfg.dpca_lags)
    pca = PCA(n_components=None, svd_solver="full")
    pca.fit(Z)
    return pca.explained_variance_ratio_


register("dpca", Spectrum(
    prefix="dpca_", label="DPCA", csv_stem="dpca_eigenvalues",
    apply=_apply_dpca,
))


def _apply_cva(X, ctx):
    """Kanonische Korrelationen zwischen Vergangenheits- und
    Zukunftsstapel: H = S_ff^(-1/2) S_fp S_pp^(-1/2), die Singulaerwerte
    von H sind die Korrelationen und liegen in [0, 1]."""
    cfg = ctx.cfg
    X = np.asarray(X, dtype=np.float64)
    past, fut, ridge = cfg.cva_past, cfg.cva_fut, cfg.ridge_rel
    T = X.shape[0]
    P = np.hstack([X[past - j: T - fut + 1 - j] for j in range(1, past + 1)])
    F = np.hstack([X[past + j: T - fut + 1 + j] for j in range(fut)])
    N = P.shape[0]
    Pc = P - P.mean(axis=0)
    Fc = F - F.mean(axis=0)
    Spp = Pc.T @ Pc / (N - 1)
    Sff = Fc.T @ Fc / (N - 1)
    Sfp = Fc.T @ Pc / (N - 1)
    H = inv_sqrt_psd(Sff, ridge) @ Sfp @ inv_sqrt_psd(Spp, ridge)
    corrs = np.linalg.svd(H, compute_uv=False)
    # Floating-Point-Rauschen kann winzige negative Werte bzw. Werte
    # knapp ueber 1 erzeugen - beides ist keine echte Korrelation.
    return np.clip(corrs, 0.0, 1.0)


register("cva", Spectrum(
    prefix="cva_", label="CVA", csv_stem="cva_eigenvalues",
    apply=_apply_cva,
    min_samples=lambda cfg: len(PROC_COLS) + cfg.cva_past + cfg.cva_fut + 5,
))


def _apply_ica(X, ctx):
    """Nicht-Gaussianitaets-Spektrum: Kurtosis der FastICA-Quellen, nach
    Betrag absteigend sortiert. Ob FastICA konvergiert ist, wird als
    eigene Spalte mitgefuehrt."""
    cfg = ctx.cfg
    X = np.asarray(X, dtype=np.float64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ica = FastICA(n_components=cfg.ica_n, whiten="unit-variance",
                      max_iter=cfg.ica_max_iter, tol=cfg.ica_tol,
                      random_state=cfg.ica_random_state)
        S = ica.fit_transform(X)
    converged = not any(issubclass(w.category, ConvergenceWarning)
                        for w in caught)
    kurt = stats.kurtosis(S, axis=0, fisher=True, bias=True)
    order = np.argsort(-np.abs(kurt))
    return kurt[order], {"converged": int(converged)}


register("ica", Spectrum(
    prefix="ica_", label="ICA", csv_stem="ica_eigenvalues",
    apply=_apply_ica, extra_cols=("converged",),
))


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


def _apply_lda(X, ctx):
    """Jeder Lauf wird gegen einen Normalbetriebslauf gestellt.

    Fault != 0 : Fehlerlauf gegen den FaultFree-Lauf gleicher Run-Nummer.
    Fault == 0 : FaultFree-Lauf gegen seinen Nachbarlauf - sonst waere die
                 Referenzklasse mit sich selbst verglichen und lambda
                 trivialerweise null.
    """
    ff = ctx.ff_by_run
    run_ids = sorted(ff)
    if ctx.fault == 0:
        X1 = ff[ctx.run]
        X0 = ff[run_ids[(run_ids.index(ctx.run) + 1) % len(run_ids)]]
    else:
        X1 = X
        X0 = ff[ctx.run]
    return np.array([lda_eigenvalue(X0, X1, ctx.cfg.ridge_rel)])


register("lda", Spectrum(
    prefix="lda_eigenvalue", label="LDA", csv_stem="lda_eigenvalues",
    apply=_apply_lda, scalar=True,
    # LDA vergleicht ZWEI Laeufe. Bei "global_mean" zoege jeder Lauf
    # seinen eigenen skalaren Mittelwert ab - die Mittelwertsdifferenz d,
    # also genau das Signal, waere dann verfaelscht.
    forced_scaling="scaler",
))


# =========================================================================
# Die gemeinsame Schleife
# =========================================================================

def run_spectra(cfg, df_all, scaler=None, ff_by_run=None,
                verbose: bool = True) -> pd.DataFrame:
    """Berechnet das Spektrum je (faultNumber, simulationRun).

    Rueckgabe: DataFrame mit 'faultNumber', 'simulationRun', den
    Spektrumsspalten und ggf. Extraspalten (ICA: 'converged').
    """
    spec = get(cfg.method)
    mode = spec.forced_scaling or cfg.scaling_mode
    if spec.forced_scaling and cfg.scaling_mode != spec.forced_scaling:
        print(f"Hinweis: {spec.label} arbeitet immer auf "
              f"scaling_mode='{spec.forced_scaling}' "
              f"(cfg sagt '{cfg.scaling_mode}').")

    ctx = Context(cfg=cfg, scaler=scaler, ff_by_run=ff_by_run or {})
    limit = spec.min_samples(cfg) if spec.min_samples else None

    grouped = df_all.groupby(["faultNumber", "simulationRun"], sort=True)
    iterator = tqdm(grouped, desc=f"Berechne {spec.label}-Spektrum pro Run") \
        if TQDM else grouped

    records, first_errors = [], []
    n_err = n_skip = 0

    for (fault, run), group in iterator:
        # Pre-Fault verwerfen - nur bei echten Faults. Der Fehler wird
        # 1 h nach Simulationsstart injiziert (20 Samples bei 3-min-
        # Sampling); die ersten Samples sind effektiv Normalbetrieb und
        # wuerden die Fault-Statistik verwaessern.
        if cfg.drop_pre_fault and fault != 0:
            group = group[group["sample"] >= cfg.pre_fault_cutoff]
        if limit is not None and len(group) < limit:
            n_skip += 1
            continue

        ctx.fault, ctx.run = int(fault), int(run)
        try:
            # Defensive Sortierung: df_all ist zwar global nach
            # (fault, run, sample) sortiert, aber DyCA und CVA brauchen die
            # zeitliche Ordnung zwingend - der Sort ist auf <= 500 Zeilen
            # billig und macht die Funktion unabhaengig vom Aufrufer.
            g = group.sort_values("sample")
            # Fensterangleichung (nur im Testsplit gesetzt): auf dieselbe
            # Anzahl Samples kuerzen, die das Training je Fault abdeckt.
            head = cfg.head_fault0 if fault == 0 else cfg.head_faulty
            if head is not None:
                g = g.head(head)
            X = scale(g[PROC_COLS].values, mode, scaler)
            out = spec.apply(X, ctx)
        except Exception as exc:
            # Ein einzelner numerisch gescheiterter Run darf eine Schleife
            # ueber 10 500 Laeufe nicht abbrechen - gezaehlt und weiter.
            n_err += 1
            if len(first_errors) < 5:
                first_errors.append(f"  fault={fault}, run={run}: {exc}")
            continue

        values, extras = out if isinstance(out, tuple) else (out, {})
        row = {"faultNumber": fault, "simulationRun": run}
        if spec.scalar:
            row[spec.prefix] = float(values[0])
        else:
            for k, val in enumerate(values, start=1):
                row[f"{spec.prefix}{k}"] = float(val)
        row.update(extras)
        records.append(row)

    if not records:
        # Ein Lauf ohne ein einziges Ergebnis ist nie beabsichtigt - frueher
        # lief das Notebook stumm weiter und exportierte eine leere CSV.
        raise RuntimeError(
            f"{spec.label}: KEIN einziger Run erfolgreich "
            f"({n_err} Fehler, {n_skip} uebersprungen). "
            + ("Erste Fehler:\n" + "\n".join(first_errors)
               if first_errors else
               "Alle Runs waren kuerzer als das Minimum."))

    df = pd.DataFrame.from_records(records)
    if not df.empty and not spec.scalar:
        # Spalten nach Komponentenindex sortieren, damit die Reihenfolge
        # nicht von der Einfuegereihenfolge abhaengt.
        cols = sorted([c for c in df.columns if c.startswith(spec.prefix)],
                      key=lambda c: int(c[len(spec.prefix):]))
        df = df[["faultNumber", "simulationRun"] + cols
                + list(spec.extra_cols)]

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
