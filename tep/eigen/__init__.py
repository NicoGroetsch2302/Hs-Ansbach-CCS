"""Eigenwert- und Spektrumsanalyse der TEP-Laeufe.

Statt zu klassifizieren wird hier CHARAKTERISIERT: was macht eine
Fehlerklasse mit dem Spektrum eines Laufs? Pro (faultNumber,
simulationRun) entsteht ein Wertevektor, der je Fehlerklasse ueber die
500 Runs gemittelt wird.

Ein Notebook besteht damit nur noch aus Konfiguration und Aufrufen:

    from tep.eigen import SpectrumConfig, load_train, merge_faults
    from tep.eigen import run_spectra, aggregate, export, plot_means

    CFG = SpectrumConfig(method="dyca", scaling_mode="global_mean",
                         dyca_m=2, dyca_n=4)

    df_ff, df_faulty = load_train(CFG)
    df_all = merge_faults(df_ff, df_faulty)
    per_run = run_spectra(CFG, df_all, scaler=fit_scaler(CFG))
    agg = aggregate(CFG, per_run)
    plot_means(CFG, agg)
    export(CFG, per_run)

Module
------
config     SpectrumConfig - alle Stellschrauben
data       Trainings-CSVs lesen und zusammenfuehren
spectra    Registry der Verfahren (pca/dyca/dpca/cva/ica/lda)
aggregate  Mittelwert/Std je Fehlerklasse, CSV-Export
plots      die vier Standardplots plus die LDA-Variante
classify   Klassifikation auf den exportierten Spektren

Spaltennamen, Cutoffs und die Vorverarbeitung kommen aus `tep.core` und
werden mit `tep.tsfresh` geteilt.
"""

from ..core import LABELS, META_COLS, PRE_FAULT_CUTOFF, PROC_COLS, versions
from ..core import fit_scaler as _core_fit_scaler
from .aggregate import aggregate, export
from .config import SpectrumConfig
from .data import faultfree_by_run, load_train, merge_faults
from .plots import (plot_bars, plot_cv, plot_dyca_mn_estimate, plot_means,
                    plot_scalar, plot_stds, transform)
from .spectra import SPECTRA, Spectrum, get, run_spectra

__all__ = [
    "SpectrumConfig",
    "load_train", "merge_faults", "faultfree_by_run", "fit_scaler",
    "run_spectra", "SPECTRA", "Spectrum", "get",
    "aggregate", "export",
    "plot_means", "plot_stds", "plot_cv", "plot_bars",
    "plot_scalar", "plot_dyca_mn_estimate", "transform",
    "PROC_COLS", "META_COLS", "PRE_FAULT_CUTOFF", "LABELS", "versions",
]


def fit_scaler(cfg: SpectrumConfig, verbose: bool = True):
    """StandardScaler fitten - oder None, wenn er nicht gebraucht wird."""
    if not cfg.needs_scaler:
        return None
    return _core_fit_scaler(cfg.data_path("TEP_FaultFree_Training.csv"),
                            verbose=verbose)
