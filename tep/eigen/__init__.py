"""Eigenwert- und Spektrumsanalyse der TEP-Laeufe.

Statt zu klassifizieren wird hier CHARAKTERISIERT: was macht eine
Fehlerklasse mit dem Spektrum eines Laufs? Pro (faultNumber,
simulationRun) entsteht ein Wertevektor, der je Fehlerklasse ueber die
500 Runs gemittelt wird.

Ein Notebook ist damit ein gerader Ablauf aus Funktionsaufrufen; die
Einstellungen stehen als Argumente daran, nicht in einem Konfigurations-
objekt:

    from tep.eigen import (aggregate, export, fit_scaler, load_train,
                           merge_faults, plot_means, run_spectra)

    METHOD, SCALING, DATA_DIR = "dyca", "global_mean", "data_csv"

    df_ff, df_faulty = load_train(data_dir=DATA_DIR)
    df_all = merge_faults(df_ff, df_faulty)
    scaler = fit_scaler(METHOD, SCALING, DATA_DIR)
    per_run = run_spectra(df_all, METHOD, scaling_mode=SCALING,
                          scaler=scaler, dyca_m=2, dyca_n=4)
    agg = aggregate(per_run, METHOD)
    plot_means(agg, METHOD, k_max=25)
    export(per_run, METHOD, SCALING, DATA_DIR)

Module
------
data       Trainings-CSVs lesen, zusammenfuehren, Scaler fitten
spectra    Registry der Verfahren (pca/dyca/dpca/cva/ica/lda) als dicts
aggregate  Mittelwert/Std je Fehlerklasse, CSV-Export
plots      die vier Standardplots plus die LDA-Variante
classify   Klassifikation auf den exportierten Spektren

Spaltennamen, Cutoffs und die Vorverarbeitung kommen aus `tep.core` und
werden mit `tep.tsfresh` geteilt.
"""

from ..core import LABELS, META_COLS, PRE_FAULT_CUTOFF, PROC_COLS, versions
from .aggregate import aggregate, export, value_columns
from .data import faultfree_by_run, fit_scaler, load_train, merge_faults
from .plots import (plot_bars, plot_cv, plot_dyca_mn_estimate, plot_means,
                    plot_scalar, plot_stds, transform)
from .spectra import (SPECTRA, csv_name, get, label, min_samples,
                      needs_scaler, prefix, run_spectra)

__all__ = [
    "load_train", "merge_faults", "faultfree_by_run", "fit_scaler",
    "run_spectra", "SPECTRA", "get", "label", "prefix", "needs_scaler",
    "csv_name", "min_samples",
    "aggregate", "export", "value_columns",
    "plot_means", "plot_stds", "plot_cv", "plot_bars",
    "plot_scalar", "plot_dyca_mn_estimate", "transform",
    "PROC_COLS", "META_COLS", "PRE_FAULT_CUTOFF", "LABELS", "versions",
]
