"""TSFresh-Pipeline: Features aus projizierten TEP-Zeitreihen klassifizieren.

Ein Notebook ist ein gerader Ablauf aus Funktionsaufrufen; die
Einstellungen stehen als Argumente daran, nicht in einem Konfigurations-
objekt:

    from tep.tsfresh import (cache_dir, compare, confusion, describe,
                             phase_a, phase_b, phase_c, plot_comparison,
                             plot_confusion_grid, validate)

    CONFIGS = [("raw",), ("pca", 6), ("dyca", 6, 12)]
    SCALING, DATA_DIR, TOP_K = "global_mean", "data_csv", 100
    CACHE = cache_dir(SCALING)
    NAMES = validate(CONFIGS)

    describe(CONFIGS, CACHE, top_k=TOP_K, scaling_mode=SCALING)
    train_top, top_names = phase_a(CONFIGS, CACHE, data_dir=DATA_DIR,
                                   top_k=TOP_K, scaling_mode=SCALING)
    test_top = phase_b(CONFIGS, CACHE, top_names, data_dir=DATA_DIR,
                       top_k=TOP_K, scaling_mode=SCALING)
    summary, boards = phase_c(CONFIGS, train_top, test_top, SUMMARY_CSV)

    cmp = compare(summary, NAMES); plot_comparison(cmp)
    cm = confusion(NAMES, PRED_CSV, train_top, test_top)
    plot_confusion_grid(cm, TOP_K)

Module
------
data         Rohdaten laden
projections  Registry der Verfahren (raw/pca/dyca/dpca/cva/ica/dycvda)
features     Cache-Ordner, Chunk-Extraktion, Feature-Ranking
pipeline     Phase A/B/C, gemeinsame Run-Menge
reporting    Vergleichstabellen und Balkenplot
confusion    Confusion-Matrizen und ihre drei Plots

Spaltennamen, Cutoffs und die Vorverarbeitung kommen aus `tep.core` und
werden mit `tep.eigen` geteilt.
"""

from ..core import (LABELS, META_COLS, PRE_FAULT_CUTOFF, PROC_COLS,
                    SPLIT_FILES, labels_from_index, run_id, versions)
from .confusion import best, confusion, names, recall_table
from .confusion import plot_detail as plot_confusion_detail
from .confusion import plot_grid as plot_confusion_grid
from .confusion import plot_recall
from .data import fit_scaler, load_runs
from .features import cache_dir, extract_config, fc_parameters, rank_features
from .pipeline import (common_runs, describe, load_summary, matrices, phase_a,
                       phase_b, phase_c)
from .projections import (PROJECTORS, channel_names, config_name, n_channels,
                          project, validate)
from .reporting import compare, plot_comparison

__all__ = [
    "cache_dir", "fc_parameters", "validate", "describe", "fit_scaler",
    "phase_a", "phase_b", "phase_c", "common_runs", "matrices",
    "load_summary",
    "PROC_COLS", "META_COLS", "SPLIT_FILES", "PRE_FAULT_CUTOFF", "LABELS",
    "load_runs", "run_id", "labels_from_index",
    "PROJECTORS", "project", "config_name", "channel_names", "n_channels",
    "extract_config", "rank_features",
    "compare", "plot_comparison",
    "confusion", "names", "best", "recall_table",
    "plot_confusion_grid", "plot_confusion_detail", "plot_recall",
    "versions",
]
