"""TSFresh-Pipeline: Features aus projizierten TEP-Zeitreihen klassifizieren.

Ein Notebook besteht nur noch aus seiner Konfiguration, seinen
Projektions-Specs und den Aufrufen:

    from tep.tsfresh import Pipeline, PipelineConfig

    CFG = PipelineConfig(
        configs=[("raw",), ("pca", 6), ("dyca", 6, 12)],
        label="PCA/DyCA",
        summary_csv="tsfresh_summary.csv",
        cm_pred_csv="tsfresh_cm_predictions.csv",
        scaling_mode="global_mean",
    )
    pipe = Pipeline(CFG); pipe.describe()
    pipe.run_phase_a(); pipe.run_phase_b(); pipe.run_phase_c()
    cmp = pipe.compare(); pipe.plot_comparison(cmp)
    cm = pipe.confusion(); pipe.plot_confusions(cm)

Module
------
config       PipelineConfig
projections  Registry der Verfahren (raw/pca/dyca/dpca/cva/ica/dycvda)
features     Chunk-Cache-Extraktion, Feature-Ranking
pipeline     Phase A/B/C
reporting    Vergleichstabellen und Balkenplot
confusion    Confusion-Matrizen und ihre drei Plots

Spaltennamen, Cutoffs und die Vorverarbeitung kommen aus `tep.core` und
werden mit `tep.eigen` geteilt.
"""

from ..core import (LABELS, META_COLS, PRE_FAULT_CUTOFF, PROC_COLS,
                    SPLIT_FILES, labels_from_index, run_id, versions)
from .config import PipelineConfig
from .confusion import ConfusionResults, confusion
from .confusion import plot_detail as plot_confusion_detail
from .confusion import plot_grid as plot_confusion_grid
from .confusion import plot_recall
from .data import load_runs
from .features import extract_config, rank_features
from .pipeline import Pipeline
from .projections import (PROJECTORS, Projector, channel_names, config_name,
                          n_channels, project, register)
from .reporting import Comparison, compare, plot_comparison

__all__ = [
    "PipelineConfig", "Pipeline",
    "PROC_COLS", "META_COLS", "SPLIT_FILES", "PRE_FAULT_CUTOFF", "LABELS",
    "load_runs", "run_id", "labels_from_index",
    "PROJECTORS", "Projector", "register", "project", "config_name",
    "channel_names", "n_channels",
    "extract_config", "rank_features",
    "compare", "plot_comparison", "Comparison",
    "confusion", "ConfusionResults", "plot_confusion_grid",
    "plot_confusion_detail", "plot_recall",
    "versions",
]
