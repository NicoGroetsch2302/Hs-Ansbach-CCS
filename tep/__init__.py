"""Gemeinsamer Unterbau der TEP-Notebooks (Tennessee Eastman Process).

Zwei Notebook-Familien, ein Fundament:

    tep.core     Spaltennamen, Splits, Cutoffs, Vorverarbeitung und die
                 linearalgebraischen Bausteine. Alles, was beide Familien
                 brauchen und was nicht in zwei Fassungen driften darf.

    tep.tsfresh  Fehler-KLASSIFIKATION: TEP-Runs projizieren, TSFresh-
                 Features extrahieren, auswaehlen, Modelle vergleichen,
                 Confusion-Matrizen.

    tep.eigen    Fehler-CHARAKTERISIERUNG: Eigenwert- bzw.
                 Nicht-Gaussianitaets-Spektren pro Run, aggregiert je
                 Fehlerklasse, als Spektren- und Balkenplots. Dazu die
                 Klassifikation auf den exportierten Spektren.

Die Notebooks importieren aus den Unterpaketen, nicht von hier:

    from tep.tsfresh import Pipeline, PipelineConfig
    from tep.eigen import SpectrumConfig, SpectrumRun

Nur die wirklich gemeinsamen Namen liegen auch direkt auf `tep`.
"""

from .core import (LABELS, META_COLS, PRE_FAULT_CUTOFF, PROC_COLS,
                   SCALING_MODES, SPLIT_FILES, XMEAS_COLS, XMV_COLS,
                   dyca_amplitudes, fit_scaler, flip_signs, inv_sqrt_psd,
                   labels_from_index, lag_stack, run_id, scale, versions)

__all__ = [
    "PROC_COLS", "XMEAS_COLS", "XMV_COLS", "META_COLS", "SPLIT_FILES",
    "PRE_FAULT_CUTOFF", "LABELS", "SCALING_MODES",
    "run_id", "labels_from_index",
    "fit_scaler", "scale",
    "flip_signs", "lag_stack", "inv_sqrt_psd", "dyca_amplitudes",
    "versions",
]
