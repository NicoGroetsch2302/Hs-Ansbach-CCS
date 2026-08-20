"""Konfiguration der TSFresh-Pipeline.

Ein einziges `PipelineConfig`-Objekt haelt alles, was frueher als lose
Modulvariablen in der Konfigurationszelle jedes Notebooks stand. Die
Defaults sind exakt die Werte der bisherigen Notebooks - ein Notebook
setzt nur noch, was es wirklich unterscheidet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Sequence

from tsfresh.feature_extraction.settings import (
    ComprehensiveFCParameters, EfficientFCParameters, MinimalFCParameters)

from ..core import SCALING_MODES

FC_MODES = {"minimal": MinimalFCParameters,
            "efficient": EfficientFCParameters,
            "comprehensive": ComprehensiveFCParameters}


@dataclass
class PipelineConfig:
    """Alle Stellschrauben eines TSFresh-Laufs.

    Pflichtfelder
    -------------
    configs : Liste von Projektions-Specs, z.B. ``[("raw",), ("pca", 6)]``.
              Welche Specs es gibt, steht in ``projections.PROJECTORS``.
    summary_csv, cm_pred_csv : Ergebnis-Dateinamen IM Cache-Ordner. Jedes
              Notebook braucht eigene, sonst ueberschreiben sich die
              Schwester-Notebooks gegenseitig.

    Cache
    -----
    ``cache_dir`` wird abgeleitet: ``tsfresh_cache`` (bzw. ``_smoke``),
    plus Suffix ``_{scaling_mode}``, falls nicht ``global_mean``. Der
    Ordner ist BEWUSST zwischen den Notebooks geteilt - die
    ``raw``-Konfiguration ist ueberall bitidentisch, ihre Chunks und die
    Top-K-Auswahl werden dadurch wiederverwendet.

    ``chunk_runs`` MUSS bei 250 bleiben, solange der Cache geteilt wird:
    die Chunk-Dateien sind ueber ihren Index an die Aufteilung gebunden,
    ein anderer Wert wuerde fremde Chunks falsch interpretieren.

    ``top_k`` dagegen darf geaendert werden: die Phase-B-Test-Chunks heissen
    seit 2026-08-18 ``{name}__test__top{top_k}__{idx}.pkl`` und kodieren den
    Wert. Ein anderer ``top_k`` extrahiert das Testset neu, statt die alten
    Chunks stumm mit 0.0 aufzufuellen.
    """

    # --- Pflicht ---
    configs: Sequence[tuple]
    summary_csv: str
    cm_pred_csv: str

    # Klartextname der Notebook-Familie, erscheint in Plottiteln
    # ("PCA/DyCA", "DPCA/CVA/ICA", "DyCVDA").
    label: str = ""

    # --- Vorverarbeitung ---
    # "global_mean": X - X.mean() (skalarer Gesamtmittelwert, die Zeile aus
    # den Eigenwert-Notebooks). "scaler": spaltenweise Standardisierung,
    # gefittet auf TEP_FaultFree_Training (Normalbetrieb).
    scaling_mode: str = "global_mean"

    # --- Datenumfang ---
    data_dir: str = "."
    runs_per_fault: int | None = None      # None = alle 500 Runs je Fault
    run_length: int | None = 480           # gemeinsame Laenge; None = aus

    # --- Feature-Extraktion und -Auswahl ---
    fc_mode: str = "efficient"             # minimal | efficient | comprehensive
    top_k: int = 100                       # Features je Konfiguration
    chunk_runs: int = 250                  # Runs je Chunk (Cache-Granularitaet)
    block_cols: int = 4000                 # Spalten je Block im Relevanztest
    n_jobs: int | None = None              # None -> os.cpu_count(); 0 = seriell

    # --- Modellvergleich ---
    lc_cv_folds: int = 5                   # Folds der Train-CV
    random_state: int = 42

    # --- Projektionsparameter (nur fuer die jeweiligen Verfahren) ---
    # ACHTUNG: Diese Werte stecken NICHT im Konfigurationsnamen und damit
    # nicht im Cache-Praefix. Bei Aenderung vorher die betroffenen Chunks
    # aus dem Cache-Ordner loeschen.
    dpca_lags: int = 2                     # DPCA: Lags L
    cva_past: int = 1                      # CVA: Vergangenheitsfenster p
    cva_fut: int = 1                       # CVA: Zukunftsfenster f
    cva_ridge_rel: float = 1e-6            # CVA/DyCVDA: relative Ridge
    ica_max_iter: int = 1000
    ica_tol: float = 1e-3
    ica_random_state: int = 42

    # --- Schnelldurchlauf ---
    # True -> winziger Lauf (Minuten) zum Pruefen der Pipeline. Eigener
    # Cache-Ordner, ueberschreibt also nichts.
    smoke_test: bool = False

    # --- abgeleitet, nicht selbst setzen ---
    cache_dir: str = field(init=False)
    fc_parameters: dict = field(init=False, repr=False)

    def __post_init__(self):
        if self.scaling_mode not in SCALING_MODES:
            raise ValueError(f"scaling_mode={self.scaling_mode!r} unbekannt")
        if self.fc_mode not in FC_MODES:
            raise ValueError(f"fc_mode={self.fc_mode!r} unbekannt")

        if self.smoke_test:
            self.runs_per_fault = 4
            self.fc_mode = "minimal"
            self.top_k = 20
            self.chunk_runs = 50

        if self.n_jobs is None:
            self.n_jobs = os.cpu_count() or 4

        self.cache_dir = ("tsfresh_cache_smoke" if self.smoke_test
                          else "tsfresh_cache")
        if self.scaling_mode != "global_mean":
            self.cache_dir += f"_{self.scaling_mode}"
        os.makedirs(self.cache_dir, exist_ok=True)

        self.fc_parameters = FC_MODES[self.fc_mode]()

    # --- Pfade ---
    def cache_path(self, *parts: str) -> str:
        return os.path.join(self.cache_dir, *parts)

    def data_path(self, fname: str) -> str:
        return os.path.join(self.data_dir, fname)

    @property
    def summary_path(self) -> str:
        return self.cache_path(self.summary_csv)

    @property
    def cm_pred_path(self) -> str:
        return self.cache_path(self.cm_pred_csv)
