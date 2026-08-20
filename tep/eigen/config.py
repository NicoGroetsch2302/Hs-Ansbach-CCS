"""Konfiguration eines Eigenwert-/Spektrum-Laufs.

Ein `SpectrumConfig` haelt alles, was frueher als lose Modulvariablen in
der Parameterzelle jedes Eigenwert-Notebooks stand. Die Defaults sind exakt
die Werte der bisherigen Notebooks - ein Notebook setzt nur noch, was es
wirklich unterscheidet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..core import PRE_FAULT_CUTOFF, SCALING_MODES


@dataclass
class SpectrumConfig:
    """Alle Stellschrauben eines Spektrum-Laufs.

    method : "pca" | "dyca" | "dpca" | "cva" | "ica" | "lda"
             Welche Verfahren es gibt, steht in `spectra.SPECTRA`.

    scaling_mode : Vorverarbeitung jedes Runs vor der Zerlegung.
        "global_mean" X - X.mean(), skalarer Gesamtmittelwert. Die
                      Rohvarianzen bleiben erhalten, varianzstarke
                      Variablen dominieren das Spektrum.
        "scaler"      spaltenweise Standardisierung in Normalbetriebs-
                      Einheiten (Fit auf TEP_FaultFree_Training).
        LDA ist die Ausnahme: es vergleicht zwei Laeufe miteinander und
        arbeitet deshalb immer auf "scaler" (siehe spectra.py).

    Der Skalierungsmodus steckt im Namen der Export-CSV, ein Umschalten
    ueberschreibt also keine alten Ergebnisse.
    """

    method: str
    scaling_mode: str = "global_mean"
    data_dir: str = "."

    # --- Datenumfang ---
    pre_fault_cutoff: int = PRE_FAULT_CUTOFF["train"]
    runs_per_fault: int | None = None   # None = alle 500 Runs je Fault

    # --- Verfahrensparameter ---
    # DyCA: m lineare Komponenten, n Dimension des det. Systems.
    # Randbedingungen aus dyca_internal._input_check: n >= m und m >= n - m.
    # Die Werte SOLLTEN aus den FaultFree-Daten geschaetzt werden -
    # dafuer gibt es plots.plot_dyca_mn_estimate().
    dyca_m: int = 2
    dyca_n: int = 4
    dpca_lags: int = 2                  # 52*(L+1) = 156 Spalten
    cva_past: int = 1
    cva_fut: int = 1
    ridge_rel: float = 1e-6             # CVA und LDA
    ica_n: int = 12
    ica_max_iter: int = 1000
    ica_tol: float = 1e-3
    ica_random_state: int = 42

    # --- Test-Split (nur fuer die Klassifikation gebraucht) ---
    # Im Test wird der Fehler nach 8 h injiziert -> erstes Post-Fault-
    # Sample 161. Ausserdem muessen die Fenster an das Training
    # angeglichen werden: dort deckt ein Run 500 (Fault 0) bzw. 480
    # (Fault != 0) Samples ab, im Test waeren es 960 bzw. 800. Ohne
    # Kuerzung waeren Train- und Test-Spektren systematisch verschieden.
    test_pre_fault_cutoff: int = PRE_FAULT_CUTOFF["test"]
    test_head_fault0: int | None = 500
    test_head_faulty: int | None = 480

    # --- Fensterkuerzung im aktuellen Lauf (von as_test gesetzt) ---
    head_fault0: int | None = None
    head_faulty: int | None = None

    # --- Darstellung ---
    k_max: int = 10                     # Komponenten in den Linienplots
    k_bar: int = 10                     # Komponenten im Balkenplot
    plot_mode: str = "linear"           # "linear" | "log" | "relative"
    plot_mode_cv: str = "linear"        # eigener Modus fuer den CV-Plot
    ncols: int = 6                      # Subplots je Zeile

    def __post_init__(self):
        if self.scaling_mode not in SCALING_MODES:
            raise ValueError(f"scaling_mode={self.scaling_mode!r} unbekannt")
        from .spectra import get                 # spaet: Zyklus vermeiden
        get(self.method)                         # wirft bei unbekanntem Namen

    # --- abgeleitet ---
    @property
    def spectrum(self):
        """Der `Spectrum`-Eintrag zu `method`."""
        from .spectra import get
        return get(self.method)

    @property
    def prefix(self) -> str:
        """Spaltenpraefix der Spektrumswerte, z.B. "lambda_" oder "dyca_"."""
        return self.spectrum.prefix

    @property
    def label(self) -> str:
        """Klartextname fuer Plottitel, z.B. "DyCA"."""
        return self.spectrum.label

    @property
    def needs_scaler(self) -> bool:
        """Ob ein StandardScaler gefittet werden muss: bei
        scaling_mode="scaler" und immer bei LDA (erzwingt ihn)."""
        return (self.scaling_mode == "scaler"
                or self.spectrum.forced_scaling == "scaler")

    def csv_name(self, split: str = "train") -> str:
        """Dateiname der Export-CSV.

        Die Namen sind eingefroren - `LazyClassifier_PCA_DyCA` liest sie.
        """
        suffix = "" if self.scaling_mode == "global_mean" else "_scaler"
        return f"{self.spectrum.csv_stem}_{split}{suffix}.csv"

    def data_path(self, fname: str) -> str:
        return os.path.join(self.data_dir, fname)

    def as_test(self) -> "SpectrumConfig":
        """Dieselbe Konfiguration, aber fuer den Testsplit: spaeterer
        Cutoff und die Fensterangleichung an das Training."""
        import dataclasses
        return dataclasses.replace(
            self,
            pre_fault_cutoff=self.test_pre_fault_cutoff,
            head_fault0=self.test_head_fault0,
            head_faulty=self.test_head_faulty)
