"""Aggregation je Fehlerklasse und Export der Spektren-CSV."""

from __future__ import annotations

import pandas as pd

from .config import SpectrumConfig


def aggregate(cfg: SpectrumConfig, per_run: pd.DataFrame,
              verbose: bool = True) -> pd.DataFrame:
    """Pro Fehlerklasse Mittelwert UND Standardabweichung jedes Werts.

    Aus einer Zeile je (Fault, Run) wird eine Zeile je Fault mit den
    Spalten `<prefix><i>_mean` und `<prefix><i>_std`. pandas rechnet die
    Std mit ddof=1 (Stichproben-Std), passend fuer N = 500 Runs.
    """
    spec = cfg.spectrum
    # Spektrumsspalten nach Komponentenindex sortiert; skalare Verfahren
    # (LDA) haben genau eine, die den Praefix selbst traegt.
    cols = ([spec.prefix] if spec.scalar else
            sorted([c for c in per_run.columns if c.startswith(spec.prefix)],
                   key=lambda c: int(c[len(spec.prefix):])))

    agg = per_run.groupby("faultNumber")[cols].agg(["mean", "std"])
    # MultiIndex ('dyca_1', 'mean') -> 'dyca_1_mean' flachklopfen.
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()

    if verbose:
        print(f"Aggregierte {spec.label}-Statistiken pro Fault: {agg.shape}")
    return agg


def export(cfg: SpectrumConfig, per_run: pd.DataFrame,
           verbose: bool = True) -> str:
    """Schreibt die Trainings-Spektren als CSV neben die Notebooks.

    Stellt sie damit fuer die Klassifikation bereit. Die TEST-Spektren
    werden bewusst NICHT hier berechnet, sondern erst im
    Klassifikations-Notebook - dieses Notebook bleibt training-only.
    """
    path = cfg.data_path(cfg.csv_name())
    per_run.to_csv(path, index=False)
    if verbose:
        print(f"{cfg.label}-Train gespeichert: {per_run.shape} -> {path}")
    return path
