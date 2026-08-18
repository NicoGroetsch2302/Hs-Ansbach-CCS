"""Aggregation je Fehlerklasse und Export/Import der Spektren-CSV."""

from __future__ import annotations

import os

import pandas as pd

from .config import SpectrumConfig
from .spectra import get


def aggregate(cfg: SpectrumConfig, per_run: pd.DataFrame,
              verbose: bool = True) -> pd.DataFrame:
    """Pro Fehlerklasse Mittelwert UND Standardabweichung jedes Werts.

    Aus einer Zeile je (Fault, Run) wird eine Zeile je Fault mit den
    Spalten `<prefix><i>_mean` und `<prefix><i>_std`. pandas rechnet die
    Std mit ddof=1 (Stichproben-Std), passend fuer N = 500 Runs.
    """
    spec = get(cfg.method)
    cols = value_columns(cfg, per_run)

    agg = per_run.groupby("faultNumber")[cols].agg(["mean", "std"])
    # MultiIndex ('dyca_1', 'mean') -> 'dyca_1_mean' flachklopfen.
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()

    if verbose:
        print(f"Aggregierte {spec.label}-Statistiken pro Fault: {agg.shape}")
    return agg


def value_columns(cfg: SpectrumConfig, df: pd.DataFrame) -> list:
    """Die Spektrumsspalten eines per-Run-DataFrames, nach Index sortiert."""
    spec = get(cfg.method)
    if spec.scalar:
        return [spec.prefix]
    cols = [c for c in df.columns if c.startswith(spec.prefix)]
    return sorted(cols, key=lambda c: int(c[len(spec.prefix):]))


def export(cfg: SpectrumConfig, per_run: pd.DataFrame,
           verbose: bool = True) -> str:
    """Schreibt die Trainings-Spektren als CSV neben die Notebooks.

    Stellt sie damit fuer die Klassifikation bereit. Die TEST-Spektren
    werden bewusst NICHT hier berechnet, sondern erst im
    Klassifikations-Notebook - dieses Notebook bleibt training-only.
    """
    path = cfg.data_path(cfg.csv_name)
    per_run.to_csv(path, index=False)
    if verbose:
        print(f"{cfg.label}-Train gespeichert: {per_run.shape} -> {path}")
    return path


def load(cfg: SpectrumConfig, verbose: bool = True) -> pd.DataFrame:
    """Liest die zuvor exportierte CSV zurueck.

    Damit laufen die Aggregations- und Plotzellen nach einem
    Kernel-Neustart ohne die teure Spektren-Schleife.
    """
    path = cfg.data_path(cfg.csv_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} fehlt -> zuerst run_spectra() und export() ausfuehren.")
    df = pd.read_csv(path)
    if verbose:
        print(f"{cfg.label}-Spektren geladen: {df.shape} aus {path}")
    return df
