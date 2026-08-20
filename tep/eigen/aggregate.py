"""Aggregation je Fehlerklasse und Export der Spektren-CSV."""

from __future__ import annotations

import os

import pandas as pd

from .spectra import csv_name, get


def value_columns(per_run: pd.DataFrame, method: str) -> list:
    """Die Spektrumsspalten eines per-Run-DataFrames, nach
    Komponentenindex sortiert. Skalare Verfahren (LDA) haben genau eine,
    die den Praefix selbst traegt."""
    spec = get(method)
    pre = spec["prefix"]
    if spec.get("scalar"):
        return [pre]
    return sorted([c for c in per_run.columns if c.startswith(pre)],
                  key=lambda c: int(c[len(pre):]))


def aggregate(per_run: pd.DataFrame, method: str,
              verbose: bool = True) -> pd.DataFrame:
    """Pro Fehlerklasse Mittelwert UND Standardabweichung jedes Werts.

    Aus einer Zeile je (Fault, Run) wird eine Zeile je Fault mit den
    Spalten `<prefix><i>_mean` und `<prefix><i>_std`. pandas rechnet die
    Std mit ddof=1 (Stichproben-Std), passend fuer N = 500 Runs.
    """
    cols = value_columns(per_run, method)
    agg = per_run.groupby("faultNumber")[cols].agg(["mean", "std"])
    # MultiIndex ('dyca_1', 'mean') -> 'dyca_1_mean' flachklopfen.
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()

    if verbose:
        print(f"Aggregierte {get(method)['label']}-Statistiken pro Fault: "
              f"{agg.shape}")
    return agg


def export(per_run: pd.DataFrame, method: str,
           scaling_mode: str = "global_mean", data_dir: str = ".",
           verbose: bool = True) -> str:
    """Schreibt die Trainings-Spektren als CSV zu den TEP-Daten.

    Stellt sie damit fuer die Klassifikation bereit. Die TEST-Spektren
    werden bewusst NICHT hier berechnet, sondern erst im
    Klassifikations-Notebook - dieses Notebook bleibt training-only.
    """
    path = os.path.join(data_dir, csv_name(method, scaling_mode, "train"))
    per_run.to_csv(path, index=False)
    if verbose:
        print(f"{get(method)['label']}-Train gespeichert: {per_run.shape} "
              f"-> {path}")
    return path
