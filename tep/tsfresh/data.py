"""Rohdaten laden.

Liest die vier TEP-CSVs und liefert je Split ein
``dict[(fault, run)] -> (T, 52) float32``.
"""

from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd

from ..core import (META_COLS, PRE_FAULT_CUTOFF, PROC_COLS, SPLIT_FILES,
                    labels_from_index, run_id)
from .config import PipelineConfig


def load_runs(cfg: PipelineConfig, split: str, verbose: bool = True) -> dict:
    """Liest einen Split ("train" | "test").

    Verwirft die Pre-Fault-Phase und kuerzt auf ``cfg.run_length``, damit
    Train (500 Samples/Run) und Test (960) dasselbe Zeitfenster nach
    Fehlereintritt abdecken. Der Cutoff gilt auch fuer Fault 0 - sonst
    wuerden laengenabhaengige TSFresh-Features (length, abs_energy, ...)
    den Normalbetrieb rein artifiziell abtrennen.
    """
    cutoff = PRE_FAULT_CUTOFF[split]
    dtypes = {c: "float32" for c in PROC_COLS}
    dtypes.update({c: "int32" for c in META_COLS})

    runs = {}
    for fname in SPLIT_FILES[split]:
        path = cfg.data_path(fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} nicht gefunden.")
        if verbose:
            print(f"  lese {fname} ...", flush=True)
        df = pd.read_csv(path, usecols=META_COLS + PROC_COLS, dtype=dtypes)

        if cfg.runs_per_fault is not None:
            df = df[df["simulationRun"] <= cfg.runs_per_fault]

        for (fault, run), g in df.groupby(["faultNumber", "simulationRun"],
                                          sort=True):
            g = g.sort_values("sample")            # klein: <= 960 Zeilen
            g = g[g["sample"] >= cutoff]
            arr = g[PROC_COLS].to_numpy(dtype=np.float32)
            if cfg.run_length is not None:
                if arr.shape[0] < cfg.run_length:
                    raise ValueError(
                        f"Run (fault={fault}, run={run}) hat nur "
                        f"{arr.shape[0]} Samples, run_length="
                        f"{cfg.run_length} verlangt mehr.")
                arr = arr[:cfg.run_length]
            runs[(int(fault), int(run))] = arr

        del df
        gc.collect()

    lens = sorted({v.shape[0] for v in runs.values()})
    if verbose:
        print(f"  {split}: {len(runs)} Runs, Laenge(n) {lens}, "
              f"{sum(v.nbytes for v in runs.values()) / 1e9:.2f} GB")
        if len(lens) > 1:
            print("  ACHTUNG: unterschiedliche Runlaengen -> laengen"
                  "abhaengige TSFresh-Features (length, abs_energy, ...) "
                  "trennen Klassen artifiziell. run_length pruefen!")
    return runs
