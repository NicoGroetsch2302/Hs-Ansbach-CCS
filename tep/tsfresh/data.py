"""Rohdaten laden.

Liest die vier TEP-CSVs und liefert je Split ein
``dict[(fault, run)] -> (T, 52) float32``.
"""

from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd

from ..core import META_COLS, PRE_FAULT_CUTOFF, PROC_COLS, SPLIT_FILES
from ..core import fit_scaler as _core_fit_scaler


def fit_scaler(scaling_mode: str = "global_mean", data_dir: str = ".",
               verbose: bool = True):
    """StandardScaler auf dem Normalbetrieb fitten - oder None.

    Nur bei scaling_mode="scaler" noetig; sonst bleiben die 250k Zeilen
    Normalbetrieb ungelesen.
    """
    if scaling_mode != "scaler":
        return None
    return _core_fit_scaler(
        os.path.join(data_dir, "TEP_FaultFree_Training.csv"), verbose)


def load_runs(split: str, data_dir: str = ".",
              runs_per_fault: int | None = None,
              run_length: int | None = 480, verbose: bool = True) -> dict:
    """Liest einen Split ("train" | "test").

    Verwirft die Pre-Fault-Phase und kuerzt auf ``run_length``, damit
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
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} nicht gefunden.")
        if verbose:
            print(f"  lese {fname} ...", flush=True)
        df = pd.read_csv(path, usecols=META_COLS + PROC_COLS, dtype=dtypes)

        if runs_per_fault is not None:
            df = df[df["simulationRun"] <= runs_per_fault]

        for (fault, run), g in df.groupby(["faultNumber", "simulationRun"],
                                          sort=True):
            g = g.sort_values("sample")            # klein: <= 960 Zeilen
            g = g[g["sample"] >= cutoff]
            arr = g[PROC_COLS].to_numpy(dtype=np.float32)
            if run_length is not None:
                if arr.shape[0] < run_length:
                    raise ValueError(
                        f"Run (fault={fault}, run={run}) hat nur "
                        f"{arr.shape[0]} Samples, run_length="
                        f"{run_length} verlangt mehr.")
                arr = arr[:run_length]
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
