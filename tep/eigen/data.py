"""Trainingsdaten der Eigenwert-Notebooks laden.

Anders als bei `tep.tsfresh` bleiben die Runs hier als EIN DataFrame
liegen (`df_all`, Fault 0..20 zusammen), weil die Spektren-Schleife
ohnehin ueber `groupby(["faultNumber", "simulationRun"])` laeuft.
"""

from __future__ import annotations

import gc

import pandas as pd

from ..core import META_COLS, PROC_COLS, scale
from .config import SpectrumConfig


def load_train(cfg: SpectrumConfig, verbose: bool = True):
    """Liest beide Trainings-CSVs und liefert (df_faultfree, df_faulty).

    TEP_FaultFree_Training.csv : nur faultNumber == 0
                                 500 Runs x 500 Samples = 250 000 Zeilen
    TEP_Faulty_Training.csv    : faultNumber 1..20
                                 20 x 500 x 500 = 5 000 000 Zeilen
    """
    # BEWUSST ohne dtype-Angabe: die urspruenglichen Notebooks lasen die
    # CSVs mit pandas-Default (float64). Ein Cast auf float32 wuerde die
    # exportierten Eigenwerte in den hinteren Stellen veraendern und
    # damit von den bereits vorliegenden *_eigenvalues_train.csv
    # abweichen. Der Speicherpreis ist rund 2,2 GB fuer den Faulty-Split.
    frames = []
    for fname in ("TEP_FaultFree_Training.csv", "TEP_Faulty_Training.csv"):
        if verbose:
            print(f"  lese {fname} ...", flush=True)
        df = pd.read_csv(cfg.data_path(fname), usecols=META_COLS + PROC_COLS)
        # Spalten-Check: ohne diese drei ist kein Gruppieren moeglich.
        missing = [c for c in META_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Spalten {missing} fehlen in {fname}")
        if cfg.runs_per_fault is not None:
            df = df[df["simulationRun"] <= cfg.runs_per_fault]
        frames.append(df)

    df_ff, df_faulty = frames
    if verbose:
        print(f"  FaultFree_Training: {df_ff.shape}")
        print(f"  Faulty_Training   : {df_faulty.shape}")
        print(f"  Prozessvariablen  : {len(PROC_COLS)}")
    return df_ff, df_faulty


def merge_faults(df_ff, df_faulty, verbose: bool = True):
    """Fault 0..20 in EINEM DataFrame, sortiert nach (fault, run, sample).

    Die Sortierung macht die spaetere Gruppenreihenfolge deterministisch
    und die Zeitreihen innerhalb jeder Gruppe bereits geordnet.
    """
    df_all = pd.concat([df_ff, df_faulty], ignore_index=True)
    df_all = df_all.sort_values(["faultNumber", "simulationRun", "sample"]) \
                   .reset_index(drop=True)
    gc.collect()
    if verbose:
        print(f"  kombiniert: {df_all.shape}, Faults "
              f"{sorted(df_all['faultNumber'].unique())}")
    return df_all


def faultfree_by_run(df_ff, cfg: SpectrumConfig, scaler) -> dict:
    """Pro FaultFree-Lauf die vorverarbeitete Messmatrix.

    Nur LDA braucht das: dort wird jeder Fehlerlauf gegen einen
    Normalbetriebslauf gestellt, und dessen Matrix soll nicht in jeder
    Iteration neu skaliert werden.
    """
    out = {}
    for run, g in df_ff.groupby("simulationRun"):
        g = g.sort_values("sample")
        out[int(run)] = scale(g[PROC_COLS].values, cfg.scaling_mode, scaler)
    return out
