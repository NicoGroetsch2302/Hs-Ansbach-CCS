"""Feature-Extraktion mit Chunk-Cache und Feature-Auswahl.

Der Chunk-Cache ist das Rueckgrat der langen Laeufe: jede Konfiguration
wird in Bloecken von cfg.chunk_runs Runs extrahiert und als Pickle
abgelegt. Ein abgebrochener Nachtlauf setzt beim naechsten Start genau
dort wieder an.

Dateinamen im Cache-Ordner:
    {name}__{split}__{tag}__{idx:04d}.pkl    Feature-Chunks
    {name}__top{top_k}.json                  ausgewaehlte Featurenamen
"""

from __future__ import annotations

import gc
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from tqdm.auto import tqdm
from tsfresh import extract_features
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.utilities.dataframe_functions import impute

from .config import PipelineConfig
from .data import run_id
from .projections import config_name, project


def chunk_path(cfg: PipelineConfig, spec, split: str, tag: str, idx: int) -> str:
    return cfg.cache_path(
        f"{config_name(spec)}__{split}__{tag}__{idx:04d}.pkl")


def top_features_path(cfg: PipelineConfig, spec) -> str:
    return cfg.cache_path(f"{config_name(spec)}__top{cfg.top_k}.json")


def _subset(part: pd.DataFrame, usecols) -> pd.DataFrame:
    """Auf usecols reduzieren. Fehlende Spalten (z.B. weil der Cache mit
    anderen fc_parameters gebaut wurde) werden mit 0.0 ergaenzt, statt mit
    einem KeyError mitten im Nachtlauf abzubrechen."""
    if usecols is None:
        return part
    missing = [c for c in usecols if c not in part.columns]
    if missing:
        for c in missing:
            part[c] = np.float32(0.0)
    return part[usecols]


def extract_config(cfg: PipelineConfig, spec, split: str, runs: dict,
                   tag: str = "full", kind_to_fc=None, usecols=None,
                   scaler=None) -> pd.DataFrame:
    """Extrahiert TSFresh-Features fuer EINE Konfiguration.

    tag        : Cache-Kennung ("full" = alle Features,
                 "top{K}" = nur die Top-K aus Phase A). Bei "top" MUSS
                 K im Tag stehen, sonst kollidieren Laeufe mit
                 unterschiedlichem top_k im selben Cache.
    kind_to_fc : wenn gesetzt, werden NUR diese Features berechnet
                 (aus tsfresh.from_columns) - der billige Weg fuer Phase B
    usecols    : beim Laden aus dem Cache nur diese Spalten behalten (RAM)

    Rueckgabe: DataFrame, Index = run_id, Spalten = Featurenamen (float32).
    """
    name = config_name(spec)
    keys = sorted(runs.keys())
    parts, n_failed = [], 0

    it = tqdm(list(range(0, len(keys), cfg.chunk_runs)),
              desc=f"{name:20s} [{split}/{tag}]")

    for ci, start in enumerate(it):
        path = chunk_path(cfg, spec, split, tag, ci)
        if os.path.exists(path):
            part = pd.read_pickle(path)
            parts.append(_subset(part, usecols))
            continue

        # --- Chunk aufbauen: projizieren und ins tsfresh-Wide-Format ---
        frames = []
        for key in keys[start:start + cfg.chunk_runs]:
            try:
                Y, names = project(runs[key], spec, cfg, scaler)
            except Exception:
                n_failed += 1      # z.B. numerisches Scheitern der DyCA-Stufe
                continue
            d = {"id": np.full(Y.shape[0], run_id(*key), dtype=np.int32),
                 "time": np.arange(Y.shape[0], dtype=np.int32)}
            for c, nm in enumerate(names):
                d[nm] = Y[:, c]
            frames.append(pd.DataFrame(d))

        if not frames:
            continue
        wide = pd.concat(frames, ignore_index=True)
        del frames

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            part = extract_features(
                wide, column_id="id", column_sort="time",
                default_fc_parameters=(None if kind_to_fc
                                       else cfg.fc_parameters),
                kind_to_fc_parameters=kind_to_fc,
                n_jobs=cfg.n_jobs, disable_progressbar=True)

        part = part.astype(np.float32)
        part.to_pickle(path)
        # _subset statt part[usecols]: schuetzt auch den Frisch-Berechnen-
        # Pfad vor einem KeyError, falls eine im Auswahl-JSON stehende
        # Spalte (z.B. nach fc_mode-Wechsel) nicht erzeugt wurde.
        parts.append(_subset(part, usecols))
        del wide, part
        gc.collect()

    if n_failed:
        print(f"    {name}: {n_failed} Runs uebersprungen "
              f"(Projektion fehlgeschlagen)")
    return pd.concat(parts) if parts else pd.DataFrame()


def rank_features(cfg: PipelineConfig, X: pd.DataFrame,
                  y: pd.Series) -> pd.DataFrame:
    """Bewertet alle Spalten von X und liefert eine sortierte Rangtabelle.

    Sortierkriterium: n_significant absteigend (Zahl der Klassen, die das
    Feature signifikant trennt), bei Gleichstand ANOVA-F absteigend.
    Blockweise, damit die Roh-Konfiguration (40k Spalten) in den RAM passt.
    """
    parts = []
    for start in range(0, X.shape[1], cfg.block_cols):
        Xb = X.iloc[:, start:start + cfg.block_cols].astype("float64")
        Xb = impute(Xb)                       # NaN/inf -> endliche Werte
        # Konstante Spalten tragen nichts bei und stoeren die Tests.
        Xb = Xb.loc[:, Xb.to_numpy().std(axis=0) > 0]
        if Xb.shape[1] == 0:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rt = calculate_relevance_table(
                Xb, y, ml_task="classification", multiclass=True,
                n_significant=1, n_jobs=cfg.n_jobs)
            f_val, _ = f_classif(Xb.to_numpy(), y.to_numpy())

        p_cols = [c for c in rt.columns if c.startswith("p_value_")]
        parts.append(pd.DataFrame({
            "feature": rt["feature"].to_numpy(),
            "n_significant": rt["n_significant"].to_numpy(),
            "p_min": rt[p_cols].min(axis=1).to_numpy(),
            "f_value": pd.Series(f_val, index=Xb.columns)
                         .reindex(rt["feature"]).to_numpy(),
        }))
        del Xb
        gc.collect()

    if not parts:
        return pd.DataFrame(
            columns=["feature", "n_significant", "p_min", "f_value"])

    rank = pd.concat(parts, ignore_index=True)
    rank["f_value"] = rank["f_value"].fillna(0.0)
    return rank.sort_values(["n_significant", "f_value"],
                            ascending=[False, False]).reset_index(drop=True)


def load_top_names(cfg: PipelineConfig, spec) -> list | None:
    """Ausgewaehlte Featurenamen aus dem Cache, oder None."""
    path = top_features_path(cfg, spec)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_top_names(cfg: PipelineConfig, spec, names: list) -> None:
    with open(top_features_path(cfg, spec), "w", encoding="utf-8") as fh:
        json.dump(names, fh, indent=1)
