"""Feature-Extraktion mit Chunk-Cache und Feature-Auswahl.

Der Chunk-Cache ist das Rueckgrat der langen Laeufe: jede Konfiguration
wird in Bloecken von `chunk_runs` Runs extrahiert und als Pickle abgelegt.
Ein abgebrochener Nachtlauf setzt beim naechsten Start genau dort wieder
an.

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
from tsfresh.feature_extraction.settings import (
    ComprehensiveFCParameters, EfficientFCParameters, MinimalFCParameters)
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.utilities.dataframe_functions import impute

from ..core import run_id
from .projections import config_name, project

FC_MODES = {"minimal": MinimalFCParameters,
            "efficient": EfficientFCParameters,
            "comprehensive": ComprehensiveFCParameters}


def fc_parameters(fc_mode: str = "efficient") -> dict:
    """Der TSFresh-Calculator-Satz zu einem Kuerzel."""
    if fc_mode not in FC_MODES:
        raise ValueError(f"fc_mode={fc_mode!r} unbekannt "
                         f"(bekannt: {sorted(FC_MODES)})")
    return FC_MODES[fc_mode]()


def cache_dir(scaling_mode: str = "global_mean",
              smoke_test: bool = False) -> str:
    """Cache-Ordner, angelegt falls noetig.

    Der Ordner ist BEWUSST zwischen den Notebooks geteilt - die
    ``raw``-Konfiguration ist ueberall bitidentisch, ihre Chunks und die
    Top-K-Auswahl werden dadurch wiederverwendet. `smoke_test` bekommt
    einen eigenen Ordner und ueberschreibt also nichts.
    """
    path = "tsfresh_cache_smoke" if smoke_test else "tsfresh_cache"
    if scaling_mode != "global_mean":
        path += f"_{scaling_mode}"
    os.makedirs(path, exist_ok=True)
    return path


def chunk_path(cache: str, spec, split: str, tag: str, idx: int) -> str:
    return os.path.join(cache,
                        f"{config_name(spec)}__{split}__{tag}__{idx:04d}.pkl")


def top_features_path(cache: str, spec, top_k: int) -> str:
    return os.path.join(cache, f"{config_name(spec)}__top{top_k}.json")


def load_top_names(cache: str, spec, top_k: int) -> list | None:
    """Ausgewaehlte Featurenamen aus dem Cache, oder None."""
    path = top_features_path(cache, spec, top_k)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_top_names(cache: str, spec, top_k: int, names: list) -> None:
    with open(top_features_path(cache, spec, top_k), "w",
              encoding="utf-8") as fh:
        json.dump(names, fh, indent=1)


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


def extract_config(spec, split: str, runs: dict, cache: str, *,
                   tag: str = "full", fc_params=None, kind_to_fc=None,
                   usecols=None, chunk_runs: int = 250,
                   n_jobs: int | None = None,
                   scaling_mode: str = "global_mean", scaler=None,
                   fix_signs: bool = True, **proj_params) -> pd.DataFrame:
    """Extrahiert TSFresh-Features fuer EINE Konfiguration.

    tag         : Cache-Kennung ("full" = alle Features, "top{K}" = nur die
                  Top-K aus select_features()). Bei "top" MUSS K im Tag
                  stehen, sonst
                  kollidieren Laeufe mit unterschiedlichem top_k im selben
                  Cache.
    fc_params   : Calculator-Satz (aus fc_parameters()); bei kind_to_fc egal
    kind_to_fc  : wenn gesetzt, werden NUR diese Features berechnet
                  (aus tsfresh.from_columns) - billig fuer apply_features()
    usecols     : beim Laden aus dem Cache nur diese Spalten behalten (RAM)
    proj_params : gehen unveraendert an project()

    Rueckgabe: DataFrame, Index = run_id, Spalten = Featurenamen (float32).
    """
    name = config_name(spec)
    keys = sorted(runs.keys())
    parts, n_failed = [], 0

    it = tqdm(list(range(0, len(keys), chunk_runs)),
              desc=f"{name:20s} [{split}/{tag}]")

    for ci, start in enumerate(it):
        path = chunk_path(cache, spec, split, tag, ci)
        if os.path.exists(path):
            part = pd.read_pickle(path)
            parts.append(_subset(part, usecols))
            continue

        # --- Chunk aufbauen: projizieren und ins tsfresh-Wide-Format ---
        frames = []
        for key in keys[start:start + chunk_runs]:
            try:
                Y, names = project(runs[key], spec, scaling_mode, scaler,
                                   fix_signs, **proj_params)
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
                default_fc_parameters=(None if kind_to_fc else fc_params),
                kind_to_fc_parameters=kind_to_fc,
                n_jobs=n_jobs if n_jobs is not None else (os.cpu_count() or 4),
                disable_progressbar=True)

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


def rank_features(X: pd.DataFrame, y: pd.Series, block_cols: int = 4000,
                  n_jobs: int | None = None) -> pd.DataFrame:
    """Bewertet alle Spalten von X und liefert eine sortierte Rangtabelle.

    Sortierkriterium: n_significant absteigend (Zahl der Klassen, die das
    Feature signifikant trennt), bei Gleichstand ANOVA-F absteigend.
    Blockweise, damit die Roh-Konfiguration (40k Spalten) in den RAM passt.
    """
    n_jobs = n_jobs if n_jobs is not None else (os.cpu_count() or 4)
    parts = []
    for start in range(0, X.shape[1], block_cols):
        Xb = X.iloc[:, start:start + block_cols].astype("float64")
        Xb = impute(Xb)                       # NaN/inf -> endliche Werte
        # Konstante Spalten tragen nichts bei und stoeren die Tests.
        Xb = Xb.loc[:, Xb.to_numpy().std(axis=0) > 0]
        if Xb.shape[1] == 0:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rt = calculate_relevance_table(
                Xb, y, ml_task="classification", multiclass=True,
                n_significant=1, n_jobs=n_jobs)
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
