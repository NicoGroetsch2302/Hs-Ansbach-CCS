"""Die drei Schritte des Laufs, jeder als eigene Funktion.

    select_features   Trainingsruns extrahieren, bewerten, Top-K behalten
    apply_features    Testruns extrahieren, aber NUR die gewaehlten Merkmale
    benchmark_models  LazyClassifier je Konfiguration -> summary-CSV

Jeder Schritt nimmt entgegen, was sie braucht, und gibt zurueck, was die
naechste braucht - kein gemeinsames Objekt, das den Zustand haelt:

    names = validate(CONFIGS)
    train_top, top_names = select_features(CONFIGS, CACHE, data_dir=..., ...)
    test_top = apply_features(CONFIGS, CACHE, top_names, data_dir=..., ...)
    summary = benchmark_models(CONFIGS, train_top, test_top, summary_path)

Nach einem Kernel-Neustart genuegt fuer die Auswertung die summary-CSV
(`load_summary`) bzw. der Vorhersage-Cache der Confusion-Matrizen.
"""

from __future__ import annotations

import gc
import os
import time

import numpy as np
import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from tsfresh.feature_extraction.settings import from_columns

from ..core import labels_from_index
from .data import load_runs
from .features import (extract_config, fc_parameters, load_top_names,
                       rank_features, save_top_names)
from .projections import config_name, n_channels, validate


def describe(configs, cache: str, *, fc_mode: str = "efficient",
             top_k: int = 100, scaling_mode: str = "global_mean",
             runs_per_fault: int | None = None,
             smoke_test: bool = False) -> None:
    """Umfang des Laufs und Kanalzahl je Konfiguration im Klartext."""
    n_calc = len(fc_parameters(fc_mode))
    print(f"smoke_test={smoke_test} | fc_mode={fc_mode} "
          f"({n_calc} Calculator) | top_k={top_k} "
          f"| scaling_mode={scaling_mode}")
    print(f"Runs je Fault: {runs_per_fault or 'alle'} | "
          f"Cache: {cache}/ (geteilt mit den Schwester-Notebooks)")
    print(f"\n{len(configs)} Konfigurationen:")
    for spec in configs:
        c = n_channels(spec)
        print(f"  {config_name(spec):22s} {c:3d} Kanaele -> "
              f"~{c * n_calc} Features/Run (Groessenordnung)")
    if not smoke_test:
        print("\n!!! VOLLER LAUF - Stunden bis Nacht. Der Chunk-Cache "
              "macht Abbrechen und Fortsetzen gefahrlos. !!!")


# =========================================================================
# Merkmale waehlen (Train)
# =========================================================================

def select_features(configs, cache: str, *, data_dir: str = ".",
            runs_per_fault: int | None = None, run_length: int | None = 480,
            fc_mode: str = "efficient", top_k: int = 100,
            chunk_runs: int = 250, block_cols: int = 4000,
            n_jobs: int | None = None, scaling_mode: str = "global_mean",
            scaler=None, fix_signs: bool = True, **proj_params):
    """Train extrahieren, Features auswaehlen, auf Top-K reduzieren.

    Konfigurationen strikt nacheinander: die volle Matrix einer
    Konfiguration wird freigegeben, bevor die naechste beginnt.

    Rueckgabe: (train_top, top_names), beide dict name -> DataFrame/Liste.
    """
    t_start = time.perf_counter()
    fc_params = fc_parameters(fc_mode)

    print("Lade Trainingsruns ...")
    runs_train = load_runs("train", data_dir, runs_per_fault, run_length)

    train_top, top_names = {}, {}
    for spec in configs:
        name = config_name(spec)
        t0 = time.perf_counter()
        names = load_top_names(cache, spec, top_k)

        common = dict(cache=cache, tag="full", fc_params=fc_params,
                      chunk_runs=chunk_runs, n_jobs=n_jobs,
                      scaling_mode=scaling_mode, scaler=scaler,
                      fix_signs=fix_signs, **proj_params)

        if names is not None:
            # Auswahl liegt vor -> nur die Top-K-Spalten aus dem Cache.
            Xtop = extract_config(spec, "train", runs_train, usecols=names,
                                  **common)
            print(f"[{name}] Auswahl aus Cache, {Xtop.shape[0]} Runs")
        else:
            X_full = extract_config(spec, "train", runs_train, **common)
            y_full = labels_from_index(X_full.index)
            print(f"[{name}] extrahiert: {X_full.shape[0]} Runs x "
                  f"{X_full.shape[1]} Features "
                  f"({time.perf_counter() - t0:.0f} s) -> selektiere ...")

            rank = rank_features(X_full, y_full, block_cols, n_jobs)
            names = rank["feature"].head(top_k).tolist()
            save_top_names(cache, spec, top_k, names)

            Xtop = X_full[names].copy()
            del X_full
            gc.collect()

        train_top[name] = Xtop
        top_names[name] = names
        print(f"[{name}] fertig: {Xtop.shape} in "
              f"{(time.perf_counter() - t0) / 60:.1f} min")

    del runs_train
    gc.collect()
    print(f"\nMerkmalsauswahl komplett in "
          f"{(time.perf_counter() - t_start) / 60:.1f} min")
    return train_top, top_names


# =========================================================================
# Merkmale anwenden (Test)
# =========================================================================

def apply_features(configs, cache: str, top_names: dict, *,
                   data_dir: str = ".",
            runs_per_fault: int | None = None, run_length: int | None = 480,
            top_k: int = 100, chunk_runs: int = 250,
            n_jobs: int | None = None, scaling_mode: str = "global_mean",
            scaler=None, fix_signs: bool = True, **proj_params) -> dict:
    """Testset extrahieren - nur die gewaehlten Merkmale."""
    if not top_names:
        raise RuntimeError("top_names fehlt -> zuerst select_features() "
                           "(laeuft aus dem Cache).")

    print("Lade Testruns ...")
    runs_test = load_runs("test", data_dir, runs_per_fault, run_length)

    test_top = {}
    for spec in configs:
        name = config_name(spec)
        t0 = time.perf_counter()
        # top_k MUSS in der Cache-Kennung stehen: die Chunks enthalten
        # genau die in select_features() ausgewaehlten Features. Ohne top_k im
        # Namen wuerde ein spaeterer Lauf mit groesserem top_k die alten
        # Chunks wiederverwenden und die fehlenden Spalten stumm mit 0.0
        # auffuellen (siehe _subset in features.py).
        Xte = extract_config(spec, "test", runs_test, cache,
                             tag=f"top{top_k}",
                             kind_to_fc=from_columns(top_names[name]),
                             chunk_runs=chunk_runs, n_jobs=n_jobs,
                             scaling_mode=scaling_mode, scaler=scaler,
                             fix_signs=fix_signs, **proj_params)

        # Reihenfolge und Vollstaendigkeit an Train angleichen: Features,
        # die tsfresh auf dem Testset nicht erzeugt, waeren sonst stumm
        # verschoben.
        missing = [c for c in top_names[name] if c not in Xte.columns]
        if missing:
            print(f"    {name}: {len(missing)} Features fehlen im Test "
                  f"-> 0.0")
            for c in missing:
                Xte[c] = 0.0
        test_top[name] = Xte[top_names[name]]

        print(f"[{name}] Test fertig: {test_top[name].shape} in "
              f"{(time.perf_counter() - t0) / 60:.1f} min")

    del runs_test
    gc.collect()
    return test_top


# =========================================================================
# Gemeinsame Run-Menge
# =========================================================================

def common_runs(train_top: dict, test_top: dict):
    """(train_index, test_index) der ueber ALLE Konfigurationen
    gemeinsamen Runs.

    Die DyCA-Stufe kann an einzelnen Runs numerisch scheitern; ohne
    diesen Schnitt waeren die Konfigurationen auf unterschiedlichen
    Testmengen bewertet.
    """
    if not train_top or not test_top:
        raise RuntimeError("train_top/test_top fehlen -> zuerst "
                           "select_features() und apply_features().")
    tr = sorted(set.intersection(*(set(d.index) for d in train_top.values())))
    te = sorted(set.intersection(*(set(d.index) for d in test_top.values())))
    return tr, te


def matrices(name: str, train_top: dict, test_top: dict):
    """(Xtr, Xte, ytr, yte, test_index) einer Konfiguration - gemeinsame
    Runs, NaN/inf aufgefuellt. Genau die Datenbasis von benchmark_models()."""
    idx_tr, idx_te = common_runs(train_top, test_top)
    Xtr = train_top[name].loc[idx_tr]
    Xte = test_top[name].loc[idx_te]
    ytr = labels_from_index(Xtr.index).to_numpy()
    yte = labels_from_index(Xte.index).to_numpy()
    # NaN/inf koennen aus Featureberechnungen stammen -> auffuellen.
    Xtr_v = np.nan_to_num(Xtr.to_numpy(dtype=np.float64),
                          posinf=0.0, neginf=0.0)
    Xte_v = np.nan_to_num(Xte.to_numpy(dtype=np.float64),
                          posinf=0.0, neginf=0.0)
    return Xtr_v, Xte_v, ytr, yte, Xte.index


# =========================================================================
# Modelle vergleichen
# =========================================================================

def benchmark_models(configs, train_top: dict, test_top: dict,
                     summary_path: str,
            *, lc_cv_folds: int = 5, random_state: int = 42):
    """LazyClassifier je Konfiguration; schreibt die summary-CSV.

    Rueckgabe: (summary, leaderboards) - die Tabelle und je Konfiguration
    das volle lazypredict-Leaderboard.
    """
    idx_tr, idx_te = common_runs(train_top, test_top)
    print(f"Gemeinsame Runs: Train {len(idx_tr)}, Test {len(idx_te)}")

    rows, leaderboards = [], {}
    for spec in configs:
        name = config_name(spec)
        Xtr_v, Xte_v, ytr, yte, _ = matrices(name, train_top, test_top)

        clf = LazyClassifier(verbose=0, ignore_warnings=True,
                             predictions=True, cv=lc_cv_folds,
                             random_state=random_state)

        t0 = time.perf_counter()
        models, preds = clf.fit(Xtr_v, Xte_v, ytr, yte)
        leaderboards[name] = models

        rows += _summary_rows(name, models, preds, yte)
        _print_best(name, rows, time.perf_counter() - t0)

    summary = pd.DataFrame(rows)
    summary.to_csv(summary_path, index=False)
    return summary, leaderboards


def load_summary(summary_path: str) -> pd.DataFrame:
    """summary aus der CSV holen - fuer Auswertungszellen nach einem
    Kernel-Neustart."""
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"{summary_path} fehlt -> zuerst "
                                f"benchmark_models().")
    summary = pd.read_csv(summary_path)
    print(f"summary aus {summary_path} geladen.")
    return summary


# =========================================================================
# Hilfsfunktionen fuer den Modellvergleich
# =========================================================================

def _cv_val(models: pd.DataFrame, col: str, model_name: str) -> float:
    """CV-Wert aus dem lazypredict-Leaderboard, NaN wenn nicht vorhanden.

    Modelle ohne predict_proba bekommen None: lazypredict rechnet alle
    CV-Scorer gebuendelt mit error_score="raise", der ROC-AUC-Scorer
    scheitert und ALLE CV-Spalten des Modells werden geleert.
    """
    if col not in models.columns:
        return np.nan
    v = models[col].get(model_name, np.nan)
    return np.nan if v is None or pd.isna(v) else float(v)


def _summary_rows(name, models, preds, yte) -> list:
    """Eine Zeile je erfolgreichem Modell.

    Macro-F1 und Balanced Accuracy werden selbst gerechnet (lazypredicts F1
    ist die gewichtete Variante). Die CV-Spalten liegen NUR in models - ohne
    diesen Schritt waere die CV reine Rechenzeitverschwendung.
    """
    rows = []
    for model_name in preds.columns:
        yp = preds[model_name].to_numpy()
        if pd.isna(yp).any():                  # Modell ist durchgefallen
            continue
        rows.append({
            "Konfiguration": name,
            "Modell": model_name,
            "BalancedAcc": balanced_accuracy_score(yte, yp),
            "MacroF1": f1_score(yte, yp, average="macro", zero_division=0),
            # Train-CV (kein Testset-Blick), NaN bei Modellen ohne CV-Werte.
            "BalancedAccCVMean": _cv_val(models, "Balanced Accuracy CV Mean",
                                         model_name),
            "BalancedAccCVStd": _cv_val(models, "Balanced Accuracy CV Std",
                                        model_name),
        })
    return rows


def _print_best(name, rows, seconds) -> None:
    hits = [s for s in rows if s["Konfiguration"] == name]
    if not hits:
        print(f"[{name:22s}] kein Modell erfolgreich!")
        return
    best = max(hits, key=lambda s: s["MacroF1"])
    print(f"[{name:22s}] {len(hits):2d} Modelle in {seconds / 60:5.1f} min "
          f"| bestes: {best['Modell']} (MacroF1 {best['MacroF1']:.4f}, "
          f"BA {best['BalancedAcc']:.4f})")
    # Zusaetzlich das CV-beste Modell: dieselbe Zeile, aber auf der Train-CV
    # ausgewaehlt statt auf dem Testmaximum (Winner's Curse).
    cv_hits = [s for s in hits if not np.isnan(s["BalancedAccCVMean"])]
    if cv_hits:
        b = max(cv_hits, key=lambda s: s["BalancedAccCVMean"])
        print(f"{'':19s}CV-bestes: {b['Modell']} "
              f"(BA CV {b['BalancedAccCVMean']:.4f} "
              f"+/- {b['BalancedAccCVStd']:.4f} -> Test-MacroF1 "
              f"{b['MacroF1']:.4f}) | {len(cv_hits)}/{len(hits)} mit "
              f"CV-Werten")
