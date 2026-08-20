"""Klassifikation auf den exportierten Spektren.

Das ist die Maschinerie von `LazyClassifier_PCA_DyCA.ipynb`: Test-Spektren
berechnen (oder aus dem CSV-Cache holen), Feature-Saetze bilden, Modelle
vergleichen und Confusion-Matrizen zeichnen.

Anders als bei `tep.tsfresh` sind die Features hier die Spektren selbst -
52 Eigenwerte je Run statt Tausender TSFresh-Kennzahlen.

Ein Feature-Satz ist ein dict:

    {"name": "PCA", "train": df, "test": df, "cols": [...]}

`matrices(fs)` macht daraus (Xtr, ytr, Xte, yte).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix)

from ..core import (LABELS, META_COLS, PRE_FAULT_CUTOFF, PROC_COLS,
                    default_estimator, fit_scaler)
from ..plotting import (counts_frame, normalize_rows, plot_grid,
                        print_top_confusions)
from .spectra import csv_name, get, needs_scaler, run_spectra

KEYS = ["faultNumber", "simulationRun"]

# Fensterangleichung Test -> Train. Im Training deckt ein Run 500 (Fault 0)
# bzw. 480 (Fault != 0) Samples ab, im Test waeren es 960 bzw. 800. Ohne
# Kuerzung waeren Train- und Test-Spektren systematisch verschieden.
TEST_HEAD_FAULT0 = 500
TEST_HEAD_FAULTY = 480


# =========================================================================
# Test-Spektren
# =========================================================================

def test_spectra(methods, scaling_mode: str = "global_mean",
                 data_dir: str = ".", runs_per_fault: int | None = None,
                 verbose: bool = True, **params) -> dict:
    """Spektren auf dem TESTsplit - aus dem CSV-Cache oder frisch gerechnet.

    methods : Liste der Verfahrensnamen, z.B. ["pca", "dyca"].
    params  : Verfahrensparameter, gehen unveraendert an run_spectra
              (dyca_m, dyca_n, ridge_rel, ...). Sie muessen zu denen
              passen, mit denen die Train-CSV erzeugt wurde.

    Die Test-CSVs sind gross (3,4 GB fuer Faulty_Testing), deshalb werden
    sie nur gelesen, wenn mindestens ein Verfahren wirklich rechnen muss.
    """
    out, todo = {}, []
    for method in methods:
        path = os.path.join(data_dir, csv_name(method, scaling_mode, "test"))
        if os.path.exists(path):
            out[method] = pd.read_csv(path)
            if verbose:
                print(f"{get(method)['label']}-Test aus Cache: "
                      f"{out[method].shape} ({path})")
        else:
            todo.append(method)

    if todo:
        scaler = None
        if any(needs_scaler(m, scaling_mode) for m in todo):
            scaler = fit_scaler(
                os.path.join(data_dir, "TEP_FaultFree_Training.csv"), verbose)

        if verbose:
            print("Lade TEP-Test-CSVs (gross, dauert) ...", flush=True)
        cols = META_COLS + PROC_COLS
        frames = [pd.read_csv(os.path.join(data_dir, f), usecols=cols)
                  for f in ("TEP_FaultFree_Testing.csv",
                            "TEP_Faulty_Testing.csv")]
        test_all = pd.concat(frames, ignore_index=True)
        del frames
        if runs_per_fault is not None:
            # Gleiche Einschraenkung wie im Training - sonst waere ein
            # Probelauf auf wenigen Runs im Test wieder vollstaendig.
            test_all = test_all[test_all["simulationRun"] <= runs_per_fault]
        test_all = test_all.sort_values(
            ["faultNumber", "simulationRun", "sample"]).reset_index(drop=True)
        if verbose:
            print("Test-Rohdaten:", test_all.shape)

        for method in todo:
            df = run_spectra(test_all, method, scaling_mode=scaling_mode,
                             scaler=scaler,
                             pre_fault_cutoff=PRE_FAULT_CUTOFF["test"],
                             head_fault0=TEST_HEAD_FAULT0,
                             head_faulty=TEST_HEAD_FAULTY,
                             verbose=verbose, **params)
            path = os.path.join(data_dir,
                                csv_name(method, scaling_mode, "test"))
            df.to_csv(path, index=False)
            out[method] = df
            if verbose:
                print(f"{get(method)['label']}-Test gespeichert: {df.shape} "
                      f"-> {path}")
        del test_all

    return out


def train_spectra(methods, scaling_mode: str = "global_mean",
                  data_dir: str = ".", verbose: bool = True) -> dict:
    """Die von den Eigenwert-Notebooks exportierten Trainings-CSVs laden."""
    out = {}
    for method in methods:
        path = os.path.join(data_dir, csv_name(method, scaling_mode, "train"))
        label = get(method)["label"]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{path}' nicht gefunden. Bitte zuerst das "
                f"{label}-Eigenwert-Notebook ausfuehren - es erzeugt "
                f"die Train-Spektren.")
        out[method] = pd.read_csv(path)
        if verbose:
            print(f"{label}-Train geladen: {out[method].shape}")
    return out


# =========================================================================
# Feature-Saetze
# =========================================================================

def matrices(fs: dict):
    """(Xtr, ytr, Xte, yte) eines Feature-Satzes."""
    return (fs["train"][fs["cols"]].values,
            fs["train"]["faultNumber"].values,
            fs["test"][fs["cols"]].values,
            fs["test"]["faultNumber"].values)


def feature_sets(methods, train: dict, test: dict,
                 combine: bool = True) -> list:
    """Einen Feature-Satz je Verfahren, optional zusaetzlich die Kombination.

    Die Kombination entsteht ueber einen INNER JOIN auf
    (faultNumber, simulationRun): nur Runs, die alle Verfahren ueberlebt
    haben. Bei DyCA scheitern einzelne Runs numerisch, deshalb ist der
    kombinierte Satz kleiner als die Einzelsaetze.
    """
    sets = []
    for method in methods:
        pre = get(method)["prefix"]
        cols = [c for c in train[method].columns if c.startswith(pre)]
        sets.append({"name": get(method)["label"], "train": train[method],
                     "test": test[method], "cols": cols})

    if combine and len(methods) > 1:
        tr, te = sets[0]["train"], sets[0]["test"]
        for s in sets[1:]:
            tr = pd.merge(tr, s["train"], on=KEYS, how="inner")
            te = pd.merge(te, s["test"], on=KEYS, how="inner")
        sets.append({"name": "+".join(s["name"] for s in sets),
                     "train": tr, "test": te,
                     "cols": [c for s in sets for c in s["cols"]]})
    return sets


def class_distribution(sets: list) -> pd.DataFrame:
    """Runs je Fehlerklasse in Train und Test, plus der Verlust gegenueber
    dem ersten Satz.

    Der Verlust zeigt, wo ein Verfahren Runs einbuesst - Klassen mit
    hohem Verlust sind Kandidaten fuer einen Selektions-Bias.
    """
    def counts(df):
        return df["faultNumber"].value_counts().sort_index()

    dist = pd.DataFrame({f"{s['name']}_{split}": counts(s[split])
                         for s in sets for split in ("train", "test")})
    dist.index.name = "faultNumber"
    base = sets[0]["name"]
    for split in ("train", "test"):
        for s in sets[1:]:
            dist[f"verlust_{s['name']}_{split}"] = (
                dist[f"{base}_{split}"] - dist[f"{s['name']}_{split}"])
    return dist


# =========================================================================
# Modellvergleich
# =========================================================================

def run_lazyclassifier(fs: dict, cv_folds: int = 5,
                       select_metric: str = "Balanced Accuracy CV Mean",
                       random_state: int = 42):
    """LazyClassifier auf einem Feature-Satz, sortiert nach der CV-Metrik.

    Ausgewaehlt wird auf der TRAIN-CV, nicht auf dem Testset - Modelle
    ohne predict_proba bekommen dabei keine CV-Werte (der ROC-AUC-Scorer
    scheitert und leert alle CV-Spalten) und landen am Tabellenende.
    """
    from lazypredict.Supervised import LazyClassifier

    Xtr, ytr, Xte, yte = matrices(fs)
    clf = LazyClassifier(verbose=0, ignore_warnings=True, predictions=False,
                         random_state=random_state, cv=cv_folds)

    print(f"=== {fs['name']} ===")
    print(f"Features: {len(fs['cols'])} | Train: {Xtr.shape[0]} | "
          f"Test: {Xte.shape[0]} | Klassen: {len(np.unique(ytr))} | "
          f"cv={cv_folds}")
    models, _ = clf.fit(Xtr, Xte, ytr, yte)

    if select_metric in models.columns:
        models = models.sort_values(select_metric, ascending=False,
                                    na_position="last")
        no_cv = models[models[select_metric].isna()]
        if len(no_cv) and "Balanced Accuracy" in no_cv.columns:
            top3 = no_cv.sort_values("Balanced Accuracy",
                                     ascending=False).head(3)
            names = ", ".join(f"{n} {r['Balanced Accuracy']:.4f}"
                              for n, r in top3.iterrows())
            print(f"Hinweis: {len(no_cv)} Modelle ohne CV-Werte am "
                  f"Tabellenende (ROC-AUC-Scorer scheitert ohne "
                  f"predict_proba). Staerkste nach Test-BA: {names}")
    print(f"Leaderboard sortiert nach '{select_metric}' "
          f"(nur im Speicher, keine CSV).")
    return models


# =========================================================================
# Confusion-Matrizen
# =========================================================================

def confusion(sets: list, random_state: int = 42, estimator=None,
              report: bool = True) -> dict:
    """Fester Modelltyp auf jedem Feature-Satz, einmal auf dem Testset
    ausgewertet. Die Unterschiede liegen damit allein an den Features."""
    results = {}
    for fs in sets:
        Xtr, ytr, Xte, yte = matrices(fs)
        model = (default_estimator(random_state) if estimator is None
                 else estimator)
        model.fit(Xtr, ytr)
        y_pred = model.predict(Xte)

        cm = confusion_matrix(yte, y_pred, labels=LABELS)
        results[fs["name"]] = {
            "cm": cm, "cm_norm": normalize_rows(cm),
            "bal_acc": balanced_accuracy_score(yte, y_pred),
            "n_features": len(fs["cols"]), "n_train": len(ytr),
            "n_test": len(yte),
        }
        print(f"=== RandomForestClassifier | {fs['name']} ===")
        print(f"Features: {len(fs['cols'])} | Train: {len(ytr)} | "
              f"Test: {len(yte)} | Balanced Accuracy (Test): "
              f"{results[fs['name']]['bal_acc']:.4f}")
        if report:
            print(classification_report(yte, y_pred, labels=LABELS,
                                        digits=3, zero_division=0))

    print("Zusammenfassung (Balanced Accuracy auf dem Testset):")
    for name, res in results.items():
        print(f"  {name:12s}: {res['bal_acc']:.4f}  "
              f"({res['n_features']} Features, {res['n_test']} Test-Runs)")
    return results


def counts(results: dict) -> dict:
    """Absolute Zaehlwerte je Feature-Satz als 21x21-DataFrame."""
    return {name: counts_frame(res["cm"]) for name, res in results.items()}


def plot_confusions(results: dict, annot_min: float = 0.05):
    """Alle Matrizen nebeneinander, gemeinsame Farbskala 0..1."""
    return plot_grid(
        [res["cm_norm"] for res in results.values()],
        [f"{name}  ({res['n_features']} Features)\n"
         f"Balanced Accuracy (Test) = {res['bal_acc']:.3f}"
         for name, res in results.items()],
        "RandomForestClassifier - Confusion-Matrizen auf dem TEP-Testset",
        big=True, annot_min=annot_min)


def report_confusions(results: dict, top_n: int = 8) -> None:
    """Die groessten Verwechslungen je Feature-Satz als Textblock."""
    for name, res in results.items():
        print()
        print_top_confusions(res["cm"], top_n, title=name)
