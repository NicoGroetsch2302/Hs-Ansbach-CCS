"""Klassifikation auf den exportierten Spektren.

Das ist die Maschinerie von `LazyClassifier_PCA_DyCA.ipynb`: Test-Spektren
berechnen (oder aus dem CSV-Cache holen), Feature-Saetze bilden, Modelle
vergleichen und Confusion-Matrizen zeichnen.

Anders als bei `tep.tsfresh` sind die Features hier die Spektren selbst -
52 Eigenwerte je Run statt Tausender TSFresh-Kennzahlen.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..core import LABELS, META_COLS, PROC_COLS, fit_scaler
from ..plotting import draw_confusion, normalize_rows, print_top_confusions
from .config import SpectrumConfig
from .spectra import get, run_spectra

KEYS = ["faultNumber", "simulationRun"]


# =========================================================================
# Test-Spektren
# =========================================================================

def test_csv_name(cfg: SpectrumConfig) -> str:
    suffix = "" if cfg.scaling_mode == "global_mean" else "_scaler"
    return f"{get(cfg.method).csv_stem}_test{suffix}.csv"


def test_spectra(cfgs, verbose: bool = True) -> dict:
    """Spektren auf dem TESTsplit - aus dem CSV-Cache oder frisch gerechnet.

    Die Test-CSVs sind gross (3,4 GB fuer Faulty_Testing), deshalb werden
    sie nur gelesen, wenn mindestens ein Verfahren wirklich rechnen muss.

    Wichtig ist die Fensterangleichung: im Training deckt ein Run 500
    (Fault 0) bzw. 480 (Fault != 0) Samples ab. Im Test sind es 960 bzw.
    800. Ohne Kuerzung auf dieselben Fensterlaengen waeren Train- und
    Test-Spektren systematisch verschieden.
    """
    out, todo = {}, []
    for cfg in cfgs:
        path = cfg.data_path(test_csv_name(cfg))
        if os.path.exists(path):
            out[cfg.method] = pd.read_csv(path)
            if verbose:
                print(f"{cfg.label}-Test aus Cache: {out[cfg.method].shape} "
                      f"({path})")
        else:
            todo.append(cfg)

    if todo:
        first = todo[0]
        scaler = None
        if any(c.scaling_mode == "scaler"
               or get(c.method).forced_scaling == "scaler" for c in todo):
            scaler = fit_scaler(
                first.data_path("TEP_FaultFree_Training.csv"), verbose)

        if verbose:
            print("Lade TEP-Test-CSVs (gross, dauert) ...", flush=True)
        cols = META_COLS + PROC_COLS
        frames = [pd.read_csv(first.data_path(f), usecols=cols)
                  for f in ("TEP_FaultFree_Testing.csv",
                            "TEP_Faulty_Testing.csv")]
        test_all = pd.concat(frames, ignore_index=True)
        del frames
        if first.runs_per_fault is not None:
            # Gleiche Einschraenkung wie im Training - sonst waere ein
            # Probelauf auf wenigen Runs im Test wieder vollstaendig.
            test_all = test_all[
                test_all["simulationRun"] <= first.runs_per_fault]
        test_all = test_all.sort_values(
            ["faultNumber", "simulationRun", "sample"]).reset_index(drop=True)
        if verbose:
            print("Test-Rohdaten:", test_all.shape)

        for cfg in todo:
            df = run_spectra(cfg.as_test(), test_all, scaler=scaler,
                             verbose=verbose)
            path = cfg.data_path(test_csv_name(cfg))
            df.to_csv(path, index=False)
            out[cfg.method] = df
            if verbose:
                print(f"{cfg.label}-Test gespeichert: {df.shape} -> {path}")
        del test_all

    return out


def train_spectra(cfgs, verbose: bool = True) -> dict:
    """Die von den Eigenwert-Notebooks exportierten Trainings-CSVs laden."""
    out = {}
    for cfg in cfgs:
        path = cfg.data_path(cfg.csv_name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{path}' nicht gefunden. Bitte zuerst das "
                f"{cfg.label}-Eigenwert-Notebook ausfuehren - es erzeugt "
                f"die Train-Spektren.")
        out[cfg.method] = pd.read_csv(path)
        if verbose:
            print(f"{cfg.label}-Train geladen: {out[cfg.method].shape}")
    return out


# =========================================================================
# Feature-Saetze
# =========================================================================

@dataclass
class FeatureSet:
    """Ein Satz Merkmale mit seiner Train/Test-Trennung."""
    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    cols: list

    @property
    def n_features(self) -> int:
        return len(self.cols)

    def matrices(self):
        return (self.train[self.cols].values,
                self.train["faultNumber"].values,
                self.test[self.cols].values,
                self.test["faultNumber"].values)


def feature_sets(cfgs, train: dict, test: dict, combine: bool = True) -> list:
    """Einen FeatureSet je Verfahren, optional zusaetzlich die Kombination.

    Die Kombination entsteht ueber einen INNER JOIN auf
    (faultNumber, simulationRun): nur Runs, die alle Verfahren ueberlebt
    haben. Bei DyCA scheitern einzelne Runs numerisch, deshalb ist der
    kombinierte Satz kleiner als die Einzelsaetze.
    """
    sets = []
    for cfg in cfgs:
        m = cfg.method
        cols = [c for c in train[m].columns if c.startswith(cfg.prefix)]
        sets.append(FeatureSet(cfg.label, train[m], test[m], cols))

    if combine and len(cfgs) > 1:
        tr = sets[0].train
        te = sets[0].test
        for s in sets[1:]:
            tr = pd.merge(tr, s.train, on=KEYS, how="inner")
            te = pd.merge(te, s.test, on=KEYS, how="inner")
        all_cols = [c for s in sets for c in s.cols]
        sets.append(FeatureSet("+".join(s.name for s in sets), tr, te,
                               all_cols))
    return sets


def class_distribution(sets: list) -> pd.DataFrame:
    """Runs je Fehlerklasse in Train und Test, plus der Verlust gegenueber
    dem ersten Satz.

    Der Verlust zeigt, wo ein Verfahren Runs einbuesst - Klassen mit
    hohem Verlust sind Kandidaten fuer einen Selektions-Bias.
    """
    def counts(df):
        return df["faultNumber"].value_counts().sort_index()

    dist = pd.DataFrame({f"{s.name}_{split}": counts(getattr(s, split))
                         for s in sets for split in ("train", "test")})
    dist.index.name = "faultNumber"
    base = sets[0]
    for split in ("train", "test"):
        for s in sets[1:]:
            dist[f"verlust_{s.name}_{split}"] = (
                dist[f"{base.name}_{split}"] - dist[f"{s.name}_{split}"])
    return dist


# =========================================================================
# Modellvergleich
# =========================================================================

@dataclass
class ClassifyConfig:
    """Einstellungen des Modellvergleichs."""
    cv_folds: int = 5
    select_metric: str = "Balanced Accuracy CV Mean"
    random_state: int = 42
    use_gpu: bool = False
    labels: list = field(default_factory=lambda: list(LABELS))


def run_lazyclassifier(fs: FeatureSet, cc: ClassifyConfig | None = None):
    """LazyClassifier auf einem Feature-Satz, sortiert nach der CV-Metrik.

    Ausgewaehlt wird auf der TRAIN-CV, nicht auf dem Testset - Modelle
    ohne predict_proba bekommen dabei keine CV-Werte (der ROC-AUC-Scorer
    scheitert und leert alle CV-Spalten) und landen am Tabellenende.
    """
    from lazypredict.Supervised import LazyClassifier

    cc = cc or ClassifyConfig()
    Xtr, ytr, Xte, yte = fs.matrices()

    kwargs = dict(verbose=0, ignore_warnings=True, predictions=False,
                  random_state=cc.random_state, cv=cc.cv_folds)
    if cc.use_gpu:
        kwargs["use_gpu"] = True
    try:
        clf = LazyClassifier(**kwargs)
    except TypeError:                       # aeltere lazypredict-Version
        kwargs.pop("use_gpu", None)
        clf = LazyClassifier(**kwargs)

    print(f"=== {fs.name} ===")
    print(f"Features: {fs.n_features} | Train: {Xtr.shape[0]} | "
          f"Test: {Xte.shape[0]} | Klassen: {len(np.unique(ytr))} | "
          f"cv={cc.cv_folds}")
    models, _ = clf.fit(Xtr, Xte, ytr, yte)

    if cc.select_metric in models.columns:
        models = models.sort_values(cc.select_metric, ascending=False,
                                    na_position="last")
        no_cv = models[models[cc.select_metric].isna()]
        if len(no_cv) and "Balanced Accuracy" in no_cv.columns:
            top3 = no_cv.sort_values("Balanced Accuracy",
                                     ascending=False).head(3)
            names = ", ".join(f"{n} {r['Balanced Accuracy']:.4f}"
                              for n, r in top3.iterrows())
            print(f"Hinweis: {len(no_cv)} Modelle ohne CV-Werte am "
                  f"Tabellenende (ROC-AUC-Scorer scheitert ohne "
                  f"predict_proba). Staerkste nach Test-BA: {names}")
    print(f"Leaderboard sortiert nach '{cc.select_metric}' "
          f"(nur im Speicher, keine CSV).")
    return models


# =========================================================================
# Confusion-Matrizen
# =========================================================================

def default_estimator(random_state: int = 42):
    """StandardScaler + RandomForest - wie lazypredict das Modell intern
    aufbaut."""
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(random_state=random_state, n_jobs=-1))


def confusion(sets: list, cc: ClassifyConfig | None = None, estimator=None,
              report: bool = True) -> dict:
    """Fester Modelltyp auf jedem Feature-Satz, einmal auf dem Testset
    ausgewertet. Die Unterschiede liegen damit allein an den Features."""
    cc = cc or ClassifyConfig()
    results = {}
    for fs in sets:
        Xtr, ytr, Xte, yte = fs.matrices()
        model = (default_estimator(cc.random_state) if estimator is None
                 else estimator)
        model.fit(Xtr, ytr)
        y_pred = model.predict(Xte)

        cm = confusion_matrix(yte, y_pred, labels=cc.labels)
        results[fs.name] = {
            "cm": cm, "cm_norm": normalize_rows(cm),
            "bal_acc": balanced_accuracy_score(yte, y_pred),
            "n_features": fs.n_features, "n_train": len(ytr),
            "n_test": len(yte),
        }
        print(f"=== RandomForestClassifier | {fs.name} ===")
        print(f"Features: {fs.n_features} | Train: {len(ytr)} | "
              f"Test: {len(yte)} | Balanced Accuracy (Test): "
              f"{results[fs.name]['bal_acc']:.4f}")
        if report:
            print(classification_report(yte, y_pred, labels=cc.labels,
                                        digits=3, zero_division=0))

    print("Zusammenfassung (Balanced Accuracy auf dem Testset):")
    for name, res in results.items():
        print(f"  {name:12s}: {res['bal_acc']:.4f}  "
              f"({res['n_features']} Features, {res['n_test']} Test-Runs)")
    return results


def counts(results: dict, labels=None) -> dict:
    """Absolute Zaehlwerte je Feature-Satz als 21x21-DataFrame."""
    labels = list(LABELS if labels is None else labels)
    return {name: pd.DataFrame(res["cm"],
                               index=[f"true_{i}" for i in labels],
                               columns=[f"pred_{i}" for i in labels])
            for name, res in results.items()}


def plot_confusions(results: dict, labels=None, annot_min: float = 0.05):
    """Alle Matrizen nebeneinander, gemeinsame Farbskala 0..1."""
    import matplotlib.pyplot as plt

    labels = list(LABELS if labels is None else labels)
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 7.2),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    im = None
    for ax, (name, res) in zip(axes, results.items()):
        im = draw_confusion(ax, res["cm_norm"], labels, annot_min=annot_min)
        ax.set_title(f"{name}  ({res['n_features']} Features)\n"
                     f"Balanced Accuracy (Test) = {res['bal_acc']:.3f}",
                     fontsize=11)
        ax.set_xlabel("Vorhergesagte Klasse")
        ax.set_ylabel("Wahre Klasse")

    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("Anteil der wahren Klasse (Zeilensumme = 1)")
    fig.suptitle("RandomForestClassifier - Confusion-Matrizen auf dem "
                 "TEP-Testset", fontsize=14)
    plt.show()
    return fig


def report_confusions(results: dict, labels=None, top_n: int = 8) -> None:
    """Die groessten Verwechslungen je Feature-Satz als Textblock."""
    labels = list(LABELS if labels is None else labels)
    for name, res in results.items():
        print()
        print_top_confusions(res["cm"], labels, top_n, title=name)
