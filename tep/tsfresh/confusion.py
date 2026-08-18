"""Confusion-Matrizen je Konfiguration.

Fester Modelltyp (RandomForest, wie im Hauptvergleich) ueber alle
Konfigurationen -> die Unterschiede zwischen den Matrizen liegen allein an
den Features. Datenbasis exakt wie Phase C: gemeinsame Runs, gleiche
NaN-Behandlung, StandardScaler + RandomForest (wie lazypredict intern).

Die Vorhersagen werden als CSV im Cache abgelegt. Nach einem Kernel-
Neustart laufen die Plotfunktionen damit ganz ohne Phase A/B/C.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..core import LABELS
from ..plotting import (draw_confusion, normalize_rows,
                        print_top_confusions)


@dataclass
class ConfusionResults:
    """Matrizen, absolute Zaehlwerte und die zugrunde liegenden Vorhersagen."""
    order: list                  # Konfigurationsnamen in Notebook-Reihenfolge
    pred: pd.DataFrame           # Konfiguration, run_id, y_true, y_pred
    results: dict                # name -> cm, cm_norm, n_test, bal_acc, macro_f1
    counts: dict                 # name -> DataFrame der absoluten Zaehlwerte
    labels: list = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = list(LABELS)

    @property
    def names(self) -> list:
        """Konfigurationen mit Matrix, in Notebook-Reihenfolge."""
        return [n for n in self.order if n in self.results]

    def best(self) -> str:
        return max(self.results, key=lambda n: self.results[n]["macro_f1"])

    def recall_table(self) -> pd.DataFrame:
        """Die Diagonalen aller Matrizen in einer Tabelle: zeigt, WELCHE
        Faults eine Projektion traegt und welche sie verliert. Der
        Skalarvergleich (Macro-F1) sagt das nicht - zwei Konfigurationen mit
        gleichem Mittel koennen an voellig verschiedenen Klassen scheitern.
        """
        tab = pd.DataFrame(
            {n: np.diag(self.results[n]["cm_norm"]) for n in self.names},
            index=self.labels).T
        tab.index.name = "Konfiguration"
        tab.columns.name = "faultNumber"
        return tab


def default_estimator(random_state: int = 42):
    """StandardScaler + RandomForest - der Modelltyp des Hauptvergleichs."""
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(random_state=random_state, n_jobs=-1))


def confusion(pipe, refit: bool = False, estimator=None,
              labels=None) -> ConfusionResults:
    """Vorhersagen holen (Cache oder Neuberechnung) und Matrizen bauen.

    refit=False nutzt den Vorhersage-Cache, falls vorhanden. refit=True
    ignoriert ihn und fittet neu - noetig, wenn estimator gewechselt wird.
    """
    cfg = pipe.cfg
    labels = list(LABELS if labels is None else labels)
    order = list(pipe.names)
    path = cfg.cm_pred_path

    if os.path.exists(path) and not refit:
        pred = pd.read_csv(path)
        print(f"Vorhersagen aus {path} geladen "
              f"({pred['Konfiguration'].nunique()} Konfigurationen, "
              f"{len(pred)} Zeilen). refit=True erzwingt Neuberechnung.")
    else:
        if not pipe.train_top or not pipe.test_top:
            raise RuntimeError(
                "train_top/test_top fehlen und es gibt keinen Vorhersage-"
                "Cache -> zuerst Phase A und B ausfuehren (laufen aus dem "
                "Chunk-Cache).")
        idx_tr, idx_te = pipe.common_runs()
        if idx_tr is not None:
            print(f"Gemeinsame Runs: Train {len(idx_tr)}, Test {len(idx_te)}")

        parts = []
        for name in order:
            t0 = time.perf_counter()
            Xtr_v, Xte_v, ytr, yte, te_index = pipe.matrices(name)

            # Fit auf dem GANZEN Trainingssatz, danach genau EINE Auswertung
            # auf dem echten Testset.
            model = (default_estimator(cfg.random_state) if estimator is None
                     else estimator)
            model.fit(Xtr_v, ytr)
            y_pred = model.predict(Xte_v)

            parts.append(pd.DataFrame({"Konfiguration": name,
                                       "run_id": np.asarray(te_index),
                                       "y_true": yte,
                                       "y_pred": y_pred}))
            print(f"[{name:22s}] MacroF1 "
                  f"{f1_score(yte, y_pred, average='macro', zero_division=0):.4f}"
                  f" | BA {balanced_accuracy_score(yte, y_pred):.4f} "
                  f"({time.perf_counter() - t0:.0f} s)")

        pred = pd.concat(parts, ignore_index=True)
        pred.to_csv(path, index=False)
        print(f"\nVorhersagen gespeichert: {path}")

    # --- Matrizen aufbauen ---------------------------------------------
    # labels erzwingt die Reihenfolge 0..20 auch fuer Klassen, die nie
    # vorhergesagt werden -> alle Matrizen sind deckungsgleich.
    results, counts = {}, {}
    for name in order:
        g = pred[pred["Konfiguration"] == name]
        if g.empty:
            continue
        cm = confusion_matrix(g["y_true"], g["y_pred"], labels=labels)
        # Zeilenweise normiert: Zelle (i,j) = Anteil der wahren Klasse i, der
        # als j vorhergesagt wurde; Diagonale = Recall.
        cm_norm = normalize_rows(cm)
        results[name] = {
            "cm": cm, "cm_norm": cm_norm, "n_test": len(g),
            "bal_acc": balanced_accuracy_score(g["y_true"], g["y_pred"]),
            "macro_f1": f1_score(g["y_true"], g["y_pred"], average="macro",
                                 zero_division=0),
        }
        counts[name] = pd.DataFrame(cm,
                                    index=[f"true_{i}" for i in labels],
                                    columns=[f"pred_{i}" for i in labels])

    # Abgleich mit Phase C: dieser RF muss die RandomForest-Zeile aus summary
    # reproduzieren. Kleine Abweichungen sind normal (RF-Zufallskomponente,
    # lazypredict haengt zusaetzlich einen SimpleImputer vor den Scaler).
    if os.path.exists(cfg.summary_path):
        rf = pd.read_csv(cfg.summary_path)
        rf = rf[rf["Modell"] == "RandomForestClassifier"] \
            .set_index("Konfiguration")
        d = {n: abs(results[n]["macro_f1"] - rf.loc[n, "MacroF1"])
             for n in results if n in rf.index}
        if d:
            w = max(d, key=d.get)
            print(f"Abgleich mit Phase C: groesste Macro-F1-Abweichung "
                  f"{d[w]:.4f} ({w})")

    if not results:
        raise RuntimeError(
            f"Keine Matrix berechenbar: {path} enthaelt keine Zeile zu den "
            f"Konfigurationen {order}. Passt der Vorhersage-Cache noch zur "
            f"Konfiguration? refit=True rechnet ihn neu.")

    res = ConfusionResults(order=order, pred=pred, results=results,
                           counts=counts, labels=labels)
    print(f"\n{len(results)} Matrizen berechnet. Absolute Zaehlwerte stehen "
          f"in .counts, z.B. cm.counts['{res.names[0]}'].")
    return res


def plot_grid(cm: ConfusionResults, cfg, ncols: int = 4):
    """Alle Matrizen im Raster, zeilenweise normiert (Diagonale = Recall),
    gemeinsame Farbskala 0..1 fuer direkte Vergleichbarkeit.

    Die Zellen werden hier NICHT beschriftet - bei vielen 21x21-Panels waere
    die Schrift unlesbar. Exakte Zahlen: cm.counts[name] oder plot_detail().
    """
    import matplotlib.pyplot as plt

    names = cm.names
    labels = cm.labels
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.9 * ncols, 4.4 * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    im = None
    for ax, name in zip(axes, names):
        res = cm.results[name]
        # annot_min=None und tick_step=2: im kleinen Panel waeren
        # Zellbeschriftung und 21 Ticks je Achse unlesbar.
        im = draw_confusion(ax, res["cm_norm"], labels, annot_min=None,
                            tick_step=2, gridlines=False, label_fontsize=6)
        ax.set_title(f"{name}\nMacro-F1 {res['macro_f1']:.3f} | "
                     f"BA {res['bal_acc']:.3f}", fontsize=10)
        ax.set_xlabel("Vorhergesagte Klasse", fontsize=8)
        ax.set_ylabel("Wahre Klasse", fontsize=8)

    for ax in axes[len(names):]:            # ungenutzte Panels ausblenden
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("Anteil der wahren Klasse (Zeilensumme = 1)")
    fig.suptitle("RandomForestClassifier - Confusion-Matrizen je "
                 f"Konfiguration (Testset, Top-{cfg.top_k} Features)",
                 fontsize=14)
    plt.show()
    return fig


def plot_detail(cm: ConfusionResults, focus: str | None = None,
                annot_min: float = 0.05, top_n: int = 8, report: bool = True):
    """Eine Konfiguration gross, mit beschrifteten auffaelligen Zellen.

    focus=None waehlt die beste nach Macro-F1. annot_min ist die Schwelle,
    ab der eine Zelle beschriftet wird - alles darunter bliebe im
    21x21-Raster ohnehin unlesbar. top_n steuert die Liste der groessten
    Verwechslungen darunter.
    """
    import matplotlib.pyplot as plt

    focus = focus or cm.best()
    res = cm.results[focus]
    labels = cm.labels
    g = cm.pred[cm.pred["Konfiguration"] == focus]

    # figsize: imshow erzwingt ein quadratisches Panel - die Hoehe ist also
    # Breite minus Colorbar/Beschriftung, plus der zweizeilige Titel. Wird
    # die Figur flacher, schneidet sie die erste Titelzeile ab.
    fig, ax = plt.subplots(figsize=(9, 9.2), constrained_layout=True)
    im = draw_confusion(ax, res["cm_norm"], labels, annot_min=annot_min,
                        tick_step=1, gridlines=True, label_fontsize=7)

    ax.set_title(f"RandomForestClassifier | {focus} - Confusion-Matrix "
                 f"(Testset)\nMacro-F1 {res['macro_f1']:.3f} | "
                 f"Balanced Accuracy {res['bal_acc']:.3f} | "
                 f"{res['n_test']} Test-Runs", fontsize=12)
    ax.set_xlabel("Vorhergesagte Klasse")
    ax.set_ylabel("Wahre Klasse")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Anteil der wahren Klasse (Zeilensumme = 1)")
    plt.show()

    if report:
        # zero_division=0: Klassen ohne einzige Vorhersage bekommen
        # Precision 0 statt einer UndefinedMetricWarning.
        print(f"=== {focus}: Bericht je Klasse ===")
        print(classification_report(g["y_true"], g["y_pred"], labels=labels,
                                    digits=3, zero_division=0))

        # Die groessten Off-Diagonal-Eintraege = die systematischen
        # Verwechslungen. Genau das, was man in der Grafik sucht, hier als
        # sortierte Liste.
        print_top_confusions(res["cm"], labels, top_n, title=focus)

    return fig, focus


def plot_recall(cm: ConfusionResults, n_worst: int = 5):
    """Recall je Fault-Klasse und Konfiguration als Heatmap."""
    import matplotlib.pyplot as plt

    tab = cm.recall_table()
    labels = cm.labels
    fig, ax = plt.subplots(figsize=(13, 0.42 * len(tab) + 1.8),
                           constrained_layout=True)
    # aspect="auto": die Tabelle ist breit und flach, quadratische Zellen
    # wuerden die Figur unnoetig hoch machen.
    im = ax.imshow(tab.to_numpy(), cmap="Blues", vmin=0.0, vmax=1.0,
                   aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(len(tab)))
    ax.set_yticklabels(tab.index, fontsize=8)
    ax.set_xlabel("Fault-Klasse")
    ax.set_title("Recall je Klasse (Diagonale der Confusion-Matrix) - "
                 "RandomForestClassifier auf dem Testset")

    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.iat[i, j]
            ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center",
                    fontsize=6, color="white" if v > 0.5 else "#1f2937")

    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02).set_label("Recall")
    plt.show()

    print("Schwierigste Klassen (mittlerer Recall ueber alle "
          "Konfigurationen):")
    print(tab.mean(axis=0).sort_values().head(n_worst).round(3).to_string())
    return tab
