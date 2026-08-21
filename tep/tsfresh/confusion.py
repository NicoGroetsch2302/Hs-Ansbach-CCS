"""Confusion-Matrizen je Konfiguration.

Fester Modelltyp (RandomForest, wie im Hauptvergleich) ueber alle
Konfigurationen -> die Unterschiede zwischen den Matrizen liegen allein an
den Features. Datenbasis exakt wie benchmark_models(): gemeinsame Runs,
gleiche NaN-Behandlung, StandardScaler + RandomForest (wie lazypredict intern).

`confusion()` liefert ein dict:

    {"order":   Konfigurationsnamen in Notebook-Reihenfolge,
     "pred":    DataFrame Konfiguration/run_id/y_true/y_pred,
     "results": name -> {cm, cm_norm, n_test, bal_acc, macro_f1},
     "counts":  name -> DataFrame der absoluten Zaehlwerte}

Die Vorhersagen werden als CSV im Cache abgelegt. Nach einem Kernel-
Neustart laufen die Plotfunktionen damit ganz ohne die drei Schritte.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from ..core import LABELS, default_estimator
from ..plotting import counts_frame, draw_confusion, normalize_rows
from ..plotting import plot_grid as grid
from ..plotting import print_top_confusions
from .pipeline import common_runs, matrices


def names(cm: dict) -> list:
    """Konfigurationen mit Matrix, in Notebook-Reihenfolge."""
    return [n for n in cm["order"] if n in cm["results"]]


def best(cm: dict) -> str:
    """Die Konfiguration mit dem hoechsten Macro-F1."""
    return max(cm["results"], key=lambda n: cm["results"][n]["macro_f1"])


def recall_table(cm: dict) -> pd.DataFrame:
    """Die Diagonalen aller Matrizen in einer Tabelle: zeigt, WELCHE
    Faults eine Projektion traegt und welche sie verliert. Der
    Skalarvergleich (Macro-F1) sagt das nicht - zwei Konfigurationen mit
    gleichem Mittel koennen an voellig verschiedenen Klassen scheitern.
    """
    tab = pd.DataFrame(
        {n: np.diag(cm["results"][n]["cm_norm"]) for n in names(cm)},
        index=LABELS).T
    tab.index.name = "Konfiguration"
    tab.columns.name = "faultNumber"
    return tab


def confusion(config_names: list, pred_path: str, train_top: dict | None =
              None, test_top: dict | None = None, *, refit: bool = False,
              estimator=None, random_state: int = 42,
              summary_path: str | None = None) -> dict:
    """Vorhersagen holen (Cache oder Neuberechnung) und Matrizen bauen.

    refit=False nutzt den Vorhersage-Cache, falls vorhanden. refit=True
    ignoriert ihn und fittet neu - noetig, wenn estimator gewechselt wird;
    dann werden train_top und test_top gebraucht.
    """
    order = list(config_names)

    if os.path.exists(pred_path) and not refit:
        pred = pd.read_csv(pred_path)
        print(f"Vorhersagen aus {pred_path} geladen "
              f"({pred['Konfiguration'].nunique()} Konfigurationen, "
              f"{len(pred)} Zeilen). refit=True erzwingt Neuberechnung.")
    else:
        if not train_top or not test_top:
            raise RuntimeError(
                "train_top/test_top fehlen und es gibt keinen Vorhersage-"
                "Cache -> zuerst select_features() und apply_features() "
                "ausfuehren (laufen aus dem Chunk-Cache).")
        idx_tr, idx_te = common_runs(train_top, test_top)
        print(f"Gemeinsame Runs: Train {len(idx_tr)}, Test {len(idx_te)}")

        parts = []
        for name in order:
            t0 = time.perf_counter()
            Xtr_v, Xte_v, ytr, yte, te_index = matrices(name, train_top,
                                                        test_top)

            # Fit auf dem GANZEN Trainingssatz, danach genau EINE Auswertung
            # auf dem echten Testset.
            model = (default_estimator(random_state) if estimator is None
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
        pred.to_csv(pred_path, index=False)
        print(f"\nVorhersagen gespeichert: {pred_path}")

    # --- Matrizen aufbauen -------------------------------------------
    # LABELS erzwingt die Reihenfolge 0..20 auch fuer Klassen, die nie
    # vorhergesagt werden -> alle Matrizen sind deckungsgleich.
    results, counts = {}, {}
    for name in order:
        g = pred[pred["Konfiguration"] == name]
        if g.empty:
            continue
        cm = confusion_matrix(g["y_true"], g["y_pred"], labels=LABELS)
        # Zeilenweise normiert: Zelle (i,j) = Anteil der wahren Klasse i, der
        # als j vorhergesagt wurde; Diagonale = Recall.
        results[name] = {
            "cm": cm, "cm_norm": normalize_rows(cm), "n_test": len(g),
            "bal_acc": balanced_accuracy_score(g["y_true"], g["y_pred"]),
            "macro_f1": f1_score(g["y_true"], g["y_pred"], average="macro",
                                 zero_division=0),
        }
        counts[name] = counts_frame(cm)

    # Abgleich mit benchmark_models(): dieser RF muss die RandomForest-
    # Zeile aus summary reproduzieren. Kleine Abweichungen sind normal
    # (RF-Zufallskomponente; lazypredict haengt zusaetzlich einen
    # SimpleImputer vor den Scaler).
    if summary_path and os.path.exists(summary_path):
        rf = pd.read_csv(summary_path)
        rf = rf[rf["Modell"] == "RandomForestClassifier"] \
            .set_index("Konfiguration")
        d = {n: abs(results[n]["macro_f1"] - rf.loc[n, "MacroF1"])
             for n in results if n in rf.index}
        if d:
            w = max(d, key=d.get)
            print(f"Abgleich mit benchmark_models(): groesste "
                  f"Macro-F1-Abweichung "
                  f"{d[w]:.4f} ({w})")

    if not results:
        raise RuntimeError(
            f"Keine Matrix berechenbar: {pred_path} enthaelt keine Zeile zu "
            f"den Konfigurationen {order}. Passt der Vorhersage-Cache noch "
            f"zur Konfiguration? refit=True rechnet ihn neu.")

    cm_res = {"order": order, "pred": pred, "results": results,
              "counts": counts}
    print(f"\n{len(results)} Matrizen berechnet. Absolute Zaehlwerte stehen "
          f"unter 'counts', z.B. cm['counts']['{names(cm_res)[0]}'].")
    return cm_res


def plot_grid(cm: dict, top_k: int = 100, ncols: int = 4):
    """Alle Matrizen im Raster, zeilenweise normiert (Diagonale = Recall),
    gemeinsame Farbskala 0..1 fuer direkte Vergleichbarkeit.

    Die Zellen werden hier NICHT beschriftet - bei vielen 21x21-Panels waere
    die Schrift unlesbar. Exakte Zahlen: cm["counts"][name] oder plot_detail().
    """
    return grid(
        [cm["results"][n]["cm_norm"] for n in names(cm)],
        [f"{n}\nMacro-F1 {cm['results'][n]['macro_f1']:.3f} | "
         f"BA {cm['results'][n]['bal_acc']:.3f}" for n in names(cm)],
        "RandomForestClassifier - Confusion-Matrizen je Konfiguration "
        f"(Testset, Top-{top_k} Features)",
        ncols=ncols)


def plot_detail(cm: dict, focus: str | None = None,
                annot_min: float = 0.05, top_n: int = 8,
                report: bool = True):
    """Eine Konfiguration gross, mit beschrifteten auffaelligen Zellen.

    focus=None waehlt die beste nach Macro-F1. annot_min ist die Schwelle,
    ab der eine Zelle beschriftet wird - alles darunter bliebe im
    21x21-Raster ohnehin unlesbar. top_n steuert die Liste der groessten
    Verwechslungen darunter.
    """
    import matplotlib.pyplot as plt

    focus = focus or best(cm)
    res = cm["results"][focus]
    g = cm["pred"][cm["pred"]["Konfiguration"] == focus]

    # figsize: imshow erzwingt ein quadratisches Panel - die Hoehe ist also
    # Breite minus Colorbar/Beschriftung, plus der zweizeilige Titel. Wird
    # die Figur flacher, schneidet sie die erste Titelzeile ab.
    fig, ax = plt.subplots(figsize=(9, 9.2), constrained_layout=True)
    im = draw_confusion(ax, res["cm_norm"], annot_min=annot_min,
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
        print(classification_report(g["y_true"], g["y_pred"], labels=LABELS,
                                    digits=3, zero_division=0))

        # Die groessten Off-Diagonal-Eintraege = die systematischen
        # Verwechslungen. Genau das, was man in der Grafik sucht, hier als
        # sortierte Liste.
        print_top_confusions(res["cm"], top_n, title=focus)

    return fig, focus


def plot_recall(cm: dict, n_worst: int = 5):
    """Recall je Fault-Klasse und Konfiguration als Heatmap.

    Rueckgabe (fig, tab) wie plot_detail - main.py speichert die Figur.
    """
    import matplotlib.pyplot as plt

    tab = recall_table(cm)
    fig, ax = plt.subplots(figsize=(13, 0.42 * len(tab) + 1.8),
                           constrained_layout=True)
    # aspect="auto": die Tabelle ist breit und flach, quadratische Zellen
    # wuerden die Figur unnoetig hoch machen.
    im = ax.imshow(tab.to_numpy(), cmap="Blues", vmin=0.0, vmax=1.0,
                   aspect="auto")

    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, fontsize=8)
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
    return fig, tab
