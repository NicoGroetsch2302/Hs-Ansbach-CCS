"""Darstellungsbausteine, die beide Notebook-Familien brauchen.

Confusion-Matrizen kommen sowohl in `tep.tsfresh` (je Projektion) als auch
in `tep.eigen` (je Feature-Satz) vor. Gezeichnet werden sie gleich - hier
steht das Wie, damit es nicht in zwei Fassungen driftet.

Die Klassenachse ist immer `LABELS` (Fault 0..20). Der TEP-Datensatz hat
genau diese 21 Klassen, und weil die Beschriftungen zugleich die
Zellindizes sind, koennen Achse und Matrix nicht auseinanderlaufen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import LABELS


def normalize_rows(cm: np.ndarray) -> np.ndarray:
    """Zeilenweise normieren: Zelle (i, j) = Anteil der wahren Klasse i, der
    als j vorhergesagt wurde. Die Diagonale ist damit der Recall.

    Leere Zeilen (Klasse kommt im Testset nicht vor) werden zu Null statt
    zu NaN - sonst reisst der Plot Loecher.
    """
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sums, out=np.zeros(cm.shape, dtype=float),
                     where=row_sums > 0)


def counts_frame(cm: np.ndarray) -> pd.DataFrame:
    """Absolute Zaehlwerte als 21x21-DataFrame mit sprechenden Achsen."""
    return pd.DataFrame(cm, index=[f"true_{i}" for i in LABELS],
                        columns=[f"pred_{i}" for i in LABELS])


def draw_confusion(ax, cm_norm: np.ndarray, annot_min: float | None,
                   tick_step: int = 1, gridlines: bool = True,
                   label_fontsize: int = 7):
    """Zeichnet EINE normierte Confusion-Matrix in eine Achse.

    annot_min : ab diesem Anteil wird eine Zelle beschriftet. None schaltet
                die Beschriftung ab - noetig, sobald viele 21x21-Panels
                nebeneinander stehen und die Schrift unlesbar wuerde.
    tick_step : nur jede n-te Klasse beschriften. Bei kleinen Panels waeren
                21 Ticks je Achse eine graue Linie.
    gridlines : duenne weisse Trennlinien auf den Zellgrenzen. Macht
                einzelne Zellen im 21x21-Raster abzaehlbar.

    Rueckgabe: das AxesImage, fuer eine gemeinsame Colorbar.
    """
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)

    ticks = LABELS[::tick_step]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, fontsize=label_fontsize)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks, fontsize=label_fontsize)

    if gridlines:
        ax.set_xticks(np.arange(-0.5, len(LABELS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(LABELS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5)
        ax.tick_params(which="minor", length=0)

    if annot_min is not None:
        # Schriftfarbe wechselt auf dunklen Zellen zu weiss (Kontrast).
        for i in LABELS:
            for j in LABELS:
                v = cm_norm[i, j]
                if v < annot_min:
                    continue
                ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center",
                        va="center", fontsize=6,
                        color="white" if v > 0.5 else "#1f2937")
    return im


def plot_grid(mats, titles, suptitle: str, ncols: int | None = None,
              big: bool = False, annot_min: float | None = None):
    """Mehrere normierte Matrizen als Raster, gemeinsame Farbskala 0..1.

    big : grosse Panels (wenige nebeneinander) bekommen jeden Tick und
          Gitterlinien, kleine nicht - dort waere beides unlesbar.
          Steuert auch Panelgroesse und Colorbar.
    annot_min : ab diesem Anteil wird eine Zelle beschriftet, None = gar
          nicht. Im kleinen Panel bleibt die Schrift ohnehin unlesbar.
    ncols : None = alles in eine Zeile.
    """
    import matplotlib.pyplot as plt

    ncols = ncols or len(mats)
    nrows = int(np.ceil(len(mats) / ncols))
    w, h = (6.0, 7.2) if big else (3.9, 4.4)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    im = None
    for ax, cm_norm, title in zip(axes, mats, titles):
        im = draw_confusion(ax, cm_norm, annot_min=annot_min,
                            tick_step=1 if big else 2, gridlines=big,
                            label_fontsize=7 if big else 6)
        ax.set_title(title, fontsize=11 if big else 10)
        ax.set_xlabel("Vorhergesagte Klasse", fontsize=8)
        ax.set_ylabel("Wahre Klasse", fontsize=8)

    for ax in axes[len(mats):]:             # ungenutzte Panels ausblenden
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.85 if big else 0.6,
                        pad=0.02)
    cbar.set_label("Anteil der wahren Klasse (Zeilensumme = 1)")
    fig.suptitle(suptitle, fontsize=14)
    plt.show()
    return fig


def print_top_confusions(cm: np.ndarray, top_n: int = 8,
                         title: str = "") -> None:
    """Die groessten Off-Diagonal-Eintraege als Textblock.

    Genau das, was man in der Grafik sucht: welche Klasse wird
    systematisch mit welcher verwechselt.
    """
    if title:
        print(f"=== {title}: groesste Verwechslungen (Top {top_n}) ===")
    err = cm.astype(float).copy()
    np.fill_diagonal(err, 0.0)              # Diagonale = Treffer
    for flat in np.argsort(err, axis=None)[::-1][:top_n]:
        i, j = np.unravel_index(flat, err.shape)
        n_runs = int(err[i, j])
        if n_runs == 0:
            break                           # ab hier nur noch Nullen
        print(f"  wahr {LABELS[i]:>2d} -> vorhergesagt {LABELS[j]:>2d}: "
              f"{n_runs:4d} Runs ({n_runs / cm[i].sum():.1%} der Klasse)")
