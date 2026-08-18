"""Darstellungsbausteine, die beide Notebook-Familien brauchen.

Confusion-Matrizen kommen sowohl in `tep.tsfresh` (je Projektion) als auch
in `tep.eigen` (je Feature-Satz) vor. Gezeichnet werden sie gleich - hier
steht das Wie, damit es nicht in zwei Fassungen driftet.
"""

from __future__ import annotations

import numpy as np


def normalize_rows(cm: np.ndarray) -> np.ndarray:
    """Zeilenweise normieren: Zelle (i, j) = Anteil der wahren Klasse i, der
    als j vorhergesagt wurde. Die Diagonale ist damit der Recall.

    Leere Zeilen (Klasse kommt im Testset nicht vor) werden zu Null statt
    zu NaN - sonst reisst der Plot Loecher.
    """
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sums, out=np.zeros(cm.shape, dtype=float),
                     where=row_sums > 0)


def draw_confusion(ax, cm_norm: np.ndarray, labels, annot_min: float | None,
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
    labels = list(labels)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_xticks(labels[::tick_step])
    ax.set_xticklabels(labels[::tick_step], fontsize=label_fontsize)
    ax.set_yticks(labels[::tick_step])
    ax.set_yticklabels(labels[::tick_step], fontsize=label_fontsize)

    if gridlines:
        ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5)
        ax.tick_params(which="minor", length=0)

    if annot_min is not None:
        # Schriftfarbe wechselt auf dunklen Zellen zu weiss (Kontrast).
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = cm_norm[i, j]
                if v < annot_min:
                    continue
                ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center",
                        va="center", fontsize=6,
                        color="white" if v > 0.5 else "#1f2937")
    return im


def top_confusions(cm: np.ndarray, labels, top_n: int = 8) -> list:
    """Die groessten Off-Diagonal-Eintraege als sortierte Liste.

    Genau das, was man in der Grafik sucht: welche Klasse wird
    systematisch mit welcher verwechselt.

    Rueckgabe: Liste von (wahr, vorhergesagt, n_runs, anteil).
    """
    labels = list(labels)
    err = cm.astype(float).copy()
    np.fill_diagonal(err, 0.0)              # Diagonale = Treffer
    out = []
    for flat in np.argsort(err, axis=None)[::-1][:top_n]:
        i, j = np.unravel_index(flat, err.shape)
        n_runs = int(err[i, j])
        if n_runs == 0:
            break                           # ab hier nur noch Nullen
        out.append((labels[i], labels[j], n_runs, n_runs / cm[i].sum()))
    return out


def print_top_confusions(cm: np.ndarray, labels, top_n: int = 8,
                         title: str = "") -> None:
    """top_confusions() als Textblock."""
    if title:
        print(f"=== {title}: groesste Verwechslungen (Top {top_n}) ===")
    for true_c, pred_c, n_runs, share in top_confusions(cm, labels, top_n):
        print(f"  wahr {true_c:>2d} -> vorhergesagt {pred_c:>2d}: "
              f"{n_runs:4d} Runs ({share:.1%} der Klasse)")
