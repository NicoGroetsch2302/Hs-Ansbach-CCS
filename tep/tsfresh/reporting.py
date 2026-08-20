"""Vergleich der Konfigurationen: Tabellen und Balkenplot.

Drei Blickwinkel auf dieselbe summary-Tabelle:

  Hauptvergleich   ein FESTES Modell (RandomForest) ueber alle
                   Konfigurationen - die Unterschiede liegen dann allein
                   an den Features, nicht an der Modellwahl.
  Explorativ       bestes Modell je Konfiguration, ausgewaehlt auf dem
                   TESTSET. Leicht optimistisch (Winner's Curse).
  CV-Auswahl       bestes Modell je Konfiguration nach der Train-CV,
                   berichtet mit seinen Testwerten. Kein Testset-Blick.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAIN_MODEL = "RandomForestClassifier"


@dataclass
class Comparison:
    """Ergebnis von compare(): drei Sichten plus die Konfigurationsordnung."""
    order: list
    summary: pd.DataFrame
    main: pd.DataFrame                    # festes Modell je Konfiguration
    best_per_config: pd.DataFrame         # Auswahl auf dem Testset
    cv_best: pd.DataFrame | None          # Auswahl auf der Train-CV
    main_model: str = MAIN_MODEL


def compare(pipe, model: str = MAIN_MODEL, verbose: bool = True) -> Comparison:
    """Baut die drei Vergleichssichten aus pipe.summary.

    Laeuft auch OHNE Phase C, sobald die summary-CSV im Cache liegt -
    pipe.load_summary() holt sie.
    """
    summary = pipe.load_summary()
    order = list(pipe.names)

    def in_order(df):
        out = df.copy()
        out["_o"] = out["Konfiguration"].map(order.index)
        return out.sort_values("_o").drop(columns="_o")

    # --- Hauptvergleich: festes Modell ueberall -------------------------
    main = summary[summary["Modell"] == model]
    if verbose:
        if main.empty:
            print(f"ACHTUNG: kein {model} in summary - Hauptvergleich "
                  f"entfaellt.")
        else:
            print(f"Hauptvergleich - {model} je Konfiguration (Testset):")
            print(in_order(main).drop(columns="Modell").to_string(index=False))

    # --- Explorativ: Maximum auf dem Testset ----------------------------
    # Ueber idxmax die GANZE Zeile holen - groupby().first() wuerde
    # spaltenweise arbeiten und koennte Modellname und Score aus
    # verschiedenen Zeilen mischen.
    best = summary.loc[
        summary.groupby("Konfiguration")["MacroF1"].idxmax()]
    best = in_order(best).reset_index(drop=True)
    if verbose:
        print("\nExplorativ - bestes Modell je Konfiguration "
              "(Auswahl auf dem Testset):")
        print(best.to_string(index=False))

    # --- Sauber: Modellauswahl ueber die Train-CV -----------------------
    # Modelle ohne predict_proba haben keine CV-Werte und fallen hier
    # heraus - im Test-Maximum oben sind sie weiter dabei. Aeltere
    # summary-CSVs haben die Spalte gar nicht.
    cv_best = None
    if ("BalancedAccCVMean" in summary.columns
            and summary["BalancedAccCVMean"].notna().any()):
        cvs = summary.dropna(subset=["BalancedAccCVMean"])
        cv_best = cvs.loc[
            cvs.groupby("Konfiguration")["BalancedAccCVMean"].idxmax()]
        cv_best = in_order(cv_best).reset_index(drop=True)
        if verbose:
            n_cv = cvs.groupby("Konfiguration").size()
            print("\nCV-Auswahl - bestes Modell je Konfiguration nach "
                  "'Balanced Accuracy CV Mean'")
            print("(Auswahl NUR auf Train; BalancedAcc/MacroF1 sind die "
                  "Testwerte):")
            print(cv_best.to_string(index=False))
            print(f"\nModelle mit CV-Werten je Konfiguration: "
                  f"{n_cv.min()}-{n_cv.max()} (ohne predict_proba gibt es "
                  f"keine).")
    elif verbose:
        print("\nKeine CV-Spalten in summary (aeltere CSV) -> CV-Auswahl "
              "entfaellt.")

    return Comparison(order=order, summary=summary, main=main,
                      best_per_config=best, cv_best=cv_best, main_model=model)


def plot_comparison(cmp: Comparison, cfg, figsize=(14.5, 5.5)):
    """Balken = festes Modell (fairer Konfigurationsvergleich). Rauten =
    bestes Modell je Konfiguration, ausgewaehlt auf dem TESTSET (leicht
    optimistisch). Offene Kreise = das auf der Train-CV gewaehlte Modell,
    ebenfalls mit seinem TESTwert - die Luecke zwischen Raute und Kreis ist
    der Preis des Winner's Curse."""
    import matplotlib.pyplot as plt

    order = cmp.order
    main_d = (cmp.main.set_index("Konfiguration").reindex(order)
              if not cmp.main.empty else pd.DataFrame(index=order))
    best_d = cmp.best_per_config.set_index("Konfiguration").reindex(order)

    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Blau/Orange: unterscheidet sich auf der Blau-Gelb-Achse, die bei Prot-
    # und Deuteranopie erhalten bleibt; Werte stehen zusaetzlich an den
    # Balken.
    nan_col = pd.Series(np.nan, index=order)
    b1 = ax.bar(x - w / 2, main_d.get("MacroF1", nan_col), w,
                label=f"Macro-F1 ({cmp.main_model})", color="#4C78A8",
                zorder=2)
    b2 = ax.bar(x + w / 2, main_d.get("BalancedAcc", nan_col), w,
                label=f"Balanced Accuracy ({cmp.main_model})",
                color="#F58518", zorder=2)
    for bars in (b1, b2):
        vals = [b.get_height() for b in bars]
        ax.bar_label(bars,
                     labels=["" if np.isnan(v) else f"{v:.3f}" for v in vals],
                     fontsize=7, padding=2)

    ax.scatter(x, best_d["MacroF1"], marker="D", s=32, color="#2f2f2f",
               zorder=3, label="bestes Modell (Macro-F1, Auswahl auf Test)")

    if cmp.cv_best is not None:
        cv_d = cmp.cv_best.set_index("Konfiguration").reindex(order)
        ax.scatter(x, cv_d["MacroF1"], marker="o", s=44, facecolors="none",
                   edgecolors="#2f2f2f", linewidths=1.2, zorder=4,
                   label="CV-gewaehltes Modell (Macro-F1 auf Test)")

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("Score auf dem Testset")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Legende ueber der Achse; im Plot selbst wuerde sie Balken oder Marker
    # verdecken. pad haelt den Titel frei.
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2,
              frameon=False)
    ax.set_title(f"TSFresh: {cmp.main_model} je Konfiguration (Balken) vs. "
                 f"bestes Modell (Raute) - {cfg.label}, Top-{cfg.top_k} "
                 f"Features, gemeinsame Runs", pad=46)
    plt.show()
    return fig
