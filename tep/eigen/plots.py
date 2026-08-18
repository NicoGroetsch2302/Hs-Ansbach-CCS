"""Die Standardplots der Eigenwert-Notebooks.

Vier Ansichten auf dieselbe Aggregattabelle, alle mit einem Subplot je
Fehlerklasse:

    plot_means   mittleres Spektrum         (blau)
    plot_stds    Standardabweichung         (orange)
    plot_cv      Variationskoeffizient      (gruen)  CV = Std / Mittelwert
    plot_bars    erste k Werte als Balken, mit Fault-0-Overlay

Fuer skalare Verfahren (LDA) gibt es stattdessen `plot_scalar`, das
Mittelwert, Streuung und CV ueber die Fehlerklassen zeigt.
"""

from __future__ import annotations

import numpy as np

from .aggregate import value_columns
from .config import SpectrumConfig
from .spectra import get


def transform(vals, mode: str = "log"):
    """Transformiert einen Wertevektor fuer die Darstellung.

    "linear"   keine Transformation. Gut lesbar, wenn die Werte ohnehin
               in [0, 1] liegen (DyCA, CVA).
    "log"      log10. Macht Groessenordnungsunterschiede sichtbar - noetig,
               sobald das volle Spektrum geplottet wird (die hinteren
               Eigenwerte liegen bei 1e-8 und darunter).
    "relative" Wert / Wert_1. Zeigt nur das Profil des Spektrums,
               unabhaengig vom Absolutniveau einer Fehlerklasse.
    """
    vals = np.asarray(vals, dtype=float)
    if mode == "linear":
        return vals
    if mode == "log":
        # log10(0) waere -inf und wuerde den Plot zerreissen.
        return np.log10(np.maximum(vals, 1e-12))
    if mode == "relative":
        if vals[0] <= 0:
            # Defensiv: bei Wert_1 <= 0 (numerisch entartet) waere die
            # Normierung undefiniert.
            return vals
        return vals / vals[0]
    raise ValueError(f"Unbekannter mode: {mode!r}")


def _stat_cols(cfg, agg_df, stat: str) -> list:
    """Die `_mean`- bzw. `_std`-Spalten, nach Komponentenindex sortiert
    und auf cfg.k_max begrenzt."""
    prefix = cfg.prefix
    cols = [c for c in agg_df.columns
            if c.startswith(prefix) and c.endswith(f"_{stat}")]
    cols = sorted(cols, key=lambda c: int(c[len(prefix):].split("_")[0]))
    return cols[:cfg.k_max]


def _grid(cfg, n_faults):
    import matplotlib.pyplot as plt
    ncols = cfg.ncols
    nrows = int(np.ceil(n_faults / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows),
                             sharex=True, sharey=True)
    return fig, np.array(axes).reshape(-1)


def _finish(fig, axes, n_used, title, xlabel, ylabel):
    import matplotlib.pyplot as plt
    for ax in axes[n_used:]:
        fig.delaxes(ax)
    fig.suptitle(title, fontsize=14)
    fig.text(0.5, 0.04, xlabel, ha="center")
    fig.text(0.04, 0.5, ylabel, va="center", rotation="vertical")
    plt.tight_layout(rect=[0.05, 0.05, 1.0, 0.95])
    plt.show()
    return fig


def _lines(cfg, agg_df, values_of, color, title, ylabel):
    """Gemeinsames Geruest der drei Linienplots."""
    faults = np.sort(agg_df["faultNumber"].unique())
    fig, axes = _grid(cfg, len(faults))
    K = None
    for i, fault in enumerate(faults):
        row = agg_df[agg_df["faultNumber"] == fault].iloc[0]
        y = values_of(row)
        K = len(y)
        axes[i].plot(np.arange(1, K + 1), y, "o-", color=color, markersize=4)
        axes[i].set_title(f"Fault {fault}", fontsize=9)
        axes[i].grid(True, alpha=0.3)
    return _finish(fig, axes, len(faults), title.format(K=K),
                   f"{cfg.label}-Komponenten-Index", ylabel)


def _ylabel(mode, base, first):
    if mode == "log":
        return f"log10({base})"
    if mode == "relative":
        return f"{first}_i / {first}_1"
    return base


def plot_means(cfg: SpectrumConfig, agg_df):
    """Mittleres Spektrum je Fehlerklasse."""
    cols = _stat_cols(cfg, agg_df, "mean")
    return _lines(
        cfg, agg_df,
        lambda row: transform([row[c] for c in cols], cfg.plot_mode),
        "tab:blue",
        f"Mittlere {cfg.label}-Werte pro Komponente fuer alle Fehlerklassen "
        f"({cfg.plot_mode}-Skala, erste {{K}} Komponenten)",
        _ylabel(cfg.plot_mode, "Mittlerer Wert", "Mittelwert"))


def plot_stds(cfg: SpectrumConfig, agg_df):
    """Standardabweichung je Komponente und Fehlerklasse.

    Struktur identisch zum Mittelwertsplot - eine breite Streuung heisst,
    dass die Komponente ueber die 500 Runs der Klasse nicht stabil ist.
    """
    cols = _stat_cols(cfg, agg_df, "std")
    return _lines(
        cfg, agg_df,
        lambda row: transform([row[c] for c in cols], cfg.plot_mode),
        "tab:orange",
        f"Standardabweichung der {cfg.label}-Werte pro Komponente fuer alle "
        f"Fehlerklassen ({cfg.plot_mode}-Skala, erste {{K}} Komponenten)",
        _ylabel(cfg.plot_mode, "Std der Werte", "Std"))


def plot_cv(cfg: SpectrumConfig, agg_df):
    """Variationskoeffizient CV = Std / Mittelwert.

    Die dimensionslose Streuung: sie macht Komponenten vergleichbar, deren
    Absolutniveau um Groessenordnungen auseinanderliegt.
    """
    mean_cols = _stat_cols(cfg, agg_df, "mean")
    std_cols = _stat_cols(cfg, agg_df, "std")

    def values_of(row):
        m = np.array([row[c] for c in mean_cols], dtype=float)
        s = np.array([row[c] for c in std_cols], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(m > 0, s / m, np.nan)
        return transform(cv, cfg.plot_mode_cv)

    return _lines(
        cfg, agg_df, values_of, "tab:green",
        f"Relative Streuung der {cfg.label}-Werte pro Komponente fuer alle "
        f"Fehlerklassen ({cfg.plot_mode_cv}-Skala, erste {{K}} Komponenten)",
        _ylabel(cfg.plot_mode_cv, "Variationskoeffizient (Std / Mittelwert)",
                "CV"))


def plot_bars(cfg: SpectrumConfig, agg_df):
    """Erste k Werte je Fehlerklasse als Balken, mit Fault-0-Referenz.

    Der schmale orange Balken ist Fault 0 - so ist auf einen Blick zu
    sehen, welche Komponenten sich unter dem Fehler wirklich verschieben.
    Die Fehlerbalken zeigen die Standardabweichung ueber die Runs.
    """
    k = cfg.k_bar
    mean_cols = [f"{cfg.prefix}{i}_mean" for i in range(1, k + 1)]
    std_cols = [f"{cfg.prefix}{i}_std" for i in range(1, k + 1)]
    faults = np.sort(agg_df["faultNumber"].unique())
    ref = agg_df[agg_df["faultNumber"] == 0].iloc[0][mean_cols] \
        .values.astype(float)
    x = np.arange(1, k + 1)

    fig, axes = _grid(cfg, len(faults))
    for i, fault in enumerate(faults):
        ax = axes[i]
        row = agg_df[agg_df["faultNumber"] == fault].iloc[0]
        means = row[mean_cols].values.astype(float)
        stds = row[std_cols].values.astype(float)
        ax.bar(x, means, width=0.8, color="C0", alpha=0.85,
               label="Fehlerklasse", zorder=2)
        ax.bar(x, ref, width=0.4, color="tab:orange", alpha=0.9,
               label="Fault 0 (Referenz)", zorder=3)
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black",
                    capsize=2, elinewidth=1, zorder=4)
        ax.set_title(f"Fault {fault}", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=10)
    return _finish(fig, axes, len(faults),
                   f"Mittlere {cfg.label}-Werte je Fault mit Fault-0-Referenz "
                   f"(orange) - erste {k} Komponenten",
                   f"{cfg.label}-Komponenten-Index",
                   "Mittlerer Wert (+/- Std)")


# =========================================================================
# Skalare Verfahren (LDA)
# =========================================================================

def plot_scalar(cfg: SpectrumConfig, agg_df):
    """Drei Ansichten fuer ein Verfahren mit EINER Kennzahl je Run:
    Mittelwert je Fehlerklasse, Mittelwert neben Streuung, und der
    Variationskoeffizient."""
    import matplotlib.pyplot as plt

    mean_col, std_col = f"{cfg.prefix}_mean", f"{cfg.prefix}_std"
    faults = agg_df["faultNumber"].to_numpy()
    means = agg_df[mean_col].to_numpy(dtype=float)
    stds = agg_df[std_col].to_numpy(dtype=float)

    fig1, ax = plt.subplots(figsize=(13, 4.2), constrained_layout=True)
    ax.bar(faults, means, yerr=stds, capsize=3, color="C0", alpha=0.85)
    ax.set_xticks(faults)
    ax.set_xlabel("Fehlerklasse")
    ax.set_ylabel(f"Mittlerer {cfg.label}-Eigenwert (+/- Std)")
    ax.set_title(f"{cfg.label}: Separierbarkeit gegen Normalbetrieb "
                 f"je Fehlerklasse")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.show()

    fig2, axes = plt.subplots(1, 2, figsize=(13, 4.0), constrained_layout=True)
    axes[0].bar(faults, means, color="C0", alpha=0.85)
    axes[0].set_title("Mittelwert")
    axes[1].bar(faults, stds, color="tab:orange", alpha=0.85)
    axes[1].set_title("Standardabweichung")
    for a in axes:
        a.set_xticks(faults)
        a.set_xlabel("Fehlerklasse")
        a.grid(True, axis="y", alpha=0.3)
        a.set_axisbelow(True)
    fig2.suptitle(f"{cfg.label}-Eigenwert: Niveau und Streuung im Vergleich")
    plt.show()

    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(means > 0, stds / means, np.nan)
    fig3, ax = plt.subplots(figsize=(13, 3.6), constrained_layout=True)
    ax.bar(faults, cv, color="tab:green", alpha=0.85)
    ax.set_xticks(faults)
    ax.set_xlabel("Fehlerklasse")
    ax.set_ylabel("CV = Std / Mittelwert")
    ax.set_title(f"{cfg.label}-Eigenwert: relative Streuung je Fehlerklasse")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.show()
    return fig1, fig2, fig3


# =========================================================================
# DyCA: m und n schaetzen
# =========================================================================

def plot_dyca_mn_estimate(cfg: SpectrumConfig, df_ff, scaler=None,
                          ref_run: int = 1):
    """Zwei Diagnoseplots zur Wahl von m und n auf einem FaultFree-Run.

    Links  generalisierte Eigenwerte ohne m-Vorgabe: die Anzahl der Werte
           nahe 1 ist das gesuchte m.
    Rechts Singulaerwerte bei vorgegebenem m: die Anzahl der deutlich von
           null verschiedenen Werte ist (n - m).

    Der Referenzlauf wird mit DEMSELBEN scaling_mode vorverarbeitet wie
    die Produktivschleife - sonst driften Schaetzung und Lauf auseinander.
    """
    import matplotlib.pyplot as plt
    from dyca import dyca

    from ..core import PROC_COLS, scale

    ref = df_ff[df_ff["simulationRun"] == ref_run].sort_values("sample")
    X = scale(ref[PROC_COLS].values, cfg.scaling_mode, scaler)
    print("Referenz-Run shape:", X.shape)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    ev = np.asarray(dyca(X)["generalized_eigenvalues"], dtype=float)
    axes[0].bar(range(1, len(ev) + 1), ev)
    axes[0].set_title("Schritt 1: generalisierte Eigenwerte "
                      f"(FaultFree-Run {ref_run})\n"
                      "-> Anzahl der Werte nahe 1 = m")
    axes[0].set_xlabel("Komponente")
    axes[0].set_ylabel("generalized eigenvalue")
    axes[0].axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    axes[0].grid(True, alpha=0.3)

    sv = np.asarray(dyca(X, m=cfg.dyca_m)["singular_values"], dtype=float)
    axes[1].bar(range(1, len(sv) + 1), sv, color="tab:orange")
    axes[1].set_title(f"Schritt 2: Singulaerwerte (mit m = {cfg.dyca_m})\n"
                      "-> (n - m) = Anzahl Werte deutlich > 0")
    axes[1].set_xlabel("Komponente")
    axes[1].set_ylabel("singular value")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"\nAktuelle Einstellung: dyca_m = {cfg.dyca_m}, "
          f"dyca_n = {cfg.dyca_n}")
    print("Legen die Plots eine andere Wahl nahe, in der Konfigurations-"
          "zelle aendern und neu ausfuehren.")
    return fig


def plot_all(cfg: SpectrumConfig, agg_df):
    """Alle zum Verfahren passenden Standardplots nacheinander."""
    if get(cfg.method).scalar:
        return plot_scalar(cfg, agg_df)
    return (plot_means(cfg, agg_df), plot_stds(cfg, agg_df),
            plot_cv(cfg, agg_df), plot_bars(cfg, agg_df))
