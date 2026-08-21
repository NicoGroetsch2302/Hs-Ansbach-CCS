#!/usr/bin/env python3
"""Der Einstiegspunkt - alles ohne Notebook.

    python main.py                      alle Stufen aus params.yaml
    python main.py eigen tsfresh        nur diese Stufen
    python main.py -p probe.yaml        andere Parameterdatei

Stufen
------
eigen       Spektrum je (Fault, Run), Aggregation, die vier Standardplots
amplitudes  niedrigdimensionale Amplituden y(t) je Lauf als NPZ
classify    Klassifikation AUF den Spektren + Confusion-Matrizen
tsfresh     Merkmale waehlen/anwenden, Modelle vergleichen, Matrizen

Zwischenergebnisse werden nicht neu gerechnet, wenn sie schon auf Platte
liegen (Spektren-CSVs, NPZ, TSFresh-Chunks, summary- und Vorhersage-CSV).
Neu rechnen heisst: die betreffende Datei loeschen.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")            # vor pyplot: keine Fenster, kein Display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
import yaml                      # noqa: E402

from tep.eigen import (aggregate, csv_name, export, faultfree_by_run,  # noqa
                       get, load_train, merge_faults, plot_bars, plot_cv,
                       plot_means, plot_scalar, plot_stds, run_spectra,
                       versions)
from tep.eigen import fit_scaler as eigen_scaler                       # noqa
from tep.eigen.classify import (class_distribution, confusion,         # noqa
                                feature_sets, plot_confusions,
                                report_confusions, run_lazyclassifier,
                                test_spectra, train_spectra)
from tep.tsfresh import (apply_features, benchmark_models, cache_dir,  # noqa
                         compare, config_name, describe, load_runs,
                         load_summary, plot_comparison,
                         plot_confusion_detail, plot_confusion_grid,
                         plot_recall, project, select_features, validate)
from tep.tsfresh import confusion as tsfresh_confusion                 # noqa
from tep.tsfresh import fit_scaler as tsfresh_scaler                   # noqa


def save(fig, name):
    """Figur(en) als PNG ablegen. Die tep-Plotfunktionen rufen plt.show(),
    das ist unter Agg ein No-op - gespeichert wird hier."""
    figs = fig if isinstance(fig, tuple) else (fig,)
    for i, f in enumerate(figs, start=1):
        suffix = "" if len(figs) == 1 else f"_{i}"
        path = os.path.join(P["plot_dir"], f"{name}{suffix}.png")
        f.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(f)
        print(f"    Bild: {path}")


# =========================================================================
# Stufen
# =========================================================================

def stage_eigen():
    """Spektren je Verfahren, plus die vier Standardplots."""
    e = P["eigen"]
    df_ff = df_all = None                 # erst lesen, wenn wirklich noetig

    for method in e["methods"]:
        path = os.path.join(P["data_dir"],
                            csv_name(method, P["scaling_mode"], "train"))
        if os.path.exists(path):
            per_run = pd.read_csv(path)
            print(f"  {method}: {per_run.shape} aus {path}")
        else:
            if df_all is None:            # die 1,9 GB nur einmal lesen
                df_ff, df_faulty = load_train(P["data_dir"],
                                              P["runs_per_fault"])
                df_all = merge_faults(df_ff, df_faulty)
            scaler = eigen_scaler(method, P["scaling_mode"], P["data_dir"])
            kw = dict(e["params"])
            if method == "lda":           # Lauf gegen Normalbetrieb
                kw["ff_by_run"] = faultfree_by_run(df_ff, "scaler", scaler)
            per_run = run_spectra(df_all, method,
                                  scaling_mode=P["scaling_mode"],
                                  scaler=scaler, **kw)
            export(per_run, method, P["scaling_mode"], P["data_dir"])

        agg = aggregate(per_run, method)
        if get(method).get("scalar"):     # LDA: eine Zahl je Lauf
            save(plot_scalar(agg, method), f"eigen_{method}_skalar")
        else:
            for fn, tag in ((plot_means, "mittel"), (plot_stds, "std"),
                            (plot_cv, "cv")):
                save(fn(agg, method, e["k_max"], e["plot_mode"], e["ncols"]),
                     f"eigen_{method}_{tag}")
            save(plot_bars(agg, method, e["k_bar"], e["ncols"]),
                 f"eigen_{method}_balken")


def stage_amplitudes():
    """Amplituden y(t) je Lauf als NPZ - eine Datei je Projektion und Split."""
    a = P["amplitudes"]
    configs = [tuple(c) for c in a["configs"]]
    validate(configs)
    scaler = tsfresh_scaler(P["scaling_mode"], P["data_dir"])

    for split in a["splits"]:
        todo = [s for s in configs
                if not os.path.exists(_npz(config_name(s), split))]
        for s in configs:
            if s not in todo:
                print(f"  {config_name(s)}/{split}: schon da")
        if not todo:
            continue

        runs = load_runs(split, P["data_dir"], P["runs_per_fault"],
                         a["run_length"])
        for spec in todo:
            keys, mats, n_failed = [], [], 0
            for key in sorted(runs):
                try:
                    Y, channels = project(runs[key], spec, P["scaling_mode"],
                                          scaler, **P["proj_params"])
                except Exception:
                    n_failed += 1         # z.B. numerisches Scheitern der DyCA
                    continue
                keys.append(key)
                mats.append(Y)            # float64, siehe project()
            if not mats:
                raise RuntimeError(f"{config_name(spec)}/{split}: kein Lauf "
                                   f"erfolgreich ({n_failed} Fehler).")
            path = _npz(config_name(spec), split)
            np.savez(path, amplitudes=np.stack(mats),
                     faultNumber=np.array([k[0] for k in keys]),
                     simulationRun=np.array([k[1] for k in keys]),
                     channels=np.array(channels))
            print(f"  {config_name(spec)}/{split}: {np.stack(mats).shape} -> "
                  f"{path} ({os.path.getsize(path) / 1e9:.2f} GB, "
                  f"{n_failed} Fehler)")


def _npz(name, split):
    return os.path.join(P["data_dir"], f"amplitudes.{name}.{split}.npz")


def stage_classify():
    """Klassifikation auf den exportierten Spektren."""
    c = P["classify"]
    train = train_spectra(c["methods"], P["scaling_mode"], P["data_dir"])
    test = test_spectra(c["methods"], P["scaling_mode"], P["data_dir"],
                        P["runs_per_fault"], **P["eigen"]["params"])
    sets = feature_sets(c["methods"], train, test, combine=True)
    for s in sets:
        print(f"  {s['name']:12s} {len(s['cols']):3d} Merkmale | "
              f"train {s['train'].shape} test {s['test'].shape}")

    print(class_distribution(sets).to_string())

    board = os.path.join(P["data_dir"], "spektren_leaderboards.csv")
    if os.path.exists(board):
        print(f"  Leaderboards aus {board}")
    else:
        rows = [b.assign(Merkmalssatz=s["name"], Modell=b.index)
                for s in sets
                for b in [run_lazyclassifier(s, c["cv_folds"],
                                             c["select_metric"],
                                             c["random_state"])]]
        pd.concat(rows).to_csv(board, index=False)
        print(f"  Leaderboards -> {board}")

    results = confusion(sets, c["random_state"])
    save(plot_confusions(results), "spektren_confusion")
    report_confusions(results)


def stage_tsfresh():
    """Merkmalsauswahl, -anwendung, Modellvergleich plus Plots."""
    t = P["tsfresh"]
    configs = [tuple(x) for x in t["configs"]]
    names = validate(configs)
    cache = cache_dir(P["scaling_mode"], t["smoke_test"])
    summary_path = os.path.join(cache, t["summary_csv"])
    pred_path = os.path.join(cache, t["cm_pred_csv"])

    describe(configs, cache, fc_mode=t["fc_mode"], top_k=t["top_k"],
             scaling_mode=P["scaling_mode"],
             runs_per_fault=P["runs_per_fault"], smoke_test=t["smoke_test"])

    train_top = test_top = None
    if os.path.exists(summary_path) and os.path.exists(pred_path):
        # Beide Caches da
        summary = load_summary(summary_path)
    else:
        common = dict(data_dir=P["data_dir"],
                      runs_per_fault=P["runs_per_fault"],
                      run_length=t["run_length"], top_k=t["top_k"],
                      chunk_runs=t["chunk_runs"],
                      scaling_mode=P["scaling_mode"],
                      scaler=tsfresh_scaler(P["scaling_mode"], P["data_dir"]),
                      **P["proj_params"])
        train_top, top_names = select_features(configs, cache,
                                               fc_mode=t["fc_mode"],
                                               **common)
        test_top = apply_features(configs, cache, top_names, **common)
        summary, _ = benchmark_models(configs, train_top, test_top,
                                      summary_path,
                                      lc_cv_folds=t["lc_cv_folds"])

    save(plot_comparison(compare(summary, names), t["label"], t["top_k"]),
         "tsfresh_vergleich")

    cm = tsfresh_confusion(names, pred_path, train_top, test_top,
                           summary_path=summary_path)
    save(plot_confusion_grid(cm, t["top_k"]), "tsfresh_confusion_raster")
    save(plot_confusion_detail(cm)[0], "tsfresh_confusion_detail")
    save(plot_recall(cm)[0], "tsfresh_recall")


STAGES = {"eigen": stage_eigen, "amplitudes": stage_amplitudes,
          "classify": stage_classify, "tsfresh": stage_tsfresh}


# =========================================================================
# Einstieg
# =========================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stages", nargs="*", choices=list(STAGES) + [],
                    help="Stufen; ohne Angabe die aus params.yaml")
    ap.add_argument("-p", "--params", default="params.yaml")
    args = ap.parse_args()

    P = yaml.safe_load(open(args.params, encoding="utf-8"))
    os.makedirs(P["plot_dir"], exist_ok=True)

    print(versions())
    for name in (args.stages or P["stages"]):
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        STAGES[name]()
    print("\nfertig.")
