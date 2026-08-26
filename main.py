#!/usr/bin/env python3
"""Der Einstiegspunkt - alles ohne Notebook.

    python main.py                      alle Stufen aus params.yaml
    python main.py eigen tsfresh        nur diese Stufen
    python main.py -p probe.yaml        andere Parameterdatei

Zwei Klassifikationswege, je eine Stufe:

eigen       Spektrum je (Fault, Run) -> Plots -> Klassifikation darauf
tsfresh     Projektion -> TSFresh-Merkmale -> Modellvergleich

Dazu eine Stufe, die zu KEINEM der beiden Wege gehoert:

amplitudes  exportiert die Amplituden y(t) je Lauf als NPZ, zum
            Anschauen und Weiterverarbeiten. tsfresh rechnet seine
            Projektionen selbst - die NPZ wird davon nicht gelesen.

Zwischenergebnisse werden nicht neu gerechnet, wenn sie schon auf Platte
liegen (Spektren-CSVs, NPZ, TSFresh-Chunks, summary- und Vorhersage-CSV).
Neu rechnen heisst: die betreffende Datei loeschen.
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")            # vor pyplot: keine Fenster, kein Display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
import yaml                      # noqa: E402

from tep.eigen import (aggregate, csv_name, export, faultfree_by_run,  # noqa
                       fit_scaler, get, load_train, merge_faults,
                       needs_scaler, plot_bars, plot_cv, plot_means,
                       plot_scalar, plot_stds, run_spectra)
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


def save(fig, name):
    """Figur(en) als PNG ablegen. Das erste Namensteil ist der Unterordner
    (eigen/, tsfresh/) - flach nebeneinander sind es fuenfundzwanzig
    Bilder aus zwei Wegen. Die tep-Plotfunktionen rufen plt.show(), das
    ist unter Agg ein No-op - gespeichert wird hier."""
    figs = fig if isinstance(fig, tuple) else (fig,)
    sub, _, stem = name.partition("_")
    d = os.path.join(Parameters["plot_dir"], sub)
    os.makedirs(d, exist_ok=True)
    for i, f in enumerate(figs, start=1):
        suffix = "" if len(figs) == 1 else f"_{i}"
        path = os.path.join(d, f"{stem}{_probe_suffix()}{suffix}.png")
        f.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(f)
        print(f"    Bild: {path}")


# =========================================================================
# Stufen
# =========================================================================

def stage_eigen():
    """Weg 1: Spektren je Verfahren, Plots, Klassifikation darauf."""
    eigen_params = Parameters["eigen"]
    df_ff = df_all = None                 # erst lesen, wenn wirklich noetig

    for method in eigen_params["methods"]:
        path = os.path.join(
            Parameters["data_dir"],
            csv_name(method, Parameters["scaling_mode"], "train",
                     Parameters["runs_per_fault"]))
        if os.path.exists(path):
            per_run = pd.read_csv(path)
            print(f"  {method}: {per_run.shape} aus {path}")
        else:
            if df_all is None:
                df_ff, df_faulty = load_train(Parameters["data_dir"],
                                              Parameters["runs_per_fault"])
                df_all = merge_faults(df_ff, df_faulty)
            scaler = (fit_scaler(Parameters["data_dir"])
                      if needs_scaler(method, Parameters["scaling_mode"])
                      else None) # TODO: es wird skaliert vor der Berechnung der Eigenwerte
            kw = dict(eigen_params["params"])
            if method == "lda":           # Lauf gegen Normalbetrieb
                kw["ff_by_run"] = faultfree_by_run(
                    df_ff, method, Parameters["scaling_mode"], scaler)
            per_run = run_spectra(df_all, method,
                                  scaling_mode=Parameters["scaling_mode"],
                                  scaler=scaler, **kw)
            export(per_run, method, Parameters["scaling_mode"],
                   Parameters["data_dir"], Parameters["runs_per_fault"])

        agg = aggregate(per_run, method)
        if get(method).get("scalar"):     # LDA: eine Zahl je Lauf
            save(plot_scalar(agg, method), f"eigen_{method}_skalar")
        else:
            for fn, tag in ((plot_means, "mittel"), (plot_stds, "std"),
                            (plot_cv, "cv")):
                save(fn(agg, method, eigen_params["k_max"], eigen_params["plot_mode"], eigen_params["ncols"]),
                     f"eigen_{method}_{tag}")
            save(plot_bars(agg, method, eigen_params["k_bar"], eigen_params["ncols"]),
                 f"eigen_{method}_balken")

    if eigen_params["classify_methods"]:
        _classify_spectra(eigen_params)


def stage_amplitudes():
    """Amplituden y(t) je Lauf als NPZ - eine Datei je Projektion und Split."""
    a = Parameters["amplitudes"]
    configs = [tuple(c) for c in a["configs"]]
    validate(configs)
    scaler = (fit_scaler(Parameters["data_dir"])
              if Parameters["scaling_mode"] == "scaler" else None)

    for split in a["splits"]:
        todo = [s for s in configs
                if not os.path.exists(_npz(config_name(s), split))]
        for s in configs:
            if s not in todo:
                print(f"  {config_name(s)}/{split}: schon da")
        if not todo:
            continue

        runs = load_runs(split, Parameters["data_dir"], Parameters["runs_per_fault"],
                         a["run_length"])
        for spec in todo:
            keys, mats, n_failed = [], [], 0
            for key in sorted(runs):
                try:
                    Y, channels = project(runs[key], spec, Parameters["scaling_mode"],
                                          scaler, **Parameters["proj_params"])
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


def _probe_suffix():
    """Was diesen Lauf von einem anderen unterscheidet.

    Gehoert in jeden erzeugten Dateinamen: sonst ueberschreibt ein
    scaler-Lauf die Bilder des global_mean-Laufs und ein Probelauf die
    des Volllaufs, waehrend die CSVs daneben korrekt getrennt liegen.
    """
    parts = []
    if Parameters["scaling_mode"] != "global_mean":
        parts.append(Parameters["scaling_mode"])
    if Parameters["runs_per_fault"] is not None:
        parts.append(f"r{Parameters['runs_per_fault']}")
    return "".join(f"_{p}" for p in parts)


def _npz(name, split):
    return os.path.join(Parameters["data_dir"],
                        f"amplitudes.{name}.{split}{_probe_suffix()}.npz")


def _classify_spectra(e):
    """Klassifikation AUF den Spektren - Abschluss des Eigenwert-Wegs.

    Keine eigene Stufe: sie liest die CSVs, die oben geschrieben wurden,
    und laeuft ohne sie nicht. Leere classify_methods = nur
    charakterisieren, nicht klassifizieren.
    """
    methods = e["classify_methods"]
    train = train_spectra(methods, Parameters["scaling_mode"],
                          Parameters["data_dir"],
                          Parameters["runs_per_fault"])
    test = test_spectra(methods, Parameters["scaling_mode"], Parameters["data_dir"],
                        Parameters["runs_per_fault"], **e["params"])
    sets = feature_sets(methods, train, test, combine=True)
    for s in sets:
        print(f"  {s['name']:12s} {len(s['cols']):3d} Merkmale | "
              f"train {s['train'].shape} test {s['test'].shape}")

    print(class_distribution(sets).to_string())

    board = os.path.join(Parameters["data_dir"],
                         f"spektren_leaderboards{_probe_suffix()}.csv")
    if os.path.exists(board):
        # Wirklich laden, nicht nur melden - sonst faellt der
        # Modellvergleich stillschweigend ganz aus.
        boards = pd.read_csv(board)
        print(f"  Leaderboards aus {board}: {boards.shape[0]} Zeilen, "
              f"{boards['Merkmalssatz'].nunique()} Merkmalssaetze")
    else:
        rows = [b.assign(Merkmalssatz=s["name"], Modell=b.index)
                for s in sets
                for b in [run_lazyclassifier(s, e["cv_folds"],
                                             e["select_metric"],
                                             e["random_state"])]]
        pd.concat(rows).to_csv(board, index=False)
        print(f"  Leaderboards -> {board}")

    results = confusion(sets, e["random_state"])
    save(plot_confusions(results), "eigen_spektren_confusion")
    report_confusions(results)


def stage_tsfresh():
    """Merkmalsauswahl, -anwendung, Modellvergleich plus Plots."""
    t = dict(Parameters["tsfresh"])
    runs_per_fault = Parameters["runs_per_fault"]
    if t["smoke_test"]:
        # Wie frueher PipelineConfig.__post_init__: der Probelauf muss
        # den Umfang schrumpfen, sonst ist er langsamer als der echte
        # Lauf (leerer Cache, aber voller Umfang).
        t.update(fc_mode="minimal", top_k=20, chunk_runs=50)
        runs_per_fault = 4
        print("  smoke_test: runs_per_fault=4, fc_mode=minimal, "
              "top_k=20, chunk_runs=50")
    # label identifiziert die Notebook-Familie (PCA/DyCA, DPCA/CVA/ICA,
    # DyCVDA) - ohne sie im Bildnamen ueberschreiben sich die Laeufe.
    family = re.sub(r"[^a-z0-9]+", "_", t["label"].lower()).strip("_")
    configs = [tuple(x) for x in t["configs"]]
    names = validate(configs)
    # Der Ordnername traegt alles, was den Chunk-Inhalt bestimmt - ein
    # geaenderter Parameter trifft damit nie die Chunks des alten Laufs,
    # sondern legt einen eigenen Ordner an.
    cache = cache_dir(Parameters["scaling_mode"], t["smoke_test"],
                      runs_per_fault, fc_mode=t["fc_mode"],
                      run_length=t["run_length"], chunk_runs=t["chunk_runs"],
                      data_dir=Parameters["data_dir"],
                      **Parameters["proj_params"])
    summary_path = os.path.join(cache, t["summary_csv"])
    pred_path = os.path.join(cache, t["cm_pred_csv"])

    describe(configs, cache, fc_mode=t["fc_mode"], top_k=t["top_k"],
             scaling_mode=Parameters["scaling_mode"],
             runs_per_fault=runs_per_fault, smoke_test=t["smoke_test"])

    train_top = test_top = None
    if os.path.exists(summary_path) and os.path.exists(pred_path):
        # Beide Caches da
        summary = load_summary(summary_path)
    else:
        common = dict(data_dir=Parameters["data_dir"],
                      runs_per_fault=runs_per_fault,
                      run_length=t["run_length"], top_k=t["top_k"],
                      chunk_runs=t["chunk_runs"],
                      scaling_mode=Parameters["scaling_mode"],
                      scaler=(fit_scaler(Parameters["data_dir"])
                              if Parameters["scaling_mode"] == "scaler"
                              else None),
                      **Parameters["proj_params"])
        train_top, top_names = select_features(configs, cache,
                                               fc_mode=t["fc_mode"],
                                               **common)
        test_top = apply_features(configs, cache, top_names, **common)
        summary, _ = benchmark_models(configs, train_top, test_top,
                                      summary_path,
                                      lc_cv_folds=t["lc_cv_folds"])

    save(plot_comparison(compare(summary, names), t["label"], t["top_k"]),
         f"tsfresh_{family}_vergleich")

    cm = tsfresh_confusion(names, pred_path, train_top, test_top,
                           summary_path=summary_path)
    save(plot_confusion_grid(cm, t["top_k"]), f"tsfresh_{family}_confusion_raster")
    save(plot_confusion_detail(cm)[0], f"tsfresh_{family}_confusion_detail")
    save(plot_recall(cm)[0], f"tsfresh_{family}_recall")


def print_params(params):
    """Print the parameters in a readable format."""
    print("Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")


STAGES = {"eigen": stage_eigen, "tsfresh": stage_tsfresh,
          "amplitudes": stage_amplitudes}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stages", nargs="*", choices=list(STAGES) + [],
                    help="Stufen; ohne Angabe die aus params.yaml")
    ap.add_argument("-p", "--params", default="params.yaml")
    args = ap.parse_args()

    Parameters = yaml.safe_load(open(args.params, encoding="utf-8"))
    print_params(Parameters)

    for name in (args.stages or Parameters["stages"]):
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        STAGES[name]()
    print("\nfertig.")
