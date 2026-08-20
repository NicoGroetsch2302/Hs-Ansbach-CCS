"""Die drei Phasen des Laufs, gebuendelt in einer Pipeline-Klasse.

    Phase A  Trainingsruns extrahieren -> Top-K auswaehlen -> reduzieren
    Phase B  Testruns extrahieren, aber NUR die in A ausgewaehlten Features
    Phase C  LazyClassifier je Konfiguration, Ergebnis als summary-CSV

Das Objekt haelt den Zustand (train_top, top_names, test_top, summary), den
die Auswertungszellen brauchen. Nach einem Kernel-Neustart genuegt fuer die
Auswertung load_summary() bzw. der Vorhersage-Cache der Confusion-Matrizen.
"""

from __future__ import annotations

import gc
import os
import time

import numpy as np
import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from tsfresh.feature_extraction.settings import from_columns

from ..core import fit_scaler
from .config import PipelineConfig
from .data import labels_from_index, load_runs
from .features import (extract_config, load_top_names, rank_features,
                       save_top_names)
from .projections import config_name, get, n_channels


class Pipeline:
    """Fuehrt einen kompletten TSFresh-Lauf fuer cfg.configs aus."""

    def __init__(self, cfg: PipelineConfig, verbose: bool = True):
        self.cfg = cfg
        for spec in cfg.configs:
            proj = get(spec)                       # wirft bei unbekannter Art
            if proj.validate is not None:
                proj.validate(spec)
        self.names = [config_name(s) for s in cfg.configs]
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"Doppelte Konfigurationsnamen: {self.names}")

        # Nur bei scaling_mode="scaler" - sonst bleiben die 250k Zeilen
        # Normalbetrieb ungelesen.
        self.scaler = (
            fit_scaler(cfg.data_path("TEP_FaultFree_Training.csv"), verbose)
            if cfg.scaling_mode == "scaler" else None)
        self.train_top: dict = {}
        self.top_names: dict = {}
        self.test_top: dict = {}
        self.leaderboards: dict = {}
        self.summary: pd.DataFrame | None = None

    # -- Bequemlichkeit -------------------------------------------------
    def __iter__(self):
        """Laeuft ueber (spec, name)-Paare."""
        return zip(self.cfg.configs, self.names)

    def describe(self) -> None:
        """Die Kopfzeilen, die frueher am Ende der Konfigurationszelle
        standen: Umfang des Laufs und Kanalzahl je Konfiguration."""
        cfg = self.cfg
        print(f"smoke_test={cfg.smoke_test} | fc_mode={cfg.fc_mode} "
              f"({len(cfg.fc_parameters)} Calculator) | top_k={cfg.top_k} "
              f"| scaling_mode={cfg.scaling_mode}")
        print(f"Runs je Fault: {cfg.runs_per_fault or 'alle'} | "
              f"Cache: {cfg.cache_dir}/ (geteilt mit den Schwester-Notebooks)")
        print(f"\n{len(cfg.configs)} Konfigurationen:")
        for spec, name in self:
            c = n_channels(spec)
            print(f"  {name:22s} {c:3d} Kanaele -> "
                  f"~{c * len(cfg.fc_parameters)} Features/Run "
                  f"(Groessenordnung)")
        if not cfg.smoke_test:
            print("\n!!! VOLLER LAUF - Stunden bis Nacht. Der Chunk-Cache "
                  "macht Abbrechen und Fortsetzen gefahrlos. !!!")

    # -- Phase A --------------------------------------------------------
    def run_phase_a(self) -> None:
        """Train extrahieren, Features auswaehlen, auf Top-K reduzieren.

        Konfigurationen strikt nacheinander: die volle Matrix einer
        Konfiguration wird freigegeben, bevor die naechste beginnt.
        """
        cfg = self.cfg
        t_start = time.perf_counter()

        print("Lade Trainingsruns ...")
        runs_train = load_runs(cfg, "train")

        for spec, name in self:
            t0 = time.perf_counter()
            names = load_top_names(cfg, spec)

            if names is not None:
                # Auswahl liegt vor -> nur die Top-K-Spalten aus dem Cache.
                Xtop = extract_config(cfg, spec, "train", runs_train,
                                      tag="full", usecols=names,
                                      scaler=self.scaler)
                print(f"[{name}] Auswahl aus Cache, {Xtop.shape[0]} Runs")
            else:
                X_full = extract_config(cfg, spec, "train", runs_train,
                                        tag="full", scaler=self.scaler)
                y_full = labels_from_index(X_full.index)
                print(f"[{name}] extrahiert: {X_full.shape[0]} Runs x "
                      f"{X_full.shape[1]} Features "
                      f"({time.perf_counter() - t0:.0f} s) -> selektiere ...")

                rank = rank_features(cfg, X_full, y_full)
                names = rank["feature"].head(cfg.top_k).tolist()
                save_top_names(cfg, spec, names)

                Xtop = X_full[names].copy()
                del X_full
                gc.collect()

            self.train_top[name] = Xtop
            self.top_names[name] = names
            print(f"[{name}] fertig: {Xtop.shape} in "
                  f"{(time.perf_counter() - t0) / 60:.1f} min")

        del runs_train
        gc.collect()
        print(f"\nPhase A komplett in "
              f"{(time.perf_counter() - t_start) / 60:.1f} min")

    # -- Phase B --------------------------------------------------------
    def run_phase_b(self) -> None:
        """Testset extrahieren - nur die in Phase A ausgewaehlten Features."""
        cfg = self.cfg
        if not self.top_names:
            raise RuntimeError("Phase B braucht die Auswahl aus Phase A -> "
                               "zuerst run_phase_a() (laeuft aus dem Cache).")

        print("Lade Testruns ...")
        runs_test = load_runs(cfg, "test")

        for spec, name in self:
            t0 = time.perf_counter()
            kind_to_fc = from_columns(self.top_names[name])
            # top_k MUSS in der Cache-Kennung stehen: die Chunks enthalten
            # genau die in Phase A ausgewaehlten Features. Ohne top_k im
            # Namen wuerde ein spaeterer Lauf mit groesserem top_k die alten
            # Chunks wiederverwenden und die fehlenden Spalten stumm mit 0.0
            # auffuellen (siehe _subset in features.py).
            Xte = extract_config(cfg, spec, "test", runs_test,
                                 tag=f"top{cfg.top_k}",
                                 kind_to_fc=kind_to_fc, scaler=self.scaler)

            # Reihenfolge und Vollstaendigkeit an Train angleichen: Features,
            # die tsfresh auf dem Testset nicht erzeugt, waeren sonst stumm
            # verschoben.
            missing = [c for c in self.top_names[name] if c not in Xte.columns]
            if missing:
                print(f"    {name}: {len(missing)} Features fehlen im Test "
                      f"-> 0.0")
                for c in missing:
                    Xte[c] = 0.0
            self.test_top[name] = Xte[self.top_names[name]]

            print(f"[{name}] Test fertig: {self.test_top[name].shape} in "
                  f"{(time.perf_counter() - t0) / 60:.1f} min")

        del runs_test
        gc.collect()

    # -- gemeinsame Run-Menge -------------------------------------------
    def common_runs(self):
        """(train_index, test_index) der ueber ALLE Konfigurationen
        gemeinsamen Runs.

        Die DyCA-Stufe kann an einzelnen Runs numerisch scheitern; ohne
        diesen Schnitt waeren die Konfigurationen auf unterschiedlichen
        Testmengen bewertet.
        """
        if not self.train_top or not self.test_top:
            raise RuntimeError("train_top/test_top fehlen -> zuerst Phase A "
                               "und B ausfuehren.")
        tr = sorted(set.intersection(*(set(d.index)
                                       for d in self.train_top.values())))
        te = sorted(set.intersection(*(set(d.index)
                                       for d in self.test_top.values())))
        return tr, te

    def matrices(self, name: str):
        """(Xtr, Xte, ytr, yte) einer Konfiguration - gemeinsame Runs,
        NaN/inf aufgefuellt. Genau die Datenbasis von Phase C."""
        idx_tr, idx_te = self.common_runs()
        Xtr = self.train_top[name].loc[idx_tr]
        Xte = self.test_top[name].loc[idx_te]
        ytr = labels_from_index(Xtr.index).to_numpy()
        yte = labels_from_index(Xte.index).to_numpy()
        # NaN/inf koennen aus Featureberechnungen stammen -> auffuellen.
        Xtr_v = np.nan_to_num(Xtr.to_numpy(dtype=np.float64),
                              posinf=0.0, neginf=0.0)
        Xte_v = np.nan_to_num(Xte.to_numpy(dtype=np.float64),
                              posinf=0.0, neginf=0.0)
        return Xtr_v, Xte_v, ytr, yte, Xte.index

    # -- Phase C --------------------------------------------------------
    def run_phase_c(self) -> pd.DataFrame:
        """LazyClassifier je Konfiguration; schreibt die summary-CSV."""
        cfg = self.cfg
        idx_tr, idx_te = self.common_runs()
        print(f"Gemeinsame Runs: Train {len(idx_tr)}, Test {len(idx_te)}")

        rows = []
        for spec, name in self:
            Xtr_v, Xte_v, ytr, yte, _ = self.matrices(name)

            clf = LazyClassifier(verbose=0, ignore_warnings=True,
                                 predictions=True, cv=cfg.lc_cv_folds,
                                 random_state=cfg.random_state)

            t0 = time.perf_counter()
            models, preds = clf.fit(Xtr_v, Xte_v, ytr, yte)
            self.leaderboards[name] = models

            rows += _summary_rows(name, models, preds, yte)
            _print_best(name, rows, time.perf_counter() - t0)

        self.summary = pd.DataFrame(rows)
        self.summary.to_csv(cfg.summary_path, index=False)
        return self.summary

    # -- Auswertung ------------------------------------------------------
    # Duenne Weiterleitungen, damit im Notebook alles am Pipeline-Objekt
    # haengt. Die Arbeit steckt in reporting.py bzw. confusion.py.
    def compare(self, **kw):
        from .reporting import compare
        return compare(self, **kw)

    def plot_comparison(self, cmp, **kw):
        from .reporting import plot_comparison
        return plot_comparison(cmp, self.cfg, **kw)

    def confusion(self, **kw):
        from .confusion import confusion
        return confusion(self, **kw)

    def plot_confusions(self, cm, **kw):
        from .confusion import plot_grid
        return plot_grid(cm, self.cfg, **kw)

    def load_summary(self) -> pd.DataFrame:
        """summary aus der CSV holen - fuer Auswertungszellen nach einem
        Kernel-Neustart."""
        if self.summary is None:
            if not os.path.exists(self.cfg.summary_path):
                raise FileNotFoundError(
                    f"{self.cfg.summary_path} fehlt -> zuerst run_phase_c().")
            self.summary = pd.read_csv(self.cfg.summary_path)
            print(f"summary aus {self.cfg.summary_path} geladen.")
        return self.summary


# =========================================================================
# Hilfsfunktionen fuer Phase C
# =========================================================================

def _cv_val(models: pd.DataFrame, col: str, model_name: str) -> float:
    """CV-Wert aus dem lazypredict-Leaderboard, NaN wenn nicht vorhanden.

    Modelle ohne predict_proba bekommen None: lazypredict rechnet alle
    CV-Scorer gebuendelt mit error_score="raise", der ROC-AUC-Scorer
    scheitert und ALLE CV-Spalten des Modells werden geleert.
    """
    if col not in models.columns:
        return np.nan
    v = models[col].get(model_name, np.nan)
    return np.nan if v is None or pd.isna(v) else float(v)


def _summary_rows(name, models, preds, yte) -> list:
    """Eine Zeile je erfolgreichem Modell.

    Macro-F1 und Balanced Accuracy werden selbst gerechnet (lazypredicts F1
    ist die gewichtete Variante). Die CV-Spalten liegen NUR in models - ohne
    diesen Schritt waere die CV reine Rechenzeitverschwendung.
    """
    rows = []
    for model_name in preds.columns:
        yp = preds[model_name].to_numpy()
        if pd.isna(yp).any():                  # Modell ist durchgefallen
            continue
        rows.append({
            "Konfiguration": name,
            "Modell": model_name,
            "BalancedAcc": balanced_accuracy_score(yte, yp),
            "MacroF1": f1_score(yte, yp, average="macro", zero_division=0),
            # Train-CV (kein Testset-Blick), NaN bei Modellen ohne CV-Werte.
            "BalancedAccCVMean": _cv_val(models, "Balanced Accuracy CV Mean",
                                         model_name),
            "BalancedAccCVStd": _cv_val(models, "Balanced Accuracy CV Std",
                                        model_name),
        })
    return rows


def _print_best(name, rows, seconds) -> None:
    hits = [s for s in rows if s["Konfiguration"] == name]
    if not hits:
        print(f"[{name:22s}] kein Modell erfolgreich!")
        return
    best = max(hits, key=lambda s: s["MacroF1"])
    print(f"[{name:22s}] {len(hits):2d} Modelle in {seconds / 60:5.1f} min "
          f"| bestes: {best['Modell']} (MacroF1 {best['MacroF1']:.4f}, "
          f"BA {best['BalancedAcc']:.4f})")
    # Zusaetzlich das CV-beste Modell: dieselbe Zeile, aber auf der Train-CV
    # ausgewaehlt statt auf dem Testmaximum (Winner's Curse).
    cv_hits = [s for s in hits if not np.isnan(s["BalancedAccCVMean"])]
    if cv_hits:
        b = max(cv_hits, key=lambda s: s["BalancedAccCVMean"])
        print(f"{'':19s}CV-bestes: {b['Modell']} "
              f"(BA CV {b['BalancedAccCVMean']:.4f} "
              f"+/- {b['BalancedAccCVStd']:.4f} -> Test-MacroF1 "
              f"{b['MacroF1']:.4f}) | {len(cv_hits)}/{len(hits)} mit "
              f"CV-Werten")
