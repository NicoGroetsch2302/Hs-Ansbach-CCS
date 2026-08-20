"""Gemeinsamer Kern beider Notebook-Familien.

Hier steht, was `tep.tsfresh` (Feature-Klassifikation) und `tep.eigen`
(Eigenwertspektren) beide brauchen: die Spaltennamen des TEP-Datensatzes,
die Cutoff-Konventionen, die Vorverarbeitung und die kleinen
linearalgebraischen Bausteine.

Der Zweck ist Driftschutz: `PROC_COLS` oder der StandardScaler-Fit dürfen
nicht in zwei Fassungen existieren, sonst rechnen die Familien
unbemerkt auf verschiedenen Daten.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# --- Spalten des TEP-Datensatzes ----------------------------------------
XMEAS_COLS = [f"xmeas_{i}" for i in range(1, 42)]
XMV_COLS = [f"xmv_{i}" for i in range(1, 12)]
PROC_COLS = XMEAS_COLS + XMV_COLS          # 52 Prozessvariablen
META_COLS = ["faultNumber", "simulationRun", "sample"]

SPLIT_FILES = {
    "train": ("TEP_FaultFree_Training.csv", "TEP_Faulty_Training.csv"),
    "test": ("TEP_FaultFree_Testing.csv", "TEP_Faulty_Testing.csv"),
}

# Pre-Fault-Cutoff, in allen Notebooks identisch:
# Train = Fehler nach 1 h -> erstes Post-Fault-Sample 21  (500 Samples/Run)
# Test  = Fehler nach 8 h -> erstes Post-Fault-Sample 161 (960 Samples/Run)
PRE_FAULT_CUTOFF = {"train": 21, "test": 161}

LABELS = list(range(21))                   # Fault-Klassen 0..20

SCALING_MODES = ("global_mean", "scaler")


# =========================================================================
# Run-Identitaet
# =========================================================================

def run_id(fault, run) -> int:
    """Eindeutige Integer-ID je (fault, run) fuer tsfresh (column_id).

    Runs laufen 1..500, deshalb ist fault*1000+run kollisionsfrei und per
    Division wieder zerlegbar.
    """
    return int(fault) * 1000 + int(run)


def labels_from_index(index) -> pd.Series:
    """faultNumber aus der run_id zurueckrechnen."""
    return pd.Series((np.asarray(index) // 1000).astype(int), index=index)


# =========================================================================
# Vorverarbeitung
# =========================================================================

def fit_scaler(faultfree_train_path: str, verbose: bool = True):
    """StandardScaler auf TEP_FaultFree_Training (Normalbetrieb) fitten.

    Der Fit sieht ausschliesslich Normalbetrieb - die Skalierung ist damit
    "in Einheiten des Normalbetriebs" und kennt keine Fehlerdaten.
    """
    import gc

    from sklearn.preprocessing import StandardScaler

    if verbose:
        print("Fitte StandardScaler auf TEP_FaultFree_Training "
              "(Normalbetrieb) ...")
    ff = pd.read_csv(faultfree_train_path, usecols=PROC_COLS)
    scaler = StandardScaler().fit(ff[PROC_COLS].values)
    del ff
    gc.collect()
    if verbose:
        print("  fertig.")
    return scaler


def scale(X: np.ndarray, mode: str, scaler=None) -> np.ndarray:
    """Vorverarbeitung eines Runs VOR der Projektion bzw. Zerlegung.

    "global_mean" : X - X.mean(), ein SKALARER Gesamtmittelwert. Die
                    spaltenweisen Mittelwerte und die Rohvarianzen bleiben
                    erhalten - varianzstarke Rohvariablen dominieren.
    "scaler"      : spaltenweise Standardisierung in Normalbetriebs-
                    Einheiten; alle 52 Variablen gehen gleichgewichtet ein.
    """
    if mode == "scaler":
        if scaler is None:
            raise RuntimeError("scaling_mode='scaler', aber kein Scaler "
                               "uebergeben (fit_scaler zuerst aufrufen).")
        return scaler.transform(X)
    if mode != "global_mean":
        raise ValueError(f"Unbekannter scaling_mode: {mode!r}")
    return X - X.mean()


# =========================================================================
# Bausteine der Verfahren
# =========================================================================

def flip_signs(Y: np.ndarray) -> np.ndarray:
    """Dreht jeden Kanal so, dass sein betragsmaessig groesster Wert positiv
    ist. Behebt die Vorzeichen-Willkuer der pro Run gefitteten Achsen
    (PCA-Komponenten, DyCA-SVD, CVA-Variaten)."""
    idx = np.argmax(np.abs(Y), axis=0)
    signs = np.sign(Y[idx, np.arange(Y.shape[1])])
    signs[signs == 0] = 1.0
    return Y * signs


def lag_stack(X: np.ndarray, lags: int) -> np.ndarray:
    """Haengt zeitverzoegerte Kopien an: Zeile t -> [x_t, x_(t-1), ...,
    x_(t-lags)]. Rueckgabe: (T - lags, d * (lags + 1))."""
    T = X.shape[0]
    cols = [X[lags - l: T - l] for l in range(lags + 1)]
    return np.hstack(cols)


def inv_sqrt_psd(S: np.ndarray, ridge_rel: float) -> np.ndarray:
    """Inverse Matrixwurzel mit relativer Ridge-Regularisierung.

    Der Ridge-Term ist an die Spur gekoppelt und damit skaleninvariant;
    das Clipping der Eigenwerte faengt numerisch negative Werte ab.
    """
    p = S.shape[0]
    S = S + ridge_rel * (np.trace(S) / p) * np.eye(p)
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, 1e-12 * w.max())
    return (V / np.sqrt(w)) @ V.T


def dyca_amplitudes(X: np.ndarray, m: int, n: int) -> np.ndarray:
    """DyCA-Amplituden als (T, n). Wirft bei numerischem Scheitern
    ("Negative eigenvalues") eine Exception."""
    from dyca import dyca

    res = dyca(X, m=m, n=n)
    amp = np.asarray(res["amplitudes"])            # (n, T)
    # scipy liefert reelle Eigenvektoren, solange alle Eigenwerte reell
    # sind - hier defensiv, weil komplexe Werte tsfresh sonst stumm
    # zerlegen wuerden.
    if np.iscomplexobj(amp):
        amp = np.real(amp)
    return amp.T


# =========================================================================
# Modell
# =========================================================================

def default_estimator(random_state: int = 42):
    """StandardScaler + RandomForest - wie lazypredict das Modell intern
    aufbaut. Beide Familien vergleichen ihre Feature-Saetze damit."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(random_state=random_state, n_jobs=-1))


# =========================================================================
# Umgebung
# =========================================================================

def versions() -> str:
    """Eine Zeile mit den Versionen, die das Ergebnis beeinflussen."""
    import sklearn
    parts = [f"pandas {pd.__version__}", f"numpy {np.__version__}",
             f"sklearn {sklearn.__version__}"]
    try:
        import tsfresh
        parts.insert(0, f"tsfresh {tsfresh.__version__}")
    except ImportError:
        pass
    parts.append(f"Kerne: {os.cpu_count()}")
    return " | ".join(parts)
