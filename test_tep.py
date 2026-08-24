"""Selbsttest der Bausteine: python test_tep.py

Deckt ab, was beim Umbau zusammengelegt wurde:

  cva_covs()   steckt jetzt in CVA (eigen), CVA (tsfresh) und DyCVDA.
               Bricht das, weichen alle drei still von den gecachten
               Ergebnissen ab.
  Registries   SPECTRA und PROJECTORS sind einfache dicts - ein Tippfehler
               im Schluessel faellt sonst erst mitten im Nachtlauf auf.
  Namen        csv_name()/needs_scaler() bestimmen, welche CSV geschrieben
               und ob der Scaler gefittet wird.
"""

import numpy as np

from tep.core import cva_covs, flip_signs, inv_sqrt_psd, lag_stack


def test_cva_covs_matches_inline():
    """cva_covs == die frueher dreimal ausgeschriebene Rechnung."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    for past, fut in [(1, 1), (2, 3), (4, 4)]:
        T = X.shape[0]
        P = np.hstack([X[past - j: T - fut + 1 - j]
                       for j in range(1, past + 1)])
        F = np.hstack([X[past + j: T - fut + 1 + j] for j in range(fut)])
        N = P.shape[0]
        Pc, Fc = P - P.mean(axis=0), F - F.mean(axis=0)

        pc, fc, spp, sff, sfp = cva_covs(X, past, fut)
        assert np.array_equal(pc, Pc), (past, fut)
        assert np.array_equal(fc, Fc), (past, fut)
        assert np.array_equal(spp, Pc.T @ Pc / (N - 1)), (past, fut)
        assert np.array_equal(sff, Fc.T @ Fc / (N - 1)), (past, fut)
        assert np.array_equal(sfp, Fc.T @ Pc / (N - 1)), (past, fut)


def test_cva_covs_shapes():
    """DyCVDA-Fall past = fut = s: T - 2s + 1 Zeilen, n*s Spalten."""
    Y = np.arange(40 * 3, dtype=float).reshape(40, 3)
    for s in (1, 2, 5):
        Pc, Fc, _, _, _ = cva_covs(Y, s, s)
        assert Pc.shape == (40 - 2 * s + 1, 3 * s), s
        assert Fc.shape == Pc.shape, s


def test_flip_signs():
    """Groesster Betrag je Kanal wird positiv, Nullkanal bleibt."""
    Y = np.array([[-5.0, 1.0, 0.0], [2.0, 3.0, 0.0]])
    out = flip_signs(Y)
    assert out[0, 0] == 5.0 and out[1, 0] == -2.0
    assert np.array_equal(out[:, 1], Y[:, 1])
    assert np.array_equal(out[:, 2], Y[:, 2])


def test_lag_stack():
    X = np.arange(10).reshape(5, 2)
    Z = lag_stack(X, 2)
    assert Z.shape == (3, 6)
    assert np.array_equal(Z[0], [4, 5, 2, 3, 0, 1])


def test_inv_sqrt_psd():
    """W @ S @ W == I fuer eine gut konditionierte S."""
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 6))
    S = A @ A.T + 6 * np.eye(6)
    W = inv_sqrt_psd(S, 0.0)
    assert np.allclose(W @ S @ W, np.eye(6), atol=1e-8)


def test_spectra_registry():
    """Jeder SPECTRA-Eintrag hat die Pflichtschluessel und callable apply."""
    from tep.eigen.spectra import SPECTRA, csv_name, get, min_samples

    assert set(SPECTRA) == {"pca", "dyca", "dpca", "cva", "ica", "lda"}
    for method, spec in SPECTRA.items():
        for key in ("prefix", "label", "csv_stem", "apply"):
            assert key in spec, (method, key)
        assert callable(spec["apply"]), method
        assert get(method) is spec
    # nur DyCA und CVA brauchen eine Mindestlaenge
    assert min_samples("pca") is None
    assert min_samples("dyca", dyca_m=6, dyca_n=12) == 17
    assert min_samples("cva", cva_past=1, cva_fut=1) == 52 + 1 + 1 + 5
    # Die CSV-Namen sind eingefroren - LazyClassifier_PCA_DyCA liest sie.
    assert csv_name("pca") == "pca_eigenvalues_train.csv"
    assert csv_name("dyca", "scaler", "test") == "dyca_eigenvalues_test_scaler.csv"
    # LDA erzwingt scaler -> Suffix auch bei global_mean, sonst hiesse eine
    # Datei mit Scaler-Daten wie eine ohne.
    assert csv_name("lda", "global_mean") == "lda_eigenvalues_train_scaler.csv"


def test_needs_scaler():
    """LDA erzwingt den Scaler, sonst entscheidet der Modus."""
    from tep.eigen.spectra import needs_scaler

    assert not needs_scaler("pca", "global_mean")
    assert needs_scaler("pca", "scaler")
    assert needs_scaler("lda", "global_mean")


def test_projector_registry():
    """Jeder PROJECTORS-Eintrag liefert Namen und Kanaele zu seiner Spec."""
    from tep.tsfresh.projections import (PROJECTORS, channel_names,
                                         config_name, n_channels, validate)

    specs = [("raw",), ("pca", 6), ("dyca", 6, 12), ("dpca", 4),
             ("cva", 8), ("ica", 12), ("dycvda", 6, 12, 2, 15)]
    assert {s[0] for s in specs} == set(PROJECTORS)
    for spec in specs:
        assert callable(PROJECTORS[spec[0]]["apply"]), spec
        assert isinstance(config_name(spec), str), spec
        assert len(channel_names(spec)) == n_channels(spec), spec
    # Die Namen sind Cache-Praefixe - eine Aenderung entwertet den Cache.
    assert config_name(("dyca", 6, 12)) == "dyca_m6_n12"
    assert config_name(("dycvda", 6, 12, 2, 15)) == "dycvda_m6n12_s2_r15"
    assert validate(specs) == [config_name(s) for s in specs]


def test_validate_rejects_bad_specs():
    """Rangbedingungen und doppelte Namen fliegen vor dem Lauf auf."""
    from tep.tsfresh.projections import validate

    for bad in [[("dyca", 2, 6)],                  # m >= n - m verletzt
                [("dycvda", 6, 12, 2, 40)],        # r <= n*s verletzt
                [("pca", 6), ("pca", 6)]]:         # doppelter Name
        try:
            validate(bad)
        except (AssertionError, ValueError):
            continue
        raise AssertionError(f"validate({bad}) haette scheitern muessen")


def test_params_yaml():
    """Jeder Schluessel aus params.yaml wird von der Zielfunktion auch
    akzeptiert - ein Tippfehler faellt sonst erst nach Stunden auf."""
    import inspect
    import yaml

    from tep.core import SPLIT_FILES
    from tep.eigen import get, run_spectra
    from tep.tsfresh import validate
    from tep.tsfresh.projections import PROJECTORS

    p = yaml.safe_load(open("params.yaml", encoding="utf-8"))

    import main                          # Stufennamen gegen den Dispatcher
    for stage in p["stages"]:
        assert stage in main.STAGES, stage
    assert p["scaling_mode"] in ("global_mean", "scaler")

    known = set(inspect.signature(run_spectra).parameters)
    for key in p["eigen"]["params"]:
        assert key in known, f"eigen.params: run_spectra kennt '{key}' nicht"
    for method in p["eigen"]["methods"]:
        get(method)                      # wirft bei unbekanntem Verfahren

    # Klassifiziert wird auf den CSVs, die derselbe Lauf schreibt - was
    # nicht in methods steht, hat keine CSV und wuerde erst zur Laufzeit
    # als FileNotFoundError auffallen.
    fehlt = set(p["eigen"]["classify_methods"]) - set(p["eigen"]["methods"])
    assert not fehlt, f"classify_methods ohne Spektrum: {sorted(fehlt)}"

    # proj_params gegen die apply-Funktionen pruefen, nicht gegen project():
    # das hat **params und wuerde jeden Tippfehler stillschweigend schlucken.
    specs = [tuple(c) for c in p["amplitudes"]["configs"]]
    specs += [tuple(c) for c in p["tsfresh"]["configs"]]
    validate([tuple(c) for c in p["amplitudes"]["configs"]])
    validate([tuple(c) for c in p["tsfresh"]["configs"]])
    accepted = set()
    for spec in specs:
        accepted |= set(inspect.signature(
            PROJECTORS[spec[0]]["apply"]).parameters)
    for key in p["proj_params"]:
        assert key in accepted, f"proj_params: '{key}' kennt keine Projektion"

    for split in p["amplitudes"]["splits"]:
        assert split in SPLIT_FILES, split
    assert p["tsfresh"]["fc_mode"] in ("minimal", "efficient", "comprehensive")


def test_cache_namen_trennen_probelauf():
    """runs_per_fault MUSS in Dateinamen und Cache-Ordner stehen - sonst
    liest ein Probelauf die Ergebnisse des Volllaufs und umgekehrt."""
    from tep.eigen.spectra import csv_name
    from tep.tsfresh.features import cache_dir

    assert csv_name("pca") == "pca_eigenvalues_train.csv"
    assert csv_name("pca", runs_per_fault=3) == "pca_eigenvalues_train_r3.csv"
    assert (csv_name("pca", "scaler", "test", 3)
            == "pca_eigenvalues_test_scaler_r3.csv")
    # Volllauf und Probelauf duerfen nie denselben Ordner treffen
    voll = cache_dir("global_mean", False, None)
    probe = cache_dir("global_mean", False, 3)
    assert voll != probe, (voll, probe)
    assert probe.endswith("_r3"), probe


def test_notebooks_entpacken_plot_recall():
    """plot_recall liefert (fig, tab). Wer nur `tab` erwartet, bekommt ein
    Tupel und scheitert erst in der letzten Zelle nach Stunden."""
    import glob
    import inspect
    import json

    from tep.tsfresh import plot_recall

    src = inspect.getsource(plot_recall)
    assert "return fig, tab" in src, "Rueckgabe geaendert?"
    for p in sorted(glob.glob("notebooks/*.ipynb")):
        for cell in json.load(open(p, encoding="utf-8"))["cells"]:
            for line in "".join(cell["source"]).splitlines():
                if "plot_recall(" in line and "=" in line:
                    assert line.strip().startswith("_,"), f"{p}: {line.strip()}"


def test_effective_mode_und_referenz():
    """LDA erzwingt scaler - beide Haelften des Vergleichs muessen denselben
    Modus benutzen, sonst liegt die Referenz in Rohwerten und der
    Fehlerlauf in z-Werten und jeder Eigenwert ist bedeutungslos."""
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    from tep.core import PROC_COLS
    from tep.eigen import faultfree_by_run, run_spectra
    from tep.eigen.spectra import effective_mode

    assert effective_mode("pca", "global_mean") == "global_mean"
    assert effective_mode("pca", "scaler") == "scaler"
    assert effective_mode("lda", "global_mean") == "scaler"

    rng = np.random.default_rng(0)
    T = 40
    d = {"faultNumber": 0, "simulationRun": 1, "sample": np.arange(1, T + 1)}
    d.update({c: rng.normal(loc=17.0, size=T) for c in PROC_COLS})
    df = pd.DataFrame(d)
    sc = StandardScaler().fit(df[PROC_COLS].values)
    # Aufrufer sagt global_mean, LDA erzwingt scaler -> z-skaliert
    ff = faultfree_by_run(df, "lda", "global_mean", sc)
    assert np.allclose(ff[1].std(axis=0), 1.0, atol=0.1), ff[1].std(axis=0)[:3]

    # ohne Referenzlaeufe frueh scheitern, statt 10 500 mal einzeln
    try:
        run_spectra(df, "lda", pre_fault_cutoff=1, verbose=False)
    except ValueError as exc:
        assert "ff_by_run" in str(exc)
    else:
        raise AssertionError("LDA ohne ff_by_run haette scheitern muessen")


def test_extract_config_lehnt_stille_fallen_ab():
    """default_fc_parameters=None heisst bei tsfresh COMPREHENSIVE, ein
    leeres kind_to_fc extrahiert gar nichts - beides faellt sonst erst an
    den Spaltenzahlen auf."""
    from tep.tsfresh import extract_config

    for kwargs, stichwort in [({}, "COMPREHENSIVE"),
                              ({"kind_to_fc": {}}, "leer")]:
        try:
            extract_config(("raw",), "train", {}, "cache", **kwargs)
        except ValueError as exc:
            assert stichwort in str(exc), str(exc)
        else:
            raise AssertionError(f"{kwargs} haette scheitern muessen")


def test_fit_scaler_wird_gemerkt():
    """Der Fit liest 94 MB; die eigen-Stufe ruft ihn je Verfahren auf.
    Einmal pro Prozess reicht - das Ergebnis ist nach dem Fit fest."""
    from tep.core import fit_scaler

    assert hasattr(fit_scaler, "cache_info"), "lru_cache verschwunden?"
    fit_scaler.cache_clear()
    assert fit_scaler.cache_info().currsize == 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("alle Checks bestanden.")
