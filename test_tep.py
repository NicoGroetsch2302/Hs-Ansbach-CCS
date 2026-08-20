"""Selbsttest der linearalgebraischen Bausteine: python test_tep.py

Deckt ab, was beim Entschlacken zusammengelegt wurde - vor allem
cva_covs(), das jetzt in CVA (eigen), CVA (tsfresh) und DyCVDA steckt.
Bricht das, weichen alle drei Verfahren still von den gecachten
Ergebnissen ab.
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


def test_config_csv_names():
    """Der Skalierungsmodus steckt im Dateinamen, Split auch."""
    from tep.eigen import SpectrumConfig

    cfg = SpectrumConfig(method="pca")
    assert cfg.csv_name() == "pca_eigenvalues_train.csv"
    assert cfg.csv_name("test") == "pca_eigenvalues_test.csv"
    assert not cfg.needs_scaler

    sc = SpectrumConfig(method="pca", scaling_mode="scaler")
    assert sc.csv_name() == "pca_eigenvalues_train_scaler.csv"
    assert sc.needs_scaler
    # LDA erzwingt den Scaler, auch wenn die Konfiguration global_mean sagt.
    assert SpectrumConfig(method="lda").needs_scaler


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("alle Checks bestanden.")
