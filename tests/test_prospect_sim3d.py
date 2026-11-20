import numpy as np

from avo import prospect_sim3d as p3


def _library(n=400, nt=30, seed=0):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0.02, 0.3, (n, nt)).astype(np.float32)
    props = {p: rng.uniform(1, 2, (n, nt)).astype(np.float32) for p in p3.LIB_PROPS}
    props["phi"] = phi
    props["ai"] = props["vp"] * props["rho"]
    order = np.argsort(phi, axis=0)
    return p3.Library(4500.0, rng.normal(0, 0.1, (n, 3, nt)).astype(np.float32), props,
                      rng.integers(0, 3, (n, nt)).astype(np.int8), (rng.random((n, nt)) < 0.3).astype(np.float32),
                      order, np.take_along_axis(phi, order, axis=0))


LK = {"sigma": np.full(3, 0.02, np.float32), "scales": np.ones(3), "temper": 0.2, "half": 4, "min_ess_target": 1}


def test_weigh_concentrates_on_the_member_that_produced_the_trace():
    lib = _library()
    score = np.ones(30, bool)
    true = 17
    out = p3.weigh(lib, lib.syn[[true]], score, LK, 3, u=np.full((1, 1, 30), 0.5, np.float32))
    assert out["beta"][0] == 1.0
    assert np.allclose(out["phi_q"][0, 1], lib.props["phi"][true], atol=0.02)
    assert (out["real_facies"][0, 0] == lib.facies[true]).mean() > 0.9
    prior = p3.weigh(lib, lib.syn[[true]], score, LK, 3, prior_only=True)
    assert np.allclose(prior["facies_prob"][0].sum(axis=0), 1.0, atol=1e-5)
    assert prior["ess"][0] > 0.99 * lib.facies.shape[0]


def test_surface_honours_control_points():
    surf = {"points": [{"x": 0.0, "y": 0.0, "twt_ms": 3000.0, "tvdss_m": 4500.0},
                       {"x": 5000.0, "y": 0.0, "twt_ms": 3100.0, "tvdss_m": 4700.0},
                       {"x": 0.0, "y": 5000.0, "twt_ms": 3050.0, "tvdss_m": 4600.0},
                       {"x": 5000.0, "y": 5000.0, "twt_ms": 3300.0, "tvdss_m": 4900.0}], "idw_power": 2.0}
    t, z = p3.eval_surface(surf, np.array([0.0, 5000.0, 0.0, 5000.0]), np.array([0.0, 0.0, 5000.0, 5000.0]))
    assert np.allclose(t, [3000, 3100, 3050, 3300], atol=0.5)
    assert np.allclose(z, [4500, 4700, 4600, 4900], atol=0.5)
