import numpy as np

from avo import forward as fw
from avo.data import StackDef


def test_zoeppritz_normal_incidence_matches_impedance_contrast():
    vp1, vs1, rho1, vp2, vs2, rho2 = 2900.0, 1400.0, 2.35, 3600.0, 2130.0, 2.32
    r0 = fw.zoeppritz_rpp(vp1, vs1, rho1, vp2, vs2, rho2, np.array([0.0]))[0]
    assert np.isclose(r0, (vp2 * rho2 - vp1 * rho1) / (vp2 * rho2 + vp1 * rho1), atol=1e-10)


def test_aki_richards_close_to_zoeppritz_for_small_contrast():
    theta = np.radians([5.0, 20.0, 30.0])
    args = (3000.0, 1500.0, 2.30, 3150.0, 1600.0, 2.33)
    exact = fw.zoeppritz_rpp(*args, theta)
    approx = fw.aki_richards(*(np.array([a]) for a in args), theta)[0]
    assert np.allclose(approx, exact, atol=2e-3)


def test_synthetic_peak_at_interface_time():
    dz, dt_model = 0.5, 0.001
    vp = np.r_[np.full(400, 2900.0), np.full(400, 3600.0)]
    vs = np.r_[np.full(400, 1400.0), np.full(400, 2130.0)]
    rho = np.r_[np.full(400, 2.35), np.full(400, 2.32)]
    twt = fw.twt_from_vp(vp, dz, 2.0)
    t_out = 2.0 + np.arange(100) * 0.004
    stacks = [StackDef("near", 5, 15)]
    trace = fw.synthesize(twt, vp, vs, rho, stacks, [fw.ricker(30, dt_model)], t_out, dt_model)[0]
    t_interface = twt[399] + dz / vp[399]
    assert abs(t_out[np.argmax(trace)] - t_interface) <= 0.004
