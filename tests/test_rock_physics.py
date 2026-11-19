import numpy as np

from avo import rock_physics as rp

MINERALS = {"quartz": [36.6, 45.0, 2.65], "clay": [21.0, 7.0, 2.58], "calcite": [76.8, 32.0, 2.71]}
RESERVOIR = dict(pressure_mpa=30.0, temperature_c=110.0, salinity_ppm=50000, oil_api=29.0, gor=90.0,
                 gas_gravity=0.74)


def test_gassmann_round_trip_reproduces_logs():
    fl = rp.fluid_set(RESERVOIR)
    phi, vsh = np.array([0.12, 0.2, 0.27]), np.array([0.2, 0.08, 0.05])
    vp, vs, rho = rp.elastic_from_rock(phi, vsh, 0.0, 1.0, fl, MINERALS, 0.4, 1.8)
    vp2, vs2, rho2 = rp.substitute_fluid(vp, vs, rho, phi, vsh, 1.0, 0.2, fl, MINERALS)
    vp3, vs3, rho3 = rp.substitute_fluid(vp2, vs2, rho2, phi, vsh, 0.2, 1.0, fl, MINERALS)
    assert np.allclose(vp3, vp) and np.allclose(vs3, vs) and np.allclose(rho3, rho)


def test_oil_substitution_lowers_vp_and_density_and_raises_vs():
    fl = rp.fluid_set(RESERVOIR)
    vp, vs, rho = rp.elastic_from_rock(0.2, 0.08, 0.0, 1.0, fl, MINERALS, 0.4, 1.8)
    vp_o, vs_o, rho_o = rp.substitute_fluid(vp, vs, rho, 0.2, 0.08, 1.0, 0.2, fl, MINERALS)
    assert vp_o < vp and rho_o < rho and vs_o > vs
    assert fl["k_oil"] < fl["k_brine"] and fl["rho_oil"] < fl["rho_brine"]


def test_porosity_change_matches_forward_rock_model():
    fl = rp.fluid_set(RESERVOIR)
    vp, vs, rho = rp.elastic_from_rock(0.20, 0.08, 0.0, 1.0, fl, MINERALS, 0.4, 1.8)
    vp2, vs2, rho2, phi2 = rp.change_porosity(vp, vs, rho, 0.20, 0.08, 1.0, -0.05, fl, MINERALS, 0.4, 1.8)
    expected = rp.elastic_from_rock(0.15, 0.08, 0.0, 1.0, fl, MINERALS, 0.4, 1.8)
    assert np.isclose(phi2, 0.15)
    assert np.allclose([vp2, vs2, rho2], expected, rtol=1e-6)
