"""Rock physics: fluid properties, mineral mixing, Gassmann and a simple dry-frame model.

Units: moduli in GPa, density in g/cm3, velocity in m/s, pressure in MPa, temperature in degC.
"""

import numpy as np

_BRINE_W = np.array([
    [1402.85, 1.524, 3.437e-3, -1.197e-5],
    [4.871, -0.0111, 1.739e-4, -1.628e-6],
    [-0.04783, 2.747e-4, -2.135e-6, 1.237e-8],
    [1.487e-4, -6.503e-7, -1.455e-8, 1.327e-10],
    [-2.197e-7, 7.987e-10, 5.230e-11, -4.614e-13],
])


def _modulus(rho, v):
    return rho * v**2 * 1e-6


def brine_properties(pressure, temperature, salinity_ppm):
    """Batzle and Wang (1992) brine bulk modulus and density."""
    P, T, S = pressure, temperature, salinity_ppm * 1e-6
    rho_w = 1 + 1e-6 * (-80 * T - 3.3 * T**2 + 0.00175 * T**3 + 489 * P - 2 * T * P
                        + 0.016 * T**2 * P - 1.3e-5 * T**3 * P - 0.333 * P**2 - 0.002 * T * P**2)
    rho_b = rho_w + S * (0.668 + 0.44 * S + 1e-6 * (300 * P - 2400 * P * S
                                                    + T * (80 + 3 * T - 3300 * S - 13 * P + 47 * P * S)))
    v_w = sum(_BRINE_W[i, j] * T**i * P**j for i in range(5) for j in range(4))
    v_b = (v_w + S * (1170 - 9.6 * T + 0.055 * T**2 - 8.5e-5 * T**3 + 2.6 * P - 0.0029 * T * P - 0.0476 * P**2)
           + S**1.5 * (780 - 10 * P + 0.16 * P**2) - 1820 * S**2)
    return _modulus(rho_b, v_b), rho_b


def live_oil_properties(pressure, temperature, api, gor, gas_gravity):
    """Batzle and Wang (1992) live-oil bulk modulus and density. GOR in L/L (~Sm3/Sm3)."""
    P, T, G, Rg = pressure, temperature, gas_gravity, gor
    rho0 = 141.5 / (api + 131.5)
    b0 = 0.972 + 0.00038 * (2.4 * Rg * np.sqrt(G / rho0) + T + 17.8) ** 1.175
    rho_pseudo = (rho0 / b0) / (1 + 0.001 * Rg)
    rho_g = (rho0 + 0.0012 * G * Rg) / b0
    rho_p = rho_g + (0.00277 * P - 1.71e-7 * P**3) * (rho_g - 1.15) ** 2 + 3.49e-4 * P
    rho = rho_p / (0.972 + 3.81e-4 * (T + 17.78) ** 1.175)
    v = (2096 * np.sqrt(rho_pseudo / (2.6 - rho_pseudo)) - 3.7 * T + 4.64 * P
         + 0.0115 * (4.12 * np.sqrt(1.08 / rho_pseudo - 1) - 1) * T * P)
    return _modulus(rho, v), rho


def fluid_set(reservoir_cfg):
    """Brine and oil properties at reservoir conditions from a config block."""
    P, T = reservoir_cfg["pressure_mpa"], reservoir_cfg["temperature_c"]
    k_b, rho_b = brine_properties(P, T, reservoir_cfg["salinity_ppm"])
    k_o, rho_o = live_oil_properties(P, T, reservoir_cfg["oil_api"], reservoir_cfg["gor"],
                                     reservoir_cfg["gas_gravity"])
    return {"k_brine": k_b, "rho_brine": rho_b, "k_oil": k_o, "rho_oil": rho_o}


def fluid_mix(sw, fluids):
    """Wood (Reuss) mix of brine and oil. sw is brine saturation."""
    sw = np.clip(np.asarray(sw, dtype=float), 0.0, 1.0)
    k = 1.0 / (sw / fluids["k_brine"] + (1 - sw) / fluids["k_oil"])
    rho = sw * fluids["rho_brine"] + (1 - sw) * fluids["rho_oil"]
    return k, rho


def mineral_mix(vsh, calcite, minerals):
    """Voigt-Reuss-Hill mix of quartz, clay and calcite. minerals: name -> [K, G, rho]."""
    vsh = np.asarray(vsh, dtype=float)
    calcite = np.asarray(calcite, dtype=float) * np.ones_like(vsh)
    fq = np.clip(1 - vsh - calcite, 0.0, 1.0)
    fr = [fq, vsh, calcite]
    names = ["quartz", "clay", "calcite"]
    k = [minerals[n][0] for n in names]
    g = [minerals[n][1] for n in names]
    r = [minerals[n][2] for n in names]
    kv = sum(f * x for f, x in zip(fr, k))
    gv = sum(f * x for f, x in zip(fr, g))
    kr = 1.0 / sum(f / x for f, x in zip(fr, k))
    gr = 1.0 / sum(f / x for f, x in zip(fr, g))
    rho = sum(f * x for f, x in zip(fr, r))
    return 0.5 * (kv + kr), 0.5 * (gv + gr), rho


def gassmann_saturated(k_dry, k_min, k_fl, phi):
    phi = np.maximum(phi, 1e-6)
    return k_dry + (1 - k_dry / k_min) ** 2 / (phi / k_fl + (1 - phi) / k_min - k_dry / k_min**2)


def gassmann_dry(k_sat, k_min, k_fl, phi):
    phi = np.maximum(phi, 1e-6)
    a = phi * k_min / k_fl
    return (k_sat * (a + 1 - phi) - k_min) / (a + k_sat / k_min - 1 - phi)


def dry_frame(k_min, g_min, phi, phic, exponent):
    """Power-law critical-porosity frame: M_dry = M_min * (1 - phi/phic)^n."""
    f = np.clip(1 - np.asarray(phi) / phic, 1e-3, 1.0) ** exponent
    return k_min * f, g_min * f


def velocities(k, g, rho):
    return np.sqrt((k + 4.0 / 3.0 * g) / rho) * 1000.0, np.sqrt(g / rho) * 1000.0


def elastic_from_rock(phi, vsh, calcite, sw, fluids, minerals, phic, exponent):
    """Vp, Vs, rho for rock described by porosity, clay, calcite and brine saturation."""
    k_min, g_min, rho_min = mineral_mix(vsh, calcite, minerals)
    k_dry, g_dry = dry_frame(k_min, g_min, phi, phic, exponent)
    k_fl, rho_fl = fluid_mix(sw, fluids)
    k_sat = gassmann_saturated(k_dry, k_min, k_fl, phi)
    rho = rho_min * (1 - phi) + rho_fl * phi
    vp, vs = velocities(k_sat, g_dry, rho)
    return vp, vs, rho


def substitute_fluid(vp, vs, rho, phi, vsh, sw_from, sw_to, fluids, minerals):
    """Gassmann fluid substitution of logs from brine saturation sw_from to sw_to."""
    k_min, _, _ = mineral_mix(vsh, 0.0, minerals)
    g = rho * (vs / 1000.0) ** 2
    k_sat = rho * (vp / 1000.0) ** 2 - 4.0 / 3.0 * g
    k_fl1, rho_fl1 = fluid_mix(sw_from, fluids)
    k_fl2, rho_fl2 = fluid_mix(sw_to, fluids)
    k_dry = np.clip(gassmann_dry(k_sat, k_min, k_fl1, phi), 0.05, k_min)
    k_new = gassmann_saturated(k_dry, k_min, k_fl2, phi)
    rho_new = rho + phi * (rho_fl2 - rho_fl1)
    vp_new, vs_new = velocities(k_new, g, rho_new)
    return vp_new, vs_new, rho_new


def change_porosity(vp, vs, rho, phi, vsh, sw, dphi, fluids, minerals, phic, exponent):
    """Perturb porosity of logs by dphi, scaling the dry frame with the critical-porosity power law."""
    k_min, _, rho_min = mineral_mix(vsh, 0.0, minerals)
    k_fl, rho_fl = fluid_mix(sw, fluids)
    g = rho * (vs / 1000.0) ** 2
    k_dry = np.clip(gassmann_dry(rho * (vp / 1000.0) ** 2 - 4.0 / 3.0 * g, k_min, k_fl, phi), 0.05, k_min)
    phi_new = np.clip(phi + dphi, 0.005, phic - 0.01)
    ratio = (np.clip(1 - phi_new / phic, 1e-3, 1) / np.clip(1 - phi / phic, 1e-3, 1)) ** exponent
    k_dry2, g2 = np.minimum(k_dry * ratio, k_min), g * ratio
    rho2 = rho + (phi_new - phi) * (rho_fl - rho_min)
    vp2, vs2 = velocities(gassmann_saturated(k_dry2, k_min, k_fl, phi_new), g2, rho2)
    return vp2, vs2, rho2, phi_new


def vs_greenberg_castagna(vp, vsh):
    """Greenberg-Castagna (1992) brine-sand/shale Vs estimate, linearly mixed by Vsh. vp in m/s."""
    v = np.asarray(vp) / 1000.0
    sand = 0.80416 * v - 0.85588
    shale = 0.76969 * v - 0.86735
    return 1000.0 * ((1 - vsh) * sand + vsh * shale)
