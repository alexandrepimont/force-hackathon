"""AVO forward modelling: wavelets, angle-dependent reflectivity, depth-to-time and synthetics."""

import numpy as np


def _hilbert(x):
    n = len(x)
    h = np.zeros(n)
    h[0] = 1
    if n % 2 == 0:
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[1:(n + 1) // 2] = 2
    return np.fft.ifft(np.fft.fft(x) * h).imag


def ricker(freq, dt, length=0.128, phase_deg=0.0):
    """Odd-length Ricker wavelet centred at zero, optionally phase rotated."""
    half = int(round(length / dt / 2))
    t = np.arange(-half, half + 1) * dt
    a = (np.pi * freq * t) ** 2
    w = (1 - 2 * a) * np.exp(-a)
    if phase_deg:
        ph = np.radians(phase_deg)
        w = w * np.cos(ph) - _hilbert(w) * np.sin(ph)
    return w


def aki_richards(vp1, vs1, rho1, vp2, vs2, rho2, theta):
    """Aki-Richards P-P reflectivity. Interface arrays (n,), angles in radians (m,) -> (n, m)."""
    vp = 0.5 * (vp1 + vp2)[:, None]
    vs = 0.5 * (vs1 + vs2)[:, None]
    rho = 0.5 * (rho1 + rho2)[:, None]
    dvp, dvs, drho = (vp2 - vp1)[:, None], (vs2 - vs1)[:, None], (rho2 - rho1)[:, None]
    s2 = np.sin(theta)[None, :] ** 2
    k = (vs / vp) ** 2
    return (0.5 * (1 - 4 * k * s2) * drho / rho + dvp / (2 * vp * np.cos(theta)[None, :] ** 2)
            - 4 * k * s2 * dvs / vs)


def zoeppritz_rpp(vp1, vs1, rho1, vp2, vs2, rho2, theta):
    """Exact P-P reflection coefficient for a single interface and an array of angles (radians)."""
    th1 = np.asarray(theta, dtype=complex)
    p = np.sin(th1) / vp1
    th2 = np.arcsin(p * vp2)
    ph1 = np.arcsin(p * vs1)
    ph2 = np.arcsin(p * vs2)
    out = np.zeros(th1.shape, dtype=complex)
    for i in range(th1.size):
        t1, t2, p1, p2 = th1.flat[i], th2.flat[i], ph1.flat[i], ph2.flat[i]
        M = np.array([
            [-np.sin(t1), -np.cos(p1), np.sin(t2), np.cos(p2)],
            [np.cos(t1), -np.sin(p1), np.cos(t2), -np.sin(p2)],
            [2 * rho1 * vs1 * np.sin(p1) * np.cos(t1), rho1 * vs1 * (1 - 2 * np.sin(p1) ** 2),
             2 * rho2 * vs2 * np.sin(p2) * np.cos(t2), rho2 * vs2 * (1 - 2 * np.sin(p2) ** 2)],
            [-rho1 * vp1 * (1 - 2 * np.sin(p1) ** 2), rho1 * vs1 * np.sin(2 * p1),
             rho2 * vp2 * (1 - 2 * np.sin(p2) ** 2), -rho2 * vs2 * np.sin(2 * p2)],
        ])
        N = np.array([
            [np.sin(t1), np.cos(p1), -np.sin(t2), -np.cos(p2)],
            [np.cos(t1), -np.sin(p1), np.cos(t2), -np.sin(p2)],
            [2 * rho1 * vs1 * np.sin(p1) * np.cos(t1), rho1 * vs1 * (1 - 2 * np.sin(p1) ** 2),
             2 * rho2 * vs2 * np.sin(p2) * np.cos(t2), rho2 * vs2 * (1 - 2 * np.sin(p2) ** 2)],
            [rho1 * vp1 * (1 - 2 * np.sin(p1) ** 2), -rho1 * vs1 * np.sin(2 * p1),
             -rho2 * vp2 * (1 - 2 * np.sin(p2) ** 2), rho2 * vs2 * np.sin(2 * p2)],
        ])
        out.flat[i] = np.linalg.solve(M, N)[0, 0]
    return out.real


def stack_angles(stack, n_angles=5):
    return np.radians(np.linspace(stack.angle_min, stack.angle_max, n_angles))


def twt_from_vp(vp, dz, t_top):
    """Two-way time at the centre of each depth sample, starting at t_top (s)."""
    dt = 2.0 * dz / np.asarray(vp)
    return t_top + np.cumsum(dt) - 0.5 * dt


def model_grid(t_out, dt_model):
    """Fine modelling grid whose sample k*r + r//2 coincides with output sample k."""
    dt_out = t_out[1] - t_out[0]
    r = int(round(dt_out / dt_model))
    t_model = t_out[0] - (r // 2) * dt_model + np.arange(len(t_out) * r) * dt_model
    return t_model, r


def to_time(twt, arrays, t_model):
    """Blocky resampling of depth-indexed arrays onto a time grid. Returns arrays and validity mask."""
    idx = np.clip(np.searchsorted(twt, t_model), 0, len(twt) - 1)
    valid = (t_model >= twt[0]) & (t_model <= twt[-1])
    return [np.asarray(a)[idx] for a in arrays], valid


def reflectivity(vp, vs, rho, stacks, n_angles=5):
    """Partial-stack reflectivity (n_stacks, n) on a regular time grid, averaged over each angle range."""
    n = len(vp)
    out = np.zeros((len(stacks), n))
    change = np.flatnonzero((np.diff(vp) != 0) | (np.diff(vs) != 0) | (np.diff(rho) != 0))
    if change.size == 0:
        return out
    i1, i2 = change, change + 1
    for k, st in enumerate(stacks):
        R = aki_richards(vp[i1], vs[i1], rho[i1], vp[i2], vs[i2], rho[i2], stack_angles(st, n_angles))
        out[k, i2] = R.mean(axis=1)
    return out


def model_reflectivity(twt, vp, vs, rho, stacks, t_out, dt_model, shift=0.0):
    t_model, r = model_grid(t_out, dt_model)
    (vpt, vst, rhot), _ = to_time(np.asarray(twt) + shift, [vp, vs, rho], t_model)
    return reflectivity(vpt, vst, rhot, stacks), r


def convolve_decimate(refl, wavelet, r):
    return np.convolve(refl, wavelet, mode="same")[r // 2::r]


def labels_on_grid(twt, labels, t_out, dt_model, n_classes, shift=0.0):
    """Majority label per output sample; -1 where the bin is not fully covered by the model."""
    t_model, r = model_grid(t_out, dt_model)
    (lab,), valid = to_time(np.asarray(twt) + shift, [labels], t_model)
    lab = np.where(valid, lab, -1).reshape(len(t_out), r)
    counts = np.stack([(lab == c).sum(axis=1) for c in range(n_classes)], axis=1)
    out = counts.argmax(axis=1)
    out[(lab < 0).any(axis=1)] = -1
    return out


def synthesize(twt, vp, vs, rho, stacks, wavelets, t_out, dt_model, shift=0.0):
    """Synthetic partial stacks (n_stacks, n_out) for one 1D model."""
    refl, r = model_reflectivity(twt, vp, vs, rho, stacks, t_out, dt_model, shift)
    return np.stack([convolve_decimate(refl[k], wavelets[k], r) for k in range(len(stacks))])


def bandlimited_noise(n, freq, dt, sigma, rng):
    """Gaussian noise filtered by a Ricker wavelet, scaled to standard deviation sigma."""
    w = ricker(freq, dt)
    x = np.convolve(rng.standard_normal(n + len(w)), w, mode="same")[len(w) // 2:len(w) // 2 + n]
    return sigma * x / (x.std() + 1e-12)
