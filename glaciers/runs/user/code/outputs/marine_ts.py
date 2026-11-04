#!/usr/bin/env python3
"""Scalar time series of a marine transient (igm user output module): marine_ts.csv in the
run folder.

A row is written at the first step, at every igm save time, and at the first step past each
multiple of `freq` years (freq 0: every step). Columns:

  t, dt (yr), step, wall (s since the module was initialised; the epoch of that instant is
                          in the header, so the start-up is wall0_epoch - .t0),
  volume (m3)             ice volume, full cells + the partial-cell reservoir Href,
  vaf (m3)                volume above flotation, sum of hAF > 0 over grounded ice,
  area_gr, area_fl (m2)   grounded / floating ice (full cells),
  calved (m3)             cumulative calved volume (thk.front calved_thk, accumulated every
                          step, incl. removed icebergs),
  unapplied (m3)          cumulative ablation the sub-grid front could not apply in its step,
  sources (m3)            cumulative smb (+ bmb) volume over ice cells (thk > 0), for the
                          budget volume(t) - volume(0) ~ sources - calved (+ boundary fluxes),
  s_gl, s_cf (m)          grounding line / calving front along the fixed flowline
                          (data/GLACIER/flowline.csv, tools/flowline.py): the downstream end of
                          the last grounded (resp. ice-covered, ice_area_fraction >= 0.5)
                          stretch at least 2 km long, placed by linear interpolation between the
                          two flowline samples around the crossing; s_gl <= s_cf (a grounded
                          front is its own grounding line),
  umax, u9999 (m/yr)      largest and 99.99th-percentile depth-averaged speed over the ice.

Grounded means hAF = thk - (rho_w/rho_i) max(wl - topg, 0) > 0 on an ice cell, the flotation
criterion of the igm thickness process (thk/masks.py). Per-step accumulations stay on the
device; rows (host syncs) only at the row times, identical for every solver so that their cost
is the same. The pure-NumPy functions are also used by tools/.
"""

import os
import time

import numpy as np
import tensorflow as tf

COLUMNS = [
    "t", "dt", "step", "wall", "volume", "vaf", "area_gr", "area_fl", "calved", "unapplied",
    "sources", "s_gl", "s_cf", "umax", "u9999",
]


def height_above_flotation(thk, topg, wl, ratio):
    """hAF (m of ice); > 0 on grounded ice. ratio = rho_i / rho_w."""
    thk = np.asarray(thk, dtype=np.float64)
    return thk - np.maximum(np.asarray(wl, np.float64) - np.asarray(topg, np.float64), 0.0) / ratio


def read_flowline(path):
    data = np.loadtxt(path, delimiter=",", comments="#", skiprows=2)
    return data[:, 0], data[:, 1], data[:, 2]


def bilinear_weights(x, y, px, py):
    """Node indices and weights of the points (px, py) on the grid x, y (ascending);
    points outside the grid get zero weights (and sample 0)."""
    dx, dy = x[1] - x[0], y[1] - y[0]
    fx, fy = (px - x[0]) / dx, (py - y[0]) / dy
    i = np.clip(np.floor(fx).astype(int), 0, len(x) - 2)
    j = np.clip(np.floor(fy).astype(int), 0, len(y) - 2)
    wx, wy = np.clip(fx - i, 0, 1), np.clip(fy - j, 0, 1)
    inside = (fx >= 0) & (fy >= 0) & (fx <= len(x) - 1) & (fy <= len(y) - 1)
    w = np.stack([(1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy], axis=1)
    w *= inside[:, None]
    jj = np.stack([j, j, j + 1, j + 1], axis=1)
    ii = np.stack([i, i + 1, i, i + 1], axis=1)
    idx = np.stack([jj, ii], axis=-1)  # (n, 4, 2): (row, column) of the 4 nodes
    return idx, w


def sample(field, idx, w):
    return np.sum(field[idx[..., 0], idx[..., 1]] * w, axis=1)


def end_of_last_run(s, g, min_length=2000.0):
    """Downstream end of the last stretch with g > 0 at least `min_length` long (m), placed by
    linear interpolation to g = 0; nan if there is none. Ignores holes upstream (thin ice that
    melts out) and small downstream features (ice rises, bergs); s[-1] if the stretch reaches
    the end of the line."""
    pos = np.concatenate([[False], g > 0.0, [False]])
    edges = np.flatnonzero(np.diff(pos.astype(int)))
    starts, ends = edges[0::2], edges[1::2] - 1  # inclusive sample indices of each stretch
    keep = [(a, b) for a, b in zip(starts, ends) if s[b] - s[a] >= min_length]
    if not keep:
        return np.nan
    _, k = keep[-1]
    if k == len(s) - 1:
        return float(s[-1])
    f = g[k] / (g[k] - g[k + 1])
    return float(s[k] + f * (s[k + 1] - s[k]))


def flowline_positions(s, idx, w, thk, topg, wl, fraction, ratio):
    """(s_gl, s_cf) along the flowline."""
    haf = height_above_flotation(thk, topg, wl, ratio)
    grounded = np.where(thk > 0.0, haf, -1.0)  # negative off the ice: the grounded stretch ends
    s_gl = end_of_last_run(s, sample(grounded, idx, w))
    s_cf = end_of_last_run(s, sample(np.asarray(fraction, np.float64), idx, w) - 0.5)
    # a grounded front (tidewater / lake cliff): the grounding line is the front itself; the
    # hAF crossing of the ice-to-water drop lands up to a cell beyond the fraction-0.5 front
    return min(s_gl, s_cf), s_cf


def integrals(thk, href, topg, wl, dx, ratio):
    """(volume, vaf, area_gr, area_fl) in m3, m3, m2, m2 (cell sums)."""
    thk = np.asarray(thk, dtype=np.float64)
    haf = height_above_flotation(thk, topg, wl, ratio)
    ice = thk > 0.0
    grounded = ice & (haf > 0.0)
    a = float(dx) ** 2
    return (
        a * float(np.sum(thk) + np.sum(np.asarray(href, np.float64))),
        a * float(np.sum(np.where(grounded, haf, 0.0))),
        a * float(np.sum(grounded)),
        a * float(np.sum(ice & ~grounded)),
    )


# --- igm output module interface ---------------------------------------------------------


def initialize(cfg, state):
    p = cfg.outputs.marine_ts
    root = str(getattr(state, "original_cwd", os.getcwd()))
    s, px, py = read_flowline(os.path.join(root, str(p.flowline)))
    x = state.x.numpy().astype(np.float64)
    y = state.y.numpy().astype(np.float64)
    idx, w = bilinear_weights(x, y, px, py)
    state._mts_file = str(p.output_file)
    state._mts_freq = float(p.freq)
    state._mts_next = None
    state._mts_step = 0
    state._mts_const = (s, idx, w, abs(float(x[1] - x[0])), float(cfg.processes.thk.ratio_density))
    state._mts_acc = [tf.Variable(0.0, dtype=tf.float64) for _ in range(3)]  # calved, unapplied, sources
    state._mts_wall0 = time.time()
    with open(state._mts_file, "w") as f:
        f.write(f"# wall0_epoch={state._mts_wall0:.3f} flowline={p.flowline}\n")
        f.write(",".join(COLUMNS) + "\n")


def _accumulate(state, dx):
    area = tf.constant(dx * dx, tf.float64)
    calved, unapplied, sources = state._mts_acc
    if hasattr(state, "calved_thk"):
        calved.assign_add(tf.reduce_sum(tf.cast(state.calved_thk, tf.float64)) * area)
    if hasattr(state, "calving_unapplied_thk"):
        unapplied.assign_add(tf.reduce_sum(tf.cast(state.calving_unapplied_thk, tf.float64)) * area)
    rate = tf.cast(state.smb, tf.float64) if hasattr(state, "smb") else 0.0
    if hasattr(state, "bmb"):
        rate = rate + tf.cast(state.bmb, tf.float64)
    dt = tf.cast(state.dt, tf.float64) if hasattr(state, "dt") else 0.0
    ice = tf.cast(state.thk, tf.float64) > 0.0
    sources.assign_add(tf.reduce_sum(tf.where(ice, rate * dt, 0.0)) * area)


def run(cfg, state):
    del cfg
    s, idx, w, dx, ratio = state._mts_const
    _accumulate(state, dx)
    t = float(state.t.numpy())
    step = state._mts_step
    state._mts_step += 1
    due = step == 0 or bool(getattr(state, "saveresult", False))
    if state._mts_freq <= 0.0:
        due = True
    elif state._mts_next is None or t >= state._mts_next - 1e-6:
        due = True
    if not due:
        return
    if state._mts_freq > 0.0:
        k = np.floor(t / state._mts_freq + 1e-6) + 1.0
        state._mts_next = k * state._mts_freq
    thk = state.thk.numpy()
    href = state.Href.numpy() if hasattr(state, "Href") else np.zeros_like(thk)
    topg, wl = state.topg.numpy(), state.water_level.numpy()
    fraction = state.ice_area_fraction.numpy() if hasattr(state, "ice_area_fraction") else (thk > 0)
    vals = integrals(thk, href, topg, wl, dx, ratio)
    s_gl, s_cf = flowline_positions(s, idx, w, thk, topg, wl, fraction, ratio)
    speed = np.sqrt(state.ubar.numpy() ** 2 + state.vbar.numpy() ** 2)[thk > 0]
    umax = float(speed.max()) if speed.size else 0.0
    u9999 = float(np.percentile(speed, 99.99)) if speed.size else 0.0
    dt = getattr(state, "dt", 0.0)
    dt = float(dt.numpy()) if hasattr(dt, "numpy") else float(dt)
    acc = [float(a.numpy()) for a in state._mts_acc]
    row = [t, dt, step, time.time() - state._mts_wall0, *vals, *acc, s_gl, s_cf, umax, u9999]
    with open(state._mts_file, "a") as f:
        f.write(",".join(f"{v:.10g}" for v in row) + "\n")
