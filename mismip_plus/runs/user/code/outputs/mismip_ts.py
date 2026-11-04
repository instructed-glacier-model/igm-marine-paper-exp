#!/usr/bin/env python3
"""
MISMIP+ scalar time series (igm user output module): mismip_ts.csv in the run folder.

A row is written at the first step, at every igm save time, and at the first step past
each multiple of `freq` years (freq 0: every step). Columns:

  t, dt (yr), step, wall (s since the module was initialised),
  iceVolume, iceVAF (m3), groundedArea (m2)     over the whole 640 x 80 km domain,
  xGL_centre, xGL_south, xGL_north (m)          grounding line along y = 0 and along the
                                                two side walls (y = -40 and +40 km),
  umax (m/yr)                                   largest depth-averaged speed.

Grounded means a positive height above flotation hAF = thk - (rho_w/rho_i) max(wl - topg, 0),
the criterion of the igm thickness process (thk/masks.py) evaluated with the bed. Domain
integrals use the trapezoid rule on the nodes (half weights on the edges); sums are in
float64. The grounding line along a row is the first grounded-to-floating crossing from
the divide, placed by linear interpolation of hAF between the two nodes.

The pure-NumPy functions are also used by tools/ (a single implementation).
"""

import time

import numpy as np

COLUMNS = [
    "t", "dt", "step", "wall", "iceVolume", "iceVAF", "groundedArea",
    "xGL_centre", "xGL_south", "xGL_north", "umax",
]


def trapezoid_weights(ny, nx):
    wy = np.ones(ny)
    wy[[0, -1]] = 0.5
    wx = np.ones(nx)
    wx[[0, -1]] = 0.5
    return np.outer(wy, wx)


def height_above_flotation(thk, topg, wl, ratio):
    """hAF (m of ice); > 0 on grounded ice. ratio = rho_i / rho_w."""
    thk = np.asarray(thk, dtype=np.float64)
    return thk - np.maximum(np.asarray(wl, np.float64) - np.asarray(topg, np.float64), 0.0) / ratio


def integrals(thk, topg, wl, dx, ratio):
    """(iceVolume, iceVAF, groundedArea) in m3, m3, m2."""
    thk = np.asarray(thk, dtype=np.float64)
    haf = height_above_flotation(thk, topg, wl, ratio)
    w = trapezoid_weights(*thk.shape) * float(dx) ** 2
    grounded = (haf > 0.0) & (thk > 0.0)
    return (
        float(np.sum(w * thk)),
        float(np.sum(w * np.where(grounded, haf, 0.0))),
        float(np.sum(w * grounded)),
    )


def gl_along_row(x, haf_row):
    """First grounded -> floating crossing from x[0] (m); nan if the row starts afloat,
    x[-1] if it is grounded throughout."""
    haf_row = np.asarray(haf_row, dtype=np.float64)
    grounded = haf_row > 0.0
    if not grounded[0]:
        return np.nan
    floating = np.flatnonzero(~grounded)
    if floating.size == 0:
        return float(x[-1])
    i = floating[0] - 1  # last grounded node before the first floating one
    f = haf_row[i] / (haf_row[i] - haf_row[i + 1])
    return float(x[i] + f * (x[i + 1] - x[i]))


def gl_rows(x, y, haf):
    """Grounding line along y = 0 (the node row nearest to it) and along both walls."""
    jc = int(np.argmin(np.abs(np.asarray(y))))
    return gl_along_row(x, haf[jc]), gl_along_row(x, haf[0]), gl_along_row(x, haf[-1])


def diagnostics(x, y, thk, topg, wl, ratio):
    dx = float(x[1] - x[0])
    haf = height_above_flotation(thk, topg, wl, ratio)
    vol, vaf, area = integrals(thk, topg, wl, dx, ratio)
    return (vol, vaf, area) + gl_rows(x, y, haf)


# --- igm output module interface ---------------------------------------------------------


def initialize(cfg, state):
    p = cfg.outputs.mismip_ts
    state._mts_file = str(p.output_file)
    state._mts_freq = float(p.freq)
    state._mts_next = None
    state._mts_step = 0
    state._mts_wall0 = time.time()
    state._mts_const = (
        state.x.numpy().astype(np.float64),
        state.y.numpy().astype(np.float64),
        float(cfg.processes.thk.ratio_density),
    )
    with open(state._mts_file, "w") as f:
        f.write(",".join(COLUMNS) + "\n")


def run(cfg, state):
    del cfg
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
    x, y, ratio = state._mts_const
    thk = state.thk.numpy()
    vals = diagnostics(x, y, thk, state.topg.numpy(), state.water_level.numpy(), ratio)
    umax = float(np.sqrt(np.max(state.ubar.numpy() ** 2 + state.vbar.numpy() ** 2)))
    dt = getattr(state, "dt", 0.0)  # a tf.Variable, or a float with a fixed step (cfl <= 0)
    dt = float(dt.numpy()) if hasattr(dt, "numpy") else float(dt)
    row = [t, dt, step, time.time() - state._mts_wall0, *vals, umax]
    with open(state._mts_file, "a") as f:
        f.write(",".join(f"{v:.10g}" for v in row) + "\n")
