#!/usr/bin/env python3
"""
Albrecht flowline time series (igm user output module): albrecht_ts.csv in the run folder.

Every step is accumulated (mass budget); a row is written at the first step, at every igm
save time and at the first step past each multiple of `freq` years (0: every step). Columns:

  t, dt (yr), step, wall (s since the module was initialised),
  volume (m3)       ice volume sum(thk + Href) dx2 (Href: partial cells of a front),
  inflow (m3)       cumulative inflow through the left (Dirichlet) face,
  calved (m3)       cumulative calving (state.calved_thk, plus the control's thin-ice removal),
  residual (m3)     volume - volume(0) - inflow + calved (mass conservation: round-off),
  x_front (m)       front on the middle row, from the first node (the inflow boundary): the
                    right face of the last full cell plus the fill fraction of the next cell
                    (state.ice_area_fraction),
  H_front (m)       thickness of the last full cell, u_front (m/yr) its velocity,
  umax (m/yr)       largest depth-averaged speed.

The pure-NumPy functions are also used by tools/ (a single implementation).
"""

import os
import time

import numpy as np

COLUMNS = [
    "t", "dt", "step", "wall", "volume", "inflow", "calved", "residual",
    "x_front", "H_front", "u_front", "umax",
]


def front_row(thk_row, fraction_row, dx):
    """(x_front, index of the last full cell) along one row; x from the first node."""
    ice = np.flatnonzero(np.asarray(thk_row) > 0.0)
    if ice.size == 0:
        return 0.0, -1
    last = int(ice[-1])
    fill = float(fraction_row[last + 1]) if last + 1 < len(fraction_row) else 0.0
    return (last + 0.5 + fill) * float(dx), last


def initialize(cfg, state):
    p = cfg.outputs.albrecht_ts
    state._ats = {
        "file": p.output_file,
        "freq": float(p.freq),
        "t0": time.time(),
        "next": None,
        "step": 0,
        "inflow": 0.0,
        "calved": 0.0,
        "volume0": None,
    }
    with open(p.output_file, "w") as f:
        f.write(",".join(COLUMNS) + "\n")


def _volume(state):
    thk = np.asarray(state.thk, np.float64)
    if hasattr(state, "Href"):
        thk = thk + np.asarray(state.Href, np.float64)
    return float(np.sum(thk)) * float(state.dx) ** 2


def run(cfg, state):
    s = state._ats
    dx = float(state.dx)
    dt = float(state.dt)
    if s["volume0"] is None:
        s["volume0"] = _volume(state)
    else:
        # thk has just advanced by dt: accumulate this step's inflow and calving
        ghost = state.thk_components.transport_options.get("left_ghost")
        if ghost is not None:
            u_edge = np.asarray(state.ubar, np.float64)[:, 0]
            s["inflow"] += float(np.sum(u_edge * np.asarray(ghost, np.float64)[:, 0])) * dx * dt
        if hasattr(state, "calved_thk"):
            s["calved"] += float(np.sum(np.asarray(state.calved_thk, np.float64))) * dx * dx
        if hasattr(state, "calved_thin_ice_step"):
            s["calved"] += float(state.calved_thin_ice_step)
        s["step"] += 1
    t = float(state.t)
    due = s["next"] is None or t >= s["next"] - 1e-9 or s["freq"] <= 0.0 or bool(state.saveresult)
    if not due:
        return
    if s["freq"] > 0.0:
        s["next"] = (np.floor(t / s["freq"] + 1e-9) + 1.0) * s["freq"]
    volume = _volume(state)
    thk = np.asarray(state.thk)
    fraction = np.asarray(getattr(state, "ice_area_fraction", thk > 0.0), np.float64)
    mid = thk.shape[0] // 2
    x_front, last = front_row(thk[mid], fraction[mid], dx)
    ubar = np.asarray(state.ubar)
    row = [
        t, dt, s["step"], time.time() - s["t0"], volume, s["inflow"], s["calved"],
        volume - s["volume0"] - s["inflow"] + s["calved"], x_front,
        float(thk[mid, last]) if last >= 0 else 0.0,
        float(ubar[mid, last]) if last >= 0 else 0.0,
        float(np.max(np.hypot(ubar, np.asarray(state.vbar)))),
    ]
    with open(s["file"], "a") as f:
        f.write(",".join(f"{v:.10g}" for v in row) + "\n")
