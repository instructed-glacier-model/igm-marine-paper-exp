#!/usr/bin/env python3
"""
steady_stop (igm user process): end an Ice0 spin-up once it is steady.

Every `check_freq` model years it records x_GL(y = 0) and the volume above flotation (the
functions of user/code/outputs/mismip_ts.py). Once the run is at least `min_years` long, it
compares the means over the last two windows of `window` years:

    |mean x_GL(last) - mean x_GL(previous)|               <= tol_gl   (m)
    |mean VAF(last) - mean VAF(previous)| / mean VAF(last) <= tol_vaf

Means over windows make the test insensitive to the step-to-step noise of the cheap
emulator (~0.4 km at 1 km). When both hold, the current step is saved (state.saveresult)
and the run ends (state.continue_run); time.end remains the cap. The decision and the last
window statistics are written to steady.txt in the run folder.

It must come after iceflow in the process list, so that the tested state is the one saved.
"""

import importlib.util
import os

import numpy as np


def _ts_module(state):
    path = os.path.join(str(state.original_cwd), "user", "code", "outputs", "mismip_ts.py")
    spec = importlib.util.spec_from_file_location("mismip_ts_functions", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def initialize(cfg, state):
    p = cfg.processes.steady_stop
    state._ss = {
        "ts": _ts_module(state),
        "x": state.x.numpy().astype(np.float64),
        "y": state.y.numpy().astype(np.float64),
        "ratio": float(cfg.processes.thk.ratio_density),
        "t0": float(cfg.processes.time.start),
        "next": float(cfg.processes.time.start),
        "freq": float(p.check_freq),
        "window": float(p.window),
        "min_years": float(p.min_years),
        "tol_gl": float(p.tol_gl),
        "tol_vaf": float(p.tol_vaf),
        "t": [], "gl": [], "vaf": [],
        "done": False,
    }
    with open("steady.txt", "w") as f:
        f.write(f"steady_stop: window {p.window} yr, min_years {p.min_years}, tol_gl {p.tol_gl} m, "
                f"tol_vaf {p.tol_vaf}, check every {p.check_freq} yr\n")


def update(cfg, state):
    ss = state._ss
    if ss["done"]:
        return
    t = float(state.t.numpy())
    if t < ss["next"] - 1e-6:
        return
    ss["next"] = (np.floor((t - ss["t0"]) / ss["freq"] + 1e-6) + 1.0) * ss["freq"] + ss["t0"]
    ts = ss["ts"]
    thk, topg, wl = state.thk.numpy(), state.topg.numpy(), state.water_level.numpy()
    haf = ts.height_above_flotation(thk, topg, wl, ss["ratio"])
    gl, _, _ = ts.gl_rows(ss["x"], ss["y"], haf)
    _, vaf, _ = ts.integrals(thk, topg, wl, float(ss["x"][1] - ss["x"][0]), ss["ratio"])
    ss["t"].append(t); ss["gl"].append(gl); ss["vaf"].append(vaf)
    if t - ss["t0"] < max(ss["min_years"], 2.0 * ss["window"]):
        return
    T = np.array(ss["t"]); G = np.array(ss["gl"]); V = np.array(ss["vaf"])
    last = T > t - ss["window"]
    prev = (T > t - 2.0 * ss["window"]) & ~last
    if last.sum() < 2 or prev.sum() < 2 or np.isnan(G[last | prev]).any():
        return
    d_gl = np.mean(G[last]) - np.mean(G[prev])
    d_vaf = (np.mean(V[last]) - np.mean(V[prev])) / np.mean(V[last])
    if abs(d_gl) <= ss["tol_gl"] and abs(d_vaf) <= ss["tol_vaf"]:
        ss["done"] = True
        state.saveresult = True   # the outputs of this step write the steady frame
        state.continue_run = False
        msg = (f"steady at t = {t:.1f} yr: window means x_GL {np.mean(G[prev]) / 1e3:.3f} -> "
               f"{np.mean(G[last]) / 1e3:.3f} km (change {d_gl:+.1f} m), VAF change {d_vaf:+.2e}\n")
        with open("steady.txt", "a") as f:
            f.write(msg)
        print("steady_stop: " + msg, end="")


def finalize(cfg, state):
    ss = getattr(state, "_ss", None)
    if ss is not None and not ss["done"]:
        with open("steady.txt", "a") as f:
            f.write(f"NOT steady: reached time.end = {cfg.processes.time.end} yr\n")
