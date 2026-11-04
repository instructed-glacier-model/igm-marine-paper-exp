#!/usr/bin/env python3
"""Replay the saved geometry of a finished transient run: one exact snapshot solve per frame, in one run.

The same-geometry snapshot solves (tools/snapshot_manifest.py) start one igm process per frame, and
each pays the start-up, the tracing and the XLA compilation of the solver again. This process does all
the frames of a source run in a single igm run:
  - it replaces state.time_save by the source's frame times in [time.start, time.end], so that with
    time.cfl 0 and a time.step_max longer than any frame interval each step lands on the next frame;
  - before each step's solve it sets thk, usurf and lsurf to the geometry the source run's ice-flow
    solve saw at that frame (iceflow_input_thk / iceflow_input_usurf of SOURCE/output.nc);
  - it restarts the exact solver from zero velocity before each frame (the zero restart of cg_newton's
    cold_restart; its own time-based schedule, cold_restart_interval, should be 0, since the solve of
    a step runs at the previous frame's time), so each frame is an independent cold solve, as in the
    single snapshot runs. The step budget nbit should then be the cold one (nbit_init).
The geometry is not evolved (no `thk` process); write_ncdf saves the velocity of each frame at its time.
The first frame is solved by iceflow's initialize on the input file, which must hold that frame's
geometry (tools/prepare_reference_input.py), and is not restarted again. With a network mapping (the emulator
replay, with load_network_weights) there is no restart: the network's weights are loaded for each frame instead.
Place it first, before iceflow_geometry and iceflow.
"""

import os

import numpy as np
import tensorflow as tf
import xarray as xr


def initialize(cfg, state):
    p = cfg.processes.replay_geometry
    root = str(getattr(state, "original_cwd", os.getcwd()))
    source = p.source if os.path.isabs(p.source) else os.path.join(root, p.source)
    t0, t1 = float(cfg.processes.time.start), float(cfg.processes.time.end)
    with xr.open_dataset(os.path.join(source, "output.nc")) as out:
        t = np.asarray(out.time.values, dtype=float)
        keep = (t >= t0 - 1.0e-4) & (t <= t1 + 1.0e-4)
        state.replay_times = t[keep]
        state.replay_thk = out.iceflow_input_thk.values[keep].astype("float32")
        state.replay_usurf = out.iceflow_input_usurf.values[keep].astype("float32")
    times = state.replay_times
    if len(times) == 0 or abs(times[0] - t0) > 1.0e-4 or abs(times[-1] - t1) > 1.0e-4:
        raise ValueError(f"replay_geometry: time.start {t0:g} and time.end {t1:g} must be frames of {source}")
    state.replay_started = False


def update(cfg, state):
    if not state.replay_started:  # time's initialize has set its own save times by now
        state.time_save = tf.constant(state.replay_times, dtype="float32")
        state.replay_started = True
    k = state.itsave + 1
    if k >= len(state.replay_times):
        return
    dtype = state.thk.dtype
    state.thk = tf.constant(state.replay_thk[k], dtype=dtype)
    state.usurf = tf.constant(state.replay_usurf[k], dtype=dtype)
    state.lsurf = state.usurf - state.thk
    optimizer = state.iceflow.optimizer
    if k > 0 and getattr(optimizer.map, "name", "") == "identity":  # the exact solver; a network keeps its weights
        optimizer.map.set_theta([tf.zeros_like(value) for value in optimizer.map.get_theta()])
        optimizer.reset_warm_start()


def finalize(cfg, state):
    pass
