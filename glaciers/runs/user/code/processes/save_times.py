#!/usr/bin/env python3
"""Irregular save times: the frames of the a-posteriori snapshot errors (emulator sweep).

igm's `time` process saves at start, start + save, ..., end (a scalar frequency). This module
replaces that list, once in `initialize`, by `times` (yr) plus every multiple of `every` (yr)
from start, always with start and end, keeping only times inside [start, end]. `time.update`
then clips dt to land on each of them (state.time_save, indexed by state.itsave) and sets
state.saveresult, so write_ncdf and marine_ts write a frame there. Output scheduling only: the
physics is unchanged, apart from the few steps shortened to reach a save time.

The dense early times (0.01 ... 0.5 yr) keep the error just after the initial training of an
online network visible. A single-step snapshot run (start Y, end Y + 0.001) reduces to [Y, end].
Place the process right after `time` (its initialize must run after time's).
"""

import numpy as np
import tensorflow as tf


def initialize(cfg, state):
    p = cfg.processes.save_times
    t0, t1 = float(cfg.processes.time.start), float(cfg.processes.time.end)
    times = {float(t) for t in p.times}
    every = float(p.every)
    if every > 0.0:
        times |= {float(t) for t in np.arange(t0, t1, every)}
    keep = sorted({t0, t1} | {round(t, 9) for t in times if t0 < t < t1})
    state.time_save = tf.constant(keep, dtype="float32")


def update(cfg, state):
    pass


def finalize(cfg, state):
    pass
