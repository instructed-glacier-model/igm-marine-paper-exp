#!/usr/bin/env python3
"""
thin_ice_calving (igm user process, the control run without a front scheme): remove floating
ice thinner than `min_thk` after each thickness update and recompute the surfaces, as
igm-mismip-plus does for its Ice2 melt. Without a sub-grid front the ice advected into an
ice-free cell spreads over the whole cell and the front disperses into a thin film (Albrecht
et al., 2011, Fig. 1b); a floating film of a few metres or less makes the Newton system
nearly singular, so it is removed. Place it right after `thk`.
"""

import tensorflow as tf


def initialize(cfg, state):
    state._tic_ratio = float(cfg.processes.thk.ratio_density)
    state._tic_min = float(cfg.processes.thin_ice_calving.min_thk)
    state.calved_thin_ice_step = 0.0  # m3 removed during the last step


def update(cfg, state):
    thk = state.thk
    ratio = tf.cast(state._tic_ratio, thk.dtype)
    floating = thk * ratio < state.water_level - state.topg
    remove = floating & (thk > 0.0) & (thk < state._tic_min)
    removed = tf.reduce_sum(tf.where(remove, thk, 0.0)) * state.dx * state.dx
    thk = tf.where(remove, tf.zeros_like(thk), thk)
    state.thk = thk
    state.lsurf = tf.maximum(state.topg, state.water_level - ratio * thk)
    state.usurf = state.lsurf + thk
    state.calved_thin_ice_step = float(removed)


def finalize(cfg, state):
    pass
