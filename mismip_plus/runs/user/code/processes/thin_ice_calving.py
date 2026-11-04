#!/usr/bin/env python3
"""
thin_ice_calving (igm user process): remove floating ice thinner than `min_thk` after each
thickness update, and recompute the surfaces (lsurf = max(topg, wl - ratio thk), usurf =
lsurf + thk, as igm thk/surfaces.py).

Why: in igm's ice flow every node with thk > 0 is ice. The Ice2 melt (100 m/yr on the
floating ice beyond x = 480 km) leaves a floating film of 1 mm - 1 m over thousands of nodes;
it has almost no stiffness, the Newton system becomes nearly singular there, cg_newton
returns speeds of ~1e6 m/yr and the CFL step collapses (outputs/superseded/ice2_cg_stall).
Removing the film makes x ~ 480 km a calving front, as the melt makes it physically. The
threshold never acts in Ice0 / Ice1: their floating ice stays thicker than 19.6 m
(1 km runs, every frame).

Place it right after `thk` in the process list.
"""

import tensorflow as tf


def initialize(cfg, state):
    state._tic_ratio = float(cfg.processes.thk.ratio_density)
    state._tic_min = float(cfg.processes.thin_ice_calving.min_thk)
    state.calved_thin_ice = tf.Variable(0.0, dtype=state.thk.dtype, trainable=False)  # m3, cumulative


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
    state.calved_thin_ice.assign_add(tf.cast(removed, state.calved_thin_ice.dtype))


def finalize(cfg, state):
    pass
