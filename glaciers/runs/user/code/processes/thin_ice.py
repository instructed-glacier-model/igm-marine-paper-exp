#!/usr/bin/env python3
"""Remove vanishing ice: thickness below `min_thickness` (m) becomes zero, after each `thk`
update and once at initialisation (a start file keeps what the run before it left).

Upwind transport takes a fixed fraction of a cell's ice at each step, so ice that flows onto a
cell with no mass balance to remove it (land outside the glacier mask, where the input SMB is
0) decays geometrically and never reaches zero: the glacier_jak relaxation left 85 cells with
0 < thk < 1e-3 m, down to 1e-45 m. Those cells count as ice, their ice-flow velocity is not
determined (up to 7.6e5 m/yr in the float32 solves), and the time step, whose CFL condition
takes the speed maximum over every cell (processes/time/time.py, no active-domain mask here),
drops to ~1e-4 yr (TRANSIENT.md, "Jakobshavn time-step collapse"). 0.01 m is PISM's
geometry.ice_free_thickness_standard.

The surfaces are then recomputed as `thk` does (flotation against the water level). The
removed volume (m3) and cell count are accumulated in state.thin_ice_removed /
state.thin_ice_cells and printed at the end. Place the process right after `thk`.
"""

import tensorflow as tf

from igm.processes.thk.surfaces import update_surfaces


def initialize(cfg, state):
    x = state.x.numpy()
    state._thin_ice_area = float(abs(x[1] - x[0])) ** 2
    state.thin_ice_removed = tf.Variable(0.0, dtype=tf.float64, trainable=False)
    state.thin_ice_cells = tf.Variable(0, dtype=tf.int64, trainable=False)
    update(cfg, state)


def update(cfg, state):
    h = float(cfg.processes.thin_ice.min_thickness)
    thin = (state.thk > 0.0) & (state.thk < h)
    state.thin_ice_removed.assign_add(
        tf.reduce_sum(tf.where(thin, tf.cast(state.thk, tf.float64), 0.0)) * state._thin_ice_area)
    state.thin_ice_cells.assign_add(tf.reduce_sum(tf.cast(thin, tf.int64)))
    state.thk = tf.where(thin, tf.zeros_like(state.thk), state.thk)
    update_surfaces(cfg, state)


def finalize(cfg, state):
    print(f"thin_ice: removed {int(state.thin_ice_cells.numpy())} cell-steps of ice thinner than "
          f"{float(cfg.processes.thin_ice.min_thickness):g} m, {float(state.thin_ice_removed.numpy()):.3g} m3 in total")
