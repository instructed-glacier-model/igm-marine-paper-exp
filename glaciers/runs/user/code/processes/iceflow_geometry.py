#!/usr/bin/env python3
"""Keep the geometry that this step's ice-flow solve sees.

Outputs are written after every process of a step, i.e. after `thk` has moved the
geometry on, so the saved velocity belongs to the previous thickness. This module
stores that geometry as `iceflow_input_thk` and `iceflow_input_usurf` (list them in
write_ncdf.vars_to_save); tools/prepare_reference_input.py builds the same-geometry
reference solves from them. Only `thk` changes the geometry within a step, so the
module can sit anywhere between `thk` and `iceflow` (here just before `iceflow`).
"""

import tensorflow as tf


def initialize(cfg, state):
    update(cfg, state)


def update(cfg, state):
    state.iceflow_input_thk = tf.identity(state.thk)
    state.iceflow_input_usurf = tf.identity(state.usurf)


def finalize(cfg, state):
    pass
