#!/usr/bin/env python3
"""Save the ice-flow network's weights at every saved frame (Rosier et al. split, studies/error_decomposition).

At a step that writes a frame (state.saveresult), the network weights after that step's ice-flow update are those
that produced the frame's velocity (igm trains, then evaluates). They are written to weights/w_T.npz in the run
folder (all Keras weights, the normaliser's included), so the network as it was at time T can be evaluated later on
another geometry (user process load_network_weights). Read-only: the run itself is unchanged.
Place it after `time`, which sets state.saveresult.
"""

import os

import numpy as np


def initialize(cfg, state):
    os.makedirs("weights", exist_ok=True)  # the run folder is the working directory


def update(cfg, state):
    if not getattr(state, "saveresult", False):
        return
    weights = state.iceflow.optimizer.map.network.get_weights()
    t = float(state.t.numpy()) if hasattr(state.t, "numpy") else float(state.t)
    np.savez(os.path.join("weights", f"w_{t:.4f}.npz"), *weights)


def finalize(cfg, state):
    pass
