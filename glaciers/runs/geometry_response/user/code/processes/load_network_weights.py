#!/usr/bin/env python3
"""Load, before each step's ice-flow evaluation, the network weights saved at that frame by save_network_weights.

With replay_geometry (which sets the geometry of the frame the step lands on) and no training (nbit 0), the run
evaluates the network as it was at time T, in the source network run, on the geometry of another run at T: the
emulator replay of Rosier et al. (egusphere-2026-5523, Sect. 3.3), V_net(I^R), for an online-trained network.
processes.load_network_weights.source: the network run whose outputs/RUN/weights/w_T.npz are loaded.
Place it right after replay_geometry (before iceflow), which replaces state.time_save by the frames.
"""

import os

import numpy as np


def initialize(cfg, state):
    p = cfg.processes.load_network_weights
    root = str(getattr(state, "original_cwd", os.getcwd()))
    source = p.source if os.path.isabs(p.source) else os.path.join(root, p.source)
    state.weights_dir = os.path.join(source, "weights")
    if not os.path.isdir(state.weights_dir):
        raise ValueError(f"load_network_weights: no weights in {source} (run it with save_network_weights)")


def update(cfg, state):
    k = state.itsave + 1
    if k >= len(state.time_save):
        return
    t = float(state.time_save[k].numpy())
    path = os.path.join(state.weights_dir, f"w_{t:.4f}.npz")
    if not os.path.exists(path):
        raise ValueError(f"load_network_weights: {path} missing")
    with np.load(path) as w:
        state.iceflow.optimizer.map.network.set_weights([w[f"arr_{i}"] for i in range(len(w.files))])


def finalize(cfg, state):
    pass
