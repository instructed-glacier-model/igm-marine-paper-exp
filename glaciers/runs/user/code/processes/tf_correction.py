#!/usr/bin/env python3
"""Per-cell thermal-forcing correction of the sub-shelf melt (igm user process).

Adds a fixed field dTF(x) (K) to ``state.ocean_thermal_forcing`` after the ``ocean`` process
and before ``bmb``:

    TF(x, t) = TF_profile(z_draft(x, t)) + delta_T(t) + dTF(x)

so that the ``bmb`` quadratic law (``averaging: local``) melts
m = F (TF + dTF) |TF + dTF| with F = gamma0 (rho_w c_o / (rho_i L))^2 rho_fw / rho_i. With
dTF from tools/pig_balance.py (``field``), the melt at t = 0 is the steady-shelf (balance)
melt m_bal = smb - div(u H) of the start state on the floating ice: an ISMIP6 / Lipscomb-style
thermal-forcing correction, but per cell. The forcing anomaly delta_T(t) of the ``ocean``
process acts through the same law. The field is extended off the shelf by nearest neighbour
from the grounding-line band, so ice that goes afloat gets grounding-line melt.

The field is read once from ``file`` (netCDF under ``core.folder_data``, variable
``variable``, on the run grid; the coordinates are checked). ``state.tf_correction`` holds it
(add it to write_ncdf.vars_to_save to save it). Only the thermal forcing is changed:
``ocean_temp`` and ``ocean_salinity`` (PICO, PICOP, plume) are not.

Process order: [..., ocean, tf_correction, ..., bmb, ...].
"""

import os

import numpy as np
import tensorflow as tf
import xarray as xr


def initialize(cfg, state):
    p = cfg.processes.tf_correction
    if "ocean" not in cfg.processes:
        raise ValueError("tf_correction corrects the ocean thermal forcing: add the ocean process")
    root = str(getattr(state, "original_cwd", os.getcwd()))
    path = os.path.join(root, str(cfg.core.folder_data), str(p.file))
    with xr.open_dataset(path) as ds:
        ds = ds.sortby(["x", "y"])
        field = np.asarray(ds[str(p.variable)].values, dtype=np.float64)
        x, y = ds.x.values.astype(np.float64), ds.y.values.astype(np.float64)
    sx, sy = state.x.numpy().astype(np.float64), state.y.numpy().astype(np.float64)
    if field.shape != (len(sy), len(sx)) or np.abs(x - sx).max() > 1e-3 or np.abs(y - sy).max() > 1e-3:
        raise ValueError(f"tf_correction: {path} is not on the run grid")
    if not np.all(np.isfinite(field)):
        raise ValueError(f"tf_correction: {path} {p.variable} has non-finite values")
    state.tf_correction = tf.constant(field, dtype=state.thk.dtype)


def update(cfg, state):
    state.ocean_thermal_forcing = state.ocean_thermal_forcing + tf.cast(
        state.tf_correction, state.ocean_thermal_forcing.dtype)


def finalize(cfg, state):
    pass
