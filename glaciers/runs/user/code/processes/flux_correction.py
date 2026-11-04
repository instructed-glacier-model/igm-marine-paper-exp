#!/usr/bin/env python3
"""Static flux correction of the grounded ice (igm user process), after Hill et al. (2023, TC
17, 3739): a fixed field C(x) (m ice / yr) is added to the surface mass balance of the grounded
ice, so that the start geometry does not drift under present-day forcing (their correction:
minus the dH/dt of the relaxed state, subtracted from the mass balance).

    smb(x, t) = smb_input(x) + C(x) [grounded(x, t)]

"grounded" is igm's flotation test (thk.masks.compute_grounded_mask) on the current geometry,
so a cell that goes afloat loses its correction and takes the ocean melt instead
(tf_correction + bmb), and one that re-grounds takes the field's value there (0 where the
start state floats). C comes from tools/pig_balance.py fluxcorr (the dH/dt of a relaxation
interval, or of the start state with this kernel) and is read once from ``file`` (netCDF under
``core.folder_data``, variable ``variable``, on the run grid).

``state.flux_correction`` holds C; ``state.smb`` is rebuilt every step from the input SMB
(``state.smb_input``, kept at initialisation), so marine_ts counts the correction in its
sources. Place the process before ``thk`` (after ``bmb``).
"""

import os

import numpy as np
import tensorflow as tf
import xarray as xr

from igm.processes.thk.masks import compute_grounded_mask


def initialize(cfg, state):
    p = cfg.processes.flux_correction
    root = str(getattr(state, "original_cwd", os.getcwd()))
    path = os.path.join(root, str(cfg.core.folder_data), str(p.file))
    with xr.open_dataset(path) as ds:
        ds = ds.sortby(["x", "y"])
        field = np.asarray(ds[str(p.variable)].values, dtype=np.float64)
        x, y = ds.x.values.astype(np.float64), ds.y.values.astype(np.float64)
    sx, sy = state.x.numpy().astype(np.float64), state.y.numpy().astype(np.float64)
    if field.shape != (len(sy), len(sx)) or np.abs(x - sx).max() > 1e-3 or np.abs(y - sy).max() > 1e-3:
        raise ValueError(f"flux_correction: {path} is not on the run grid")
    if not np.all(np.isfinite(field)):
        raise ValueError(f"flux_correction: {path} {p.variable} has non-finite values")
    dtype = state.thk.dtype
    state.flux_correction = tf.constant(field, dtype=dtype)
    state.smb_input = tf.identity(tf.cast(state.smb, dtype))
    state._fc_rho_ratio = float(cfg.processes.iceflow.physics.water_density) / float(
        cfg.processes.iceflow.physics.ice_density)
    update(cfg, state)


def update(cfg, state):
    grounded = compute_grounded_mask(state.thk, state.topg, state.water_level, state._fc_rho_ratio)
    state.smb = state.smb_input + tf.where(grounded & (state.thk > 0.0), state.flux_correction,
                                           tf.zeros_like(state.flux_correction))


def finalize(cfg, state):
    pass
