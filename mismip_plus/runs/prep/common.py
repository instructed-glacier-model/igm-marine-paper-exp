"""
MISMIP+ protocol constants, grid and bed geometry shared by prep/ and tools/.

Protocol: Asay-Davis et al. (2016, GMD 9, 2471), results: Cornford et al. (2020, TC 14, 2283).
Coordinates: x in [0, 640] km along flow, y in [-40, 40] km across flow (Cornford et al.,
2020; the submission files use y in [0, 80] km, see Y_SHIFT). Nodes sit on both side walls
and on the centre line, so the igm `periodic_ns` condition (last row = first row) closes the
80 km period exactly and the mirror symmetry of the bed makes it the free-slip wall.

Units: m, yr, MPa (igm units).
"""

import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- physical constants (Asay-Davis et al., 2016) ------------------------------------------
RHO_ICE = 918.0  # kg m-3
RHO_WATER = 1028.0  # kg m-3
GRAVITY = 9.81  # m s-2
RATIO = RHO_ICE / RHO_WATER  # 0.892996 (igm thk.ratio_density)
SEA_LEVEL = 0.0  # m
ACCUMULATION = 0.3  # m ice eq. yr-1
GLEN_N = 3.0
# A = 8.0e-25 Pa-3 s-1 = 2.525e-17 Pa-3 a-1 = 25.25 MPa-3 yr-1: the value of the PISM
# submissions (data sheet JFE_PISM), 26 % above the protocol's suggested 2.0e-17 Pa-3 a-1
# (Cornford et al., 2020, Sect. 2.3: Weertman models raised A by up to 25 % to put the Ice0
# grounding line at 450 +- 10 km). User decision 2026-09-26, after IGM MOLHO at 1 km gave
# x_GL(Ice0) 464 km with 20 MPa-3 yr-1 and 446 km with this value (PISM SSA+SIA: 450 km).
ARRHENIUS = 8.0e-25 * 31556926.0 * 1e18  # MPa-3 yr-1 (25.2455)
# Weertman (Cornford et al., 2020, Eq. 3 and Sect. 2.4): tau_b = beta2 |u|^(1/m) with
# beta2 = 3.160e6 Pa m-1/3 s1/3 = 1.0e4 Pa m-1/3 a1/3, i.e. igm tau_ref = 0.01 MPa at
# u_ref = 1 m/yr
WEERTMAN_M = 3.0
TAU_REF = 0.01  # MPa at U_REF
U_REF = 1.0  # m/yr

# --- bed (Cornford et al., 2020, Eq. 1 and Table 1; Asay-Davis et al., 2016) ---------------
# B6 = -50.57 m as in Asay-Davis et al. (2016); Cornford et al. (2020, Table 1) prints -50.75.
X_BAR = 300.0e3
B0, B2, B4, B6 = -150.0, -728.8, 343.91, -50.57
F_C, D_C, W_C = 4.0e3, 500.0, 24.0e3
Z_B_DEEP = -720.0
LX, LY = 640.0e3, 80.0e3
Y_SHIFT = 40.0e3  # y_submission = y + Y_SHIFT

# --- Ice2 melt (Cornford et al., 2020, Eq. 8) -----------------------------------------------
ICE2_X = 480.0e3  # m
ICE2_MELT = 100.0  # m ice eq. yr-1, on floating ice with x > ICE2_X

# --- resolutions ----------------------------------------------------------------------------
RESOLUTIONS = {"4km": 4000.0, "2km": 2000.0, "1km": 1000.0, "500m": 500.0, "250m": 250.0}


def grid(res):
    """Node coordinates (x, y) in m for a resolution label or a spacing in m."""
    dx = RESOLUTIONS[res] if isinstance(res, str) else float(res)
    nx, ny = int(round(LX / dx)) + 1, int(round(LY / dx)) + 1
    return np.linspace(0.0, LX, nx), np.linspace(-LY / 2, LY / 2, ny)


def res_label(dx):
    for k, v in RESOLUTIONS.items():
        if abs(v - dx) < 1e-6:
            return k
    return f"{dx:g}m"


def bed(x, y):
    """MISMIP+ bed elevation (m) on the meshgrid of x, y (1-D, m); shape (ny, nx)."""
    X, Y = np.meshgrid(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    xs = X / X_BAR
    bx = B0 + B2 * xs**2 + B4 * xs**4 + B6 * xs**6
    by = D_C / (1.0 + np.exp(-2.0 * (Y - W_C) / F_C)) + D_C / (1.0 + np.exp(2.0 * (Y + W_C) / F_C))
    return np.maximum(bx + by, Z_B_DEEP)


def surfaces(topg, thk, wl=SEA_LEVEL, ratio=RATIO):
    """Lower and upper surfaces from hydrostatic equilibrium (igm thk/surfaces.py)."""
    lsurf = np.maximum(topg, wl - ratio * thk)
    return lsurf, lsurf + thk


def ice2_melt(x, y):
    """Ice2 melt field (m/yr), before the restriction to floating ice done by igm bmb."""
    X, _ = np.meshgrid(x, y)
    return np.where(X > ICE2_X, ICE2_MELT, 0.0)
