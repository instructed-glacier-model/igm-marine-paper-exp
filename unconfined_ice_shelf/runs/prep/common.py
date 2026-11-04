"""
Albrecht et al. (2011) unconfined ice-shelf flowline: constants, grid and analytic solutions,
shared by prep/ and the analysis scripts.

Albrecht, T., Martin, M., Haseloff, M., Winkelmann, R., and Levermann, A.: Parameterization
for subgrid-scale motion of ice-shelf calving fronts, The Cryosphere, 5, 35-44, 2011
(Sect. 4). A shelf fed at x = 0 with H0 = 600 m and u0 = 300 m/yr spreads under its own
weight, without friction, accumulation or melt, with constant hardness B0. Its steady profile
is Weertman's (1957) solution (Eqs. 10-12),

    du/dx = C H^3,  C = (rho g (1 - rho/rho_w) / (4 B0))^3,
    H(x) = (4 C x / Q0 + H0^-4)^(-1/4),  u(x) = Q0 / H(x),  Q0 = u0 H0,

and a front advancing from x = 0 at t = 0 reaches (volume balance, Eq. 13)

    x_c(t) = Q0 / (4 C) [(3 C t + H0^-3)^(4/3) - H0^-4].

H(x) does not depend on where the front is, so a front moving with the lateral ablation
rate c obeys dx_c/dt = u(x_c) - c(x_c) behind the analytic profile: the steady front of a
calving law sits where c = u, e.g. H* = Q0 / c for a constant rate c, and, for von Mises
calving (e1 = C H^3, e2 = 0), where sqrt(3) B (C H^3 / sqrt(2))^(1/3) = sigma_max.

Coordinates: x along flow (m) from the first node, where igm imposes the inflow velocity
(the finite-element boundary). The finite-volume cell of that node extends dx/2 upstream of
it, to the face through which the Dirichlet thickness H0 enters with the flux Q0; the
analytic solutions put H0 and u0 at x = 0, and the transient starts from the model's initial
volume in x >= 0 (analytic_start). Units: m, yr, MPa (igm units).
"""

import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- physical constants (Albrecht et al., 2011, Table 1 and Sect. 4) ---------------------
RHO_ICE = 910.0  # kg m-3
RHO_WATER = 1028.0  # kg m-3
GRAVITY = 9.81  # m s-2
RATIO = RHO_ICE / RHO_WATER  # igm thk.ratio_density
SECONDS_PER_YEAR = 31556926.0  # igm's year
B0_PA_S = 1.9e8  # Pa s^(1/3), ice hardness (MacAyeal et al., 1996)
GLEN_N = 3.0
H0 = 600.0  # m, thickness at the inflow boundary
U0 = 300.0  # m/yr, velocity at the inflow boundary
Q0 = H0 * U0  # m2/yr
H_CALVING = 250.0  # m, thickness rule of Albrecht et al. (2011)

# B0 in MPa yr^(1/3) and the igm rate factor A = B^-3 in MPa-3 yr-1
B0 = B0_PA_S * 1e-6 / SECONDS_PER_YEAR ** (1.0 / GLEN_N)
ARRHENIUS = B0 ** (-GLEN_N)
# Weertman's spreading constant (1/yr/m3): C = (rho g (1 - rho/rho_w) / (4 B))^3, B in Pa yr^(1/3)
C = (RHO_ICE * GRAVITY * (1.0 - RATIO) / (4.0 * B0 * 1e6)) ** GLEN_N

# --- grid ----------------------------------------------------------------------------------
LX = 500.0e3  # m, domain length from the first node
NY = 2  # nodes across flow; iceflow periodic_ns (last row = first row): one independent row, a y-uniform flowline
RESOLUTIONS = {"10km": 10.0e3, "5km": 5.0e3, "2.5km": 2.5e3, "1.25km": 1.25e3}
BED = -5000.0  # m, deep enough for the whole shelf to float


def grid(dx):
    """Node coordinates (m): x from 0 (the first node) to LX, NY rows spaced dx."""
    nx = int(round(LX / dx)) + 1
    return np.arange(nx) * dx, np.arange(NY) * dx


# --- analytic solutions (x from the inflow face) --------------------------------------------
def thickness(x):
    """Steady profile H(x) (m), Albrecht et al. (2011), Eq. 11."""
    return (4.0 * C * np.asarray(x, float) / Q0 + H0**-4) ** -0.25


def velocity(x):
    """Steady velocity u(x) = Q0 / H(x) (m/yr), Eq. 12."""
    return Q0 / thickness(x)


def position(H):
    """Where the steady profile has the thickness H (m)."""
    return Q0 / (4.0 * C) * (np.asarray(H, float) ** -4 - H0**-4)


def front(t):
    """Front advancing from x = 0 at t = 0 (m), Eq. 13."""
    return Q0 / (4.0 * C) * ((3.0 * C * np.asarray(t, float) + H0**-3) ** (4.0 / 3.0) - H0**-4)


def front_time(x):
    """Inverse of front(): time at which the advancing front reaches x (yr)."""
    V = Q0 / (4.0 * C) * 4.0 / 3.0 * ((4.0 * C * np.asarray(x, float) / Q0 + H0**-4) ** 0.75 - H0**-3)
    return V / Q0


def volume(x):
    """Ice volume per unit width between 0 and x on the steady profile (m2)."""
    return Q0 / (3.0 * C) * ((4.0 * C * np.asarray(x, float) / Q0 + H0**-4) ** 0.75 - H0**-3)


def inverse_volume(V):
    """Where the volume per unit width from x = 0 on the steady profile is V (m)."""
    return Q0 / (4.0 * C) * ((3.0 * C * np.asarray(V, float) / Q0 + H0**-3) ** (4.0 / 3.0) - H0**-4)


def analytic_start(thk_row, dx):
    """Time (yr) at which the analytic front holds the initial volume in x >= 0.

    ``thk_row`` is the initial thickness on the nodes; half of the first cell lies
    upstream of x = 0.
    """
    thk_row = np.asarray(thk_row, float)
    volume0 = np.sum(thk_row) * dx - 0.5 * dx * thk_row[0]
    return volume0 / Q0


def von_mises_rate(x, sigma_max):
    """von Mises calving rate on the steady profile (m/yr), sigma_max in MPa."""
    e1 = C * thickness(x) ** 3
    tensile = e1 / np.sqrt(2.0)
    sigma = np.sqrt(3.0) * B0 * tensile ** (1.0 / GLEN_N)
    return velocity(x) * sigma / sigma_max


def von_mises_front(sigma_max):
    """Steady von Mises front: sigma(H*) = sigma_max (m)."""
    H = (np.sqrt(2.0) * (sigma_max / (np.sqrt(3.0) * B0)) ** GLEN_N / C) ** (1.0 / 3.0)
    return position(H)


def front_ode(x0, t, rate):
    """Front position under dx/dt = u(x) - rate(x), from x0 (m), at the times t (yr).

    ``rate(x)`` is the ablation rate (m/yr) on the steady profile; RK4 with 0.1-yr steps.
    """
    t = np.asarray(t, float)
    out = np.empty_like(t)
    x, now, h = float(x0), 0.0, 0.1
    for k, target in enumerate(t):
        while now < target - 1e-12:
            s = min(h, target - now)
            f = lambda xx: velocity(max(xx, 0.0)) - rate(max(xx, 0.0))  # noqa: E731
            k1 = f(x)
            k2 = f(x + 0.5 * s * k1)
            k3 = f(x + 0.5 * s * k2)
            k4 = f(x + s * k3)
            x += s * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            now += s
        out[k] = x
    return out


if __name__ == "__main__":
    print(f"B0 = {B0:.5f} MPa yr^1/3, A = {ARRHENIUS:.4f} MPa-3 yr-1, C = {C:.4e} 1/yr/m3 ({C / SECONDS_PER_YEAR:.4e} 1/s/m3)")
    print(f"steady front at H = {H_CALVING} m: x = {position(H_CALVING) / 1e3:.2f} km, u = {velocity(position(H_CALVING)):.1f} m/yr")
    print(f"front reaches it at t = {front_time(position(H_CALVING)):.1f} yr; x_c(300 yr) = {front(300.0) / 1e3:.2f} km")
    print(f"constant c = 600 m/yr: x* = {position(Q0 / 600.0) / 1e3:.2f} km")
    for s in (0.1, 0.12):
        print(f"von Mises sigma_max = {s} MPa: x* = {von_mises_front(s) / 1e3:.2f} km")
