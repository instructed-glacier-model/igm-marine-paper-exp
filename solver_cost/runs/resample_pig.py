#!/usr/bin/env python3
"""Resample the relaxed 1-km Pine Island state (glaciers/runs/data/glacier_pig/start_1km.nc, t = 0) to 4, 2 and
0.5 km for the solver-scaling study (Sect. 5.4 of the paper). Writes data/<res>/initial.nc and data/audit.json.

- 1 km: the native file's ice-flow fields, unchanged (469 x 564).
- 2 and 4 km: nested subsampling of the 1-km nodes, after padding one ice-free northern row
  (y_max 113 km) so that 4 km divides the extent: exact point values. Where subsampling leaves
  domain ice touching a rock-wall node with no quasi-rigid band in between, that domain node
  becomes band (A x 1e-3, tau_ref 1 MPa); the number of such nodes is recorded.
- 0.5 km: class-aware interpolation. Classes (1 km): ice (domain and band), wall, open water,
  dry land; each 0.5-km node takes the class of its nearest 1-km node (ties: increasing
  coordinate) and, within ice, band or domain the same way. Thickness and bed are bilinear over
  same-class sources only (walls from walls: no trench between ice and wall), tau_ref log-bilinear
  over domain sources, A, tau_ref of the band and the water level categorical.
- Surfaces: lsurf = max(topg, wl - rho_i/rho_w thk), usurf = lsurf + thk (IGM convention).
- Masks: grounded node, active node/element and front faces with the rules of IGM 058558bc
  (igm.processes.thk.masks, floating.py; numpy re-implementation verified against IGM in
  ua_benchmark_pig/scripts/derive_fields.py).

Usage: python resample_pig.py   (needs numpy, scipy, netCDF4; no GPU)
"""

import hashlib
import json
import os

import netCDF4
import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(ROOT, "..", "..", "glaciers", "runs", "data", "glacier_pig", "start_1km.nc"))
OUT = os.path.join(ROOT, "data")
RHO_I, RHO_W = 917.0, 1027.0
A_ICE, A_BAND, TAU_BAND = 11.0, 0.011, 1.0
EIGHT = np.ones((3, 3), bool)


def load():
    with netCDF4.Dataset(SRC) as d:
        f = {v: np.asarray(d[v][:], np.float64) for v in d.variables if d[v].ndim == 2}
        f["x"] = np.asarray(d["x"][:], np.float64)
        f["y"] = np.asarray(d["y"][:], np.float64)
        attrs = {k: d.getncattr(k) for k in d.ncattrs()}
    return f, attrs


def classes(f):
    ice = f["thk"] > 0
    band = ice & (f["bandmask"] > 0.5)
    wall = (~ice) & (f["wallmask"] > 0.5)
    water = (~ice) & (~wall) & (f["water_level"] == 0.0)
    land = (~ice) & (~wall) & (~water)
    return ice, band, wall, water, land


def pad_north(f):
    """One ice-free row at y_max + 1 km: fields copied from the last row (asserted ice-free)."""
    if (f["thk"][-1] > 0).any():
        raise SystemExit("northern row carries ice")
    g = {}
    for k, a in f.items():
        if k == "x":
            g[k] = a
        elif k == "y":
            g[k] = np.append(a, a[-1] + (a[1] - a[0]))
        else:
            g[k] = np.vstack([a, a[-1:]])
    return g


def surfaces(thk, topg, wl):
    lsurf = np.maximum(topg, wl - RHO_I / RHO_W * thk)
    return lsurf + thk


def subsample(f, step):
    g = {k: (a[::step, ::step] if a.ndim == 2 else a[::step]) for k, a in f.items()}
    ice = g["thk"] > 0
    band = ice & (g["bandmask"] > 0.5)
    dom = ice & ~band
    wall = (~ice) & (g["wallmask"] > 0.5)
    touch = dom & ndimage.binary_dilation(wall, EIGHT)
    g["bandmask"] = np.where(touch, 1.0, g["bandmask"])
    g["domainmask"] = np.where(touch, 0.0, g["domainmask"])
    g["arrhenius"] = np.where(touch, A_BAND, g["arrhenius"])
    g["tau_ref"] = np.where(touch, TAU_BAND, g["tau_ref"])
    return g, int(touch.sum())


def refine_half(f):
    """0.5-km grid from the padded 1-km grid by class-aware interpolation."""
    x1, y1 = f["x"], f["y"]
    x = np.arange(x1[0], x1[-1] + 1.0, 500.0)
    y = np.arange(y1[0], y1[-1] + 1.0, 500.0)
    fx = (x - x1[0]) / 1000.0
    fy = (y - y1[0]) / 1000.0
    i0 = np.minimum(np.floor(fx).astype(int), len(x1) - 2)
    j0 = np.minimum(np.floor(fy).astype(int), len(y1) - 2)
    wx, wy = fx - i0, fy - j0
    # nearest source (ties -> increasing coordinate)
    inear = np.minimum(np.floor(fx + 0.5).astype(int), len(x1) - 1)
    jnear = np.minimum(np.floor(fy + 0.5).astype(int), len(y1) - 1)
    J, I = np.meshgrid(jnear, inear, indexing="ij")

    ice, band, wall, water, land = classes(f)
    cls1 = np.select([ice, wall, water, land], [0, 1, 2, 3])
    cls = cls1[J, I]
    is_band = band[J, I]

    corners = []
    for dj, di in ((0, 0), (0, 1), (1, 0), (1, 1)):
        jj = (j0 + dj)[:, None] * np.ones((1, len(x)), int)
        ii = np.ones((len(y), 1), int) * (i0 + di)[None, :]
        w = ((wy if dj else 1 - wy)[:, None]) * ((wx if di else 1 - wx)[None, :])
        corners.append((jj, ii, w))

    def interp(field, same, log=False):
        num = np.zeros((len(y), len(x)))
        den = np.zeros((len(y), len(x)))
        for jj, ii, w in corners:
            ok = same(jj, ii)
            val = np.log(field[jj, ii]) if log else field[jj, ii]
            num += np.where(ok, w * np.where(ok, val, 0.0), 0.0)
            den += np.where(ok, w, 0.0)
        out = np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)
        near = np.log(field[J, I]) if log else field[J, I]
        out = np.where(np.isnan(out), near, out)  # nearest node is always of the right class
        return np.exp(out) if log else out

    same_cls = lambda jj, ii: cls1[jj, ii] == cls  # noqa: E731
    dom1 = ice & ~band
    same_dom = lambda jj, ii: dom1[jj, ii]  # noqa: E731

    thk = np.where(cls == 0, np.maximum(interp(f["thk"], same_cls), 0.0), 0.0)
    topg = interp(f["topg"], same_cls)
    wl = f["water_level"][J, I]
    tau = np.where(is_band, TAU_BAND, interp(np.where(dom1, f["tau_ref"], 1.0), same_dom, log=True))
    tau = np.where(cls == 0, tau, f["tau_ref"][J, I])
    arr = np.where(is_band & (cls == 0), A_BAND, A_ICE)
    g = dict(
        x=x,
        y=y,
        thk=thk,
        topg=topg,
        water_level=wl,
        tau_ref=tau,
        arrhenius=arr,
        bandmask=(is_band & (cls == 0)).astype(float),
        domainmask=((cls == 0) & ~is_band).astype(float),
        wallmask=(cls == 1).astype(float),
    )
    return g


# --- IGM 058558bc mask rules (numpy; verified in ua_benchmark_pig/scripts/derive_fields.py) ---
def grounded(thk, lower, wl):
    r = RHO_W / RHO_I
    t, l, w = (a.astype(np.float32) for a in (thk, lower, wl))
    phi = t + np.float32(r) * (l - w)
    tol = np.float32(64.0 * 2.0**-23) * (t + np.abs(l))
    return (phi - tol) > 0.0


def all4(m):
    return m[:-1, :-1] & m[:-1, 1:] & m[1:, :-1] & m[1:, 1:]


def any4(m):
    return m[:-1, :-1] | m[:-1, 1:] | m[1:, :-1] | m[1:, 1:]


def masks(thk, usurf, wl):
    ice = thk > 0
    lower = usurf - thk
    gr = grounded(thk, lower, wl)
    cells = all4(ice) | (any4(ice) & all4(gr))
    sup = np.zeros(ice.shape, bool)
    sup[:-1, :-1] |= cells
    sup[:-1, 1:] |= cells
    sup[1:, :-1] |= cells
    sup[1:, 1:] |= cells
    land = (~ice) & (lower > wl)
    cell_ice = all4(ice)
    wet = ~cell_ice & ~any4(land)
    wp = np.pad(wet, 1, constant_values=False)
    faces = dict(
        east=cell_ice & wp[1:-1, 2:],
        west=cell_ice & wp[1:-1, :-2],
        north=cell_ice & wp[2:, 1:-1],
        south=cell_ice & wp[:-2, 1:-1],
    )
    return dict(grounded=gr & ice, cells=cells, nodes=ice & sup, support=sup, faces=faces)


def audit(g, m, dx):
    ice = g["thk"] > 0
    s = g["usurf"]
    gx = 0.5 * ((s[:-1, 1:] - s[:-1, :-1]) + (s[1:, 1:] - s[1:, :-1])) / dx
    gy = 0.5 * ((s[1:, :-1] - s[:-1, :-1]) + (s[1:, 1:] - s[:-1, 1:])) / dx
    slope = np.hypot(gx, gy)
    trench = np.zeros(ice.shape, bool)
    for dj, di in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        sh = lambda a: np.roll(np.roll(a, dj, 0), di, 1)  # noqa: E731
        trench |= ~ice & sh(ice) & (sh(s) - s > 500.0)
    band = g["bandmask"] > 0.5
    dom = ice & ~band
    wall = g["wallmask"] > 0.5
    lab, ncomp = ndimage.label(m["cells"])
    cell = dx * dx / 1e6
    return dict(
        nx=int(ice.shape[1]),
        ny=int(ice.shape[0]),
        spacing_m=float(dx),
        ice_nodes=int(ice.sum()),
        active_ice_nodes=int(m["nodes"].sum()),
        active_elements=int(m["cells"].sum()),
        ice_area_km2=float(ice.sum() * cell),
        grounded_area_km2=float(m["grounded"].sum() * cell),
        floating_area_km2=float((ice & ~m["grounded"]).sum() * cell),
        band_area_km2=float(band.sum() * cell),
        volume_km3=float(g["thk"].sum() * dx * dx / 1e9),
        active_components=int(ncomp),
        max_active_surface_slope=float(slope[m["cells"]].max()),
        trench_nodes=int(trench.sum()),
        domain_ice_touching_wall=int((dom & ndimage.binary_dilation(wall, EIGHT)).sum()),
        front_faces=int(sum(v.sum() for v in m["faces"].values())),
    )


def write(path, g, m, attrs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with netCDF4.Dataset(path, "w") as d:
        for k, v in attrs.items():
            d.setncattr(k, v)
        d.createDimension("x", len(g["x"]))
        d.createDimension("y", len(g["y"]))
        d.createDimension("xc", len(g["x"]) - 1)
        d.createDimension("yc", len(g["y"]) - 1)
        for k in ("x", "y"):
            v = d.createVariable(k, "f8", (k,))
            v[:] = g[k]
            v.units = "m"
        for k, units in (
            ("thk", "m"),
            ("topg", "m"),
            ("usurf", "m"),
            ("water_level", "m"),
            ("arrhenius", "MPa-3 yr-1"),
            ("tau_ref", "MPa"),
        ):
            v = d.createVariable(k, "f4", ("y", "x"), zlib=True)
            v[:] = g[k].astype(np.float32)
            v.units = units
        for k in ("bandmask", "domainmask", "wallmask"):
            v = d.createVariable(k, "i1", ("y", "x"), zlib=True)
            v[:] = (g[k] > 0.5).astype(np.int8)
        for k, a, dims in (
            ("ice_mask", g["thk"] > 0, ("y", "x")),
            ("grounded_node", m["grounded"], ("y", "x")),
            ("active_node", m["nodes"], ("y", "x")),
            ("active_element", m["cells"], ("yc", "xc")),
        ):
            v = d.createVariable(k, "i1", dims, zlib=True)
            v[:] = a.astype(np.int8)
        for side, a in m["faces"].items():
            v = d.createVariable(f"front_face_{side}", "i1", ("yc", "xc"), zlib=True)
            v[:] = a.astype(np.int8)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def main():
    f, attrs = load()
    keep = ("thk", "topg", "usurf", "water_level", "arrhenius", "tau_ref", "bandmask", "domainmask", "wallmask", "x", "y")
    base = dict(
        title="PIG t = 0 state for the Newton-CG scalability study (tools/resample_pig.py)",
        source="data/glacier_pig/start_1km.nc",
        source_sha256=sha256(SRC),
        pyproj_srs="EPSG:3031",
    )
    report = {}
    grids = {"1km": ({k: f[k] for k in keep}, "native, unchanged", 0)}
    fp = pad_north({k: f[k] for k in keep})
    for res, step in (("2km", 2), ("4km", 4)):
        g, nfix = subsample(fp, step)
        grids[res] = (g, f"nested subsampling every {step} nodes of the north-padded 1-km grid", nfix)
    grids["500m"] = (refine_half(fp), "class-aware interpolation of the north-padded 1-km grid", 0)
    for res, (g, how, nfix) in grids.items():
        g = dict(g)
        if res != "1km":
            g["usurf"] = surfaces(g["thk"], g["topg"], g["water_level"])
        dx = float(g["x"][1] - g["x"][0])
        m = masks(g["thk"], g["usurf"], g["water_level"])
        a = audit(g, m, dx)
        a["band_nodes_added"] = nfix
        a["method"] = how
        path = os.path.join(OUT, res, "initial.nc")
        write(path, g, m, dict(base, resampling=how, spacing_m=dx))
        a["sha256"] = sha256(path)
        report[res] = a
        print(res, json.dumps({k: v for k, v in a.items() if k not in ("method", "sha256")}))
    with open(os.path.join(OUT, "audit.json"), "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
