"""
Write the experiment files experiment/*.yaml (self-contained, one per run; never edit them).

The physics comes from prep/common.py, the velocity solver from prep/methods/fast.yaml: a
cold-started Newton-CG in float64, so every solve is independent of the previous step
(prep/methods/warm.yaml, a warm start, gives the same fronts and profiles to the printed
digits at CFL 0.5 for 35 % less per step; README "Solver"). CFL 0.5: at 0.9 the errors
behind the front grow from < 1 % to 2-4 %. The online-trained emulator is not cheaper on
this grid (results/nn_study.md).

  adv_<front>_<res>      advance from a two-cell shelf and steady state, 0-500 yr, with the
                         250-m front thickness rule (thk.front.min_thickness), front
                         sub_grid | level_set, res 10km | 5km | 2.5km | 1.25km
  adv_none_5km           the same without a front scheme (control: the front disperses;
                         floating ice thinner than 10 m is removed), 0-300 yr
  adv_<front>_5km_dt1    advance with 1-yr steps (Albrecht et al., 2011, Fig. 6), 0-300 yr
  retreat_<front>_<res>  from the steady adv_<front>_<res>, without the thickness rule:
                         ice_speed law c = |u| + 300 m/yr, the front retreats at exactly
                         300 m/yr, 0-300 yr, 5km
  advance_<front>_<res>  ice_speed c = |u| - 200 m/yr: the front advances at 200 m/yr, 0-300 yr
  const_<front>_<res>    constant calving c = 800 m/yr (> u at the front: an accelerating
                         retreat, reference dx/dt = u(x) - c), 0-300 yr
  vm_<front>_<res>       von Mises calving, sigma_max 0.09 MPa (the same kind of retreat), 0-300 yr

On this unconfined shelf u grows downstream, so the fronts of a constant rate or of von
Mises calving (c/u ~ H) are unstable fixed points; the thickness rule gives the steady front,
and the rate laws are verified along their transients. The rate runs start from
data/restart/adv_<front>_<res>.nc (make_restart.py).

Usage:  python prep/make_experiments.py
"""

import copy
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

FRONTS = ("sub_grid", "level_set")
RATE_RES = ("5km",)
CONSTANT_RATE = 800.0  # m/yr
SIGMA_MAX = 0.09  # MPa
RETREAT = 300.0  # m/yr, prescribed front retreat (ice_speed law)
ADVANCE = 200.0  # m/yr, prescribed front advance (ice_speed law)

VARS = ["topg", "thk", "usurf", "lsurf", "ubar", "vbar", "velbar_mag", "Href",
        "ice_area_fraction", "calved_thk"]


def merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


class Flow(list):
    """A list dumped in YAML flow style."""


yaml.SafeDumper.add_representer(
    Flow, lambda d, v: d.represent_sequence("tag:yaml.org,2002:seq", v, flow_style=True)
)


def flowify(x):
    if isinstance(x, dict):
        return {k: flowify(v) for k, v in x.items()}
    if isinstance(x, list):
        items = [flowify(v) for v in x]
        return items if any(isinstance(v, (dict, list)) for v in x) else Flow(items)
    return x


def iceflow():
    return {
        "method": "unified",
        "physics": {
            "energy_components": ["viscosity", "gravity", "sliding", "floating"],
            "ice_density": C.RHO_ICE,
            "water_density": C.RHO_WATER,
            "gravity_cst": C.GRAVITY,
            "cf_eswn": [],  # the front is inside the domain
            "viscosity": {"arrhenius": round(C.ARRHENIUS, 6), "exponent": C.GLEN_N},
            "sliding": {"law": "weertman", "tau_ref": 0.01, "u_ref": 1.0, "exponent": 3.0,
                        "regularization": 1.0, "use_mask_gr": True},  # all floating: no friction
        },
        # MOLHO (Nz 2): no vertical shear on a floating shelf, so it is the SSA solution;
        # float64 as in igm-mismip-plus (the banded Newton-CG breaks down in float32)
        "numerics": {"Nz": 2, "basis_vertical": "molho", "basis_horizontal": "q1", "precision": "double"},
        "unified": {
            "inputs": ["thk", "usurf", "arrhenius", "tau_ref", "dX", "water_level"],
            # inflow u = u0, v = 0 at x = 0; periodic_ns makes the y-uniform shelf unconfined
            "bcs": ["dirichlet", "periodic_ns"],
            "bc": {"dirichlet": {"left": [C.U0, 0.0], "right": None, "top": None, "bottom": None}},
        },
    }


def thk_block(front, min_thickness):
    block = {
        "scheme": "explicit",
        "slope_type": "superbee",
        "ratio_density": round(C.RATIO, 6),
        # inflow thickness H0 at x = 0 (the initial edge column), open front side, and
        # zero-flux sides across flow (the shelf is uniform in y)
        "boundary": {"left": "dirichlet", "right": "zero", "top": "symmetric", "bottom": "symmetric"},
    }
    if front != "none":
        block["front"] = {"method": front, "min_thickness": min_thickness}
    return block


def experiment(name, res, t0, t1, save, front, min_thickness=0.0, restart=None,
               law=None, step_max=20.0, cfl=0.5, control=False):
    processes = (["calving_rate"] if law else []) + ["time", "thk"]
    processes += (["thin_ice_calving"] if control else []) + ["iceflow"]
    vars_to_save = VARS + (["psi"] if front == "level_set" else []) + (
        ["calving_rate", "frontal_melt_rate"] if law else [])
    if front == "none":
        vars_to_save = [v for v in vars_to_save if v not in ("Href", "ice_area_fraction", "calved_thk")]
    cfg = {
        "hydra": {"run": {"dir": f"outputs/{name}"}},
        "defaults": [
            {"/user/conf/outputs@outputs": "albrecht_ts"},
            {"override /inputs": ["local"]},
            {"override /processes": processes},
            {"override /outputs": ["write_ncdf", "albrecht_ts"]},
        ],
        "core": {"print_info": False, "print_comp": True},
        "inputs": {"local": {"filename": restart or f"albrecht_{res}.nc"}},
        "processes": {
            "time": {"start": t0, "end": t1, "save": save, "cfl": cfl, "step_max": step_max},
            "thk": thk_block(front, min_thickness),
            "iceflow": iceflow(),
        },
        "outputs": {
            "write_ncdf": {"output_file": "output.nc", "complevel": 1, "vars_to_save": vars_to_save},
            "albrecht_ts": {"output_file": "albrecht_ts.csv", "freq": 0.0},
        },
    }
    if law:
        cfg["processes"]["calving_rate"] = law
    if control:
        cfg["processes"]["thin_ice_calving"] = {"min_thk": 10.0}
    return cfg


METHODS = ("fast",)  # cold-started Newton-CG (user, 2026-09-27)


def write(name, cfg, what):
    for method in METHODS:
        with open(os.path.join(C.ROOT, "prep", "methods", f"{method}.yaml")) as f:
            cfg = merge(cfg, yaml.safe_load(f))
    head = (
        "# @package _global_\n"
        f"# Albrecht et al. (2011) flowline, {name}: {what}\n"
        "# GENERATED by prep/make_experiments.py: edit prep/ and rerun it, not this file.\n"
    )
    path = os.path.join(C.ROOT, "experiment", f"{name}.yaml")
    with open(path, "w") as f:
        f.write(head + yaml.safe_dump(flowify(cfg), sort_keys=False, width=100))
    return path


def all_experiments():
    runs = {}
    for front in FRONTS:
        for res in C.RESOLUTIONS:
            runs[f"adv_{front}_{res}"] = (
                experiment(f"adv_{front}_{res}", res, 0.0, 500.0, 25.0, front, C.H_CALVING),
                f"advance and steady state, {front} front, 250-m thickness rule, {res}",
            )
        runs[f"adv_{front}_5km_dt1"] = (
            experiment(f"adv_{front}_5km_dt1", "5km", 0.0, 300.0, 25.0, front, C.H_CALVING, step_max=1.0),
            f"advance with 1-yr steps, {front} front, 5km",
        )
        for res in RATE_RES:
            restart = f"restart/adv_{front}_{res}.nc"
            laws = {
                "retreat": ({"law": "ice_speed", "ice_speed": {"factor": 1.0, "front_velocity": -RETREAT}},
                            f"front retreating at {RETREAT:g} m/yr (ice_speed)"),
                "advance": ({"law": "ice_speed", "ice_speed": {"factor": 1.0, "front_velocity": ADVANCE}},
                            f"front advancing at {ADVANCE:g} m/yr (ice_speed)"),
                "const": ({"law": "constant", "constant": {"value": CONSTANT_RATE}},
                          f"constant calving {CONSTANT_RATE:g} m/yr"),
                "vm": ({"law": "von_mises", "von_mises": {"sigma_max_floating": SIGMA_MAX}},
                       f"von Mises calving, sigma_max {SIGMA_MAX} MPa"),
            }
            for key, (law, what) in laws.items():
                runs[f"{key}_{front}_{res}"] = (
                    experiment(f"{key}_{front}_{res}", res, 0.0, 300.0, 25.0, front,
                               restart=restart, law=law),
                    f"{what} from the steady state, {front} front, {res}",
                )
    runs["adv_none_5km"] = (
        experiment("adv_none_5km", "5km", 0.0, 300.0, 25.0, "none", control=True),
        "advance without a front scheme (control), 5km",
    )
    return runs


def main():
    os.makedirs(os.path.join(C.ROOT, "experiment"), exist_ok=True)
    for name, (cfg, what) in all_experiments().items():
        print(write(name, cfg, what))


if __name__ == "__main__":
    main()
