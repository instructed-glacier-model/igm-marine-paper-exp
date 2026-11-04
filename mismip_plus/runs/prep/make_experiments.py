"""
Write the MISMIP+ experiment files experiment/*.yaml (self-contained, one per run).

The physics comes from prep/common.py (protocol constants), the solver block from
prep/methods/ (fast.yaml: the igm-marine "exact, fast" Newton-CG; tight.yaml: the solver
check). The files are self-contained on purpose: igm reads the process order from the
`defaults` list of the launched experiment file itself.

Per resolution RES in 2km, 1km, 500m, 250m:

  ice0_nn_RES       Ice0 spin-up, stage 1: cheap hybrid emulator (prep/methods/nn.yaml) from
                    a uniform 100-m slab (data/mismip_plus_RES_slab100.nc), 0.3 m/yr
                    accumulation, no melt, CFL 0.8, 15 kyr (converged by ~10 kyr at 2/1 km)
  ice0_cg_RES       Ice0 spin-up, stage 2: MOLHO + cg_newton with converged solves until
                    steady (250-yr window means: <= 0.1 km and 2.5e-4; >= 500 yr, cap 3 kyr)
                    (prep/methods/cgfinish.yaml on top of fast.yaml) from the end of
                    ice0_nn_RES (data/restart/cg_RES_init.nc); its last frame is the Ice0
                    state (data/restart/ice0_RES.nc); see run.sh.
  timing_RES        20-yr cost probe on data/restart/timing_RES_init.nc (a row every step)
  ice0_RES          Ice0 control, 0-100 yr, from data/restart/ice0_RES.nc
  ice1r_RES         Ice1r, 0-100 yr, Ice1 melt (Cornford et al., 2020, Eq. 7), from Ice0
  ice1rr_RES        Ice1rr, 100-200 yr, Ice1 melt, from data/restart/ice1r_RES_t100.nc
  ice1ra_RES        Ice1ra, 100-200 yr, no melt, from the same restart
  ice2r_RES, ice2rr_RES, ice2ra_RES   the same with the Ice2 melt (Eq. 8)
  (1km only) ice{1,2}r_1km_{pmp,fmp}  grounding-line melt treatment sensitivity
             ice{1,2}r_1km_tight      same run with the tight solver (prep/methods/tight.yaml)

Usage:  python prep/make_experiments.py      (rerun after editing prep/methods/ or this file)
"""

import copy
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

# the spin-up stages end when steady (user/code/processes/steady_stop.py); the lengths are caps
NN_YEARS = {"4km": 30000.0, "2km": 30000.0, "1km": 30000.0, "500m": 30000.0, "250m": 30000.0}
NN_SAVE = {"4km": 500.0, "2km": 500.0, "1km": 500.0, "500m": 250.0, "250m": 250.0}
CG_YEARS = 3000.0
# steady_stop settings: window-mean changes of x_GL(y=0) and VAF between the last two windows
STEADY_NN = {"check_freq": 10.0, "window": 1000.0, "min_years": 5000.0, "tol_gl": 500.0, "tol_vaf": 1.0e-3}
STEADY_CG = {"check_freq": 10.0, "window": 250.0, "min_years": 500.0, "tol_gl": 100.0, "tol_vaf": 2.5e-4}
# steps above ~1.5 yr make the thickness-velocity coupling unstable whatever the CFL number
# (4 km with 2.4-yr steps and one Newton iteration per step blew up after 20 yr)
STEP_MAX = {"4km": 1.5, "2km": 1.5, "1km": 1.0, "500m": 1.0, "250m": 1.0}  # CFL 0.8 governs below

VARS = [
    "topg", "thk", "usurf", "lsurf", "water_level", "smb",
    "ubar", "vbar", "velbar_mag", "uvelsurf", "vvelsurf", "velsurf_mag",
    "uvelbase", "vvelbase", "velbase_mag",
]
VARS_BMB = ["bmb", "shelf_melt_rate", "grounded_fraction"]

MELT = {
    "ice1": {"form": "mismip_plus"},  # omega 0.2 yr-1, H0 75 m, z0 -100 m (igm defaults = protocol)
    "ice2": {"form": "field", "field": "prescribed_melt_rate"},  # 100 m/yr for x > 480 km
}


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
            "cf_eswn": ["E"],  # calving front on the east edge, x = 640 km
            "viscosity": {"arrhenius": round(C.ARRHENIUS, 4), "exponent": C.GLEN_N},
            "sliding": {
                "law": "weertman",
                "tau_ref": C.TAU_REF,
                "u_ref": C.U_REF,
                "exponent": C.WEERTMAN_M,
                "regularization": 1.0,
                "use_mask_gr": True,  # no friction under floating ice (live flotation mask)
            },
        },
        # float64: in float32 the banded Newton-CG breaks down on MISMIP+ (the CG does 0-2
        # iterations with relres 1, the Newton step is 0 and the change-based halt reports
        # convergence), so the velocity freezes or, after a cold restart, stays ~0
        # (outputs/superseded/, README "Precision"); float64 costs ~1.5x per Newton iteration
        "numerics": {"Nz": 2, "basis_vertical": "molho", "basis_horizontal": "q1", "precision": "double"},
        "unified": {
            "inputs": ["thk", "usurf", "arrhenius", "tau_ref", "dX", "water_level"],
            # no-slip u = v = 0 at the divide x = 0; the other edges must be null (default 0);
            # periodic_ns (last row = first row) + the mirror-symmetric bed = free-slip walls
            "bcs": ["dirichlet", "periodic_ns"],
            "bc": {"dirichlet": {"left": 0.0, "right": None, "top": None, "bottom": None}},
        },
    }


# protocol runs (not the spin-up): floating ice thinner than this is removed after each
# thickness update (user/code/processes/thin_ice_calving.py); the Ice2 melt otherwise leaves
# a mm-thin floating film on which cg_newton diverges. Never acts in Ice0 / Ice1 (floating ice
# >= 19.6 m thick in every frame of the 1 km runs).
THIN_ICE_MIN_THK = 10.0  # m


def base(name, input_rel, t0, t1, save, ts_freq, melt=None, treatment="nmp", steady=None):
    calving = steady is None
    processes = ["time", "smb"] + (["bmb"] if melt else []) + ["thk"] + (["thin_ice_calving"] if calving else []) + ["iceflow"]
    processes += ["steady_stop"] if steady else []  # after iceflow: tests the saved state
    cfg = {
        "hydra": {"run": {"dir": f"outputs/{name}"}},
        "defaults": [
            {"/user/conf/outputs@outputs": "mismip_ts"},
        ] + ([{"/user/conf/processes@processes": "steady_stop"}] if steady else []) + [
            {"override /inputs": ["local"]},
            {"override /processes": processes},
            {"override /outputs": ["write_ncdf", "mismip_ts"]},
        ],
        "core": {"print_info": False, "print_comp": True},
        "inputs": {"local": {"filename": input_rel}},
        "processes": {
            "time": {"start": t0, "end": t1, "save": save, "cfl": 0.3, "step_max": 1.0},
            "smb": {
                "method": "simple",
                "simple": {
                    "update_freq": 1.0e9,  # constant: computed once
                    "array": [
                        ["time", "gradabl", "gradacc", "ela", "accmax"],
                        [0.0, 0.0, 1.0, -1.0e4, C.ACCUMULATION],  # min(usurf + 1e4, 0.3) = 0.3 m/yr
                        [1.0e6, 0.0, 1.0, -1.0e4, C.ACCUMULATION],
                    ],
                },
            },
            "thk": {
                "scheme": "explicit",
                "ratio_density": round(C.RATIO, 6),
                # thk rows: top = y[0] (south wall); divide closed, front open, walls closed
                "boundary": {"left": "symmetric", "right": "zero", "top": "symmetric", "bottom": "symmetric"},
            },
            "iceflow": iceflow(),
            **({"steady_stop": dict(steady)} if steady else {}),
            **({"thin_ice_calving": {"min_thk": THIN_ICE_MIN_THK}} if calving else {}),
        },
        "outputs": {
            "write_ncdf": {"output_file": "output.nc", "complevel": 1, "vars_to_save": VARS + (VARS_BMB if melt else [])},
            "mismip_ts": {"output_file": "mismip_ts.csv", "freq": ts_freq},
        },
    }
    if melt:
        cfg["processes"]["bmb"] = {
            "method": "prescribed",
            "update_freq": 0.0,
            "include_grounded_melt": False,
            "grounding_line": {"treatment": treatment, "sub_samples": 4},
            "prescribed": MELT[melt],
        }
    return cfg


def write(name, cfg, what, methods):
    for m in methods:
        with open(os.path.join(C.ROOT, "prep", "methods", f"{m}.yaml")) as f:
            cfg = merge(cfg, yaml.safe_load(f))
    dx_off = cfg.pop("dX_offset", None)
    if "nn" in methods:  # the network replaces the Newton-CG block entirely
        for k in ("cg_newton", "halt", "line_search"):
            cfg["processes"]["iceflow"]["unified"].pop(k, None)
        norm = cfg["processes"]["iceflow"]["unified"]["normalization"]["fixed"]
        if dx_off is not None:
            norm["inputs_offsets"]["dX"] = dx_off
        # the constant rate factor maps to 0 (offset A, variance A^2)
        norm["inputs_offsets"]["arrhenius"] = round(C.ARRHENIUS, 4)
        norm["inputs_variances"]["arrhenius"] = round(C.ARRHENIUS ** 2, 2)
    p = cfg["processes"]
    head = (
        "# @package _global_\n"
        f"# MISMIP+ {name}: {what}.\n"
        f"# Input data/{cfg['inputs']['local']['filename']}; t = {p['time']['start']:g} -> {p['time']['end']:g} yr.\n"
        f"# MOLHO (Nz 2, Q1, {cfg['processes']['iceflow']['numerics']['precision']}), Weertman, A {C.ARRHENIUS:.4g} MPa-3 yr-1 (8e-25 Pa-3 s-1), tau_ref {C.TAU_REF:g} MPa at "
        f"{C.U_REF:g} m/yr; solver prep/methods/{' + '.join(methods)}.yaml.\n"
        "# GENERATED by prep/make_experiments.py: edit prep/ and rerun it, not this file.\n"
    )
    out = os.path.join(C.ROOT, "experiment", f"{name}.yaml")
    with open(out, "w") as f:
        f.write(head + yaml.safe_dump(flowify(cfg), sort_keys=False, width=110))
    return out


def experiments():
    """name -> (config, description, methods)."""
    exps = {}
    for res in C.RESOLUTIONS:
        nn = base(f"ice0_nn_{res}", f"mismip_plus_{res}_slab100.nc", 0.0, NN_YEARS[res], NN_SAVE[res], 10.0,
                  steady=STEADY_NN)
        nn["processes"]["time"]["cfl"] = 0.8
        nn["dX_offset"] = C.RESOLUTIONS[res]  # network input dX normalised to 0 at every grid
        exps[f"ice0_nn_{res}"] = (
            nn, "Ice0 spin-up stage 1, cheap hybrid emulator from a uniform 100-m slab\n# (0.3 m/yr "
                "accumulation, no melt, CFL 0.8)", ["nn"])
        cg = base(f"ice0_cg_{res}", f"restart/cg_{res}_init.nc", 0.0, CG_YEARS, 100.0, 1.0, steady=STEADY_CG)
        cg["processes"]["time"]["step_max"] = STEP_MAX[res]
        exps[f"ice0_cg_{res}"] = (
            cg, "Ice0 spin-up stage 2, converged MOLHO + cg_newton from the end of ice0_nn", ["fast", "cgfinish"])
        exps[f"timing_{res}"] = (
            base(f"timing_{res}", f"restart/timing_{res}_init.nc", 0.0, 20.0, 10.0, 0.0),
            "20-yr cost probe (a mismip_ts row every step)", ["fast"])
        ice0 = f"restart/ice0_{res}.nc"
        exps[f"ice0_{res}"] = (base(f"ice0_{res}", ice0, 0.0, 100.0, 10.0, 1.0),
                               "Ice0 control, no melt", ["fast"])
        c0 = base(f"ice0_{res}_nn", ice0, 0.0, 100.0, 10.0, 1.0)
        c0["dX_offset"] = C.RESOLUTIONS[res]
        exps[f"ice0_{res}_nn"] = (c0, "Ice0 control, no melt, hybrid emulator instead of cg_newton", ["nn", "nn_fwd"])
        for k in ("1", "2"):
            melt = f"ice{k}"
            r100 = f"restart/ice{k}r_{res}_t100.nc"
            exps[f"ice{k}r_{res}"] = (base(f"ice{k}r_{res}", ice0, 0.0, 100.0, 10.0, 1.0, melt),
                                      f"Ice{k}r, Ice{k} melt from the Ice0 steady state", ["fast"])
            exps[f"ice{k}rr_{res}"] = (base(f"ice{k}rr_{res}", r100, 100.0, 200.0, 10.0, 1.0, melt),
                                       f"Ice{k}rr, Ice{k} melt continued from Ice{k}r at 100 yr", ["fast"])
            exps[f"ice{k}ra_{res}"] = (base(f"ice{k}ra_{res}", r100, 100.0, 200.0, 10.0, 1.0),
                                       f"Ice{k}ra, melt switched off at 100 yr (re-advance)", ["fast"])
            for b in ("r", "rr", "ra"):  # the same with the emulator (prep/methods/nn.yaml + nn_fwd.yaml)
                name = f"ice{k}{b}_{res}"
                cfg_nn = copy.deepcopy(exps[name][0])
                cfg_nn["hydra"]["run"]["dir"] = f"outputs/{name}_nn"
                cfg_nn["dX_offset"] = C.RESOLUTIONS[res]
                if b != "r":  # the ra / rr branches start from the emulator's own r run
                    cfg_nn["inputs"]["local"]["filename"] = f"restart/ice{k}r_{res}_nn_t100.nc"
                exps[f"{name}_nn"] = (cfg_nn, exps[name][1] + ", hybrid emulator instead of cg_newton", ["nn", "nn_fwd"])
            if res == "1km":
                for tr in ("pmp", "fmp"):
                    n = f"ice{k}r_1km_{tr}"
                    exps[n] = (base(n, ice0, 0.0, 100.0, 10.0, 1.0, melt, treatment=tr),
                               f"Ice{k}r with the {tr} grounding-line melt treatment", ["fast"])
                n = f"ice{k}r_1km_tight"
                exps[n] = (base(n, ice0, 0.0, 100.0, 10.0, 1.0, melt),
                           f"Ice{k}r with the tight solver (solver-error check)", ["fast", "tight"])
    return exps


def main():
    # --only SUBSTR writes only the experiments whose name contains SUBSTR (e.g. while chains
    # are running, so that the files they will still read are not rewritten)
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    exps = experiments()
    n = 0
    for name, (cfg, what, methods) in exps.items():
        if only is None or only in name:
            write(name, cfg, what, methods)
            n += 1
    print(f"written {n} files in experiment/")


if __name__ == "__main__":
    main()
