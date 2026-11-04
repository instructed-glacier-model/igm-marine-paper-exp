# Error split of Sect. 5.5 (Fig. 8)

How much of the error of a neural-network run against the reference run is the network's own error, and how much
comes from the two trajectories having drifted apart? Following Rosier et al. (2026, Sect. 3.3), with E the state of
the network run at a saved frame, R the state of the reference run at the same time, and V the mean surface speed
over the cells with more than 1 m of ice in both states (Pine Island's quasi-rigid band left out):

- e_tot = V_net(E) − V_ex(R), the error of the network run; e_tot = e_inst + e_geom with
- e_inst = V_net(R) − V_ex(R), the instantaneous error: the network of time t evaluated on the reference state;
- e_geom = V_net(E) − V_net(R), the geometry-induced term: the same network on the two geometries;
- e_phys = V_ex(E) − V_ex(R), the response of the identity mapping (Newton–CG) to the same difference of state.

Fig. 8 plots e_geom against e_phys for the N = 20 runs. This needs the network of every frame evaluated on the
reference state, and the Newton–CG solution on every state of both runs, hence three kinds of extra runs:

| Run | What | Experiment, user modules |
|---|---|---|
| `w_<run>` | the N = 20 network run again, saving the network weights at every frame (bitwise the original run) | `glacier_*_T2_init_stop_w` (the original experiment + `save_network_weights`) |
| `netreplay_exact_<run>` | the saved weights of each frame evaluated, without training, on the reference state of that frame | `glacier_*_netreplay` (`replay_geometry`, `load_network_weights`) |
| `replay_<reference>`, `replay_<run>` | cold Newton–CG solves of every saved frame of a run (CG tolerance 10⁻⁴, halt at 0.1 m yr⁻¹, single precision) | `glacier_*_replay` (`replay_geometry`) |

A replay reads the frames of its source run (`processes.replay_geometry.source`) and starts from an input file holding
the geometry of its first frame, made by `prepare_reference_input.py`. The replay of the reference run means that
V_ex(E) and V_ex(R) use the same numerics, so e_phys = 0 at t = 0. On Pine Island, the single-precision replay of the
network run slows to minutes per frame from 57 yr on a nearly detached floating block; its frames from 57 yr were
solved in double precision (`replay_pig_n50_20_mt5_100yr_dbl`). On that block the Newton–CG speed on the network
state runs away, and the cells where it exceeds twice the largest V_ex(R) of the frame are left out of e_phys.

## Reproduce

After `../run_all.sh`, from `glaciers/runs`:

```bash
geometry_response/run.sh                       # the runs above, one GPU: about 70 min (Sermeq Kujalleq), 1 h (Pine Island), 12 min (Mendenhall)
python geometry_response/geometry_response.py  # -> ../figure/data/geometry_response.csv (CPU, a minute)
python ../figure/fig_gmd_geometry_response.py  # Fig. 8
```

`geometry_response.csv` has one row per frame: `cells`, `excluded_phys`, `V_exR`, the four terms `e_*` (m yr⁻¹), the
same split of the field as relative RMSE (`R_*`, %), and the cosines between the error fields.
