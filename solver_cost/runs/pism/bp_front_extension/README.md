# Front adapter for PISM's Blatter–Pattyn solver

`FrontMatched.hh` subclasses PISM's `Blatter` solver (PISM 2.3.1, GPL-3.0-or-later) so that the marine-front
pressure is applied on the faces that IGM classifies as the front, instead of PISM's native classification; PISM's
volume, basal, pressure and Jacobian kernels are unchanged. IGM's rule (`igm/processes/iceflow/energy/components/
floating.py`): a horizontal element is icy when all four corners have ice, land when an ice-free corner has its lower
surface above the local water level, and wet otherwise; pressure acts on icy faces adjoining wet elements. PISM's
native classifier only looks at boundary-node types and at whether the ice bottom is below the water level, so on
the 1 km grid it loaded a few faces bordering interior rock holes, giving a few very large velocities absent from
IGM. The subclass also takes a nodal mask of stationary columns; `bp_diag.py` passes an empty mask, so the
quasi-rigid band keeps its finite drag as in IGM.

`bp_front_boundary.i` is the SWIG interface; `../build_bp_front_extension.sh` builds `_bp_front_boundary.so`
against the PISM of `../env`. `bp_diag.py` computes the face mask from the input fields, records its size in
`audit.json`, and passes an empty face list with `--front-treatment native` to recover PISM's classifier.
