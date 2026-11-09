#!/usr/bin/env bash
# Build the Blatter-Pattyn front adapter (bp_front_extension/FrontMatched.hh, a subclass of PISM's Blatter solver that
# takes IGM's classification of the marine front) as the Python module bp_front_extension/_bp_front_boundary.so.
# Needs the two environments of setup.sh.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
export PATH="$ROOT/tools-env/bin:$ROOT/env/bin:$PATH"
export LD_LIBRARY_PATH="$ROOT/env/lib:$ROOT/tools-env/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT/bp_front_extension"
swig -c++ -python -I"$ROOT/env/include" -o bp_front_boundary_wrap.cxx bp_front_boundary.i
CXX="$ROOT/tools-env/bin/x86_64-conda-linux-gnu-c++"
"$CXX" -O3 -DNDEBUG -shared -fPIC -std=c++17 \
  -I"$ROOT/env/include" -I"$ROOT/env/include/python3.11" -I"$ROOT/env/include/pism" -I"$ROOT/env/include/petsc" \
  -I"$ROOT/env/lib/python3.11/site-packages/mpi4py/include" -I"$ROOT/env/include/openmpi" \
  bp_front_boundary_wrap.cxx -L"$ROOT/env/lib" -lpism -lpetsc -lmpi -Wl,-rpath,"$ROOT/env/lib" -o _bp_front_boundary.so
sha256sum FrontMatched.hh bp_front_boundary.i _bp_front_boundary.so > build-sha256.txt
"$CXX" --version > compiler-version.txt
