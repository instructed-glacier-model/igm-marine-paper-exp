# Source before running PISM or MPI: puts the restored environment (setup.sh) first on PATH, one thread per rank.
PISM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PATH="$PISM_ROOT/env/bin:$PATH"
export UDUNITS2_XML_PATH="$PISM_ROOT/env/share/udunits/udunits2.xml"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
