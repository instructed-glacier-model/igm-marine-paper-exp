#!/usr/bin/env bash
# Restore the two pinned conda environments from their explicit package lists (environment/*.txt), with micromamba:
#   env/        PISM 2.3.1 (PETSc 3.24.6, Open MPI 5.0.10, Python 3.11) -- the runs
#   tools-env/  the compiler and SWIG that build the Blatter-Pattyn front adapter
# Existing environments are kept as they are.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$ROOT/bin"
if ! test -x "$ROOT/bin/micromamba"; then
  curl -fLs https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$ROOT" bin/micromamba
fi
restore_environment() {
  local destination="$1" lock="$2"
  if test -x "$destination/bin/python"; then
    return
  fi
  local spec
  spec=$(mktemp)
  { printf '@EXPLICIT\n'; grep '^https://' "$lock"; } > "$spec"
  "$ROOT/bin/micromamba" create -y -p "$destination" -f "$spec"
  rm -f "$spec"
}
restore_environment "$ROOT/env" "$ROOT/environment/pism-explicit.txt"
restore_environment "$ROOT/tools-env" "$ROOT/environment/bp-tools-explicit.txt"
