#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$script_dir"
julia_bin="${JULIA:-julia}"
mpirun_bin="${MPIEXEC:-/opt/ompi-cuda/bin/mpirun}"

if [[ ! -x "$mpirun_bin" ]]; then
    echo "MPI launcher not found or not executable: $mpirun_bin" >&2
    exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi was not found" >&2
    exit 2
fi

visible_devices="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES="$visible_devices"
export JULIA_PKG_PRECOMPILE_AUTO=0

"$julia_bin" --startup-file=no --project="$project_dir" \
    -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

"$mpirun_bin" --bind-to core -np 2 \
    "$julia_bin" --startup-file=no --project="$project_dir" \
    "$project_dir/runtests.jl"
