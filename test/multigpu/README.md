# Two-H100 multi-GPU test

This is an opt-in integration test. It launches two MPI ranks on one node, binds rank 0
to CUDA device 0 and rank 1 to CUDA device 1 through
`select_device_by_mpi_rank!`, and requires both devices to be NVIDIA
H100 GPUs (compute capability 9.0).

The test checks:

- two distinct physical GPU UUIDs are used;
- JACC allocates `CUDA.CuArray` storage;
- MPI halo exchange across the two GPU-owned lattice partitions;
- distributed `mul!`, shifted-lattice materialization, gather, and reduction results
  against CPU reference values.
- staggered D and D-dagger against a dense CPU oracle with an odd local x
  extent, checking global staggered phases across the rank boundary.

Run from the `LatticeMatrices.jl` directory:

```bash
test/multigpu/run_h100_2gpu.sh
```

The runner expects the CUDA-enabled Open MPI installation at `/opt/ompi-cuda`,
sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`, and uses GPUs `0,1`. Override the
executable paths or selected GPUs when needed:

```bash
JULIA=/path/to/julia \
MPIEXEC=/opt/ompi-cuda/bin/mpirun \
CUDA_VISIBLE_DEVICES=2,3 \
LATTICEMATRICES_GPU_TEST_ITERS=10 \
test/multigpu/run_h100_2gpu.sh
```

The H100 name and compute-capability checks are enabled by default. To exercise the same
multi-GPU correctness test on a non-H100 pair, explicitly set
`LATTICEMATRICES_REQUIRE_H100=false`.

If `MPIEXEC` points at another MPI installation, update the matching `libmpi` and
`mpiexec` entries in `LocalPreferences.toml` as well; the launcher and MPI library must
come from the same implementation.

To compare pinned host staging with CUDA-aware MPI on two GPUs and verify that
both halo exchange and long-distance shifts produce identical results:

```bash
JULIA_CUDA_MEMORY_POOL=none \
CUDA_VISIBLE_DEVICES=0,1 \
/opt/ompi-cuda/bin/mpirun --bind-to core -np 2 \
    julia --project=test/multigpu test/multigpu/mpi_transport.jl
```

For the four-GPU direct-shift regression on the global `12×12×12×24` lattice:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/opt/ompi-cuda/bin/mpirun --bind-to core -np 4 \
    julia --project=test/multigpu test/multigpu/direct_shift_4gpu.jl
```

This test covers process grids `(4,1,1,1)`, `(2,2,1,1)`, and `(1,1,2,2)`,
halo widths `0`, `1`, and `2`, positive/negative diagonal shifts, phase-wrapped
shifts, lease release, and preallocated-pool reuse. It also reports synchronized
rank-maximum median timings for an in-halo shift and a direct long shift.

For a single-GPU correctness check and steady-state staggered benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 \
LATTICEMATRICES_STAGGERED_BENCH_L=24 \
LATTICEMATRICES_STAGGERED_BENCH_PRECISION=Float32 \
julia --project=test/multigpu test/multigpu/staggered_bench.jl
```

`LATTICEMATRICES_STAGGERED_BENCH_ITERS` and
`LATTICEMATRICES_STAGGERED_BENCH_SAMPLES` control timing repetitions.  Use
`Float64` for the double-precision run.

For a single-GPU Möbius/generalized-domain-wall forward and Enzyme-pullback
benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 \
LATTICEMATRICES_DOMAINWALL_BENCH_L=16 \
LATTICEMATRICES_DOMAINWALL_BENCH_L5=12 \
LATTICEMATRICES_DOMAINWALL_BENCH_PRECISION=Float32 \
julia --project=test/multigpu \
    test/multigpu/domainwall_pullback_bench.jl
```

`LATTICEMATRICES_DOMAINWALL_BENCH_ITERS` and
`LATTICEMATRICES_DOMAINWALL_BENCH_SAMPLES` control the timing repetitions.
The reported pullback time includes the primal application and scalar loss,
as required by the complete reverse-mode evaluation.  The driver reports the
generalized forward overhead relative to the legacy Möbius specialization and
times the generalized pullback including gradients of all `a_s`, `b_s`, and
`c_s` coefficients.

## MPI device mapping on multiple nodes

`LatticeMatrix` uses `device_mapping=:auto` by default.  It creates a temporary
MPI shared-memory communicator and assigns devices by the rank within the local
node, not by `MPI.COMM_WORLD` rank.  For example, with two MPI ranks and two
visible GPUs on every node, local ranks 0 and 1 select devices 0 and 1 on every
node independently.  The same mapping supports CUDA, AMDGPU, and oneAPI through
the JACC backend selected before loading `LatticeMatrices`.

If a scheduler exposes exactly one physical GPU to each rank, every rank selects
the sole device visible in its process instead.  Pass `device_mapping=:current`
to `LatticeMatrix` only when the application or launcher has already selected
the device and the automatic mapping should be skipped.
