# Two-H100 multi-GPU test

This is an opt-in integration test. It launches two MPI ranks on one node, binds rank 0
to CUDA device 0 and rank 1 to CUDA device 1, and requires both devices to be NVIDIA
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

The runner expects the CUDA-enabled Open MPI installation at `/opt/ompi-cuda` and uses
GPUs `0,1`. Override the executable paths or selected GPUs when needed:

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

For a single-GPU Möbius/domain-wall forward and Enzyme-pullback benchmark:

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
as required by the complete reverse-mode evaluation.
