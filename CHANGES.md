# Changes

This file records the user-visible changes in the stable v1 release line.
LatticeMatrices follows semantic versioning; releases in the stable v1 series
preserve the public v1 API.

## v1.2.0

### Optional MPI backend

- MPI.jl is now a weak dependency. Loading LatticeMatrices without MPI uses a
  type-stable `SerialCommunicator` for one-process lattices and local periodic
  halo updates.
- MPI applications must add MPI.jl to their own environment, load it with
  `using MPI`, and call `MPI.Init()` before constructing an MPI-backed lattice.
  With MPI loaded, `comm0=nothing` continues to select `MPI.COMM_WORLD`.
- LatticeMatrices does not call `MPI.Init()` or `MPI.Finalize()` automatically;
  MPI lifecycle management remains the application's responsibility.
- One-rank MPI applications can explicitly select either implementation with
  `comm0=SerialCommunicator()` or `comm0=MPI.COMM_WORLD`.
- One-process lattices no longer allocate unused packed MPI halo buffers on the
  host or accelerator.
- Loading MPI.jl activates `LatticeMatricesMPIExt` and preserves the existing
  `MPI.COMM_WORLD`, Cartesian decomposition, halo exchange, reductions,
  gathers, and direct-shift behavior.
- CUDA/ROCm device-aware MPI detection is isolated in combined GPU+MPI
  extensions, while non-MPI accelerator use remains available.
- MPIPreferences is no longer a direct dependency; MPI.jl continues to provide
  it transitively when MPI support is installed.
- `mpi_transport_info` reports `resolved=:local` and
  `reason=:serial_communicator` when no inter-process transport is present.

### Validation

- Version 1.2.0 was validated with Gaugefields 1.0.4 and
  LatticeDiracOperators 1.0.0 using threaded CPU execution, two-rank MPI, and
  CUDA execution on an NVIDIA H100. CUDA execution without MPI was also
  validated with `SerialCommunicator`. Enzyme AD smoke tests run both with and
  without MPI.jl installed.

## v1.1.6

### Generic SU(N) normalization

- `normalize_matrix!` now applies a determinant-phase correction after the
  generic modified Gram–Schmidt path, so color counts above three are projected
  to SU(N) rather than only U(N).
- Rank-deficient inputs receive a stable orthonormal basis completion, and the
  generic path remains allocation-free inside CPU and accelerator kernels.

## v1.1.5

### MPI transport selection

- `LatticeMatrix(...; mpi_transport=...)` accepts `:auto`, `:host_staged`, and
  `:device_direct`. CUDA and ROCm use MPI.jl's official device-buffer support;
  unsupported accelerator backends fall back to host staging in `:auto` mode.
- Halo exchange, long-distance `Alltoallv!` shifts, and reverse halo exchanges
  use the same per-lattice transport policy. `mpi_transport_info` reports the
  requested and resolved route for reproducible benchmark output.

## v1.1.4

### GPU normalization

- SU(3) normalization accumulates squared row norms with `abs2`, keeping them
  real.  This avoids the checked conversion path in `sqrt(::ComplexF32)` and
  its AMDGPU `malloc_hostcall` without changing the normalization algorithm.
- Float32 and Float64 normalization are checked for unitarity and unit
  determinant with halo widths one and three.

## v1.1.3

### Domain-wall adjoint performance

- The physical `NC=3` Möbius and generalized domain-wall adjoints use a
  backend-independent two-stage path: one four-dimensional Wilson-adjoint
  evaluation per fifth slice followed by element-owned fifth-direction
  mixing.
- The intermediate field is borrowed from the input field's existing
  temporary pool and returned after every application. There is no global
  cache, field-layout change, backend-specific launch branch, or public API
  change.
- Other color counts retain the generic implementation.

### Validation

- Möbius and nonuniform generalized adjoints are checked against dense
  references and the adjoint inner-product identity.
- The optimized path is covered by CUDA, JACC Threads, and optional Enzyme
  reverse-mode tests. Tests also verify that the borrowed temporary is
  released after use.

## v1.1.2

### Fermion operators and pullbacks

- The physical `NC=3` Wilson forward, adjoint, and hopping-only paths use a
  backend-independent half-spin implementation. The generic implementation
  remains available for other color counts and halo-free fields.
- The `NC=3` clover field-strength kernels evaluate each four-link leaf with
  factorized SU(3) matrix products.
- `wilson_clover_link_pullback!` provides the complete analytic link pullback
  for Wilson hopping and four-leaf clover contributions. The optional Enzyme
  reverse rules delegate to the same core implementation.
- Möbius and generalized domain-wall operators gained optimized five-dimensional
  kernels and updated link pullbacks.

### Domain-wall measurement building blocks

- Reusable APIs import a four-dimensional Shamir physical source into a
  five-dimensional field and export physical or midpoint projections.
- `PP` and `J5q` projected-bilinear slice contractions are available for
  residual-mass and propagator measurements.

### Validation

- Wilson half-spin kernels are checked against the generic Wilson
  implementation for forward, adjoint, and hopping-only applications.
- Wilson--clover link derivatives are checked against finite differences and
  through the optional Enzyme extension.
- Domain-wall physical projections are checked against direct reference
  contractions.

## v1.1.1

### HISQ improvements

- `hisq_link_pullback!` exposes the analytic thin-link pullback through the
  Dirac stencil, Naik links, level-2 Fat7/Lepage smearing, U(3) projection, and
  level-1 Fat7 smearing.
- The physical `NC=3` HISQ kernels use row-owned and factorized implementations
  to reduce repeated path work and improve accelerator execution.
- `HISQDiracCache4D` tracks thin-link epochs and reuses the complete smearing
  chain until a source link changes.
- The HISQ stencil and smearing chain were cross-checked against independent
  SIMULATeQCD reference programs. Benchmark methodology and results are kept in
  [`benchmark/HISQ_DIRAC_BENCHMARK_2026-08-17.md`](benchmark/HISQ_DIRAC_BENCHMARK_2026-08-17.md).

## v1.1.0

### Measurement building blocks

- `set_global_component!` sets one matrix component using global lattice
  coordinates. Under MPI, only the rank that owns the site writes it; the same
  site-local JACC kernel works on CPU and accelerator backends.
- `projected_bilinear_slices` contracts spin-color propagators into hyperplane
  slices, with optional Fourier-momentum and staggered-parity projections, and
  performs the final global MPI reduction. QCDMeasurements.jl uses these
  primitives for generalized connected-meson two-point measurements.

### Stable SU(2)/SU(3) exponential pullbacks

- `exp_ta_pullback!(output, cotangent, A, t=1)` is a public pullback for
  `exp(t * TA(A))`, restricted to traceless anti-Hermitian variations. SU(2)
  and SU(3) use site-local JACC kernels shared by CPU and GPU execution.
- The active Enzyme reverse rule for `expt_TA!` calls these core kernels. The
  SU(3) coefficient derivatives are analytic rather than finite-difference
  estimates.
- The SU(3) implementation follows the Cayley--Hamilton coefficients and
  derivatives of
  [Morningstar--Peardon](https://arxiv.org/abs/hep-lat/0311018), including the
  reflection relations for negative cubic invariant. Stable series are used
  near the origin and near degenerate eigenvalues.
- The pullbacks were checked against the block-matrix exponential identity of
  [Al-Mohy--Higham](https://eprints.maths.manchester.ac.uk/1218/) for SU(2) and
  SU(3), including zero and very small fields, negative `t`, both signs of the
  SU(3) degenerate-eigenvalue case, and `ComplexF32`/`ComplexF64`.

### Compatibility

- Enzyme remains an optional weak dependency; the existing `expt_TA!`
  interface is unchanged.
- Existing v1.0 code requires no source changes. QCDMeasurements.jl's
  generalized meson measurement requires LatticeMatrices v1.1 or later.

## v1.0.0

Compared with v0.3.13, v1.0.0 added and stabilized:

- Production parallel execution with
  [JACC.jl](https://github.com/JuliaORNL/JACC.jl) threaded CPU kernels, MPI,
  hybrid MPI+threads, single GPU, and multi-GPU execution. Multi-GPU jobs
  normally use one MPI rank per GPU, with automatic node-local device mapping
  for CUDA, AMDGPU/ROCm, and oneAPI backends.
- Safe automatic halo synchronization using core and halo epochs. Public
  mutations mark halos stale, and shifted reads synchronize them on demand.
- Arbitrary-distance periodic shifts, including direct MPI redistribution,
  boundary phases, reusable preallocated storage, and `nw=0` operation.
- Wilson, Wilson--clover, one-link staggered, HISQ, Möbius domain-wall, and
  generalized domain-wall operators, together with adjoints and cached paths.
- Optional Enzyme reverse-mode AD for lattice operations, Dirac operators, and
  HISQ smearing. The correctly spelled `Wirtinger` API was added while the old
  `Wiltinger` names remained compatibility aliases.
- Decomposition-independent site utilities and random-number streams, an
  allocation-conscious CG solver, and a distributed non-QCD Heisenberg-model
  example.

### Upgrading from v0.3

Existing constructors and legacy entry points remain available. Code that
writes directly to `M.A` must call `mark_halo_dirty!(M)` after completing its
core writes; mutations through the public LatticeMatrices API do this
automatically. Device selection is automatic by default; pass
`device_mapping=:current` when the job launcher has already assigned a device
to each MPI rank.
