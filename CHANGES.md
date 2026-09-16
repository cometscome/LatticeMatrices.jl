# Changes

This file records the user-visible changes in the stable v1 release line.
LatticeMatrices follows semantic versioning; releases in the stable v1 series
preserve the public v1 API.

## v1.2.7

### Enzyme coefficient gradients

- Extend the custom reverse rule for `add_matrix!(C, A, coefficient)` so an
  active real `coefficient` receives its real Frobenius-inner-product
  cotangent while preserving the existing lattice-input cotangent.
- Compute the coefficient cotangent with a JACC reduction and the field
  communicator's sum reduction, giving the same global parameter gradient in
  serial and MPI execution.
- Add focused Enzyme coverage for the active coefficient, including the MPI
  reduction path.

## v1.2.6

### Fused force primitives

- Add `traceless_antihermitian_product_add!` to accumulate the Lie-algebra
  projection of a local matrix product without materializing the product or
  launching a separate projection kernel. Both operands may independently be
  adjointed; SU(2) and SU(3) retain their established coefficient order and
  the generic SU(N) path follows the standard generator order.
- Add `wilson_clover_link_pullback_pair!` to combine two weighted
  Wilson--Clover cotangents before the six clover-plane path scatters. This
  preserves the result of two independent pullbacks while sharing link reads,
  cache validation, scratch fields, and kernel launches.

### Validation and performance

- Check all four operand-adjoint combinations of the fused product projection
  for SU(2) and SU(3), and compare the paired Clover pullback with two separate
  calls to maximum differences of `5e-13` and `4.44e-16`, respectively.
- On an otherwise idle NVIDIA H100 NVL, the SU(3) product/projection primitive
  is 1.30x faster and the paired Clover pullback is 2.02x faster than their
  separate-call equivalents in the focused benchmarks.

## v1.2.5

### APE, stout/EXP, HYP, and HEX smearing

- Add `APEParameters`, `StoutParameters`, `HYPParameters`, and
  `HEXParameters` together with allocating and preallocated four-dimensional
  link-smearing interfaces. Their defaults follow the standard matched
  choices: `alpha=0.6` for APE, `rho=0.1` for stout/EXP,
  `(alpha_outer, alpha_middle, alpha_inner)=(0.75, 0.6, 0.3)` for HYP, and
  `(0.125, 0.15, 0.15)` for HEX.
- Implement stout/EXP and all three restricted geometric levels of HEX with
  matrix-exponential SU(N) retractions. Both transformations have analytic
  reverse passes, exposed through `stout_pullback!` and `hex_pullback!`, for
  SU(2) and SU(3) links. HEX applies the standard `1/6`, `1/4`, and `1/2`
  geometric normalization to its outer, middle, and inner exponent
  coefficients.
- Give APE and all three restricted geometric levels of HYP two explicit SU(N)
  projection choices. `projection=:max_retr` is the default and follows the
  Bridge++ Cabibbo--Marinari SU(2)-subgroup iteration and global convergence
  criterion; `max_retr_iterations` and `max_retr_tolerance` control it.
  `projection=:polar` selects a principal-branch polar projection with an
  analytic reverse pass through both the U(N) polar factor and determinant
  phase. The derivative is defined away from the negative-real determinant
  branch cut.
- Keep iterative MaxReTr APE/HYP forward-only. Requesting its pullback throws
  an `ArgumentError` directing differentiable callers to `projection=:polar`,
  so interoperability and molecular-dynamics definitions are never silently
  mixed.
- Add the common `smearing_cache`, `smear_links`, `smear_links!`, and
  `smear_links_pullback!` protocol for nHYP and all four new schemes.
  `IteratedSmearing(parameters, iterations)` repeats a complete smearing
  transformation and reverses analytic schemes through every cached stage.
  This iteration count is distinct from the three nested levels of nHYP,
  HYP, and HEX.
- Reuse the existing target-centric staple reverse kernels and JACC
  matrix-exponential pullback. Reusable caches retain the required forward
  state, convert coefficients to the link precision, reject aliased storage,
  and detect thin links modified between a forward pass and its pullback.
- Require four-dimensional square floating-point link matrices and halo width
  `nw >= 1`. The native stout/EXP, HEX, APE, and HYP paths currently support
  SU(2) and SU(3); nHYP retains its existing generic U(N) projection path.

### Validation

- Add more than 900 focused CPU regression checks covering parameter validation,
  Float32 cache specialization, special-unitary output, both APE/HYP
  projections, zero-coefficient identity transformations, stale-cache
  rejection, MaxReTr pullback errors, and two-stage iteration.
- Compare the analytic stout/EXP, HEX, and polar APE/HYP pullbacks with central
  finite-difference directional derivatives on fixed-seed hot SU(3) fields.
  All pullbacks agree within the test tolerances, including a two-iteration
  stout chain.
- Compare MaxReTr APE/HYP and analytic HEX links site by site with Bridge++
  2.1.3 on a deterministic `4^4` SU(3) field. Maximum absolute differences
  are `1.78e-15`, `1.78e-15`, and `4.85e-12`, respectively.
- Compare stout at `rho=0.1` with QEX commit
  `f93ce40d9d88acf6c8dea1477e85e07bd50ecc01`; the forward and analytic
  pullback maximum absolute differences are `5.90e-16` and `1.89e-15`.
  Reproducible Bridge++ and QEX comparison drivers are included under
  `test/reference`.

## v1.2.4

### Normalized HYP smearing and HMC pullback

- The new `NHYPParameters`, `NHYPSmearingCache4D`, `nhyp_smear`,
  `nhyp_smear!`, and `nhyp_pullback!` APIs implement four-dimensional
  normalized HYP smearing with a U(N) polar projection at each of its three
  nested levels. The default outer, middle, and inner coefficients are 0.5,
  0.5, and 0.4.
- Coefficients are named by geometric nesting level to avoid ambiguity.
  QEX's `(alpha1, alpha2, alpha3)` convention maps to
  `(alpha_inner, alpha_middle, alpha_outer)` in LatticeMatrices.
- `nhyp_smear(U, parameters)` is the allocating interface and returns
  `(smeared, cache)`. For repeated HMC trajectories, callers can allocate
  `NHYPSmearingCache4D(U, parameters)` and the output links once and use
  `nhyp_smear!(smeared, U, cache)` on subsequent forward passes.
- The forward pass fuses the central link and all symmetric staples for each
  direction into one site kernel, reducing its kernel launches from 104 to
  56. Accelerator launches are queued across each nesting level and
  synchronized only at the three inner, middle, and outer stage boundaries.
- The reusable cache retains the unprojected and projected inner and middle
  links, the unprojected outer links, and reverse-pass scratch fields. Its
  coefficients are converted to the real element type of the links. A cache
  records its source objects and their core epochs and rejects a pullback
  after the thin links have changed. One cache must not be used concurrently
  by multiple tasks.
- `nhyp_pullback!(dU, left, U, cache)` analytically differentiates all staple
  products and U(N) projections, overwriting `dU` with the thin-link
  cotangent. Its convention is
  `real(sum(dot(dU[mu], deltaU[mu]) for mu in 1:4))`, so it can be used in
  HMC force construction without finite differencing.
- The NC=3 polar-projection pullback uses the QEX Cayley--Hamilton closed-form
  Sylvester solver instead of forming and LU-factorizing a 9-by-9 complex
  system at every site. This optimization is shared by nHYP and HISQ.
- Add public `staggered_link_pullback!` for the one-link staggered operator.
  It is a direct JACC analytic kernel used by LDO's dedicated staggered
  action; the Enzyme rule now shares the same implementation.
- Inputs and outputs may be vectors or four-tuples of compatible
  `LatticeMatrix{4}` links. The implementation requires periodic square
  floating-point matrix links with halo width `nw >= 1`, checks layout and
  aliasing constraints, and uses target-centric reverse kernels so MPI halo
  contributions are accumulated on the correct owning rank.

### Validation

- Forward links were compared site by site with QEX `hypsmear` at commit
  `f93ce40d9d88acf6c8dea1477e85e07bd50ecc01` on a fixed-seed hot SU(3) field
  on a 4^4 lattice. With QEX coefficients `(0.4, 0.5, 0.5)`, the maximum
  absolute difference was `6.62e-15` on threaded CPU.
- The analytic pullback was compared with QEX's `smearGetForce` on the same
  hot field and a nonzero random cotangent; its maximum absolute difference
  was `1.41e-14`. Independent finite-difference directional derivatives are
  also covered by the regression tests.
- Targeted nHYP tests pass with one and four CPU threads and with two MPI
  ranks. On an NVIDIA H100 NVL (compute capability 9.0), both the forward and
  pullback execute on `CuArray{ComplexF64}` through JACC's CUDA backend; the
  maximum QEX differences were `5.35e-15` and `1.98e-14`, respectively.
- The complete one-rank CPU regression suite passes 28,876/28,876 tests with
  the closed-form projection and public staggered pullback enabled.
- On a fixed-seed hot SU(3) `16^4` field on the H100, optimized median times
  over 20 synchronized runs are 9.825 ms for the forward pass, 36.125 ms for
  the pullback, and 50.721 ms for the combined chain.

## v1.2.3

### Lazy per-lattice scratch storage

- `LatticeMatrix` scratch fields are now allocated on first use instead of at
  construction time. The default `numtemps` is therefore zero; passing a
  positive `numtemps` continues to request eager preallocation.
- `similar(lattice)` creates an independent, initially empty scratch pool
  instead of copying the source lattice's scratch high-water mark. Scratch
  pools remain object-local so `get_block`/`unused!` lease accounting cannot
  alias between otherwise independent lattice objects.
- The new lazy pool preserves the unlabeled `PreallocatedArrays` operations
  used by LatticeMatrices and downstream diagnostics, including `get_block`,
  `unused!`, indexed access, `length`, and the existing `_data` and
  `_flagusing` inspection fields. Normal on-demand growth no longer emits the
  capacity warnings previously produced by `LatticeMatrix.temps`.
- The concrete type of `LatticeMatrix.temps` is now `LatticeScratchPool`.
  Downstream code should use the pool operations above rather than requiring a
  concrete `PreallocatedArray` type annotation.

### Memory diagnostics

- The exported `lattice_memory_report(lattice)` reports core data, halo
  padding, scratch capacity and in-use count, backend halo buffers, host
  staging buffers, and tracked totals in bytes.
- Reported byte counts cover array payloads. Julia object headers and backend
  allocator overhead are intentionally excluded.

### Validation

- Lazy allocation, explicit eager reservation, scratch release, and
  high-water-mark reset through `similar` are covered by serial and MPI
  long-shift regressions. Domain-wall adjoint tests also verify that borrowed
  scratch storage is returned after use.

## v1.2.2

### Portable oneAPI halo exchange

- oneAPI arrays use full preallocated face buffers for MPI halo exchange,
  avoiding compact derived device views that are not reliable on Intel GPUs.
- The oneAPI fallback retains host-staged MPI and the current boundary-phase
  convention. CUDA and ROCm keep the existing compact-buffer and
  device-direct transport paths unchanged.
- CPU MPI regression coverage compares the full-buffer fallback with the
  optimized exchange for halo width two, multidimensional corners, and
  complex boundary phases.

## v1.2.1

### HISQ extensions

- HISQ reunitarization and its analytic pullback now support the generic U(N)
  path used by `NC=2` and `NC=4`, while retaining the specialized physical
  `NC=3` kernels. `hisq_project_un` and `hisq_project_un!` are the generic
  names; the existing `hisq_project_u3` APIs remain backward-compatible
  aliases.
- Precomputed HISQ links with halo widths one or two use a correct
  shift-materializing Dirac fallback. The fused resident-halo stencil remains
  the performance path for `nw>=3`. Complete thin-link smearing supports
  `nw=0` or `nw>=2`.
- The core analytic U(N) projection pullback is shared with the Enzyme reverse
  rule, and the GPU smoke coverage now includes smearing, the Dirac stencil,
  link pullback, generic color counts, and low-halo execution.
- Nonzero-halo Fat7 construction now uses the same three-stage factorization
  for `NC=2`, `NC=3`, and `NC=4`, while retaining the unrolled `NC=3`
  specialization. Its analytic pullback reverses those stages and reuses
  cached intermediates instead of re-enumerating every path for every matrix
  element.

### Portable kernel-launch specialization

- Mutating lattice kernels bind their function and heterogeneous arguments in
  a concrete, Adapt-compatible callable before entering `JACC.parallel_for`.
  The public `parallel_for_mutating!` launcher provides the same specialization
  to Gaugefields and other lattice consumers while preserving one backend-
  independent path whose fields are adapted to each accelerator's device
  types.
- On the one-thread Threads backend, an `8^4` SU(3) lattice `mul!` now performs
  zero heap allocation after warm-up, down from 254,544 bytes per call. The
  numerical result is unchanged. Multi-threaded execution, Gaugefields
  heatbath comparisons with its legacy backend, and CUDA execution on an
  NVIDIA H100 are covered by regression tests.

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
