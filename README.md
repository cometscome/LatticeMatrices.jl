# LatticeMatrices.jl

[![Build Status](https://github.com/cometscome/LatticeMatrices.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cometscome/LatticeMatrices.jl/actions/workflows/CI.yml?query=branch%3Amain)

High-performance **matrix fields on arbitrary D-dimensional lattices** in Julia.

🎉 **LatticeMatrices.jl v1 is available!**

Version 1.2.2 is the current backward-compatible release in the stable v1 line.
It supports Julia 1.11 and later, threaded CPU execution, MPI decomposition,
and accelerator execution through JACC.

Version 1.2.2 adds a portable full-buffer halo fallback for oneAPI while
preserving the optimized CUDA and ROCm communication paths; see
[CHANGES.md](CHANGES.md) for details.

Version 1.1.6 added generic SU(N) normalization for NC > 3.

Version 1.1.5 added selectable host-staged and device-direct MPI transports for CUDA/ROCm-aware MPI.

Existing serial code requires no source changes. MPI applications must declare
MPI.jl in their own environment, load it, and initialize MPI explicitly; see
[CHANGES.md](CHANGES.md) for the complete v1 release history and upgrade notes.

## Installation

```julia
pkg> add LatticeMatrices
```

Requirements:

- Julia ≥ 1.11

MPI.jl is optional. Without MPI.jl, a one-process lattice uses the built-in
`SerialCommunicator`. Add and load MPI.jl only when using an MPI process grid:

```julia
pkg> add MPI

julia> using LatticeMatrices, MPI
julia> MPI.Init()
```

Call `MPI.Init()` before constructing an MPI-backed `LatticeMatrix`.
LatticeMatrices does not call `MPI.Init()` or `MPI.Finalize()` automatically.

---

## What you can do

- Store real or complex `NC1×NC2` matrices at every site of an arbitrary
  D-dimensional lattice, including square and rectangular site matrices.
- Decompose lattices over MPI Cartesian process grids with configurable halos,
  boundary phases, and distributed gather/reduction operations.
- Run the same JACC kernels on threaded CPUs and supported accelerators,
  including hybrid MPI+threads and multi-GPU configurations.
- Apply local and shifted matrix algebra, matrix exponentials, traceless
  anti-Hermitian projections, even/odd updates, and iterative solvers.
- Build Wilson, Wilson--clover, staggered, HISQ, Möbius domain-wall, and
  generalized domain-wall operators, together with their adjoints and cached
  execution paths.
- Use LatticeMatrices directly for structured-lattice models or as the
  MPI/JACC backend for
  [Gaugefields.jl](https://github.com/akio-tomiya/Gaugefields.jl) and
  [LatticeDiracOperators.jl](https://github.com/akio-tomiya/LatticeDiracOperators.jl).

---

## Quick tour

### 1) D-dimensional indexing helpers (GPU-kernel friendly)

```julia
using LatticeMatrices

# Build an indexer for a D-dimensional lattice (1-based indices)
gsize = (16, 16, 16, 16)     # global lattice size
d = DIndexer(gsize)          # computes column-major strides internally

# Convert between linear and multi-index (1-based)
L  = linearize(d, (1, 1, 1, 1))   # -> 1
ix = delinearize(d, 4)            # -> (4, 1, 1, 1) on this shape

# Apply shifts componentwise
p = shiftindices((4, 1, 1, 1), (1, 0, 0, 0))   # -> (5, 1, 1, 1)
```

**Signatures**
```julia
struct DIndexer{D,dims,strides} end
DIndexer(dims_in::NTuple{D,<:Integer}) where {D}
DIndexer(dims_in::AbstractVector{<:Integer})

# 1-based linearization/delinearization (no heap allocs; GPU-friendly)
linearize(::DIndexer{D,dims,strides}, idx::NTuple{D,T})::Int32 where {D,T<:Integer}
delinearize(::DIndexer{D,dims,strides}, L::Integer, offset::Integer=0)::NTuple{D,Int}

# elementwise shifting for index tuples
shiftindices(indices, shift)
```

- `delinearize(..., offset)` is handy to **map into halo regions**, e.g. pass `offset = nw`.

---

### 2) Lattice containers (serial/MPI + halos + JACC arrays)

The core container stores a **halo-padded** array on each rank. Without MPI.jl,
`PEs = (1, ..., 1)` selects the MPI-free serial communicator and periodic halos
are copied locally. When MPI.jl is loaded, the default communicator is
`MPI.COMM_WORLD`; distributed faces are packed into contiguous buffers without
MPI derived datatypes.

Serial construction does not require MPI.jl:

```julia
using LatticeMatrices

dim = 4
gsize = (16, 16, 16, 16)
M = LatticeMatrix(3, 3, dim, gsize, (1, 1, 1, 1); nw=1)
```

For MPI decomposition:

```julia
using LatticeMatrices, MPI, JACC, LinearAlgebra
JACC.@init_backend
MPI.Init()  # required before constructing an MPI-backed lattice

dim   = 4
nw    = 1                      # ghost width
NC    = 3                      # per-site matrix size (NC×NC)

# This decomposition is valid for any positive MPI process count.
nprocs = MPI.Comm_size(MPI.COMM_WORLD)
gsize = (4 * nprocs, 16, 16, 16)
PEs = (nprocs, 1, 1, 1)

# Construct an empty lattice matrix (device array via JACC.zeros)
M = LatticeMatrix(NC, NC, dim, gsize, PEs; nw, elementtype=ComplexF64)

# Or initialize from an existing array (broadcast to ranks)
A = rand(ComplexF64, NC, NC, gsize...)
M2 = LatticeMatrix(A, dim, PEs; nw, numtemps=2)

# Halo exchange across all spatial dimensions
set_halo!(M)

# Global gather helpers (host reconstruction on rank 0)
G = gather_matrix(M; root=0)                # rank 0: Array(NC, NC, gsize...)
Gall = gather_and_bcast_matrix(M; root=0)   # all ranks receive the same Array
```

**Key type**
```julia
abstract type LatticeMatrix{D,T,AT,NC1,NC2,nw,DI} <:
    Lattice{D,T,AT,NC1,NC2,nw}

mutable struct HaloEpoch
    core::UInt64
    halo::UInt64
end

struct LatticeMatrix_standard{D,T,AT,NC1,NC2,nw,DI,C} <:
       LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}
    nw::Int                       # ghost width
    phases::SVector{D,T}          # per-direction phase
    NC1::Int
    NC2::Int
    gsize::NTuple{D,Int}
    cart::C                       # serial or MPI Cartesian communicator
    coords::NTuple{D,Int}         # 0-based Cartesian coordinates
    dims::NTuple{D,Int}           # process grid (PEs)
    nbr::NTuple{D,NTuple{2,Int}}  # neighbors (minus, plus)
    A::AT                         # local halo-padded array
    buf::Vector{AT}               # device-side communication buffers
    buf_host::Vector{Array{T}}    # host communication buffers (pinned when supported)
    shift_buf_host::DirectShiftHostBuffers{T}
    mpi_transport::MPITransportConfig
    myrank::Int
    PN::NTuple{D,Int}             # local interior size per dimension
    comm::C                       # original communicator
    indexer::DI                   # DIndexer for global sizes
    temps::PreallocatedArray{AT,Union{Nothing,String},false}
    halo_epoch::HaloEpoch
end
```

`LatticeMatrix` is the abstract interface; the `LatticeMatrix(...)`
constructors return a `LatticeMatrix_standard`. `HaloEpoch` records the core
and halo epochs so shifted reads can synchronize stale halo data automatically.

**Constructors**
```julia
LatticeMatrix(NC1, NC2, dim, gsize, PEs;
              nw=1, elementtype=ComplexF64, phases=ones(dim),
              comm0=nothing, numtemps=1, device_mapping=:auto,
              mpi_transport=:auto)

LatticeMatrix(A, dim, PEs; nw=1, phases=ones(dim),
              comm0=nothing, numtemps=1, device_mapping=:auto,
              mpi_transport=:auto)
```

- **Layout**: `(NC1, NC2, X, Y, Z, …)`; halos are the outer `nw` cells on each spatial dim.
- **Phases**: wrap-around phases per dimension. A positive-direction wrap applies `phase`,
  while a negative-direction wrap applies `inv(phase)`.
- **Communicator**: `comm0=nothing` selects `MPI.COMM_WORLD` when MPI.jl is
  loaded, otherwise `SerialCommunicator()`. Pass either communicator explicitly
  to override the default. In particular, a one-rank MPI process can choose
  either path explicitly:

  ```julia
  serial = LatticeMatrix(3, 3, 4, gsize, (1, 1, 1, 1);
                         comm0=SerialCommunicator())
  mpi = LatticeMatrix(3, 3, 4, gsize, (1, 1, 1, 1);
                      comm0=MPI.COMM_WORLD)
  ```
- **Exchange**: `set_halo!(ls)` calls `exchange_dim!(ls, d)` for each spatial dimension `d`.

Halo exchange uses receive-before-send nonblocking MPI without a per-direction
global barrier.  Sequential directions send a staircase cross-section: halos
from directions already exchanged are included to preserve corners, while
not-yet-exchanged directions send only their core range.

`mpi_transport` selects how accelerator buffers reach MPI:

- a serial lattice resolves to `:local`, because it performs no inter-process
  transport;
- `:auto` uses device-direct MPI when both MPI.jl and the selected MPI library
  report support, and otherwise falls back to host staging;
- `:host_staged` always copies accelerator buffers through host memory;
- `:device_direct` requires device-aware MPI and throws during construction if
  the capability cannot be confirmed.

CPU arrays are already passed directly to MPI. CUDA and ROCm device buffers use
MPI.jl's official GPU-buffer integration together with `MPI.has_cuda()` or
`MPI.has_rocm()`. oneAPI and Metal remain on the portable host-staged path until
MPI.jl provides a corresponding direct-buffer integration. The resolved route
and the MPI implementation can be recorded in benchmark output:

```julia
info = mpi_transport_info(M)
# (requested=:auto, resolved=:device_direct, backend=:cuda, ...)
```

For reproducible transport comparisons, construct separate fields with
`mpi_transport=:host_staged` and `mpi_transport=:device_direct`; do not infer a
performance comparison from `:auto`.

#### Halo epochs and automatic synchronization

Each `LatticeMatrix` tracks a core-data epoch and a halo epoch. Public
mutating operations advance the core epoch. A nonzero `Shifted_Lattice` read
calls `ensure_halo!`, which exchanges halos only when the two epochs differ.
Reusing a shifted wrapper after another mutation is safe: the wrapper checks
the source lattice again when its data is read.

```julia
add_matrix!(M, M2)               # core epoch advances; halo is now dirty
@assert halo_is_dirty(M)

Mp = Shifted_Lattice(M, (1, 0, 0, 0))
@assert !halo_is_dirty(M)        # the shift synchronized the halo on demand

epochs = halo_epochs(M)          # (core=..., halo=...)
ensure_halo!(M)                  # no communication while already clean
@assert halo_epochs(M) == epochs
```

Writing through the storage field bypasses the public mutating API. Call
`mark_halo_dirty!` once after the core writes are complete:

```julia
@views M.A[:, :, interior_ranges...] .= new_values
mark_halo_dirty!(M)

# Optional eager synchronization. Usually a later shift can do this lazily.
ensure_halo!(M)
```

Backend packages wrapping `LatticeMatrix` have the same obligation: a kernel
that writes directly to `M.A` must call `mark_halo_dirty!(M)` before a shifted
read or `set_halo!`. Operations that use the exported LatticeMatrices mutating
API are marked automatically. With `nw=0`, the core and halo epochs remain
equal and `ensure_halo!` is a no-op.

Halo exchange is an MPI collective operation. All ranks in the Cartesian
communicator must therefore execute mutations and later shifted reads in the
same control flow. Do not let only a subset of ranks enter `ensure_halo!` or a
nonzero shifted operation.

---

### 3) Linear algebra on lattices

Per-site matrix operations follow BLAS-like semantics. The test suite shows full coverage (plain/adjoint inputs, shifted views):

```julia
# Random per-site matrices
A1 = rand(ComplexF64, NC, NC, gsize...)
A2 = rand(ComplexF64, NC, NC, gsize...)
A3 = rand(ComplexF64, NC, NC, gsize...)

M1 = LatticeMatrix(NC, NC, dim, gsize, PEs; nw)
M2 = LatticeMatrix(A2, dim, PEs; nw, numtemps=2)
M3 = LatticeMatrix(A3, dim, PEs; nw)

# Choose a site (using DIndexer + halos)
indexer = DIndexer(gsize)
L = 4
idx_halo = Tuple(delinearize(indexer, L, Int32(nw)))  # with halo offset
idx_core = Tuple(delinearize(indexer, L, Int32(0)))   # core (no halo)

# Reference (host) product at a single site:
a1 = A1[:, :, idx_core...]
a2 = A2[:, :, idx_core...]
a3 = A3[:, :, idx_core...]
mul!(a1, a2, a3)

# Lattice product (device-backed); updates M1.A at that site:
mul!(M1, M2, M3)
m1 = M1.A[:, :, idx_halo...]
@assert a1 ≈ m1

# Matrix exponential at each site (in-place):
expt!(M1, M2, 1)
m1 = M1.A[:, :, idx_halo...]
a1 = exp(a2)
@assert a1 ≈ m1

# Trace and sum over all sites (returns a scalar)
 println(tr(M1))

```

Adjoints and **shifted** operands are supported via wrappers:

```julia
M2p = Shifted_Lattice(M2, (1, 0, 0, 0))    # shift by +1 along X (periodic)
M3p = Shifted_Lattice(M3, (0, 1, 0, 0))    # shift by +1 along Y
mul!(M1, M2', M3p)                          # all combinations in tests:
                                            # (A, B, C), (A, B', C), (A, B, C'), etc.
```

For `nw > 0`, an in-halo shift is a lightweight view and therefore observes later
changes to its source lattice. For `nw == 0`, a nonzero shift is materialized when
`Shifted_Lattice` is constructed, so it is a snapshot. This eager behavior keeps every
public operation safe even though a halo-free lattice has no boundary storage.

If any component of a shift is larger than `nw`, the shift is materialized in
one direct MPI redistribution instead of extending the halo one cell at a time.
The result borrows storage from the source lattice's `PreallocatedArray` pool.
Return that slot deterministically after use:

```julia
long_shift = Shifted_Lattice(M2, (nw + 2, 0, 0, 0))
try
    mul!(M1, long_shift, M3)
finally
    release!(long_shift)          # `close(long_shift)` is equivalent
end

# The scoped helper performs the same release even if the callback throws.
with_shifted_lattice(M2, (nw + 2, 0, 0, 0)) do shifted
    mul!(M1, shifted, M3)
end
```

`release!` is idempotent and is a no-op for lightweight in-halo shifts. A
finalizer on the internal lease is a safety net if a materialized wrapper is
dropped, but explicit/scoped release is preferred because finalizer timing is
nondeterministic. Set the constructor keyword `numtemps` high enough for the
maximum number of simultaneously live long shifts; the pool grows on demand
when exhausted and then reuses those arrays. The constructor rejects
`nw > minimum(local lattice extents)`.

**Convenience**
```julia
# Reduced sums (interior region only)
s = allsum(M)   # MPI.Reduce to root (returns the global sum on rank 0)
```


### 4) Non-QCD example: 3D classical Heisenberg model

As an example outside lattice QCD, the repository includes a simulation of
the classical Heisenberg ferromagnet on a periodic three-dimensional
simple-cubic lattice,

```math
H/J=-\sum_{x,\mu=1}^{3}\boldsymbol{s}(x)\cdot
\boldsymbol{s}(x+\hat\mu), \qquad |\boldsymbol{s}(x)|=1.
```

Each three-component spin is stored as a `3×1` `LatticeMatrix`. The complete
[`examples/classical_heisenberg.jl`](examples/classical_heisenberg.jl)
program uses

- halo exchange for the six nearest neighbors,
- deterministic global-site Philox streams,
- parallel even/odd heat-bath updates,
- optional microcanonical over-relaxation, and
- MPI reductions for the energy, magnetization, and Binder parameter.

Run the default calculation at the literature critical coupling with

```sh
julia --project=. examples/classical_heisenberg.jl
```

For example, a four-rank run can be launched with

```sh
mpiexec -n 4 julia --project=. examples/classical_heisenberg.jl \
    --L=16 --pes=2,2,1 --thermalization=10000 --sweeps=50000
```

The measured Binder parameter is

```math
U_L=1-\frac{\langle m^4\rangle}{3\langle m^2\rangle^2}.
```

At the high-precision critical coupling `K_c=0.693002(2)` reported by
[Deng, Blöte, and Nightingale](https://doi.org/10.1103/PhysRevE.72.016128),
small validation runs gave

| `L` | `U_L` |
| ---: | ---: |
| 8 | `0.6228(23)` |
| 12 | `0.6232(11)` |
| 16 | `0.6226(16)` |

These results are consistent, including statistical and finite-size effects,
with the finite-size-scaling value `U*=0.6217(8)` of
[Holm and Janke](https://doi.org/10.1016/0375-9601(93)90077-D). A separate
two-size Binder-crossing check gave `K_cross=0.69276 ± 0.00155`, also
consistent with the literature critical coupling. Exact checks, run lengths,
error estimation, and limitations are recorded in
[`examples/heisenberg_validation.md`](examples/heisenberg_validation.md).


### 5) Dirac operators

LatticeMatrices.jl currently provides the following fermion operators.  Gauge
links are supplied as a four-element `Vector` of four-dimensional
`LatticeMatrix` objects with per-site shape `NC×NC`.  Wilson and clover
fermions have per-site shape `NC×4`; staggered fermions have shape `NC×1`.

| Type | Meaning | Halo support |
| --- | --- | --- |
| `WilsonDiracOperator4D(U, kappa)` or `WilsonDiracOperator4D(U1, U2, U3, U4, kappa)` | Wilson operator, including the on-site identity term; the explicit-link form is suitable inside AD callbacks | `nw=0` or `nw>=1` |
| `WilsonDiracOperator4D_Donly(U)` | Nearest-neighbor Wilson hopping part only, with coefficient `1/2` and no on-site identity term | `nw=0` or `nw>=1` |
| `WilsonDiracCloverOperator4D(U, kappa, cSW)` | Wilson operator plus the cached four-leaf clover term | `nw=0` or `nw>=1` |
| `StaggeredDiracOperator4D(U, mass)` | Four-dimensional one-link staggered operator in the Bridge++ mass normalization | `nw=0` or `nw>=1` |
| `HISQDiracOperator4D(X, L, mass; naik_epsilon)` or `HISQDiracOperator4D(U, mass; naik_epsilon)` | HISQ stencil for precomputed links, or complete construction from thin links `U` | precomputed: any `nw`; thin-link builder: `nw=0` or `nw>=2` |
| `D5DW_MobiusDomainwallOperator5D(U, L5, mass, M, b, c)` | Five-dimensional Möbius/domain-wall operator with a four-dimensional gauge field | `nw>=1` only |
| `D5DW_GeneralizedDomainwallOperator5D(U, L5, mass, M, a, b, c)` | Generalized domain-wall operator with independent slice coefficients `a_s`, `b_s`, and `c_s` | `nw>=1` only |

All of these operators support both `mul!(out, D, psi)` and
`mul!(out, adjoint(D), psi)`.  Wilson, clover, and domain-wall spinors use the
chiral basis represented by the exported matrices `γ1`, ..., `γ4`; a
staggered field has no explicit spin index beyond its singleton second axis.
Adjoint wrappers are involutive: `adjoint(adjoint(D)) === D` for every
operator listed above.  The same property holds for `DiracOp`, while the
self-adjoint `DdagDOp` satisfies `adjoint(DdagD) === DdagD`.

#### Wilson and Wilson--clover example

The following is a complete single- or multi-rank setup.  It uses unit gauge
links as a small smoke test; consequently its clover field strength is zero.
Replacing `unit_link` with a nontrivial gauge configuration activates the
clover contribution without changing the API.

```julia
using MPI, LinearAlgebra, Random
import JACC

JACC.@init_backend
using LatticeMatrices
MPI.Init()

nprocs = MPI.Comm_size(MPI.COMM_WORLD)
NC = 3
gsize = (4 * nprocs, 4, 4, 4)
PEs = (nprocs, 1, 1, 1)
nw = 1

# U[mu] is an NC×NC gauge matrix at every four-dimensional lattice site.
unit_link = zeros(ComplexF64, NC, NC, gsize...)
for site in CartesianIndices(gsize), color in 1:NC
    unit_link[color, color, Tuple(site)...] = 1
end
U = [LatticeMatrix(unit_link, 4, PEs; nw) for _ in 1:4]

# The second per-site index is the four-component spin index.  Here the time
# direction is antiperiodic for the fermion and the spatial directions are
# periodic.
Random.seed!(1234)
psi_host = randn(ComplexF64, NC, 4, gsize...)
psi = LatticeMatrix(psi_host, 4, PEs;
    nw, phases=(1, 1, 1, -1))
out = similar(psi)

kappa = 0.12

D_wilson = WilsonDiracOperator4D(U, kappa)
mul!(out, D_wilson, psi)
mul!(out, adjoint(D_wilson), psi)

# In a callback differentiated with respect to U1, ..., U4, use the
# explicit-link constructor. It preserves concrete link types on Julia 1.12.
D_wilson_callback = WilsonDiracOperator4D(
    U[1], U[2], U[3], U[4], kappa)

D_clover = WilsonDiracCloverOperator4D(U, kappa, 1.0)
mul!(out, D_clover, psi)
mul!(out, adjoint(D_clover), psi)

# The hopping-only operator is available separately when building composite
# formulations or preconditioners.
D_hopping = WilsonDiracOperator4D_Donly(U)
mul!(out, D_hopping, psi)
```

For `NC=3` with a halo, the Wilson forward, adjoint, and hopping-only paths
use a backend-independent half-spin kernel.  It applies each rank-two Wilson
projector before the SU(3) multiplication and accumulates all eight neighbors
before writing the spinor once.  This optimization does not change the
`LatticeMatrix` memory layout or require CUDA-, ROCm-, or Threads-specific user
code.  The `NC != 3` and `nw=0` implementations remain generic fallbacks.

The Wilson operator is normalized as

```math
(D_W\psi)(x) = \psi(x)
- \kappa\sum_{\mu=1}^{4}\left[
U_\mu(x)(1-\gamma_\mu)\psi(x+\hat\mu)
+ U_\mu^\dagger(x-\hat\mu)(1+\gamma_\mu)\psi(x-\hat\mu)
\right].
```

The clover implementation follows the Bridge++ chiral-basis convention:

```math
D_{\mathrm{clover}} = D_W
- \kappa c_{\mathrm{SW}}\sum_{\mu<\nu}
\gamma_\mu\gamma_\nu F_{\mu\nu}, \qquad
F_{\mu\nu} = \frac{Q_{\mu\nu}-Q_{\mu\nu}^\dagger}{8},
```

where `Q` is the sum of the four plaquettes touching the site in the
`mu`--`nu` plane.  The six anti-Hermitian components are cached in the order
`(12, 13, 14, 23, 24, 34)` and can also be constructed directly:

For `NC=3`, each four-link clover leaf is evaluated as three factorized SU(3)
matrix products.  The generic element-by-element construction remains the
fallback for other color counts.

```julia
field_strength = CloverFieldStrength4D(U)
F12 = field_strength[1]
```

Constructing the clover field is considerably more expensive than applying
its local term, so it is not rebuilt by each `mul!`.  After changing any gauge
link, explicitly refresh the cache before applying the operator again:

```julia
# mutate U with substitute!, a LatticeMatrices mutating operation, or a kernel
update_clover!(D_clover)
mul!(out, D_clover, psi)
```

`update_clover!` is unnecessary while `U` is unchanged.  Direct writes to a
lattice's storage (`U[mu].A`) must additionally be followed by
`mark_halo_dirty!(U[mu])`, as described in the halo-epoch section above.

The link derivative of a Wilson--clover bilinear is also available directly:

```julia
left = psi
dU = [similar(link) for link in U]
clear_matrix!.(dU)
wilson_clover_link_pullback!(dU, D_clover, U, left, psi)
```

`wilson_clover_link_pullback!` accumulates the Wilson hopping and four-leaf
clover contributions into `dU`. It is an analytic backend operation and does
not require an automatic-differentiation package. A halo width of at least one
is required.

#### One-link staggered example (Bridge++ convention)

Continue with `U`, `gsize`, `PEs`, `NC`, and `nw` from the Wilson example.
Only the per-site fermion shape changes from `NC×4` to `NC×1`:

```julia
psi_staggered_host = randn(ComplexF64, NC, 1, gsize...)
psi_staggered = LatticeMatrix(psi_staggered_host, 4, PEs;
    nw, phases=(1, 1, 1, -1))
out_staggered = similar(psi_staggered)

mass = 0.01
D_staggered = StaggeredDiracOperator4D(U, mass)
mul!(out_staggered, D_staggered, psi_staggered)
mul!(out_staggered, adjoint(D_staggered), psi_staggered)
```

The operator is normalized exactly as Bridge++ `Fopr_Staggered`:

```math
(D_{\mathrm{stag}}\psi)(x) = m\psi(x)
+ \frac{1}{2}\sum_{\mu=1}^{4}\eta_\mu(x)
\left[U_\mu(x)\psi(x+\hat\mu)
- U_\mu^\dagger(x-\hat\mu)\psi(x-\hat\mu)\right],
```

with zero-based global coordinates
`eta_mu(x) = (-1)^(sum(x[nu] for nu=1:mu-1))`.  `adjoint(D_staggered)`
changes the sign of the hopping term and keeps the real mass term.  The phase
is evaluated from global MPI coordinates, so odd local extents do not reset
the staggered sign at rank boundaries.

Fermion boundary conditions are supplied through `psi_staggered.phases`; the
example is antiperiodic in time.  Boundary phases must have unit magnitude,
and the gauge links themselves must be periodic. This particular
implementation is the standard one-link/unimproved staggered operator; HISQ
is exposed separately through `HISQDiracOperator4D`.
For the production `nw>=1` path, all eight neighbor terms and the mass are
fused into one JACC kernel, with an unrolled `NC=3` matrix-vector path.  The
`nw=0` fallback is supported but materializes shifted fields and is primarily
useful for compatibility and small tests.

The regression suite compares D and D-dagger against an independent dense
host implementation, adjoint and epsilon-Hermiticity identities, the free
field spectrum, gauge covariance, odd-local-extent MPI decomposition, and
fixed numerical fingerprints generated directly by Bridge++ 2.1.3.  The
Bridge++ oracle source is
[`test/reference/bridgepp_staggered_reference.cpp`](test/reference/bridgepp_staggered_reference.cpp).

#### Complete HISQ smearing and stencil (SIMULATeQCD convention)

The first HISQ smearing level can be constructed directly from periodic thin
links. The complete stencil needs `nw>=3`, so use a separate set of thin
links and a staggered field with that halo width. The smearing sums the 1-,
3-, 5-, and 7-link paths with the SIMULATeQCD
coefficients `1/8`, `1/16`, `1/64`, and `1/384`:

```julia
nw_hisq = 3
U_hisq = [LatticeMatrix(unit_link, 4, PEs; nw=nw_hisq) for _ in 1:4]
psi_hisq = LatticeMatrix(psi_staggered_host, 4, PEs;
    nw=nw_hisq, phases=(1, 1, 1, -1))

V = hisq_fat7_level1(U_hisq)

# An allocation-controlling form is also available.
V_preallocated = [similar(link) for link in U_hisq]
hisq_fat7_level1!(V_preallocated, U_hisq)
```

`V` is the unprojected level-1 field. This builder accepts `nw>=1`; a slower
`nw=0` compatibility path is also provided. Input and output gauge links are
periodic and do not contain staggered or fermion boundary phases.

The complete thin-link builder applies level-1 Fat7, U(N) polar
reunitarization, level-2 Fat7 with the Lepage correction, and the Naik
three-link product:

```julia
epsilon_N = -0.083
hisq_links = hisq_links_from_thin(U_hisq; naik_epsilon=epsilon_N)
D_hisq = HISQDiracOperator4D(
    hisq_links, mass; naik_epsilon=epsilon_N)

# Equivalent convenience constructor.
D_hisq_from_U = HISQDiracOperator4D(
    U_hisq, mass; naik_epsilon=epsilon_N)
```

For repeated construction, all output and work storage can be caller-owned:

```julia
V = [similar(link) for link in U_hisq] # level-1 work
W = [similar(link) for link in U_hisq] # reunitarized work
X = [similar(link) for link in U_hisq] # corrected fat links
L = [similar(link) for link in U_hisq] # forward-anchored Naik links

hisq_links_from_thin!(X, L, V, W, U_hisq; naik_epsilon=epsilon_N)
```

For a Krylov solve, retain all four smearing stages in a transparent cache:

```julia
cache = HISQDiracCache4D(U_hisq, mass; naik_epsilon=epsilon_N)
result = similar(psi_hisq)

mul_cached_hisq!(
    result, cache, U_hisq[1], U_hisq[2], U_hisq[3], U_hisq[4], psi_hisq)
mul_cached_hisq_adjoint!(
    result, cache, U_hisq[1], U_hisq[2], U_hisq[3], U_hisq[4], psi_hisq)

# Analytic thin-link pullback; no Enzyme import is required.
dU_hisq = [similar(link) for link in U_hisq]
clear_matrix!.(dU_hisq)
hisq_link_pullback!(
    dU_hisq, cache, U_hisq, result_cotangent, psi_hisq;
    coefficient=1)
```

The first call after a thin link changes rebuilds level-1, reunitarized, fat,
and Naik links; later calls reuse them. Public lattice mutations advance the
epoch used by this check. After writing through `link.A` directly, call
`mark_halo_dirty!(link)`. For every nonzero-halo color count, the factorized
Fat7 workspace is not global: six same-layout matrix fields are owned by this
`cache` object and reused on every refresh. The cache also owns the primal and
cotangent work fields needed to reverse the factorized stages. Thus ordinary
cached CG/HMC code needs no workspace argument or `runtime_activity=true`
setting.

`hisq_link_pullback!` propagates a Dirac-output cotangent through the Naik
term, both Fat7 levels, and the U(N) projection without Enzyme. It accumulates
into its four destination fields, so clear them before the first contribution
and leave them intact when summing rational or flavor terms. Force evaluation
requires `nw>=3`; `NC=2`, `NC=3`, and `NC=4` use the same public API.
The generic polar pullback solves an `NC^2 × NC^2` static Sylvester system,
so it is intended for the small compile-time color counts used by this
package; the performance-critical `NC=3` projection keeps its specialized
kernel. Fat7 forward and reverse propagation are factorized for all nonzero-
halo color counts, with an additional fully unrolled forward specialization
for `NC=3`.

The optimized fused Dirac stencil still reads the resident halo directly when
`nw>=3`. With precomputed `X` and `L`, `nw=1` and `nw=2` are also correct: the
operator materializes one- and three-hop neighbors through the arbitrary-shift
communication path. This fallback trades additional communication and kernel
launches for a smaller halo. Constructing an operator with `nw<3` emits a
once-per-process performance warning. Complete smearing needs `nw=0` or
`nw>=2`, because level-2 Fat7 and Naik-link construction reach two sites away.

The complete unphased `X` and forward-anchored `L` results, including their
layout-sensitive numerical fingerprints, have been cross-checked against
SIMULATeQCD `HisqSmearing::SmearAll`.

##### End-to-end SIMULATeQCD validation

The complete cached HISQ path was also cross-checked against an independently
built, unmodified SIMULATeQCD tree at commit `767a1b1` (double precision,
CUDA 12.4, NVIDIA H100).  Both codes used the same deterministic, noncommuting
SU(3) gauge field on a `4^4` lattice, mass `m=0.37`, Naik correction
`epsilon_N=-0.083`, temporal antiperiodic boundary conditions, and identical
sources.  The comparison exercised the full two-level smearing and U(3)
projection, Naik links, staggered phases, `D`, `D' * D`, and CG inversion.

The HMC-related contractions `norm(D*eta)^2/V`,
`eta'*(D'*D)*eta/V`, and `eta'*inv(D'*D)*eta/V` agreed with relative errors
no larger than `6.5e-16`.  A point-source Goldstone pseudoscalar correlator,
using the SIMULATeQCD `measureHadrons` M2 contraction,

```math
C(t)=\sum_{\vec{x},c,c'}\left|D^{-1}_{cc'}(\vec{x},t;0)\right|^2,
```

gave the following double-precision values:

| `t` | SIMULATeQCD | LatticeMatrices.jl |
| ---: | ---: | ---: |
| 0 | 1.0065142452009632 | 1.0065142452009666 |
| 1 | 0.12466004270358608 | 0.12466004270358633 |
| 2 | 0.066499548830371014 | 0.066499548830371125 |
| 3 | 0.12509337682203300 | 0.12509337682203336 |

The largest relative correlator difference was `3.4e-15`.  SIMULATeQCD tests
the squared relative CG residual, so its `precision=1e-13` was compared with
`rtol=sqrt(1e-13)` here.  This is a fixed-configuration implementation check;
an ensemble-level physics comparison additionally requires matching gauge
ensembles, source statistics, and taste normalization.

##### GPU smearing implementation

The halo-based Fat7 and Naik builders use fixed-size row kernels. For every
nonzero-halo color count, the allocating complete builder and
`HISQDiracCache4D` factor repeated Fat7 staples into three stages using six
same-layout matrix fields. Caller-owned repeated construction can select the
same path by passing `HISQFat7Workspace` to `hisq_links_from_thin!`. The
physical `NC=3` forward path remains fully unrolled; `NC=2`, `NC=4`, and other
small compile-time color counts use the same factorization with generic static
matrices. This changes neither `LatticeMatrix` storage nor the Dirac
operator's memory layout. `nw=0` retains the direct implementation.
Forward kernels enumerate a matrix row before the lattice site, matching
the first (contiguous) `LatticeMatrix` array dimension. Fat7 pullback reverses
the same three factorized staple stages, accumulating complete static matrices
per site and reusing primal/cotangent intermediates owned by the cache.
The index arithmetic and factorized stages use JACC launch abstractions and
contain no backend-specific thread launch.
The same source path was checked with the JACC Threads backend using four Julia
threads: direct/factorized/U(3) smearing completed ten explicit-GC stress
iterations with maximum difference `5.55e-16`, and the `NC=3` Enzyme pullback
passed all six finite-difference checks. On the validation host, the locally
built OpenMPI links UCX 1.20, whose default error-signal list includes
`SIGSEGV`; that conflicts with Julia task-stack growth independently of JACC
and also crashes plain `Threads.@threads`. For that MPI installation the
threaded command must exclude `SEGV`, for example
`UCX_ERROR_SIGNALS=ILL,BUS,FPE JULIA_NUM_THREADS=4 julia ...`. MPICH_jll passed
the same stress test without this UCX setting.
The backend-neutral JACC `NC=3` forward path uses flattened arrays and fully
unrolled three-color products. The color-generic factorized forward and reverse
paths use compile-time-sized static matrices. They avoid device-side matrix
allocation and complex-valued atomics. No global cache, `CUDA.limit!`, or Enzyme
`runtime_activity=true` setting is used.  `HISQDiracCache4D` is an ordinary
caller-owned object; its epoch check transparently refreshes the derived links
on the first multiply after `U` changes and reuses them in subsequent CG
iterations.

A stress test also completed repeated full rebuilds with CUDA's default
device heap, without any user-side heap configuration.

The cached thin-link force was additionally exercised in a `4^4` HISQ HMC
trajectory with `m=0.5` and `epsilon_N=-0.05`.  Reducing the molecular
dynamics step size at fixed trajectory length gave

| MD steps | `dt` | `Delta H` | `Delta H / dt^2` |
| ---: | ---: | ---: | ---: |
| 1 | 0.05 | -0.393101 | -157.240 |
| 2 | 0.025 | -0.0978934 | -156.629 |
| 4 | 0.0125 | -0.0244497 | -156.478 |
| 8 | 0.00625 | -0.00611096 | -156.441 |

Successive `|Delta H|` ratios were `4.016`, `4.004`, and `4.001`, corresponding
to observed orders `2.006`, `2.001`, and `2.000`.  Thus the force gives the
expected second-order Hamiltonian-error scaling.  Cache refresh counters also
showed one smearing refresh per gauge update, rather than one per CG
iteration; the final action evaluation performed one additional refresh.

The Dirac stencil remains separate from smearing. Supply the corrected fat
links `X[mu]`, which connect `x` to `x+mu`, and the forward-anchored Naik
transporters `L[mu]`, which connect `x` to `x+3mu`:

```julia
# Continue with the caller-owned `X` and `L` built above.

psi_hisq = LatticeMatrix(psi_staggered_host, 4, PEs;
    nw=nw_hisq, phases=(1, 1, 1, -1))
out_hisq = similar(psi_hisq)

hisq_links = HISQLinks4D(X, L)
D_hisq = HISQDiracOperator4D(
    hisq_links, mass; naik_epsilon=epsilon_N)
mul!(out_hisq, D_hisq, psi_hisq)
mul!(out_hisq, adjoint(D_hisq), psi_hisq)
```

The normalization follows SIMULATeQCD `HisqDSlash`:

```math
(D_{\mathrm{HISQ}}\psi)(x) = m\psi(x)
+ \sum_{\mu=1}^{4}\eta_\mu(x)\left\{
\frac{1}{2}\left[X_\mu(x)\psi(x+\hat\mu)
-X_\mu^\dagger(x-\hat\mu)\psi(x-\hat\mu)\right]
-\frac{1+\epsilon_N}{48}\left[L_\mu(x)\psi(x+3\hat\mu)
-L_\mu^\dagger(x-3\hat\mu)\psi(x-3\hat\mu)\right]
\right\}.
```

The fused path and complete halo-based builder require a halo width of at
least three and each local lattice extent must be at least that width. A
halo-free `nw=0` compatibility path is also available.

#### Five-dimensional Möbius and generalized domain-wall examples

The gauge field remains four-dimensional, while the fermion field has a fifth
extent `L5`.  Keeping the fifth process-grid dimension equal to one is the
recommended setup.  Both the conventional Möbius operator and a generalized
operator with independently selectable slice coefficients `a_s`, `b_s`, and
`c_s` are available.

```julia
# Continue with U, gsize, PEs, and NC from the previous example.
L5 = 8
gsize5 = (gsize..., L5)
PEs5 = (PEs..., 1)

psi5_host = randn(ComplexF64, NC, 4, gsize5...)
psi5 = LatticeMatrix(psi5_host, 5, PEs5;
    nw=1, phases=(1, 1, 1, -1, 1))
out5 = similar(psi5)

mass = 0.01
M = -1.0                    # Wilson-kernel mass parameter
b, c = 2.0, 1.0             # scaled-Shamir Möbius kernel
D5 = D5DW_MobiusDomainwallOperator5D(U, L5, mass, M, b, c)

mul!(out5, D5, psi5)
mul!(out5, adjoint(D5), psi5)
```

The constructor keeps the package's legacy `(b,c)` convention.  Its
coefficients in the standard Möbius formula are
`b5=(b+c)/2` and `c5=(b-c)/2`:

| `LatticeMatrices` `(b,c)` | standard/Bridge++ `(b5,c5)` | kernel |
| --- | --- | --- |
| `(1,1)` | `(1,0)` | Shamir |
| `(2,0)` | `(1,1)` | Borici/truncated-overlap |
| `(2,1)` | `(1.5,0.5)` | scaled Shamir |

This mapping follows the Möbius operator of
[Brower, Neff, and Orginos](https://arxiv.org/abs/1206.5214).  The gauge-link
pullback uses the same reduction over fifth-dimensional slices as the
[Bridge++ domain-wall force](https://bridge.kek.jp/Lattice-code/docs/html.2.1.x/force__F__Domainwall_8cpp_source.html).
Five-dimensional operators reject `nw=0`.

The generalized constructor accepts three real vectors of length `L5` and
implements

```math
D_5 = A\left[I-F_m+D_W(B+C F_m)\right],
```

where `A=diag(a)`, `B=diag(b)`, `C=diag(c)`, and `F_m` is the fifth-direction
hop including the boundary mass.  This convention is the generalized
five-dimensional operator used in the
[Lattice 2025 formulation](https://indico.global/event/14504/contributions/137859/attachments/64183/124021/lattice_2025.pdf).
For example, the following uses genuinely slice-dependent coefficients:

```julia
a5 = 1 .+ 0.05 .* sin.(2pi .* (0:L5-1) ./ L5)
b5 = 1.5 .+ 0.10 .* cos.(2pi .* (0:L5-1) ./ L5)
c5 = 0.5 .+ 0.08 .* sin.(2pi .* (0:L5-1) ./ L5)

D5general = D5DW_GeneralizedDomainwallOperator5D(
    U, L5, mass, M, a5, b5, c5)
mul!(out5, D5general, psi5)
mul!(out5, adjoint(D5general), psi5)
```

The coefficient vectors are promoted to a common real type and copied to the
active JACC backend.  The existing Möbius operator is recovered exactly with

```julia
a5 = ones(L5)
b5 = fill((b + c) / 2, L5)
c5 = fill((b - c) / 2, L5)
D5general = D5DW_GeneralizedDomainwallOperator5D(
    U, L5, mass, M, a5, b5, c5)
```

For nonuniform coefficients the adjoint applies the slice factors in the
mathematically required shifted order,
`D5general' = (I-F_m' + (B+F_m'C)D_W')A`; it is not obtained by simply
reusing the forward slice index.

For `NC=3`, both adjoint operators use the same backend-independent two-stage
path.  The first stage computes the four-dimensional Wilson adjoint once for
each fifth-dimensional slice.  The second stage applies the chiral
fifth-direction mixing with one work item per spin-color element, which keeps
adjacent work items aligned with the existing component-major field layout.
The intermediate field is borrowed from `psi.temps` and returned in a
`finally` block, so it is local to the input field rather than a global cache.
The generic single-stage implementation remains the fallback for `NC != 3`.

The optimized path was checked for both Möbius and genuinely nonuniform
generalized coefficients against the dense reference implementation and the
adjoint inner-product identity.  CUDA tests pass on H100, Enzyme reverse-mode
tests pass for `NC=3`, and the same implementation passes the JACC Threads
test with four Julia threads.  The cross-implementation check uses a general,
non-diagonal SU(3) gauge field read independently from the same ILDG file.

Loading Enzyme enables reverse rules for both `D5` and `adjoint(D5)`.  The
following example differentiates `real(dot(left, D5*psi))` with respect to
the four gauge links and `psi`:

```julia
using Enzyme

loss(D, psi, left, out) = (mul!(out, D, psi); real(dot(left, out)))

left = LatticeMatrix(randn(ComplexF64, NC, 4, gsize5...), 5, PEs5;
    nw=1, phases=(1, 1, 1, -1, 1))
dU = [similar(link) for link in U]
dpsi, dout = similar(psi5), similar(out5)
clear_matrix!.(dU)
clear_matrix!.((dpsi, dout))
dD5 = D5DW_MobiusDomainwallOperator5D(dU, L5, mass, M, b, c)

Enzyme.autodiff(
    Enzyme.Reverse, Enzyme.Const(loss), Enzyme.Active,
    enzyme_duplicated(D5, dD5),
    enzyme_duplicated(psi5, dpsi),
    Enzyme.Const(left),
    enzyme_duplicated(out5, dout),
)
```

`enzyme_duplicated` selects `Enzyme.Duplicated` on Julia 1.11 and the
root-safe `Enzyme.MixedDuplicated` calling convention on Julia 1.12 and
later. `Enzyme_derivative!` additionally converts its fixed-size vector
workspaces to tuples when required by Julia 1.12's `MemoryRef` lowering.

For the adjoint operator, pass `adjoint(D5)` and `adjoint(dD5)` as the
primal and shadow operator annotations, respectively.

For the legacy Möbius type, the pullback treats `mass`, `M`, `b`, and `c` as
constants and accumulates cotangents into `dU` and `dpsi`.  The generalized
type additionally accumulates cotangents for all slice coefficients:

```julia
dD5general = D5DW_GeneralizedDomainwallOperator5D(
    dU, L5, mass, M, zeros(L5), zeros(L5), zeros(L5))

Enzyme.autodiff(
    Enzyme.Reverse, Enzyme.Const(loss), Enzyme.Active,
    enzyme_duplicated(D5general, dD5general),
    enzyme_duplicated(psi5, dpsi),
    Enzyme.Const(left),
    enzyme_duplicated(out5, dout),
)

da5 = Array(dD5general.a)
db5 = Array(dD5general.b)
dc5 = Array(dD5general.c)
```

Both pullbacks require `nw>=1`, identical primal and
shadow layouts, and `PEs5[5] == 1`.  The link kernel directly reduces all
`L5` slices into each four-dimensional link element, uses no atomics, and
does not allocate a five-dimensional link-gradient temporary.  Generalized
coefficient gradients use accelerator reductions followed by one MPI
Allreduce of the `3L5` real values.

#### Generic operator wrappers

`DiracOp(U, apply, apply_dag, parameters, prototype)` wraps user-supplied
forward and adjoint kernels in the same `mul!` interface.  Normal operators
and CG use caller-owned temporary fields explicitly, so the low-level solver
does not allocate fields or acquire hidden storage from a pool:

```julia
# Solve (D' * D) x = rhs.  D can be any operator implementing D and D'.
x = similar(rhs)       # also supplies the initial guess
clear_matrix!(x)       # zero initial guess for this example
Dpsi = similar(rhs)    # temporary used while applying D' * D
r = similar(rhs)       # the three CG work fields
p = similar(rhs)
Ap = similar(rhs)

normal = DdagDOp(D, Dpsi)
status = cg!(x, normal, rhs, r, p, Ap;
    rtol=1e-10, atol=0, maxiter=5000)
status.converged || error("CG failed: $(status.reason)")
```

`x`, `rhs`, `Dpsi`, `r`, `p`, and `Ap` must not alias.  `cg!` returns a
`CGResult` containing the convergence flag, iteration count, absolute and
relative residuals, and termination reason.  It does not print or throw on
ordinary non-convergence.  `solve!(x, normal, rhs, r, p, Ap; ...)` is an
equivalent convenience entry point.

The original pool-based interface remains available for compatibility.  It
borrows three fields from `temps`, returns `nothing` on convergence, and
throws on failure:

```julia
LatticeMatrices.cg(x, normal, rhs, temps;
    eps=1e-10, maxsteps=5000, verboselevel=2)
```

The pre-v1 `DiracOp` convenience interfaces are also retained. They borrow
their work fields from `D.phitemps`; the explicit-workspace forms above are
recommended for new code and concurrent applications:

```julia
legacy_normal = DdagDOp(D)
solve!(x, legacy_normal, rhs; verboselevel=2)
S = pseudofermion_action(D, phi)
```

The same operator code runs on the JACC backend selected for the current
project.  The two-rank integration test also covers staggered D/D-dagger on
two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 test/multigpu/run_h100_2gpu.sh
```

## Examples: matrix multiplication on lattices

### 1) Plain matrix multiplication at each lattice site

```julia
using LatticeMatrices, MPI, JACC, LinearAlgebra
JACC.@init_backend
MPI.Init()

dim   = 2
NC    = 3
nprocs = MPI.Comm_size(MPI.COMM_WORLD)
gsize = (8 * nprocs, 8)
PEs   = (nprocs, 1)     # valid for both single- and multi-rank runs

M1 = LatticeMatrix(NC, NC, dim, gsize, PEs)
M2 = LatticeMatrix(rand(ComplexF64, NC, NC, gsize...), dim, PEs)
M3 = LatticeMatrix(rand(ComplexF64, NC, NC, gsize...), dim, PEs)

mul!(M1, M2, M3)        # per-site product: M1 = M2 * M3
```

### 2) Multiplication with a shifted lattice

```julia
shift = (1, 0)                  # shift by +1 along X
M2s = Shifted_Lattice(M2, shift)

mul!(M1, M2s, M3)                # M1 = (M2 shifted) * M3
```

The shift is applied with periodic wrapping across the global lattice size.

---



### 3) Multiplication with conjugate-transposed matrices

```julia
mul!(M1, M2', M3)                # M1 = adjoint(M2) * M3
mul!(M1, M2, M3')                # M1 = M2 * adjoint(M3)
mul!(M1, M2', M3')               # M1 = adjoint(M2) * adjoint(M3)
```

All combinations of shifted and adjoint operands are supported and tested in `test/runtests.jl`.

---

## Automatic differentiation (Enzyme)

Enzyme support is provided as an optional dependency loaded through a package
extension. Install and load Enzyme explicitly when AD is needed. This complete
example differentiates a shifted trace on one or more MPI ranks:

```julia
using Enzyme
using LatticeMatrices, MPI, JACC
JACC.@init_backend
MPI.Init()

nprocs = MPI.Comm_size(MPI.COMM_WORLD)
gsize = (4 * nprocs, 2, 2, 2)
PEs = (nprocs, 1, 1, 1)
host = rand(ComplexF64, 2, 2, gsize...)
U = [LatticeMatrix(host, 4, PEs; nw=1) for _ in 1:4]
set_halo!.(U)

dU = [similar(link) for link in U]
temp = [similar(U[1])]
dtemp = [similar(U[1])]
clear_matrix!.(dU)
clear_matrix!.(temp)
clear_matrix!.(dtemp)

function shifted_trace(U1, U2, U3, U4, temp)
    mul_AshiftB!(temp[1], U1, U2, (1, 0, 0, 0))
    return realtrace(temp[1])
end

Enzyme_derivative!(
    shifted_trace, U[1], U[2], U[3], U[4],
    dU[1], dU[2], dU[3], dU[4]; temp, dtemp)
```

The executable finite-difference, halo-epoch, long-shift pool, and fused
SU(3)-exponential regressions are in `test/mpi_enzyme.jl`.

Note: the AD result here follows Enzyme's complex differentiation convention. For a complex variable
`U = X + iY`, the gradient reported by Enzyme is
`dS/dUij = dS/dXij + i dS/dYij`.

`Enzyme_derivative!` requires `nw >= 1` for every lattice argument and work buffer.
Halo-free (`nw=0`) lattices can be used for ordinary calculations, but are rejected before AD starts.

Custom reverse rules are provided for the staggered and HISQ Dirac stencils
and for every complete HISQ smearing stage: level-1 Fat7, U(N) projection,
level-2 Fat7/Lepage, and Naik links. Consequently a real action can be
differentiated from `HISQDiracOperator4D` all the way back to the thin links.
The complete HISQ AD path requires `nw >= 3`; pass the caller-owned `V`, `W`,
`X`, and `L` work vectors with `enzyme_duplicated` when differentiating
through `hisq_links_from_thin!`.

For HMC code that only needs the cached HISQ thin-link force, prefer the core
`hisq_link_pullback!` API described above. It implements the same complex
gradient convention and is available when Enzyme is not installed or loaded.

`mul_cached_hisq!` has a dedicated static Enzyme reverse rule for the complete
Dirac → Naik/level-2 → U(N) → level-1 → thin-link force chain. The cache is
treated as derived storage, so `runtime_activity=true` is not required and
smearing is not rebuilt on each CG iteration.

On Julia 1.12, use this cached rule when the entire smearing-plus-Dirac chain
must be differentiated in one call. The individual smearing-stage and Dirac
rules also work separately, but Enzyme cannot currently type-analyze a generic
differentiated function that constructs the immutable `HISQDiracOperator4D`
between those stages.

---

## Running the test example

Exactly what `test/runtests.jl` does:

```bash
# CPU single process
julia --project -e 'using Pkg; Pkg.test("LatticeMatrices")'

# MPI (choose ranks and an MPI launcher)
mpiexec -n 4 julia --project test/runtests.jl

# Executable smoke test for the README quick-tour snippets
mpiexec -n 2 julia --project test/readmetest.jl

# Focused two-rank halo/epoch regression used by CI
mpiexec -n 2 julia --project test/mpi_halo.jl

# Focused Enzyme regression (run from a project containing Enzyme)
mpiexec -n 2 julia --project test/mpi_enzyme.jl

# With GPUs (example; make sure CUDA/ROCm works and select a JACC backend)
julia --project -e 'using JACC; JACC.@init_backend; using Pkg; Pkg.test()'
```

Internally, the tests:
- sweep `dim = 1:4` and `NC = 2:4`,
- construct `LatticeMatrix` objects on a Cartesian grid `PEs`,
- verify `mul!` for all nine combinations with/without adjoint and with/without shifts,
- use `DIndexer` to map between linear and multi-indices, including halo offsets.

---

## API reference (selected)

```julia
# Indexing
DIndexer(::NTuple{D,<:Integer})
DIndexer(::AbstractVector{<:Integer})
linearize(::DIndexer{D,dims,strides}, ::NTuple{D,T})::Int32 where {D,T<:Integer}
delinearize(::DIndexer{D,dims,strides}, ::Integer, ::Integer=0)::NTuple{D,Int}
shiftindices(indices, shift)

# Lattice
LatticeMatrix(NC1, NC2, dim, gsize, PEs; nw=1, elementtype=ComplexF64,
              phases=ones(dim), comm0=nothing, numtemps=1,
              device_mapping=:auto, mpi_transport=:auto)
LatticeMatrix(A, dim, PEs; nw=1, phases=ones(dim),
              comm0=nothing, numtemps=1, device_mapping=:auto,
              mpi_transport=:auto)
mpi_transport_info(ls)

set_halo!(ls)
ensure_halo!(ls)
mark_halo_dirty!(ls)
halo_is_dirty(ls)::Bool
halo_epochs(ls)::NamedTuple{(:core, :halo)}
LatticeMatrices.exchange_dim!(ls, d::Int)  # internal single-dimension primitive

gather_matrix(ls; root=0)::Union{Array{T},Nothing}
gather_and_bcast_matrix(ls; root=0)::Array{T}

allsum(ls)  # Reduce(SUM) to root over interior

set_global_component!(ls, value, row, column, global_position)
projected_bilinear_slices(
    propagators1, propagators2, left, right;
    axis=4, origin, momentum, parity_mask, coefficient=-1)
exp_ta_pullback!(output, cotangent, A, t=1)

# Lightweight wrappers
Shifted_Lattice(data, shift)
adjoint(data)
release!(shifted)
with_shifted_lattice(f, data, shift)

# Dirac operators
WilsonDiracOperator4D(U, kappa)
WilsonDiracOperator4D_Donly(U)
WilsonDiracCloverOperator4D(U, kappa, cSW)
StaggeredDiracOperator4D(U, mass)
hisq_fat7_level1(U)
hisq_fat7_level1!(V, U)
HISQFat7Workspace(U[1])
hisq_project_un(V)
hisq_project_un!(W, V)
# Backward-compatible aliases:
hisq_project_u3(V)
hisq_project_u3!(W, V)
hisq_fat7_level2(W; naik_epsilon=0)
hisq_fat7_level2!(X, W; naik_epsilon=0)
hisq_naik_links(W)
hisq_naik_links!(L, W)
hisq_links_from_thin(U; naik_epsilon=0)
hisq_links_from_thin!(X, L, V, W, U; naik_epsilon=0)
HISQLinks4D(X, L)
HISQDiracOperator4D(X, L, mass; naik_epsilon=0)
HISQDiracOperator4D(U, mass; naik_epsilon=0)
HISQDiracCache4D(U, mass; naik_epsilon=0)
update_hisq_cache!(cache, U)
mul_cached_hisq!(out, cache, U1, U2, U3, U4, psi)
mul_cached_hisq_adjoint!(out, cache, U1, U2, U3, U4, psi)
hisq_link_pullback!(dU, cache, U, result_cotangent, psi; coefficient=1)
CloverFieldStrength4D(U)
update_clover!(field_strength, U)
update_clover!(clover_operator)
D5DW_MobiusDomainwallOperator5D(U, L5, mass, M, b, c)
D5DW_GeneralizedDomainwallOperator5D(U, L5, mass, M, a, b, c)

# Callback-based operator composition and solver helpers
DiracOp(U, apply, apply_dag, parameters, prototype;
        numtemp=4, numphitemp=4)
DdagDOp(D, Dpsi)
cg!(x, A, rhs, r, p, Ap; rtol=1e-10, atol=0, maxiter=5000)
solve!(x, DdagD, rhs, r, p, Ap; rtol=1e-10, atol=0, maxiter=5000)
pseudofermion_action(D, phi, eta, Deta, r, p, Ap)

# Enzyme annotation for lattice objects and fixed-size workspaces
enzyme_duplicated(primal, shadow)

# Compatibility interface using a PreallocatedArray pool
LatticeMatrices.cg(x, A, rhs, temps;
                   eps=1e-10, maxsteps=5000, verboselevel=2)
DdagDOp(D)
solve!(x, DdagD, rhs; verboselevel=2)
pseudofermion_action(D, phi)
```

---


## License

MIT (see `LICENSE`).

---

## Acknowledgements

LatticeMatrices.jl is built on the excellent Julia HPC stack: **MPI.jl**,
**[JACC.jl](https://github.com/JuliaORNL/JACC.jl)**, and the Julia standard
libraries. In particular, we sincerely thank the JACC.jl developers for the
performance-portable kernel abstraction that enables the same lattice code to
run on threaded CPUs and multiple GPU backends.

---

### References

- MPI.jl: https://github.com/JuliaParallel/MPI.jl  
- JACC.jl: https://github.com/JuliaORNL/JACC.jl
- Sheikholeslami--Wohlert improved Wilson action: https://doi.org/10.1016/0550-3213(85)90002-1
- Kogut--Susskind staggered fermions: https://doi.org/10.1103/PhysRevD.11.395
- Bridge++ source and releases: https://bridge.kek.jp/Lattice-code/source.html
- Bridge++ 2.1.x documentation: https://bridge.kek.jp/Lattice-code/docs/html.2.1.x/index.html



---

## Selecting & switching GPU/CPU backends (via JACC.jl)

LatticeMatrices.jl uses [JACC.jl](https://github.com/JuliaORNL/JACC.jl) for
performance-portable execution. Follow JACC’s recommended flow to select
**one** backend per project/session:

1) **Set a backend** (writes/updates `LocalPreferences.toml` and adds the backend package):
```julia
julia> import JACC
julia> JACC.set_backend("cuda")     # or "amdgpu" or "threads" (default)
```
2) **Initialize at top level** so your code doesn’t need backend‑specific imports:
```julia
import JACC
JACC.@init_backend                  # must be at top-level scope
```

3) **Switching backends.** Re-run `JACC.set_backend("amdgpu")` (or `"threads"`, `"cuda"`) in the same project to switch; this updates `LocalPreferences.toml`. Restart your Julia session so extensions load for the new backend, then call `JACC.@init_backend` again.

> Notes:
> - Without calling `@init_backend`, using a non-`"threads"` backend will raise
>   errors like `get_backend(::Val(:cuda))` when invoking JACC functions.
> - `JACC.array` / `JACC.array_type()` help you stay backend‑agnostic in your APIs.


References: JACC quick start and usage in the upstream README.  

---

## Citation

If you use LatticeMatrices.jl in research, please cite the following papers:

- Yuki Nagai and Akio Tomiya, [“JuliaQCD: Portable lattice QCD package in
  Julia language”](https://arxiv.org/abs/2409.03030), arXiv:2409.03030.
- Yuki Nagai, Akio Tomiya, and Hiroshi Ohno, [“Lattice Gauge Theory via
  LLVM-Level Automatic Differentiation”](https://arxiv.org/abs/2602.20516),
  arXiv:2602.20516.
