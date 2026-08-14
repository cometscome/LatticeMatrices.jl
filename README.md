# LatticeMatrices.jl

[![Build Status](https://github.com/cometscome/MPILattice.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cometscome/MPILattice.jl/actions/workflows/CI.yml?query=branch%3Amain)

High-performance **matrix fields on arbitrary D-dimensional lattices** in Julia.

- Per-site matrices (size `NC1×NC2`) stored in **column-major layout**:  
  `(NC1, NC2, X, Y, Z, …)`
- **MPI** domain decomposition via a Cartesian communicator (halo width `nw`, periodic BCs).
- **GPU-ready** through **[JACC.jl](https://github.com/JuliaORNL/JACC.jl)** (portable CPU/GPU kernels; CUDA/ROCm/Threads).
- Fast, allocation-free **indexing helpers** for kernels: `DIndexer`, `linearize`, `delinearize`, `shiftindices`.
- Lattice-QCD fermion operators including Wilson, clover, staggered, and a **complete HISQ thin-link smearing and Dirac pipeline**.

> This package focuses on scalable, halo-exchange–based lattice algorithms with minimal allocations and clean multi-backend execution.

**Applications**: This package is designed to support large-scale simulations on structured lattices. A key application area is lattice QCD, where gauge fields and fermion fields are represented as matrix-valued objects on a multi-dimensional lattice. In future developments, LatticeMatrices.jl is planned to be integrated into [Gaugefields.jl](https://github.com/akio-tomiya/Gaugefields.jl) and [LatticeDiracOperators.jl](https://github.com/akio-tomiya/LatticeDiracOperators.jl), providing the underlying data structures and linear algebra kernels for gauge and fermion dynamics.



**Current limitation.** Multi‑GPU execution and hybrid MPI+threads parallelism are **experimental** and **not yet thoroughly tested**; treat them as provisional.


---

## Installation

```julia
pkg> add LatticeMatrices
```

Requirements:
- Julia ≥ 1.11

---

## Quick tour

### 1) D-dimensional indexing helpers (GPU-kernel friendly)

```julia
using LatticeMatrices

# Build an indexer for a D-dimensional lattice (1-based indices)
gsize = (16, 16, 16, 16)     # global lattice size
d = DIndexer(gsize)          # computes row-major "strides" internally

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
linearize(::DIndexer{D,dims,strides}, idx::NTuple{D,Int32})::Int32
delinearize(::DIndexer{D,dims,strides}, L::Integer, offset::Int32=0)::NTuple{D,Int32}

# elementwise shifting for index tuples
shiftindices(indices, shift)
```

- `delinearize(...; offset)` is handy to **map into halo regions**, e.g. pass `offset = nw`.

---

### 2) Lattice containers (MPI + halos + JACC arrays)

The core container stores a **halo-padded** array on each rank and manages halo exchange without MPI derived datatypes (faces are packed into contiguous buffers).

```julia
using LatticeMatrices, MPI, JACC, LinearAlgebra
JACC.@init_backend
MPI.Init()

dim   = 4
gsize = ntuple(_ -> 16, dim)   # global spatial size per dimension
nw    = 1                      # ghost width
NC    = 3                      # per-site matrix size (NC×NC)

# Choose a Cartesian process grid (PEs) of length `dim`
nprocs = MPI.Comm_size(MPI.COMM_WORLD)
n1 = max(nprocs ÷ 2, 1)
PEs = ntuple(i -> i == 1 ? n1 : (i == 2 ? nprocs ÷ n1 : 1), dim)

# Construct an empty lattice matrix (device array via JACC.zeros)
M = LatticeMatrix(NC, NC, dim, gsize, PEs; nw, elementtype=ComplexF64)

# Or initialize from an existing array (broadcast to ranks)
A = rand(ComplexF64, NC, NC, gsize...)
M2 = LatticeMatrix(A, dim, PEs; nw)

# Halo exchange across all spatial dimensions
set_halo!(M)

# Global gather helpers (host reconstruction on rank 0)
G = gather_matrix(M; root=0)                # rank 0: Array(NC, NC, gsize...)
Gall = gather_and_bcast_matrix(M; root=0)   # all ranks receive the same Array
```

**Key type**
```julia
struct LatticeMatrix{D,T,AT,NC1,NC2,nw,DI} <: Lattice{D,T,AT}
    nw::Int
    phases::SVector{D,T}         # per-direction phase (applied at wrap boundaries)
    NC1::Int
    NC2::Int
    gsize::NTuple{D,Int}
    cart::MPI.Comm               # Cartesian communicator
    coords::NTuple{D,Int}        # 0-based Cartesian coords
    dims::NTuple{D,Int}          # process grid (PEs)
    nbr::NTuple{D,NTuple{2,Int}} # neighbors (minus, plus)
    A::AT                        # local array (NC1, NC2, X, Y, Z, …) with halos
    buf::Vector{AT}              # four face buffers per spatial dim
    myrank::Int
    PN::NTuple{D,Int}            # local interior size per dim (no halos)
    comm::MPI.Comm               # original communicator
    indexer::DI                  # DIndexer for global sizes
end
```

**Constructors**
```julia
LatticeMatrix(NC1, NC2, dim, gsize, PEs;
              nw=1, elementtype=ComplexF64, phases=ones(dim), comm0=MPI.COMM_WORLD)

LatticeMatrix(A, dim, PEs; nw=1, phases=ones(dim), comm0=MPI.COMM_WORLD)
```

- **Layout**: `(NC1, NC2, X, Y, Z, …)`; halos are the outer `nw` cells on each spatial dim.
- **Phases**: wrap-around phases per dimension. A positive-direction wrap applies `phase`,
  while a negative-direction wrap applies `inv(phase)`.
- **Exchange**: `set_halo!(ls)` calls `exchange_dim!(ls, d)` for each spatial dimension `d`.

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
M2 = LatticeMatrix(A2, dim, PEs; nw)
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
mul!(M1, M2', M3p)                          # all combinations in tests:
                                            # (A, B, C), (A, B', C), (A, B, C'), etc.
```

For `nw > 0`, an in-halo shift is a lightweight view and therefore observes later
changes to its source lattice. For `nw == 0`, a nonzero shift is materialized when
`Shifted_Lattice` is constructed, so it is a snapshot. This eager behavior keeps every
public operation safe even though a halo-free lattice has no boundary storage.

**Convenience**
```julia
# Reduced sums (interior region only)
s = allsum(M)   # MPI.Reduce to root (returns the global sum on rank 0)
```


### 4) Dirac operators

LatticeMatrices.jl currently provides the following fermion operators.  Gauge
links are supplied as a four-element `Vector` of four-dimensional
`LatticeMatrix` objects with per-site shape `NC×NC`.  Wilson and clover
fermions have per-site shape `NC×4`; staggered fermions have shape `NC×1`.

| Type | Meaning | Halo support |
| --- | --- | --- |
| `WilsonDiracOperator4D(U, kappa)` | Wilson operator, including the on-site identity term | `nw=0` or `nw>=1` |
| `WilsonDiracOperator4D_Donly(U)` | Nearest-neighbor Wilson hopping part only, with coefficient `1/2` and no on-site identity term | `nw=0` or `nw>=1` |
| `WilsonDiracCloverOperator4D(U, kappa, cSW)` | Wilson operator plus the cached four-leaf clover term | `nw=0` or `nw>=1` |
| `StaggeredDiracOperator4D(U, mass)` | Four-dimensional one-link staggered operator in the Bridge++ mass normalization | `nw=0` or `nw>=1` |
| `HISQDiracOperator4D(X, L, mass; naik_epsilon)` or `HISQDiracOperator4D(U, mass; naik_epsilon)` | HISQ stencil for precomputed links, or complete construction from thin links `U` | `nw=0` or `nw>=3` |
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

D_clover = WilsonDiracCloverOperator4D(U, kappa, 1.0)
mul!(out, D_clover, psi)
mul!(out, adjoint(D_clover), psi)

# The hopping-only operator is available separately when building composite
# formulations or preconditioners.
D_hopping = WilsonDiracOperator4D_Donly(U)
mul!(out, D_hopping, psi)
```

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
links. It sums the 1-, 3-, 5-, and 7-link paths with the SIMULATeQCD
coefficients `1/8`, `1/16`, `1/64`, and `1/384`:

```julia
V = hisq_fat7_level1(U)

# An allocation-controlling form is also available.
V_preallocated = [similar(link) for link in U]
hisq_fat7_level1!(V_preallocated, U)
```

`V` is the unprojected level-1 field. This builder accepts `nw>=1`; a slower
`nw=0` compatibility path is also provided. Input and output gauge links are
periodic and do not contain staggered or fermion boundary phases.

The complete thin-link builder applies level-1 Fat7, U(3) polar
reunitarization, level-2 Fat7 with the Lepage correction, and the Naik
three-link product:

```julia
epsilon_N = -0.083
hisq_links = hisq_links_from_thin(U; naik_epsilon=epsilon_N)
D_hisq = HISQDiracOperator4D(
    hisq_links, mass; naik_epsilon=epsilon_N)

# Equivalent convenience constructor.
D_hisq_from_U = HISQDiracOperator4D(
    U, mass; naik_epsilon=epsilon_N)
```

For repeated construction, all output and work storage can be caller-owned:

```julia
V = [similar(link) for link in U] # level-1 work
W = [similar(link) for link in U] # reunitarized work
X = [similar(link) for link in U] # corrected fat links
L = [similar(link) for link in U] # forward-anchored Naik links

hisq_links_from_thin!(X, L, V, W, U; naik_epsilon=epsilon_N)
```

For a Krylov solve, retain all four smearing stages in a transparent cache:

```julia
cache = HISQDiracCache4D(U, mass; naik_epsilon=epsilon_N)
result = similar(psi)

mul_cached_hisq!(
    result, cache, U[1], U[2], U[3], U[4], psi)
mul_cached_hisq_adjoint!(
    result, cache, U[1], U[2], U[3], U[4], psi)
```

The first call after a thin link changes rebuilds level-1, reunitarized, fat,
and Naik links; later calls reuse them. Public lattice mutations advance the
epoch used by this check. After writing through `link.A` directly, call
`mark_halo_dirty!(link)`.

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

The Dirac stencil remains separate from smearing. Supply the corrected fat
links `X[mu]`, which connect `x` to `x+mu`, and the forward-anchored Naik
transporters `L[mu]`, which connect `x` to `x+3mu`:

```julia
nw_hisq = 3
X = [LatticeMatrix(X_host[mu], 4, PEs; nw=nw_hisq) for mu in 1:4]
L = [LatticeMatrix(L_host[mu], 4, PEs; nw=nw_hisq) for mu in 1:4]

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

#### Five-dimensional Möbius/domain-wall example

The gauge field remains four-dimensional, while the fermion field has a fifth
extent `L5`.  Keeping the fifth process-grid dimension equal to one is the
recommended setup.

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
    Enzyme.Duplicated(D5, dD5),
    Enzyme.Duplicated(psi5, dpsi),
    Enzyme.Const(left),
    Enzyme.Duplicated(out5, dout),
)
```

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
    Enzyme.Duplicated(D5general, dD5general),
    Enzyme.Duplicated(psi5, dpsi),
    Enzyme.Const(left),
    Enzyme.Duplicated(out5, dout),
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
Allreduce of the `3L5` real values.  A CUDA correctness/performance driver is
available as `test/multigpu/domainwall_pullback_bench.jl`.

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

The same operator code runs on the JACC backend selected for the current
project.  For the CUDA correctness check and staggered benchmark, run:

```bash
CUDA_VISIBLE_DEVICES=0 \
LATTICEMATRICES_STAGGERED_BENCH_L=24 \
LATTICEMATRICES_STAGGERED_BENCH_PRECISION=Float32 \
julia --project=test/multigpu test/multigpu/staggered_bench.jl
```

Set the precision to `Float64` for double precision.  The script reports
milliseconds per D/D-dagger application, lattice sites per second, the
Bridge++ flop-count convention, and a minimum-traffic bandwidth estimate.
The two-rank integration test also covers staggered D/D-dagger on two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 test/multigpu/run_h100_2gpu.sh
```

For the Wilson--clover CUDA benchmark, run:

```bash
julia --project=test/multigpu test/multigpu/wilson_clover_bench.jl
```


## Examples: matrix multiplication on lattices

### 1) Plain matrix multiplication at each lattice site

```julia
using LatticeMatrices, MPI, JACC, LinearAlgebra
JACC.@init_backend
MPI.Init()

dim   = 2
gsize = (8, 8)
NC    = 3
PEs   = (2, 2)          # process grid (2×2)

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

(above v0.3: experimental) Enzyme is an optional dependency loaded through a package extension.
Install and load Enzyme explicitly when AD is needed. We provide Enzyme-based AD extensions and test cases. See `test/adtest/ad.jl` for a concrete comparison between
automatic differentiation and numerical differentiation using `calc_action_loopfn`. The loop body is factored
into a small helper function (`_calc_action_step!`), which makes Enzyme AD more reliable for loop-heavy code.

Example (runs the AD vs numerical comparison with `calc_action_loopfn`):

```julia
using Enzyme
using LatticeMatrices, MPI, JACC
JACC.@init_backend
MPI.Init()

include("test/adtest/ad.jl") # runs main() in the script
```

Note: the AD result here follows Enzyme's complex differentiation convention. For a complex variable
`U = X + iY`, the gradient reported by Enzyme is
`dS/dUij = dS/dXij + i dS/dYij`.

`Enzyme_derivative!` requires `nw >= 1` for every lattice argument and work buffer.
Halo-free (`nw=0`) lattices can be used for ordinary calculations, but are rejected before AD starts.

Custom reverse rules are provided for the staggered and HISQ Dirac stencils
and for every complete HISQ smearing stage: level-1 Fat7, U(3) projection,
level-2 Fat7/Lepage, and Naik links. Consequently a real action can be
differentiated from `HISQDiracOperator4D` all the way back to the thin links.
The complete HISQ AD path requires `nw >= 3`; pass the caller-owned `V`, `W`,
`X`, and `L` work vectors as `Enzyme.Duplicated` arguments when differentiating
through `hisq_links_from_thin!`.

`mul_cached_hisq!` has a dedicated static Enzyme reverse rule for the complete
Dirac → Naik/level-2 → U(3) → level-1 → thin-link force chain. The cache is
treated as derived storage, so `runtime_activity=true` is not required and
smearing is not rebuilt on each CG iteration.

---

## Running the test example

Exactly what `test/runtests.jl` does:

```bash
# CPU single process
julia --project -e 'using Pkg; Pkg.test("LatticeMatrices")'

# MPI (choose ranks and an MPI launcher)
mpiexec -n 4 julia --project test/runtests.jl

# Focused two-rank halo/epoch regression used by CI
mpiexec -n 2 julia --project test/mpi_halo.jl

# With GPUs (example; make sure CUDA/ROCm works and select a JACC backend)
julia --project -e 'using JACC; JACC.@init_backend; using Pkg; Pkg.test()'
```

Internally, the tests:
- sweep `dim = 1:4` and `NC = 2:4`,
- construct `LatticeMatrix` objects on a Cartesian grid `PEs`,
- verify `mul!` for all nine combinations with/without adjoint and with/without shifts,
- use `DIndexer` to map between linear and multi-indices, including halo offsets.

The epoch overhead benchmark has no extra package dependencies:

```bash
# Single rank
julia --project benchmark/halo_epochs.jl

# Include real inter-rank halo communication
mpiexec -n 2 julia --project benchmark/halo_epochs.jl
```

It reports the median time for `mark_halo_dirty!`, a clean `ensure_halo!`, a
dirty synchronization, and an unconditional `set_halo!`. Environment variables
`LM_BENCH_FAST_ITERS`, `LM_BENCH_SYNC_ITERS`, `LM_BENCH_SAMPLES`, and
`LM_BENCH_LOCAL_X` control the workload.

---

## API reference (selected)

```julia
# Indexing
DIndexer(::NTuple{D,<:Integer})
DIndexer(::AbstractVector{<:Integer})
linearize(::DIndexer{D,dims,strides}, ::NTuple{D,Int32})::Int32
delinearize(::DIndexer{D,dims,strides}, ::Integer, ::Int32=0)::NTuple{D,Int32}
shiftindices(indices, shift)

# Lattice
LatticeMatrix(NC1, NC2, dim, gsize, PEs; nw=1, elementtype=ComplexF64,
              phases=ones(dim), comm0=MPI.COMM_WORLD)
LatticeMatrix(A, dim, PEs; nw=1, phases=ones(dim), comm0=MPI.COMM_WORLD)

set_halo!(ls)
ensure_halo!(ls)
mark_halo_dirty!(ls)
halo_is_dirty(ls)::Bool
halo_epochs(ls)::NamedTuple{(:core, :halo)}
exchange_dim!(ls, d::Int)

gather_matrix(ls; root=0)::Union{Array{T},Nothing}
gather_and_bcast_matrix(ls; root=0)::Array{T}

allsum(ls)  # Reduce(SUM) to root over interior

# Lightweight wrappers
struct Shifted_Lattice{D,shift}; data::D; end
struct Adjoint_Lattice{D};       data::D; end
# Base.adjoint(::Lattice) and Base.adjoint(::Shifted_Lattice) return Adjoint_Lattice

# Dirac operators
WilsonDiracOperator4D(U, kappa)
WilsonDiracOperator4D_Donly(U)
WilsonDiracCloverOperator4D(U, kappa, cSW)
StaggeredDiracOperator4D(U, mass)
hisq_fat7_level1(U)
hisq_fat7_level1!(V, U)
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

# Compatibility interface using a PreallocatedArray pool
LatticeMatrices.cg(x, A, rhs, temps;
                   eps=1e-10, maxsteps=5000, verboselevel=2)
```

---


## License

MIT (see `LICENSE`).

---

## Acknowledgements

Built on the excellent Julia HPC stack: **MPI.jl**, **JACC.jl**, and the Julia standard libraries.

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

LatticeMatrices.jl uses [JACC.jl] for performance‑portable execution. Follow JACC’s
recommended flow to select **one** backend per project/session:

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
