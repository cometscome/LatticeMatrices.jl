# LatticeMatrices.jl

[![Build Status](https://github.com/cometscome/MPILattice.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cometscome/MPILattice.jl/actions/workflows/CI.yml?query=branch%3Amain)

High-performance **matrix fields on arbitrary D-dimensional lattices** in Julia.

- Per-site matrices (size `NC1×NC2`) stored in **column-major layout**:  
  `(NC1, NC2, X, Y, Z, …)`
- **MPI** domain decomposition via a Cartesian communicator (halo width `nw`, periodic BCs).
- **GPU-ready** through **[JACC.jl]** (portable CPU/GPU kernels; CUDA/ROCm/Threads).
- Fast, allocation-free **indexing helpers** for kernels: `DIndexer`, `linearize`, `delinearize`, `shiftindices`.

> This package focuses on scalable, halo-exchange–based lattice algorithms with minimal allocations and clean multi-backend execution.

**Applications**: This package is designed to support large-scale simulations on structured lattices. A key application area is lattice QCD, where gauge fields and fermion fields are represented as matrix-valued objects on a multi-dimensional lattice. In future developments, LatticeMatrices.jl is planned to be integrated into [Gaugefields.jl](https://github.com/akio-tomiya/Gaugefields.jl) and [LatticeDiracOperators.jl](https://github.com/akio-tomiya/LatticeDiracOperators.jl), providing the underlying data structures and linear algebra kernels for gauge and fermion dynamics.



**Current limitation.** Multi‑GPU execution and hybrid MPI+GPU parallelism are **experimental** and **not yet thoroughly tested**; treat them as provisional.


---

## Installation

```julia
pkg> add https://github.com/cometscome/LatticeMatrices.jl
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

# Apply periodic shifts componentwise
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
- **Phases**: wrap-around phases per dimension (applied on the boundary faces during exchange).  
- **Exchange**: `set_halo!(ls)` calls `exchange_dim!(ls, d)` for each spatial dimension `d`.

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

Adjoints and **shifted** operands are supported via lightweight wrappers:

```julia
M2p = Shifted_Lattice(M2, (1, 0, 0, 0))    # shift by +1 along X (periodic)
mul!(M1, M2', M3p)                          # all combinations in tests:
                                            # (A, B, C), (A, B', C), (A, B, C'), etc.
```

**Convenience**
```julia
# Reduced sums (interior region only)
s = allsum(M)   # MPI.Reduce to root (returns the global sum on rank 0)
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

### 3) Multiplication with conjugate-transposed matrices

```julia
mul!(M1, M2', M3)                # M1 = adjoint(M2) * M3
mul!(M1, M2, M3')                # M1 = M2 * adjoint(M3)
mul!(M1, M2', M3')               # M1 = adjoint(M2) * adjoint(M3)
```

All combinations of shifted and adjoint operands are supported and tested in `test/runtests.jl`.

---


---

## Running the test example

Exactly what `test/runtests.jl` does:

```bash
# CPU single process
julia --project -e 'using Pkg; Pkg.test("LatticeMatrices")'

# MPI (choose ranks and an MPI launcher)
mpiexec -n 4 julia --project test/runtests.jl

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
linearize(::DIndexer{D,dims,strides}, ::NTuple{D,Int32})::Int32
delinearize(::DIndexer{D,dims,strides}, ::Integer, ::Int32=0)::NTuple{D,Int32}
shiftindices(indices, shift)

# Lattice
LatticeMatrix(NC1, NC2, dim, gsize, PEs; nw=1, elementtype=ComplexF64,
              phases=ones(dim), comm0=MPI.COMM_WORLD)
LatticeMatrix(A, dim, PEs; nw=1, phases=ones(dim), comm0=MPI.COMM_WORLD)

set_halo!(ls)
exchange_dim!(ls, d::Int)

gather_matrix(ls; root=0)::Union{Array{T},Nothing}
gather_and_bcast_matrix(ls; root=0)::Array{T}

allsum(ls)  # Reduce(SUM) to root over interior

# Lightweight wrappers
struct Shifted_Lattice{D,shift}; data::D; end
struct Adjoint_Lattice{D};       data::D; end
# Base.adjoint(::Lattice) and Base.adjoint(::Shifted_Lattice) return Adjoint_Lattice
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
