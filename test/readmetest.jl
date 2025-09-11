using LatticeMatrices

using LatticeMatrices, MPI, JACC, LinearAlgebra
import JACC
JACC.@init_backend

function main()
    # Build an indexer for a D-dimensional lattice (1-based indices)
    gsize = (16, 16, 16, 16)     # global lattice size
    d = DIndexer(gsize)          # computes row-major "strides" internally

    # Convert between linear and multi-index (1-based)
    L = linearize(d, (1, 1, 1, 1))   # -> 1
    ix = delinearize(d, 4)            # -> (4, 1, 1, 1) on this shape

    # Apply periodic shifts componentwise
    p = shiftindices((4, 1, 1, 1), (1, 0, 0, 0))   # -> (5, 1, 1, 1)


    MPI.Init()

    dim = 4
    gsize = ntuple(_ -> 16, dim)   # global spatial size per dimension
    nw = 1                      # ghost width
    NC = 3                      # per-site matrix size (NC×NC)

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
    #display(a1)
    #display(m1)

    @assert a1 ≈ m1

    expt!(M1, M2, 1)
    m1 = M1.A[:, :, idx_halo...]
    m2 = M2.A[:, :, idx_halo...]
    display(m2)
    display(m1)
    a1 = exp(a2)

    display(a1)


    @assert a1 ≈ m1




    NC = 3

    M1 = LatticeMatrix(NC, NC, dim, gsize, PEs)
    M2 = LatticeMatrix(rand(ComplexF64, NC, NC, gsize...), dim, PEs)
    M3 = LatticeMatrix(rand(ComplexF64, NC, NC, gsize...), dim, PEs)

    mul!(M1, M2, M3)        # per-site product: M1 = M2 * M3

    shift = (1, 0, 0, 0)[1:dim]                  # shift by +1 along X
    M2s = Shifted_Lattice(M2, shift)

    mul!(M1, M2s, M3)                # M1 = (M2 shifted) * M3

    mul!(M1, M2', M3)                # M1 = adjoint(M2) * M3
    mul!(M1, M2, M3')                # M1 = M2 * adjoint(M3)
    mul!(M1, M2', M3')               # M1 = adjoint(M2) * adjoint(M3)

    println(tr(M1))


end
main()