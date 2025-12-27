##############################################################################
#  LatticeMatrix (no derived datatypes version)
#  --------------------------------------
#  * column-major layout   :  (NC , X , Y , …)
#  * halo width            :  nw
#  * per–direction phases  :  φ
#  * internal DoF          :  NC  (fastest dim)
#  * ALWAYS packs faces into contiguous buffers and sends them as
#    plain arrays (no MPI_Type_create_subarray, no commit/free hustle).
#
#  Back-end: CPU threads / CUDA / ROCm via JACC.
##############################################################################
using StaticArrays, JACC



abstract type LatticeMatrix{D,T,AT,NC1,NC2,nw,DI} <: Lattice{D,T,AT,NC1,NC2,nw} end

function get_comm end

function LatticeMatrix_construct end

# ---------------------------------------------------------------------------
# constructor + heavy init (still cheap to call)
# ---------------------------------------------------------------------------
function LatticeMatrix(NC1, NC2, dim, gsize, PEs; nw=1, elementtype=ComplexF64, phases=ones(dim), comm0=nothing)
    if isdefined(Base, :get_extension)
        ext = Base.get_extension(LatticeMatrices, :LatticeMatricesMPIExt)
        if ext !== nothing
            #println("MPI extension is available")
            comm_i = get_comm(comm0)
            return LatticeMatrix_construct(NC1, NC2, dim, gsize, PEs; nw, elementtype, phases, comm0=comm_i)
            #return LatticeMatrix_standard(NC1, NC2, dim, gsize, PEs; nw, elementtype, phases, comm_i)
        end
    end

end

function LatticeMatrix(A, dim, PEs; nw=1, phases=ones(dim), comm0=nothing)
    if isdefined(Base, :get_extension)
        ext = Base.get_extension(LatticeMatrices, :LatticeMatricesMPIExt)
        if ext !== nothing
            #println("MPI extension is available")
            comm_i = get_comm(comm0)
            return LatticeMatrix_construct(A, dim, PEs; nw, phases, comm0=comm_i)
        end
    end
end



function Base.similar(ls::TL) where {D,T,AT,NC1,NC2,TL<:LatticeMatrix{D,T,AT,NC1,NC2}}
    return LatticeMatrix(NC1, NC2, D, ls.gsize, ls.dims; nw=ls.nw, elementtype=T, phases=ls.phases, comm0=ls.comm)
end

function get_nprocs(x)
    error("get_nprocs is not implemented for $(typeof(x))")
end

function get_myrank(x)
    error("get_myrank is not implemented for $(typeof(x))")
end

function barrier(x)
    error("barrier is not implemented for $(typeof(x))")
end

function reduce_sum(x, b, rank)
    error("reducesum is not implemented for $(typeof(x))")
end

function get_coords_r(x)
    error("get_coords_r is not implemented for $(typeof(x))")
end

function get_coords_r(x, r)
    error("get_coords_r is not implemented for $(typeof(x))")
end

function isend_L(x, bufSM, rankM, d)
    error("isend_L is not implemented for $(typeof(x))")
end

function irecv_L!(x, bufRP, rankP, d)
    error("irecv_L! is not implemented for $(typeof(x))")
end

function recv_L!(x, recvbuf, r, tag)
    error("recv_L! is not implemented for $(typeof(x))")
end

function send_L(x, sendbuf, root, tag)
    error("send_L is not implemented for $(typeof(x))")
end

function bcast_L!(x, G, root)
    error("bcast_L! is not implemented for $(typeof(x))")
end


function Base.display(ls::TL) where {T,AT,NC1,NC2,TL<:LatticeMatrix{4,T,AT,NC1,NC2}}

    NN = size(ls.A)
    nprocs = get_nprocs(ls)

    for rank = 0:nprocs-1 #MPI.Comm_size(ls.cart)-1
        if ls.myrank == rank
            println("LatticeMatrix (rank $rank):")
            indices = map(d -> get_globalrange(ls, d), 1:4)
            println("Global indices: ", indices)
            #println(ls.nw+1:NN[4]-ls.nw)
            for it in 1:ls.PN[4]
                for iz in 1:ls.PN[3]
                    for iy in 1:ls.PN[2]
                        for ix in 1:ls.PN[1]
                            println((indices[1][ix], indices[2][iy], indices[3][iz], indices[4][it]))
                            display(ls.A[:, :, ls.nw+ix, ls.nw+iy, ls.nw+iz, ls.nw+it])
                            #print("$(ls.A[:, :, ix, iy, iz, it]) ")
                        end
                    end
                end
            end
            #display(ls.A[:, :, ls.nw+1:end-ls.nw, ls.nw+1:end-ls.nw, ls.nw+1:end-ls.nw, ls.nw+1:end-ls.nw])
        end
        barrier(ls)
        #MPI.Barrier(ls.cart)
    end
end



function allsum(ls::TL) where {D,T,AT,NC1,NC2,TL<:LatticeMatrix{D,T,AT,NC1,NC2}}
    NN = ls.PN
    indices = ntuple(i -> (i == 1 || i == 2) ? Colon() : (ls.nw+1):(ls.nw+NN[i-2]), D + 2)
    # sum all elements in the local array
    local_sum = sum(ls.A[indices...])
    #local_sum = sum(ls.A[:, :, ls.nw+1:ls.nw+NN[1], ls.nw+1:ls.nw+NN[2], ls.nw+1:ls.nw+NN[3], ls.nw+1:ls.nw+NN[4]])
    # reduce to all processes
    global_sum = reduce_sum(ls, local_sum, 0)#.cart)
    #global_sum = MPI.Reduce(local_sum, MPI.SUM, 0, ls.cart)
    return global_sum
end

export allsum

function get_globalrange(ls::TL, dim) where {TL<:LatticeMatrix}
    #coords_r = MPI.Cart_coords(ls.cart, ls.myrank)
    coords_r = get_coords_r(ls)
    istart = get_globalindex(ls, 1, dim, coords_r[dim])
    #if dim == 1
    #    println(" $( ls.PN[dim])")
    # end
    iend = get_globalindex(ls, ls.PN[dim], dim, coords_r[dim])
    return istart:iend
end

function get_globalindex(ls::TL, i, dim, myrank_dim) where {D,T,AT,NC1,NC2,nw,DI,TL<:LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}}
    ix = i + ls.PN[dim] * myrank_dim
    return ix
end



function set_halo!(ls::TL) where {D,T,AT,NC1,NC2,nw,DI,TL<:LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}}
    for id = 1:D
        exchange_dim!(ls, id)
    end
end
export set_halo!

# ---------------------------------------------------------------------------
# helpers that build proper “view tuples” without parsing errors
# ---------------------------------------------------------------------------
"""
    _faceMatrix(A, nw, d, side)

Return a view of the halo–1 slab (width = `nw`) in spatial
dimension `d` on `side = :minus | :plus`.

* Array ordering is `(NC1, NC2, X, Y, Z, …)` so the spatial
  dimension maps to index `d + 2`.
"""
function _faceMatrix(A, nw, d, side::Symbol)
    # (1) decide the range WITHOUT the ternary-inside-range trick
    face_rng = if side === :minus
        (nw+1):(2*nw)
    else
        sz = size(A, d + 2)
        (sz-2*nw+1):(sz-nw)
    end

    # (2) build an indexing tuple, replacing only index d+2
    idx = ntuple(i -> i == d + 2 ? face_rng : Colon(), ndims(A))
    @views return A[idx...]            # a view, no copy
end

"""
    _ghostMatrix(A, nw, d, side)

Return a `@view` of the *internal* ghost layer (width `nw`) for
dimension `d` on the requested `side`.
"""
function _ghostMatrix(A, nw, d, side::Symbol)
    ghost_rng = if side === :minus
        1:nw
    else
        sz = size(A, d + 2)
        (sz-nw+1):sz
    end

    idx = ntuple(i -> i == d + 2 ? ghost_rng : Colon(), ndims(A))
    @views return A[idx...]
end


function exchange_dim! end

# ---------------------------------------------------------------------------
# hooks (user overrides)
# ---------------------------------------------------------------------------
compute_interior!(ls::LatticeMatrix) = nothing
compute_boundary!(ls::LatticeMatrix) = nothing

export LatticeMatrix

# ---------------------------------------------------------------------------
# gather_matrix: collect local (halo-stripped) blocks to rank=0
# Reconstruct a global array of shape (NC1, NC2, gsize...)
# Communication is done on host memory for portability (CPU/GPU back-ends).
# ---------------------------------------------------------------------------
function gather_matrix(ls::TL;
    root::Int=0) where {D,T,AT,NC1,NC2,TL<:LatticeMatrix{D,T,AT,NC1,NC2}}
    #comm = ls.cart
    #me = ls.myrank
    me = get_myrank(ls)
    nprocs = get_nprocs(ls)
    #nprocs = MPI.Comm_size(comm)

    # 1) Build view of the interior block (without halos)
    #    Spatial dims are shifted by +2 because array layout = (NC1, NC2, X, Y, Z, ...)
    interior_idx = ntuple(i -> (i <= 2 ? Colon() : (ls.nw+1):(ls.nw+ls.PN[i-2])), D + 2)
    @views local_view = ls.A[interior_idx...]   # a view on device/host
    local_block_cpu = Array(local_view)        # ensure host memory for MPI

    # Flatten to 1D send buffer for simple point-to-point
    sendbuf = reshape(local_block_cpu, :)
    count = length(sendbuf)

    # Helper: place a received block into the correct global offsets
    # coords are 0-based along each cart dimension
    function _place_block!(G, block, coords::NTuple{D,Int})
        # Compute global spatial ranges for this coords
        ranges = ntuple(d -> begin
                start = coords[d] * ls.PN[d] + 1
                stop = start + ls.PN[d] - 1
                start:stop
            end, D)
        # Build indexing tuple = (Colon, Colon, ranges...)
        idx = (Colon(), Colon(), ranges...)
        @views G[idx...] = block
        return nothing
    end

    if me == root
        # 2) Allocate the global array on root
        gshape = (ls.NC1, ls.NC2, ls.gsize...)
        G = Array{T}(undef, gshape)

        # 2a) Place root's own block
        _place_block!(G, reshape(sendbuf, size(local_block_cpu)), ls.coords)

        # 2b) Receive all other ranks and place
        #     For simplicity use a fixed tag per direction.
        tag = 900
        recvbuf = similar(sendbuf)  # reuse buffer
        for r in 0:nprocs-1
            r == root && continue
            #MPI.Recv!(recvbuf, r, tag, comm)
            recv_L!(ls, recvbuf, r, tag)
            #coords_r = Tuple(MPI.Cart_coords(comm, r))  # 0-based coords
            coords_r = Tuple(get_coords_r(ls, r))  # 0-based coords
            blk = reshape(recvbuf, size(local_block_cpu))
            _place_block!(G, blk, coords_r)
        end
        return G
    else
        # Non-root: send and return nothing
        tag = 900
        send_L(ls, sendbuf, root, tag)
        #MPI.Send(sendbuf, root, tag, comm)
        return nothing
    end
end

export gather_matrix

# ---------------------------------------------------------------------------
# gather_and_bcast_matrix:
#   Collect halo-stripped blocks to root, reconstruct global matrix,
#   then broadcast it so all ranks receive the same array.
#   Returns Array{T}(NC1, NC2, gsize...)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# gather_and_bcast_matrix:
#   Collect local halo-free blocks to root, reconstruct global matrix on root,
#   then broadcast the global matrix so that ALL ranks return the same Array.
# ---------------------------------------------------------------------------
function gather_and_bcast_matrix(ls::TL;
    root::Int=0) where {D,T,AT,NC1,NC2,TL<:LatticeMatrix{D,T,AT,NC1,NC2}}
    #comm = ls.cart
    #me = ls.myrank
    #nprocs = MPI.Comm_size(comm)
    me = get_myrank(ls)
    nprocs = get_nprocs(ls)

    # --- 1) local interior (no halo) on HOST ---
    interior_idx = ntuple(i -> (i <= 2 ? Colon() : (ls.nw+1):(ls.nw+ls.PN[i-2])), D + 2)
    @views local_view = ls.A[interior_idx...]
    local_block_cpu = Array(local_view)              # host buffer
    sendbuf = reshape(local_block_cpu, :)

    # helper to place a block at correct global offsets
    function _place_block!(G, block, coords::NTuple{D,Int})
        ranges = ntuple(d -> begin
                s = coords[d] * ls.PN[d] + 1
                e = s + ls.PN[d] - 1
                s:e
            end, D)
        idx = (Colon(), Colon(), ranges...)
        @views G[idx...] = block
        return nothing
    end

    G = nothing
    if me == root
        # --- 2) reconstruct on root ---
        gshape = (ls.NC1, ls.NC2, ls.gsize...)
        G = Array{T}(undef, gshape)

        # root’s own block
        _place_block!(G, reshape(sendbuf, size(local_block_cpu)), ls.coords)

        # receive others
        recvbuf = similar(sendbuf)
        for r in 0:nprocs-1
            r == root && continue
            #MPI.Recv!(recvbuf, r, 900, comm)
            recv_L!(ls, recvbuf, r, 900)
            #coords_r = Tuple(MPI.Cart_coords(comm, r))
            coords_r = Tuple(get_coords_r(ls, r))  # 0-based coords
            blk = reshape(recvbuf, size(local_block_cpu))
            _place_block!(G, blk, coords_r)
        end
    else
        # non-root: send local block
        #MPI.Send(sendbuf, root, 900, comm)
        send_L(ls, sendbuf, root, 900)
    end

    # --- 3) broadcast ONLY the data (shape is deterministic) ---
    gshape = (ls.NC1, ls.NC2, ls.gsize...)   # same on all ranks
    if me != root
        G = Array{T}(undef, gshape)          # allocate receive buffer
    end
    #MPI.Bcast!(G, root, comm)                # broadcast the global array
    bcast_L!(ls, G, root)
    return G
end
export gather_and_bcast_matrix

@inline _mul_phase!(buf, ϕ) =
    JACC.parallel_for(length(buf)) do i
        buf[i] *= ϕ
    end
