module LatticeMatricesMPIExt
using LatticeMatrices
using MPI
using StaticArrays
using JACC
const comm = MPI.COMM_WORLD
export comm


import LatticeMatrices: get_nprocs, reduce_sum, barrier,
    get_coords_r, isend_L, irecv_L!, recv_L!, get_myrank, get_comm, LatticeMatrix_construct,
    get_globalrange, exchange_dim!, _ghostMatrix, _faceMatrix, _mul_phase!, compute_interior!

export LatticeMatrix_standard

# ---------------------------------------------------------------------------
# container  (faces / derived datatypes are GONE)
# ---------------------------------------------------------------------------
#struct LatticeMatrix{D,T,AT,NC1,NC2,nw} <: Lattice{D,T,AT}
struct LatticeMatrix_standard{D,T,AT,NC1,NC2,nw,DI} <: LatticeMatrix{D,T,AT,NC1,NC2,nw,DI} #Lattice{D,T,AT,NC1,NC2,nw}
    nw::Int                          # ghost width
    phases::SVector{D,T}                 # phases
    NC1::Int
    NC2::Int                        # internal DoF
    gsize::NTuple{D,Int}                # global size

    cart::MPI.Comm
    coords::NTuple{D,Int}
    dims::NTuple{D,Int}
    nbr::NTuple{D,NTuple{2,Int}}

    A::AT                           # main array (NC first)
    buf::Vector{AT}                   # 2D work buffers (minus/plus)
    myrank::Int
    PN::NTuple{D,Int}
    comm::MPI.Comm
    indexer::DI
    #stride::NTuple{D,Int}
end



@inline function get_comm(comm0)
    if comm0 === nothing
        return comm
    else
        return comm0
    end
end

@inline function get_nprocs(x::TL) where {TL<:LatticeMatrix_standard}
    return MPI.Comm_size(x.cart)
end

@inline function get_myrank(x::TL) where {TL<:LatticeMatrix_standard}
    return x.myrank
end


@inline function barrier(x::TL) where {TL<:LatticeMatrix_standard}
    MPI.Barrier(x.cart)
end

@inline function reduce_sum(x::TL, b, rank) where {TL<:LatticeMatrix_standard}
    return MPI.Reduce(b, MPI.SUM, rank, x.cart)
end

@inline function get_coords_r(x::TL) where {TL<:LatticeMatrix_standard}
    return MPI.Cart_coords(x.cart, x.myrank)
end

@inline function get_coords_r(x::TL, r) where {TL<:LatticeMatrix_standard}
    return MPI.Cart_coords(x.cart, r)
end

@inline function isend_L(x::TL, bufSM, rankM, d) where {TL<:LatticeMatrix_standard}
    return MPI.Isend(bufSM, rankM, d, x.cart)
end

@inline function irecv_L!(x::TL, bufRM, rankM, d) where {TL<:LatticeMatrix_standard}
    return MPI.Irecv!(bufRM, rankM, d, x.cart)
end

@inline function recv_L!(x::TL, recvbuf, r, tag) where {TL<:LatticeMatrix_standard}
    MPI.Recv!(recvbuf, r, tag, x.cart)
end

@inline function send_L(x::TL, sendbuf, root, tag) where {TL<:LatticeMatrix_standard}
    MPI.Send(sendbuf, root, tag, x.cart)
end

@inline function bcast_L!(x::TL, G, root) where {TL<:LatticeMatrix_standard}
    MPI.Bcast!(G, root, x.cart)
end




function Base.similar(ls::TL) where {D,T,AT,NC1,NC2,DI,nw,TL<:LatticeMatrix_standard{D,T,AT,NC1,NC2,nw,DI}}
    return LatticeMatrix_standard{D,T,AT,NC1,NC2,nw,DI}(ls.nw,
        ls.phases,
        ls.NC1,
        ls.NC2,
        ls.gsize,
        ls.cart,
        ls.coords,
        ls.dims,
        ls.nbr,
        zero(ls.A),
        ls.buf,
        ls.myrank,
        ls.PN,
        ls.comm,
        ls.indexer)
end


function LatticeMatrix_construct(NC1, NC2, dim, gsize, PEs; nw=1, elementtype=ComplexF64, phases=ones(dim), comm0=MPI.COMM_WORLD)
    return LatticeMatrix_standard(NC1, NC2, dim, gsize, PEs; nw, elementtype, phases, comm0)
end
# ---------------------------------------------------------------------------
# constructor + heavy init (still cheap to call)
# ---------------------------------------------------------------------------
function LatticeMatrix_standard(NC1, NC2, dim, gsize, PEs; nw=1, elementtype=ComplexF64, phases=ones(dim), comm0=MPI.COMM_WORLD)

    # Cartesian grid
    D = dim
    T = elementtype
    dims = PEs #MPI.dims_create(MPI.Comm_size(MPI.COMM_WORLD), D)
    periodic = ntuple(_ -> true, D)
    #println(dims)
    #println(periodic)
    cart = MPI.Cart_create(comm0, dims; periodic=periodic)
    coords = MPI.Cart_coords(cart, MPI.Comm_rank(cart))

    #comm  = MPI.Cart_create(MPI.COMM_WORLD, dims; periods=ntuple(_->true,D))
    #coords= MPI.Cart_coords(cart, MPI.Comm_rank(cart))
    nbr = ntuple(d -> ntuple(s -> MPI.Cart_shift(cart, d - 1, ifelse(s == 1, -1, 1))[2], 2), D)
    # local array (NC first)
    #println(gsize)
    locS = ntuple(i -> gsize[i] ÷ dims[i] + 2nw, D)
    loc = (NC1, NC2, locS...)
    A = JACC.zeros(T, loc...)
    #stride = ntuple(i -> (i == 1 ? 1 : prod(locS[1:i-1])), D)

    # contiguous buffers for each face
    buf = Vector{typeof(A)}(undef, 4D)
    for d in 1:D
        shp = ntuple(i -> i == d ? nw : locS[i], D)   # halo slab shape
        buf[4d-3] = JACC.zeros(T, (NC1, NC2, shp...)...)  # minus side
        buf[4d-2] = JACC.zeros(T, (NC1, NC2, shp...)...)  # plus  side
        buf[4d-1] = JACC.zeros(T, (NC1, NC2, shp...)...)  # minus side
        buf[4d] = JACC.zeros(T, (NC1, NC2, shp...)...)  # plus  side
    end


    PN = ntuple(i -> gsize[i] ÷ dims[i], D)
    #println("LatticeMatrix: $dims, $gsize, $PN, $nw")
    #indexer = DIndexer(gsize)
    indexer = DIndexer(PN)
    DI = typeof(indexer)

    #return LatticeMatrix{D,T,typeof(A),NC1,NC2,nw}(nw, phases, NC1, NC2, gsize,
    #    cart, Tuple(coords), dims, nbr,
    #    A, buf, MPI.Comm_rank(cart), PN, comm0)
    return LatticeMatrix_standard{D,T,typeof(A),NC1,NC2,nw,DI}(nw, phases, NC1, NC2, gsize,
        cart, Tuple(coords), dims, nbr,
        A, buf, MPI.Comm_rank(cart), PN, comm0, indexer)
end

function LatticeMatrix_construct(A, dim, PEs; nw=1, phases=ones(dim), comm0=MPI.COMM_WORLD)
    return LatticeMatrix_standard(A, dim, PEs; nw, phases, comm0)
end

function LatticeMatrix_standard(A, dim, PEs; nw=1, phases=ones(dim), comm0=MPI.COMM_WORLD)

    NC1, NC2, NN... = size(A)
    #println(NN)
    elementtype = eltype(A)

    @assert dim == length(NN) "Dimension mismatch: expected $dim, got $(length(NN))"
    #if dim == 1
    #    gsize = (NN,)
    #else
    #    gsize = NN
    #end
    gsize = NN

    ls = LatticeMatrix(NC1, NC2, dim, gsize, PEs; elementtype, nw, phases, comm0)
    MPI.Bcast!(A, ls.cart)
    Acpu = Array(ls.A)

    idx = ntuple(i -> (i == 1 || i == 2) ? Colon() : (ls.nw+1):(size(ls.A, i)-ls.nw), dim .+ 2)



    idx_global = ntuple(i -> (i == 1 || i == 2) ? Colon() : get_globalrange(ls, i - 2), dim .+ 2)

    #println(idx)
    #=
    for i = 1:MPI.Comm_size(ls.cart)
        if ls.myrank == i
            println(get_globalrange(ls, 1))
        end
        MPI.Barrier(ls.cart)
    end
    =#



    #println(idx_global)
    Acpu[idx...] = A[idx_global...]
    #println(Acpu)


    Agpu = JACC.array(Acpu)
    ls.A .= Agpu

    set_halo!(ls)
    #println(ls.A)

    return ls

    #coords_r = MPI.Cart_coords(ls.cart, ls.myrank)
    # 0-based coords
    #println(coords_r)

end

##############################################################################
# exchange_dim!  –  no-derived-datatype version that never aliases buffers
#                   (works with MPI.jl v0.20.x)
#
#  * four contiguous buffers per spatial dimension:
#        bufSM (send minus), bufRM (recv minus),
#        bufSP (send plus) , bufRP (recv plus)
#  * send-buffers are filled with `_faceMatrix`, optionally phase-multiplied,
#    then passed to MPI.Isend
#  * recv-buffers are passed to MPI.Irecv!  and finally copied into `_ghostMatrix`
##############################################################################
function exchange_dim!(ls::LatticeMatrix{D}, d::Int) where D
    # buffer indices
    iSM, iRM = 4d - 3, 4d - 2
    iSP, iRP = 4d - 1, 4d

    bufSM, bufRM = ls.buf[iSM], ls.buf[iRM]      # minus side: send / recv
    bufSP, bufRP = ls.buf[iSP], ls.buf[iRP]      # plus  side: send / recv

    rankM, rankP = ls.nbr[d]                     # neighbour ranks
    me = ls.myrank
    reqs = MPI.Request[]

    # --- self-neighbor on BOTH sides (happens iff dims[d] == 1) -------------
    if rankM == me && rankP == me
        # minus ghost <= plus face
        copy!(_ghostMatrix(ls.A, ls.nw, d, :minus),
            _faceMatrix(ls.A, ls.nw, d, :plus))
        _mul_phase!(_ghostMatrix(ls.A, ls.nw, d, :minus), ls.phases[d])

        # plus  ghost <= minus face
        copy!(_ghostMatrix(ls.A, ls.nw, d, :plus),
            _faceMatrix(ls.A, ls.nw, d, :minus))
        _mul_phase!(_ghostMatrix(ls.A, ls.nw, d, :plus), ls.phases[d])

        # no MPI in the self-case; just compute the interior and return
        compute_interior!(ls)
        return
    end


    #baseT = MPI.Datatype(eltype(ls.A))           # elementary datatype
    #println("M ", rankM, "\t $me")
    barrier(ls)
    #MPI.Barrier(ls.cart)
    #println("P ", rankP, "\t $me")
    # ---------------- minus direction -------------------
    if rankM == me
        copy!(_ghostMatrix(ls.A, ls.nw, d, :minus),
            _faceMatrix(ls.A, ls.nw, d, :minus))
        if ls.coords[d] == 0                     # wrap ⇒ phase
            _mul_phase!(_ghostMatrix(ls.A, ls.nw, d, :minus), ls.phases[d])
        end
    else
        copy!(bufSM, _faceMatrix(ls.A, ls.nw, d, :minus))
        if ls.coords[d] == 0
            _mul_phase!(bufSM, ls.phases[d])
        end

        cnt = length(bufSM)

        #push!(reqs, MPI.Isend(bufSM, rankM, d, ls.cart))#;
        push!(reqs, isend_L(ls, bufSM, rankM, d))
        #count=cnt, datatype=baseT))

        #push!(reqs, MPI.Irecv!(bufRM, rankM, d + D, ls.cart))#;
        push!(reqs, irecv_L!(ls, bufRM, rankM, d + D))#;
        #count=cnt, datatype=baseT))
    end

    # ---------------- plus direction --------------------
    if rankP == me
        copy!(_ghostMatrix(ls.A, ls.nw, d, :plus),
            _faceMatrix(ls.A, ls.nw, d, :plus))
        if ls.coords[d] == ls.dims[d] - 1
            _mul_phase!(_ghostMatrix(ls.A, ls.nw, d, :plus), ls.phases[d])
        end
    else
        copy!(bufSP, _faceMatrix(ls.A, ls.nw, d, :plus))
        if ls.coords[d] == ls.dims[d] - 1
            _mul_phase!(bufSP, ls.phases[d])
        end

        cnt = length(bufSP)

        push!(reqs, isend_L(ls, bufSP, rankP, d + D))
        #push!(reqs, MPI.Isend(bufSP, rankP, d + D, ls.cart))#;
        #count=cnt, datatype=baseT))
        #push!(reqs, MPI.Irecv!(bufRP, rankP, d, ls.cart))
        push!(reqs, irecv_L!(ls, bufRP, rankP, d))
        #count=cnt, datatype=baseT))
    end

    # -------- overlap bulk computation -----------------
    compute_interior!(ls)
    isempty(reqs) || MPI.Waitall!(reqs)

    # -------- copy received data into ghosts -----------
    if rankM != me
        copy!(_ghostMatrix(ls.A, ls.nw, d, :minus), bufRM)
    end
    if rankP != me
        copy!(_ghostMatrix(ls.A, ls.nw, d, :plus), bufRP)
    end
end


end

