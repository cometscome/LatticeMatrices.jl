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
using PreallocatedArrays

include("LatticeScratchPool.jl")

abstract type LatticeMatrix{D,T,AT,NC1,NC2,nw,DI} <: Lattice{D,T,AT,NC1,NC2,nw} end

# ---------------------------------------------------------------------------
# container  (faces / derived datatypes are GONE)
# ---------------------------------------------------------------------------
mutable struct HaloEpoch
    core::UInt64
    halo::UInt64
end

HaloEpoch() = HaloEpoch(0, 0)

mutable struct DirectShiftHostBuffers{T}
    send::Vector{T}
    recv::Vector{T}
end

DirectShiftHostBuffers(::Type{T}) where {T} =
    DirectShiftHostBuffers{T}(Vector{T}(), Vector{T}())

@inline _prepare_mpi_host_buffer(::Any, buffer::Array) = buffer

@enum _MPITransportRoute::UInt8 begin
    _LOCAL
    _MPI_HOST_DIRECT
    _MPI_HOST_STAGED
    _MPI_DEVICE_DIRECT
end

struct MPITransportConfig
    requested::Symbol
    resolved::_MPITransportRoute
    backend::Symbol
    reason::Symbol
end

@inline _mpi_device_buffer_supported(::Any) = false
@inline _mpi_device_aware_available(::Any) = false
@inline _mpi_device_kind(::Any) = Symbol(JACC.backend)

# Most accelerator arrays support the compact prefix views used by the
# optimized halo exchange. Backends that do not can opt into the legacy full
# face buffers from their package extension without changing CUDA/ROCm paths.
@inline _uses_full_halo_buffers(::Any) = false

@inline function _mpi_transport_symbol(route::_MPITransportRoute)
    route === _LOCAL && return :local
    route === _MPI_HOST_DIRECT && return :host_direct
    route === _MPI_HOST_STAGED && return :host_staged
    return :device_direct
end

@inline _uses_direct_mpi(config::MPITransportConfig) =
    config.resolved === _MPI_HOST_DIRECT ||
    config.resolved === _MPI_DEVICE_DIRECT

function _validate_mpi_transport(mpi_transport)
    mpi_transport isa Symbol || throw(ArgumentError(
        "mpi_transport must be :auto, :host_staged, or :device_direct, " *
        "got $(repr(mpi_transport))"))
    mpi_transport in (:auto, :host_staged, :device_direct) || throw(ArgumentError(
        "mpi_transport must be :auto, :host_staged, or :device_direct, " *
        "got $(repr(mpi_transport))"))
    return mpi_transport
end

@inline function _mpi_transport_code(mpi_transport::Symbol)
    mpi_transport === :auto && return Int32(1)
    mpi_transport === :host_staged && return Int32(2)
    return Int32(3)
end

function _resolve_mpi_transport(mpi_transport, array, comm)
    requested = _validate_mpi_transport(mpi_transport)
    backend = Symbol(JACC.backend)

    if !_is_mpi_communicator(comm)
        requested === :device_direct && throw(ArgumentError(
            "mpi_transport=:device_direct requires an MPI communicator; " *
            "the serial communicator performs no inter-process transport"))
        return MPITransportConfig(
            requested, _LOCAL, backend, :serial_communicator)
    end

    # Construction is already collective because it creates a Cartesian
    # communicator. Detect a mismatched command-line/configuration choice here
    # instead of letting ranks enter different communication paths later.
    code = _mpi_transport_code(requested)
    minimum_code = _allreduce_min(code, comm)
    maximum_code = _allreduce_max(code, comm)
    minimum_code == maximum_code || throw(ArgumentError(
        "all MPI ranks must use the same mpi_transport setting"))

    local_host = array isa Array
    all_host = _allreduce_min(local_host ? Int32(1) : Int32(0), comm) == 1
    any_host = _allreduce_max(local_host ? Int32(1) : Int32(0), comm) == 1
    if all_host
        requested === :device_direct && throw(ArgumentError(
            "mpi_transport=:device_direct requires accelerator arrays; " *
            "CPU arrays are already passed directly to MPI"))
        return MPITransportConfig(
            requested, _MPI_HOST_DIRECT, backend, :host_array)
    end
    any_host && throw(ArgumentError(
        "mixing host and accelerator arrays in one LatticeMatrix communicator " *
        "is not supported"))

    requested === :host_staged && return MPITransportConfig(
        requested, _MPI_HOST_STAGED, backend, :explicit_host_staging)

    local_buffer_supported = _mpi_device_buffer_supported(array)
    local_runtime_available = local_buffer_supported &&
        _mpi_device_aware_available(array)
    all_buffer_supported = _allreduce_min(
        local_buffer_supported ? Int32(1) : Int32(0), comm) == 1
    all_runtime_available = _allreduce_min(
        local_runtime_available ? Int32(1) : Int32(0), comm) == 1

    if requested === :device_direct
        all_buffer_supported || throw(ArgumentError(
            "mpi_transport=:device_direct is not supported for the $backend " *
            "array type; use mpi_transport=:host_staged"))
        all_runtime_available || throw(ArgumentError(
            "mpi_transport=:device_direct was requested for " *
            "$(_mpi_device_kind(array)), but $(_mpi_library_info(comm).library) did not report " *
            "device-aware support; configure a matching GPU-aware system MPI " *
            "or use mpi_transport=:host_staged"))
        return MPITransportConfig(
            requested, _MPI_DEVICE_DIRECT, backend, :explicit_device_direct)
    end

    if all_buffer_supported && all_runtime_available
        return MPITransportConfig(
            requested, _MPI_DEVICE_DIRECT, backend, :device_aware_detected)
    elseif !all_buffer_supported
        return MPITransportConfig(
            requested, _MPI_HOST_STAGED, backend, :mpi_buffer_unsupported)
    else
        return MPITransportConfig(
            requested, _MPI_HOST_STAGED, backend, :device_aware_not_detected)
    end
end

function _ensure_direct_shift_host_buffers!(
    buffers::DirectShiftHostBuffers{T}, count::Int, device_array,
) where {T}
    if length(buffers.send) < count
        buffers.send = _prepare_mpi_host_buffer(
            device_array, Vector{T}(undef, count))
    end
    if length(buffers.recv) < count
        buffers.recv = _prepare_mpi_host_buffer(
            device_array, Vector{T}(undef, count))
    end
    return buffers
end

#struct LatticeMatrix{D,T,AT,NC1,NC2,nw} <: Lattice{D,T,AT}
struct LatticeMatrix_standard{D,T,AT,NC1,NC2,nw,DI,C} <: LatticeMatrix{D,T,AT,NC1,NC2,nw,DI} #Lattice{D,T,AT,NC1,NC2,nw}
    nw::Int                          # ghost width
    phases::SVector{D,T}                 # phases
    NC1::Int
    NC2::Int                        # internal DoF
    gsize::NTuple{D,Int}                # global size

    cart::C
    coords::NTuple{D,Int}
    dims::NTuple{D,Int}
    nbr::NTuple{D,NTuple{2,Int}}

    A::AT                           # main array (NC first)
    buf::Vector{AT}                   # 2D work buffers (minus/plus)
    buf_host::Vector{Array{T}}      # host send/receive arrays (backend may pin)
    shift_buf_host::DirectShiftHostBuffers{T}
    mpi_transport::MPITransportConfig

    myrank::Int
    PN::NTuple{D,Int}
    comm::C
    indexer::DI
    temps::LatticeScratchPool{AT}
    halo_epoch::HaloEpoch
    #stride::NTuple{D,Int}
end

@inline _array_payload_bytes(array) = length(array) * sizeof(eltype(array))

@inline function _arrays_payload_bytes(arrays)
    bytes = 0
    for array in arrays
        bytes += _array_payload_bytes(array)
    end
    return bytes
end

"""
    lattice_memory_report(lattice)

Return a named tuple describing the array payload memory owned by `lattice`.
The report separates the main lattice (including halo padding), lazily allocated
scratch fields, backend halo buffers, and host staging buffers. Julia object
headers and backend allocator overhead are intentionally not included.
"""
function lattice_memory_report(ls::LatticeMatrix)
    data_bytes = _array_payload_bytes(ls.A)
    core_data_bytes = ls.NC1 * ls.NC2 * prod(ls.PN) * sizeof(eltype(ls.A))

    scratch_pool = _scratch_inner(ls.temps)
    scratch_bytes = scratch_pool === nothing ? 0 :
        _arrays_payload_bytes(scratch_pool._data)
    halo_backend_buffer_bytes = _arrays_payload_bytes(ls.buf)
    halo_host_buffer_bytes = _arrays_payload_bytes(ls.buf_host)
    direct_shift_host_buffer_bytes =
        _array_payload_bytes(ls.shift_buf_host.send) +
        _array_payload_bytes(ls.shift_buf_host.recv)

    backend_array_bytes =
        data_bytes + scratch_bytes + halo_backend_buffer_bytes
    host_auxiliary_bytes =
        halo_host_buffer_bytes + direct_shift_host_buffer_bytes

    return (
        core_data_bytes,
        data_bytes,
        halo_padding_bytes=data_bytes - core_data_bytes,
        scratch_bytes,
        scratch_capacity=scratch_capacity(ls.temps),
        scratch_inuse=scratch_inuse(ls.temps),
        halo_backend_buffer_bytes,
        halo_host_buffer_bytes,
        direct_shift_host_buffer_bytes,
        backend_array_bytes,
        host_auxiliary_bytes,
        total_tracked_bytes=backend_array_bytes + host_auxiliary_bytes,
    )
end

export lattice_memory_report

"""
    mark_halo_dirty!(lattice)

Advance the core-data epoch after modifying the local lattice data. Public
mutating operations call this automatically. Call it explicitly after writing
through `lattice.A` directly.

On an MPI lattice, rank-local mutations and later shifted reads must follow the
same control flow on every rank because halo synchronization is collective.
"""
@inline function mark_halo_dirty!(ls::LatticeMatrix)
    epoch = ls.halo_epoch
    epoch.core += UInt64(1)
    if iszero(ls.nw)
        epoch.halo = epoch.core
    end
    return nothing
end

@inline function _mark_halo_clean!(ls::LatticeMatrix)
    ls.halo_epoch.halo = ls.halo_epoch.core
    return nothing
end

"""Return whether the stored halo is older than the lattice core data."""
@inline halo_is_dirty(ls::LatticeMatrix) = ls.halo_epoch.core != ls.halo_epoch.halo

"""Return the current `(core, halo)` epochs as a named tuple."""
@inline halo_epochs(ls::LatticeMatrix) = (core=ls.halo_epoch.core, halo=ls.halo_epoch.halo)

struct _BoundMutatingKernel{F,A<:Tuple}
    kernel::F
    arguments::A
end

Adapt.@adapt_structure _BoundMutatingKernel

@inline function (bound::_BoundMutatingKernel)(i)
    bound.kernel(i, bound.arguments...)
    return nothing
end

@inline function _parallel_for_mutating!(
        destination::LatticeMatrix, count::Integer, kernel::F, arguments...; kwargs...,
    ) where {F}
    mark_halo_dirty!(destination)
    bound = _BoundMutatingKernel(kernel, arguments)
    return JACC.parallel_for(count, bound; kwargs...)
end

@inline function _parallel_for_mutating!(destination::LatticeMatrix, args...; kwargs...)
    mark_halo_dirty!(destination)
    return JACC.parallel_for(args...; kwargs...)
end

"""
    parallel_for_mutating!(destination, count, kernel, arguments...; kwargs...)

Launch a portable JACC kernel that mutates `destination`. The kernel and its
arguments are bound into one concrete, recursively adaptable functor before
launch, so the same call is specialized on CPU and converted by CUDA, AMDGPU,
and oneAPI device adaptors. The destination halo is marked dirty before the
kernel runs.
"""
@inline parallel_for_mutating!(
    destination::LatticeMatrix, args...; kwargs...,
) = _parallel_for_mutating!(destination, args...; kwargs...)

export mark_halo_dirty!, halo_is_dirty, halo_epochs, parallel_for_mutating!


function Base.similar(ls::TL) where {D,T,AT,NC1,NC2,DI,nw,TL<:LatticeMatrix_standard{D,T,AT,NC1,NC2,nw,DI}}
    tA = zero(ls.A)
    # Scratch capacity is a workload high-water mark, not structural lattice
    # metadata. A similar lattice therefore starts with no scratch allocation
    # and grows lazily if a materialized long shift actually needs storage.
    temps = LatticeScratchPool(tA)
    buf = similar(ls.buf)
    buf_host = similar(ls.buf_host)
    for i in eachindex(ls.buf)
        buf[i] = zero(ls.buf[i])
        host_buffer = Array{T}(undef, size(ls.buf_host[i]))
        direction = cld(i, 4)
        buf_host[i] = ls.dims[direction] == 1 ? host_buffer :
            _prepare_mpi_host_buffer(tA, host_buffer)
    end

    return LatticeMatrix_standard{D,T,AT,NC1,NC2,nw,DI,typeof(ls.cart)}(ls.nw,
        ls.phases,
        ls.NC1,
        ls.NC2,
        ls.gsize,
        ls.cart,
        ls.coords,
        ls.dims,
        ls.nbr,
        tA,
        buf,
        buf_host,
        DirectShiftHostBuffers(T),
        ls.mpi_transport,
        ls.myrank,
        ls.PN,
        ls.comm,
        ls.indexer,
        temps,
        HaloEpoch()
    )
end

@inline function _lattice_alias_with_array(
    ls::TL,
    A::AT;
    phases=ls.phases,
    halo_epoch=HaloEpoch(),
    temps=ls.temps,
    shift_buf_host=ls.shift_buf_host,
) where {D,T,AT,NC1,NC2,nw,DI,TL<:LatticeMatrix_standard{D,T,AT,NC1,NC2,nw,DI}}
    phase_vector = phases isa typeof(ls.phases) ? phases : typeof(ls.phases)(phases)
    return LatticeMatrix_standard{D,T,AT,NC1,NC2,nw,DI,typeof(ls.cart)}(
        ls.nw,
        phase_vector,
        ls.NC1,
        ls.NC2,
        ls.gsize,
        ls.cart,
        ls.coords,
        ls.dims,
        ls.nbr,
        A,
        ls.buf,
        ls.buf_host,
        shift_buf_host,
        ls.mpi_transport,
        ls.myrank,
        ls.PN,
        ls.comm,
        ls.indexer,
        temps,
        halo_epoch,
    )
end

# ---------------------------------------------------------------------------
# constructor + heavy init (still cheap to call)
# ---------------------------------------------------------------------------
function LatticeMatrix(NC1, NC2, dim, gsize, PEs; nw=1, elementtype=ComplexF64, phases=ones(dim),
    comm0=nothing, numtemps=0, device_mapping=:auto, mpi_transport=:auto)
    return LatticeMatrix_standard(NC1, NC2, dim, gsize, PEs;
        nw, elementtype, phases, comm0, numtemps, device_mapping, mpi_transport)
end

function LatticeMatrix(A, dim, PEs; nw=1, phases=ones(dim), comm0=nothing, numtemps=0,
    device_mapping=:auto, mpi_transport=:auto)
    return LatticeMatrix_standard(A, dim, PEs;
        nw, phases, comm0, numtemps, device_mapping, mpi_transport)
end

# ---------------------------------------------------------------------------
# constructor + heavy init (still cheap to call)
# ---------------------------------------------------------------------------
function LatticeMatrix_standard(NC1, NC2, dim, gsize, PEs; nw=1, elementtype=ComplexF64, phases=ones(dim), comm0=nothing,
    numtemps=0, device_mapping=:auto, mpi_transport=:auto)

    nw >= 0 || throw(ArgumentError("nw must be non-negative, got $nw"))
    dim > 0 || throw(ArgumentError("dim must be positive, got $dim"))
    length(gsize) == dim || throw(ArgumentError(
        "global size must have $dim entries, got $(length(gsize))"))
    length(PEs) == dim || throw(ArgumentError(
        "process grid must have $dim entries, got $(length(PEs))"))
    length(phases) == dim || throw(ArgumentError(
        "phases must have $dim entries, got $(length(phases))"))
    NC1 > 0 && NC2 > 0 || throw(ArgumentError("matrix dimensions must be positive"))

    gsize = ntuple(i -> Int(gsize[i]), dim)
    dims = ntuple(i -> Int(PEs[i]), dim)
    all(>(0), gsize) || throw(ArgumentError("global lattice sizes must be positive, got $gsize"))
    all(>(0), dims) || throw(ArgumentError("process-grid sizes must be positive, got $dims"))
    any(iszero, phases) && throw(ArgumentError(
        "boundary phases must be nonzero because negative wraps use inv(phase)"))
    for d in 1:dim
        iszero(gsize[d] % dims[d]) || throw(ArgumentError(
            "global size $(gsize[d]) in dimension $d is not divisible by process-grid size $(dims[d])"))
    end
    comm0 = _resolve_communicator(comm0)
    _communicator_ready(comm0) || throw(ArgumentError(
        "MPI must be initialized and not finalized before constructing an MPI lattice"))
    comm_size = _comm_size(comm0)
    prod(dims) == comm_size || throw(ArgumentError(
        "process grid $dims contains $(prod(dims)) ranks, but communicator contains $comm_size"))
    PN = ntuple(i -> gsize[i] ÷ dims[i], dim)
    for d in 1:dim
        nw <= PN[d] || throw(ArgumentError(
            "halo width nw=$nw exceeds local lattice size $(PN[d]) in dimension $d; " *
            "the nearest-neighbor halo exchange requires nw <= local size"))
    end

    _prepare_backend_device!(comm0, device_mapping)

    # Cartesian grid
    D = dim
    T = elementtype
    periodic = ntuple(_ -> true, D)
    #println(dims)
    #println(periodic)
    cart = _cart_create(comm0, dims; periodic=periodic)
    coords = _cart_coords(cart, _comm_rank(cart), Val(D))

    #comm  = MPI.Cart_create(MPI.COMM_WORLD, dims; periods=ntuple(_->true,D))
    #coords= MPI.Cart_coords(cart, MPI.Comm_rank(cart))
    nbr = ntuple(d -> ntuple(
        s -> _cart_shift(cart, d - 1, ifelse(s == 1, -1, 1))[2], 2), D)
    # local array (NC first)
    #println(gsize)
    locS = ntuple(i -> gsize[i] ÷ dims[i] + 2nw, D)
    loc = (NC1, NC2, locS...)
    A = JACC.zeros(T, loc...)
    mpi_transport_config = _resolve_mpi_transport(mpi_transport, A, comm0)
    #stride = ntuple(i -> (i == 1 ? 1 : prod(locS[1:i-1])), D)

    # contiguous buffers for each face
    # A one-process lattice always updates halos by local periodic copies, so
    # the packed MPI face buffers would never be used. Avoid allocating them,
    # especially on accelerators where they otherwise consume device and host
    # memory for every field.
    nbuf = iszero(nw) || comm_size == 1 ? 0 : 4D
    buf = Vector{typeof(A)}(undef, nbuf)
    buf_host = Vector{Array{elementtype}}(undef, nbuf)
    if !iszero(nbuf)
        for d in 1:D
            shp = ntuple(i -> i == d ? nw : locS[i], D)   # halo slab shape
            buf[4d-3] = JACC.zeros(T, (NC1, NC2, shp...)...)  # minus side
            buf[4d-2] = JACC.zeros(T, (NC1, NC2, shp...)...)  # plus  side
            buf[4d-1] = JACC.zeros(T, (NC1, NC2, shp...)...)  # minus side
            buf[4d] = JACC.zeros(T, (NC1, NC2, shp...)...)  # plus  side

            for buffer_index in (4d-3):(4d)
                host_buffer = Array{T}(undef, size(buf[buffer_index]))
                buf_host[buffer_index] = dims[d] == 1 ? host_buffer :
                    _prepare_mpi_host_buffer(A, host_buffer)
            end
        end
    end


    #println("LatticeMatrix: $dims, $gsize, $PN, $nw")
    #indexer = DIndexer(gsize)
    indexer = DIndexer(PN)
    DI = typeof(indexer)

    temps = LatticeScratchPool(A; num=numtemps)

    #return LatticeMatrix{D,T,typeof(A),NC1,NC2,nw}(nw, phases, NC1, NC2, gsize,
    #    cart, Tuple(coords), dims, nbr,
    #    A, buf, MPI.Comm_rank(cart), PN, comm0)
    return LatticeMatrix_standard{D,T,typeof(A),NC1,NC2,nw,DI,typeof(cart)}(nw, phases, NC1, NC2, gsize,
        cart, Tuple(coords), dims, nbr,
        A, buf, buf_host, DirectShiftHostBuffers(T), mpi_transport_config,
        _comm_rank(cart), PN, comm0,
        indexer, temps, HaloEpoch())
end

function LatticeMatrix_standard(A, dim, PEs; nw=1, phases=ones(dim), comm0=nothing, numtemps=0,
    device_mapping=:auto, mpi_transport=:auto)

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

    ls = LatticeMatrix(NC1, NC2, dim, gsize, PEs;
        elementtype, nw, phases, comm0, numtemps, device_mapping, mpi_transport)
    _broadcast!(A, 0, ls.cart)
    Acpu = Array(ls.A)

    idx = ntuple(i -> (i == 1 || i == 2) ? Colon() : (ls.nw+1):(size(ls.A, i)-ls.nw), dim .+ 2)



    idx_global = ntuple(i -> (i == 1 || i == 2) ? Colon() : get_globalrange(ls, i - 2), dim .+ 2)

    #println(idx)
    #=
    for i = 1:_comm_size(ls.cart)
        if ls.myrank == i
            println(get_globalrange(ls, 1))
        end
        _barrier(ls.cart)
    end
    =#



    #println(idx_global)
    Acpu[idx...] = A[idx_global...]
    #println(Acpu)


    Agpu = JACC.array(Acpu)
    ls.A .= Agpu

    mark_halo_dirty!(ls)
    set_halo!(ls)
    #println(ls.A)

    return ls

    #coords_r = MPI.Cart_coords(ls.cart, ls.myrank)
    # 0-based coords
    #println(coords_r)

end

function Base.similar(ls::TL) where {D,T,AT,NC1,NC2,TL<:LatticeMatrix{D,T,AT,NC1,NC2}}
    return LatticeMatrix(NC1, NC2, D, ls.gsize, ls.dims;
        nw=ls.nw, elementtype=T, phases=ls.phases, comm0=ls.comm, numtemps=0,
        device_mapping=:current, mpi_transport=ls.mpi_transport.requested)
end

"""
    mpi_transport_info(lattice)

Report the requested and resolved MPI communication transport for `lattice`.
`resolved` is one of `:local`, `:host_direct`, `:host_staged`, or
`:device_direct`. `:local` means that the lattice uses `SerialCommunicator` and
there is no inter-process transport.
"""
function mpi_transport_info(ls::LatticeMatrix)
    config = ls.mpi_transport
    library_info = _mpi_library_info(ls.comm)
    return (
        requested=config.requested,
        resolved=_mpi_transport_symbol(config.resolved),
        backend=config.backend,
        device_aware=config.resolved === _MPI_DEVICE_DIRECT,
        reason=config.reason,
        mpi_library=library_info.library,
        mpi_library_version=library_info.version,
    )
end

export mpi_transport_info



function Base.display(ls::TL) where {T,AT,NC1,NC2,TL<:LatticeMatrix{4,T,AT,NC1,NC2}}

    NN = size(ls.A)
    for rank = 0:_comm_size(ls.cart)-1
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
        _barrier(ls.cart)
    end
end



function allsum(ls::TL) where {D,T,AT,NC1,NC2,TL<:LatticeMatrix{D,T,AT,NC1,NC2}}
    NN = ls.PN
    indices = ntuple(i -> (i == 1 || i == 2) ? Colon() : (ls.nw+1):(ls.nw+NN[i-2]), D + 2)
    # sum all elements in the local array
    local_sum = sum(ls.A[indices...])
    #local_sum = sum(ls.A[:, :, ls.nw+1:ls.nw+NN[1], ls.nw+1:ls.nw+NN[2], ls.nw+1:ls.nw+NN[3], ls.nw+1:ls.nw+NN[4]])
    # reduce to all processes
    global_sum = _reduce_sum(local_sum, 0, ls.cart)
    return global_sum
end

export allsum

function get_globalrange(ls::TL, dim) where {D,TL<:LatticeMatrix{D}}
    coords_r = _cart_coords(ls.cart, ls.myrank, Val(D))
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



Base.@noinline function set_halo!(ls::TL) where {D,T,AT,NC1,NC2,DI,TL<:LatticeMatrix{D,T,AT,NC1,NC2,0,DI}}
    _mark_halo_clean!(ls)
    return nothing
end

Base.@noinline function set_halo!(ls::TL) where {D,T,AT,NC1,NC2,nw,DI,TL<:LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}}
    # Single-process lattices do not need MPI communication. Keep halo updates local.
    if _comm_size(ls.cart) == 1
        for id = 1:D
            exchange_dim_local!(ls, id)
        end
        _mark_halo_clean!(ls)
        return nothing
    end
    for id = 1:D
        exchange_dim!(ls, id)
    end
    _mark_halo_clean!(ls)
    return nothing
end
export set_halo!

"""
    ensure_halo!(lattice)

Synchronize the halo only when its epoch is older than the core-data epoch.
Shifted lattice operations call this automatically.
"""
Base.@noinline function ensure_halo!(ls::LatticeMatrix)
    halo_is_dirty(ls) && set_halo!(ls)
    return nothing
end

@inline function _ensure_halo_for_shift!(ls::LatticeMatrix, shift)
    any(s -> !iszero(s), shift) && ensure_halo!(ls)
    return nothing
end

export ensure_halo!

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

# 4D-specialized slab view builders to avoid per-call ntuple construction.
function _faceMatrix(A::AbstractArray{T,6}, nw, d, side::Symbol) where T
    @views if d == 1
        return side === :minus ? A[:, :, (nw+1):(2*nw), :, :, :] : A[:, :, (end-2*nw+1):(end-nw), :, :, :]
    elseif d == 2
        return side === :minus ? A[:, :, :, (nw+1):(2*nw), :, :] : A[:, :, :, (end-2*nw+1):(end-nw), :, :]
    elseif d == 3
        return side === :minus ? A[:, :, :, :, (nw+1):(2*nw), :] : A[:, :, :, :, (end-2*nw+1):(end-nw), :]
    else
        return side === :minus ? A[:, :, :, :, :, (nw+1):(2*nw)] : A[:, :, :, :, :, (end-2*nw+1):(end-nw)]
    end
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

function _ghostMatrix(A::AbstractArray{T,6}, nw, d, side::Symbol) where T
    @views if d == 1
        return side === :minus ? A[:, :, 1:nw, :, :, :] : A[:, :, (end-nw+1):end, :, :, :]
    elseif d == 2
        return side === :minus ? A[:, :, :, 1:nw, :, :] : A[:, :, :, (end-nw+1):end, :, :]
    elseif d == 3
        return side === :minus ? A[:, :, :, :, 1:nw, :] : A[:, :, :, :, (end-nw+1):end, :]
    else
        return side === :minus ? A[:, :, :, :, :, 1:nw] : A[:, :, :, :, :, (end-nw+1):end]
    end
end

# A sequential halo exchange does not need the full padded cross-section in
# every direction. At step d, halos in dimensions 1:d-1 are already valid and
# must be propagated to form corners; dimensions d+1:D still need only their
# core ranges. The final padded lattice is identical to a full-slab exchange,
# while early-direction messages and pack kernels are substantially smaller.
function _exchange_slab_matrix(ls::LatticeMatrix{D}, d::Int, side::Symbol,
    ghost::Bool) where {D}
    slab_range = if ghost
        side === :minus ? (1:ls.nw) :
            ((size(ls.A, d + 2)-ls.nw+1):size(ls.A, d + 2))
    else
        side === :minus ? ((ls.nw+1):(2 * ls.nw)) :
            ((size(ls.A, d + 2)-2 * ls.nw+1):(size(ls.A, d + 2)-ls.nw))
    end
    indices = ntuple(D + 2) do array_dimension
        array_dimension <= 2 && return Colon()
        lattice_dimension = array_dimension - 2
        lattice_dimension == d && return slab_range
        lattice_dimension < d && return Colon()
        return (ls.nw + 1):(ls.nw + ls.PN[lattice_dimension])
    end
    @views return ls.A[indices...]
end

@inline _exchange_face_matrix(ls, d, side) =
    _exchange_slab_matrix(ls, d, side, false)
@inline _exchange_ghost_matrix(ls, d, side) =
    _exchange_slab_matrix(ls, d, side, true)

@inline function _active_face_buffer(buffer, slab)
    count = length(slab)
    return @view(vec(buffer)[1:count])
end

@inline _mpi_transfer_parent(buffer::SubArray) = parent(buffer)
@inline _mpi_transfer_parent(buffer) = buffer

@inline function _copy_active_mpi_buffer!(destination, source)
    length(destination) == length(source) || throw(DimensionMismatch(
        "MPI staging buffers have different lengths: " *
        "$(length(destination)) and $(length(source))"))
    return copyto!(
        _mpi_transfer_parent(destination), 1,
        _mpi_transfer_parent(source), 1,
        length(destination),
    )
end

function exchange_dim_local!(ls::LatticeMatrix{D}, d::Int) where D
    gminus = _exchange_ghost_matrix(ls, d, :minus)
    gplus = _exchange_ghost_matrix(ls, d, :plus)
    fminus = _exchange_face_matrix(ls, d, :minus)
    fplus = _exchange_face_matrix(ls, d, :plus)

    # minus ghost <= plus face
    copy!(gminus, fplus)
    _mul_phase!(gminus, inv(ls.phases[d]))

    # plus ghost <= minus face
    copy!(gplus, fminus)
    _mul_phase!(gplus, ls.phases[d])

    compute_interior!(ls)
    return
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

# Post one pair of minus/plus receives and sends using the transport selected
# when the lattice was constructed. This helper is also used by reverse halo
# exchanges in the AD extension so those paths cannot accidentally require a
# device-aware MPI implementation while the primal path uses host staging.
function _post_packed_halo_exchange!(
    ls::LatticeMatrix,
    d::Int,
    device_buffers,
    host_buffers,
    tags,
)
    rankM, rankP = ls.nbr[d]
    me = ls.myrank
    requests = _request_vector(ls.cart)

    # Packing kernels and phase multiplication may be asynchronous. MPI is not
    # assumed to understand JACC streams, so make the send buffers visible
    # before either a direct device send or a device-to-host copy.
    JACC.synchronize()

    if _uses_direct_mpi(ls.mpi_transport)
        rankM != me && push!(requests, _irecv!(
            device_buffers.recv_minus, rankM, tags.recv_minus, ls.cart))
        rankP != me && push!(requests, _irecv!(
            device_buffers.recv_plus, rankP, tags.recv_plus, ls.cart))
        rankM != me && push!(requests, _isend(
            device_buffers.send_minus, rankM, tags.send_minus, ls.cart))
        rankP != me && push!(requests, _isend(
            device_buffers.send_plus, rankP, tags.send_plus, ls.cart))
    else
        rankM != me && push!(requests, _irecv!(
            host_buffers.recv_minus, rankM, tags.recv_minus, ls.cart))
        rankP != me && push!(requests, _irecv!(
            host_buffers.recv_plus, rankP, tags.recv_plus, ls.cart))
        rankM != me && _copy_active_mpi_buffer!(
            host_buffers.send_minus, device_buffers.send_minus)
        rankP != me && _copy_active_mpi_buffer!(
            host_buffers.send_plus, device_buffers.send_plus)
        rankM != me && push!(requests, _isend(
            host_buffers.send_minus, rankM, tags.send_minus, ls.cart))
        rankP != me && push!(requests, _isend(
            host_buffers.send_plus, rankP, tags.send_plus, ls.cart))
    end
    return requests
end

function _finish_packed_halo_exchange!(
    ls::LatticeMatrix,
    d::Int,
    requests,
    device_buffers,
    host_buffers,
)
    isempty(requests) || _waitall!(requests)
    _uses_direct_mpi(ls.mpi_transport) && return nothing

    rankM, rankP = ls.nbr[d]
    me = ls.myrank
    rankM != me && _copy_active_mpi_buffer!(
        device_buffers.recv_minus, host_buffers.recv_minus)
    rankP != me && _copy_active_mpi_buffer!(
        device_buffers.recv_plus, host_buffers.recv_plus)
    return nothing
end

# Full-buffer staging deliberately uses the ordinary two-argument `copyto!`.
# That is the portable device/host transfer API used by the pre-1.1 halo path;
# the five-argument prefix copy above is needed only for compact views.
function _post_full_halo_exchange!(
    ls::LatticeMatrix,
    d::Int,
    device_buffers,
    host_buffers,
    tags,
)
    rankM, rankP = ls.nbr[d]
    me = ls.myrank
    requests = _request_vector(ls.cart)

    JACC.synchronize()
    if _uses_direct_mpi(ls.mpi_transport)
        rankM != me && push!(requests, _irecv!(
            device_buffers.recv_minus, rankM, tags.recv_minus, ls.cart))
        rankP != me && push!(requests, _irecv!(
            device_buffers.recv_plus, rankP, tags.recv_plus, ls.cart))
        rankM != me && push!(requests, _isend(
            device_buffers.send_minus, rankM, tags.send_minus, ls.cart))
        rankP != me && push!(requests, _isend(
            device_buffers.send_plus, rankP, tags.send_plus, ls.cart))
    else
        rankM != me && push!(requests, _irecv!(
            host_buffers.recv_minus, rankM, tags.recv_minus, ls.cart))
        rankP != me && push!(requests, _irecv!(
            host_buffers.recv_plus, rankP, tags.recv_plus, ls.cart))
        rankM != me && copyto!(
            host_buffers.send_minus, device_buffers.send_minus)
        rankP != me && copyto!(
            host_buffers.send_plus, device_buffers.send_plus)
        rankM != me && push!(requests, _isend(
            host_buffers.send_minus, rankM, tags.send_minus, ls.cart))
        rankP != me && push!(requests, _isend(
            host_buffers.send_plus, rankP, tags.send_plus, ls.cart))
    end
    return requests
end

function _finish_full_halo_exchange!(
    ls::LatticeMatrix,
    d::Int,
    requests,
    device_buffers,
    host_buffers,
)
    isempty(requests) || _waitall!(requests)
    _uses_direct_mpi(ls.mpi_transport) && return nothing

    rankM, rankP = ls.nbr[d]
    me = ls.myrank
    rankM != me && copyto!(
        device_buffers.recv_minus, host_buffers.recv_minus)
    rankP != me && copyto!(
        device_buffers.recv_plus, host_buffers.recv_plus)
    return nothing
end

function _exchange_packed_halo_buffers!(
    ls::LatticeMatrix,
    d::Int,
    send_minus,
    recv_minus,
    send_plus,
    recv_plus;
    send_minus_tag,
    recv_minus_tag,
    send_plus_tag,
    recv_plus_tag,
)
    iSM, iRM = 4d - 3, 4d - 2
    iSP, iRP = 4d - 1, 4d
    device_buffers = (; send_minus, recv_minus, send_plus, recv_plus)

    if _uses_full_halo_buffers(ls.A)
        host_buffers = (
            send_minus=ls.buf_host[iSM],
            recv_minus=ls.buf_host[iRM],
            send_plus=ls.buf_host[iSP],
            recv_plus=ls.buf_host[iRP],
        )
        tags = (; send_minus=send_minus_tag, recv_minus=recv_minus_tag,
            send_plus=send_plus_tag, recv_plus=recv_plus_tag)
        requests = _post_full_halo_exchange!(
            ls, d, device_buffers, host_buffers, tags)
        _finish_full_halo_exchange!(
            ls, d, requests, device_buffers, host_buffers)
        return nothing
    end

    host_buffers = (
        send_minus=_active_face_buffer(ls.buf_host[iSM], send_minus),
        recv_minus=_active_face_buffer(ls.buf_host[iRM], recv_minus),
        send_plus=_active_face_buffer(ls.buf_host[iSP], send_plus),
        recv_plus=_active_face_buffer(ls.buf_host[iRP], recv_plus),
    )
    tags = (; send_minus=send_minus_tag, recv_minus=recv_minus_tag,
        send_plus=send_plus_tag, recv_plus=recv_plus_tag)
    requests = _post_packed_halo_exchange!(
        ls, d, device_buffers, host_buffers, tags)
    _finish_packed_halo_exchange!(
        ls, d, requests, device_buffers, host_buffers)
    return nothing
end

# Portable fallback for accelerator arrays that cannot reliably construct the
# compact `vec(buffer)[1:count]` views used by the optimized exchange. The
# preallocated buffers already have the full padded face shape, so this path
# can pack, stage, and unpack them without deriving another device array.
function _exchange_dim_full_buffers!(
    ls::LatticeMatrix{D}, d::Int, rankM, rankP, me,
) where {D}
    iSM, iRM = 4d - 3, 4d - 2
    iSP, iRP = 4d - 1, 4d

    bufSM, bufRM = ls.buf[iSM], ls.buf[iRM]
    bufSP, bufRP = ls.buf[iSP], ls.buf[iRP]
    bufSM_host, bufRM_host = ls.buf_host[iSM], ls.buf_host[iRM]
    bufSP_host, bufRP_host = ls.buf_host[iSP], ls.buf_host[iRP]

    gminus = _ghostMatrix(ls.A, ls.nw, d, :minus)
    gplus = _ghostMatrix(ls.A, ls.nw, d, :plus)
    fminus = _faceMatrix(ls.A, ls.nw, d, :minus)
    fplus = _faceMatrix(ls.A, ls.nw, d, :plus)

    if rankM == me
        copy!(gminus, fminus)
        ls.coords[d] == 0 && _mul_phase!(gminus, ls.phases[d])
    else
        copy!(bufSM, fminus)
        ls.coords[d] == 0 && _mul_phase!(bufSM, ls.phases[d])
    end

    if rankP == me
        copy!(gplus, fplus)
        ls.coords[d] == ls.dims[d] - 1 &&
            _mul_phase!(gplus, inv(ls.phases[d]))
    else
        copy!(bufSP, fplus)
        ls.coords[d] == ls.dims[d] - 1 &&
            _mul_phase!(bufSP, inv(ls.phases[d]))
    end

    device_buffers = (
        send_minus=bufSM,
        recv_minus=bufRM,
        send_plus=bufSP,
        recv_plus=bufRP,
    )
    host_buffers = (
        send_minus=bufSM_host,
        recv_minus=bufRM_host,
        send_plus=bufSP_host,
        recv_plus=bufRP_host,
    )
    tags = (send_minus=d, recv_minus=d + D, send_plus=d + D, recv_plus=d)
    requests = _post_full_halo_exchange!(
        ls, d, device_buffers, host_buffers, tags)

    compute_interior!(ls)
    _finish_full_halo_exchange!(
        ls, d, requests, device_buffers, host_buffers)

    rankM != me && copy!(gminus, bufRM)
    rankP != me && copy!(gplus, bufRP)
    return nothing
end

function exchange_dim!(ls::LatticeMatrix{D}, d::Int) where D
    rankM, rankP = ls.nbr[d]                     # neighbour ranks
    me = ls.myrank

    # --- self-neighbor on BOTH sides (happens iff dims[d] == 1) -------------
    # Check this before indexing communication buffers: one-process lattices
    # deliberately do not allocate those unused buffers.
    if rankM == me && rankP == me
        exchange_dim_local!(ls, d)
        return
    end

    _uses_full_halo_buffers(ls.A) &&
        return _exchange_dim_full_buffers!(ls, d, rankM, rankP, me)

    # buffer indices
    iSM, iRM = 4d - 3, 4d - 2
    iSP, iRP = 4d - 1, 4d

    bufSM, bufRM = ls.buf[iSM], ls.buf[iRM]      # minus side: send / recv
    bufSP, bufRP = ls.buf[iSP], ls.buf[iRP]      # plus  side: send / recv
    bufSM_host, bufRM_host = ls.buf_host[iSM], ls.buf_host[iRM]      # minus side: send / recv
    bufSP_host, bufRP_host = ls.buf_host[iSP], ls.buf_host[iRP]      # plus  side: send / recv

    gminus = _exchange_ghost_matrix(ls, d, :minus)
    gplus = _exchange_ghost_matrix(ls, d, :plus)
    fminus = _exchange_face_matrix(ls, d, :minus)
    fplus = _exchange_face_matrix(ls, d, :plus)
    activeSM = _active_face_buffer(bufSM, fminus)
    activeRM = _active_face_buffer(bufRM, gminus)
    activeSP = _active_face_buffer(bufSP, fplus)
    activeRP = _active_face_buffer(bufRP, gplus)
    activeSM_host = _active_face_buffer(bufSM_host, fminus)
    activeRM_host = _active_face_buffer(bufRM_host, gminus)
    activeSP_host = _active_face_buffer(bufSP_host, fplus)
    activeRP_host = _active_face_buffer(bufRP_host, gplus)
    packedSM = reshape(activeSM, size(fminus))
    packedRM = reshape(activeRM, size(gminus))
    packedSP = reshape(activeSP, size(fplus))
    packedRP = reshape(activeRP, size(gplus))

    # Pack both outgoing faces before posting communication. One backend-
    # neutral synchronization makes the packed buffers visible to device-aware
    # MPI without introducing CUDA/Threads branches.
    if rankM == me
        copy!(gminus, fminus)
        if ls.coords[d] == 0                     # wrap ⇒ phase
            _mul_phase!(gminus, ls.phases[d])
        end
    else
        copy!(packedSM, fminus)
        if ls.coords[d] == 0
            _mul_phase!(activeSM, ls.phases[d])
        end
    end

    if rankP == me
        copy!(gplus, fplus)
        if ls.coords[d] == ls.dims[d] - 1
            _mul_phase!(gplus, ls.phases[d])
        end
    else
        copy!(packedSP, fplus)
        if ls.coords[d] == ls.dims[d] - 1
            _mul_phase!(activeSP, inv(ls.phases[d]))
        end
    end

    device_buffers = (
        send_minus=activeSM,
        recv_minus=activeRM,
        send_plus=activeSP,
        recv_plus=activeRP,
    )
    host_buffers = (
        send_minus=activeSM_host,
        recv_minus=activeRM_host,
        send_plus=activeSP_host,
        recv_plus=activeRP_host,
    )
    tags = (send_minus=d, recv_minus=d + D, send_plus=d + D, recv_plus=d)
    reqs = _post_packed_halo_exchange!(
        ls, d, device_buffers, host_buffers, tags)

    compute_interior!(ls)
    _finish_packed_halo_exchange!(
        ls, d, reqs, device_buffers, host_buffers)

    # Copy received faces into the padded lattice. Subsequent JACC work uses
    # the same backend ordering, so no backend-specific synchronization is
    # needed here.
    rankM != me && copy!(gminus, packedRM)
    rankP != me && copy!(gplus, packedRP)
end

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
    comm = ls.cart
    me = ls.myrank
    nprocs = _comm_size(comm)

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
            _recv!(recvbuf, r, tag, comm)
            coords_r = _cart_coords(comm, r, Val(D))  # 0-based coords
            blk = reshape(recvbuf, size(local_block_cpu))
            _place_block!(G, blk, coords_r)
        end
        return G
    else
        # Non-root: send and return nothing
        tag = 900
        _send(sendbuf, root, tag, comm)
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
    comm = ls.cart
    me = ls.myrank
    nprocs = _comm_size(comm)

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
            _recv!(recvbuf, r, 900, comm)
            coords_r = _cart_coords(comm, r, Val(D))
            blk = reshape(recvbuf, size(local_block_cpu))
            _place_block!(G, blk, coords_r)
        end
    else
        # non-root: send local block
        _send(sendbuf, root, 900, comm)
    end

    # --- 3) broadcast ONLY the data (shape is deterministic) ---
    gshape = (ls.NC1, ls.NC2, ls.gsize...)   # same on all ranks
    if me != root
        G = Array{T}(undef, gshape)          # allocate receive buffer
    end
    _broadcast!(G, root, comm)                # broadcast the global array

    return G
end
export gather_and_bcast_matrix

@inline function _mul_phase!(buf, ϕ)
    isone(ϕ) && return nothing
    buf .*= ϕ
    return
end
