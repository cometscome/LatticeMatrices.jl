module LatticeMatrices
using MPI
using LinearAlgebra
using JACC
#using Enzyme

include("utilities/randomgenerator.jl")

abstract type AbstractLattice end

abstract type Lattice{D,T,AT,NC1,NC2,NW} <: AbstractLattice end

include("device_selection.jl")

#include("HaloComm.jl")
#include("1D/1Dlatticevector.jl")
#include("1D/1Dlatticematrix.jl")

mutable struct ShiftLease{P}
    # Keep the concrete pool type available to Enzyme. Shifted_Lattice stores
    # the parametric lease behind the ShiftLease UnionAll, so this internal
    # detail does not add a type parameter to the public shifted-lattice type.
    pool::P
    index::Int
    active::Bool
end

mutable struct Shifted_Lattice{D,Dim} <: AbstractLattice
    data::D
    shift::NTuple{Dim,Int64}
    lease::Union{Nothing,ShiftLease}

    @inline function Shifted_Lattice(data, shift, ::Val{Dim}) where {Dim}
        return new{typeof(data),Dim}(data, shift, nothing)
    end

    @inline function Shifted_Lattice(
        data, shift, ::Val{Dim}, lease::ShiftLease,
    ) where {Dim}
        return new{typeof(data),Dim}(data, shift, lease)
    end
end

# Internal lazy representation used only by halo-free kernels that implement
# periodic indexing themselves. Public shift constructors never return it.
struct _LazyShifted_Lattice{D,Dim} <: AbstractLattice
    data::D
    shift::NTuple{Dim,Int64}
end

struct Traceless_AntiHermitian{D} <: AbstractLattice
    data::D
end
export Traceless_AntiHermitian


export Shifted_Lattice
export shift_L
export add_matrix_shiftedA!
export release!, with_shifted_lattice

struct Adjoint_Lattice{D} <: AbstractLattice
    data::D
end



function Base.adjoint(data::TD) where {D,T,AT,TD<:Lattice{D,T,AT}}
    return Adjoint_Lattice{typeof(data)}(data)
end

@inline function Base.adjoint(data::T) where {D,Dim,T<:Shifted_Lattice{D,Dim}}
    return Adjoint_Lattice{typeof(data)}(data)
end

@inline function Base.adjoint(data::T) where {D,Dim,T<:_LazyShifted_Lattice{D,Dim}}
    return Adjoint_Lattice{typeof(data)}(data)
end

function Base.adjoint(data::TD) where {TD<:Adjoint_Lattice}
    return data.data
end




include("Latticeindices.jl")
include("LatticeMatrices_core.jl")
include("measurements/global_source.jl")
include("measurements/projected_bilinear_slices.jl")
include("LinearAlgebras/linearalgebra.jl")
include("TA/TA.jl")
#include("AD/AD.jl")
include("ND.jl")
include("LinearAlgebras/staggered.jl")



@inline _shift_has_lease(x::Shifted_Lattice) = getfield(x, :lease) !== nothing

@inline function _assert_shift_open(x::Shifted_Lattice)
    lease = getfield(x, :lease)
    if lease !== nothing && !lease.active
        throw(ArgumentError("this materialized Shifted_Lattice has already been released"))
    end
    return nothing
end

@inline _release_lease!(::Nothing) = nothing

function _release_lease!(lease::ShiftLease)
    if lease.active
        unused!(lease.pool, lease.index)
        lease.active = false
    end
    return nothing
end

@inline function _new_shift_lease(pool, index::Integer)
    lease = ShiftLease(pool, Int(index), true)
    # Explicit release remains the normal path. The finalizer is a safety net
    # for callers which drop a materialized shift without returning its slot.
    finalizer(_release_lease!, lease)
    return lease
end

"""
    release!(shifted)

Return storage borrowed by a materialized long-distance shift to its
`PreallocatedArray`. The operation is idempotent. Short halo-backed shifts do
not own storage, so releasing them is a no-op.
"""
function release!(shifted::Shifted_Lattice)
    _release_lease!(getfield(shifted, :lease))
    return nothing
end

function release!(shifted::Adjoint_Lattice{<:Shifted_Lattice})
    release!(getfield(shifted, :data))
    return nothing
end

Base.close(shifted::Shifted_Lattice) = release!(shifted)
Base.close(shifted::Adjoint_Lattice{<:Shifted_Lattice}) = release!(shifted)

function Base.isopen(shifted::Shifted_Lattice)
    lease = getfield(shifted, :lease)
    return lease === nothing || lease.active
end

Base.isopen(shifted::Adjoint_Lattice{<:Shifted_Lattice}) =
    isopen(getfield(shifted, :data))

"""
    with_shifted_lattice(f, data, shift)

Construct a shifted lattice, call `f` with it, and deterministically release
any borrowed long-shift storage when `f` returns or throws.
"""
function with_shifted_lattice(f::F, data, shift) where {F}
    shifted = Shifted_Lattice(data, shift)
    try
        return f(shifted)
    finally
        release!(shifted)
    end
end

@inline function _shift_requires_materialization(
    data::LatticeMatrix{D,T,AT,NC1,NC2,nw},
    shift_in,
) where {D,T,AT,NC1,NC2,nw}
    shift = _as_shift_tuple(shift_in, Val(D))
    return any(s -> abs(s) > nw, shift)
end

function _borrow_shift_storage(data::LatticeMatrix{D}) where {D}
    array, index = get_block(data.temps)
    lease = _new_shift_lease(data.temps, index)
    try
        storage = _lattice_alias_with_array(data, array)
        clear_matrix!(storage)
        zeroshift = ntuple(_ -> 0, D)
        return Shifted_Lattice(storage, zeroshift, Val(D), lease)
    catch
        _release_lease!(lease)
        rethrow()
    end
end

function _materialized_shift_shadow(data::LatticeMatrix, shift)
    return _borrow_shift_storage(data)
end

@inline function _adjoint_shift_phases(data::LatticeMatrix)
    return typeof(data.phases)(map(phase -> inv(conj(phase)), data.phases))
end

function _accumulate_shift_pullback!(
    destination::LatticeMatrix{D,T,AT,NC1,NC2,nw},
    shadow::Shifted_Lattice,
    shift_in,
    metadata::LatticeMatrix=destination,
) where {D,T,AT,NC1,NC2,nw}
    _assert_shift_open(shadow)
    shift = _as_shift_tuple(shift_in, Val(D))
    inverse_shift = map(-, shift)
    source_data = getfield(shadow, :data)
    source_lease = getfield(shadow, :lease)
    source_lease isa ShiftLease || throw(ArgumentError(
        "the materialized shift pullback requires borrowed shadow storage"))

    # Enzyme shadows may poison non-differentiable metadata fields, including
    # the phase vector of the destination cotangent. Use only shadow arrays;
    # communicator, topology, buffers, and phases come from the primal input.
    phases = _adjoint_shift_phases(metadata)
    source = _lattice_alias_with_array(
        metadata,
        source_data.A;
        phases,
        halo_epoch=HaloEpoch(UInt64(1), UInt64(0)),
        temps=destination.temps,
        shift_buf_host=metadata.shift_buf_host,
    )
    shifted = nothing
    try
        shifted = _materialize_direct_shift_reusing_source(
            source, inverse_shift, source_lease)
        add_matrix!(destination, getfield(shifted, :data))
        return nothing
    finally
        shifted === nothing || release!(shifted)
    end
end


@inline function get_shift(x::Shifted_Lattice{Tx,D}) where {D,T,AT,NC1,NC2,nw,Tx<:LatticeMatrix{D,T,AT,NC1,NC2,nw}}
    ensure_halo!(x)
    return x.shift
end

@inline function get_shift(x::Adjoint_Lattice{<:Shifted_Lattice{Tx,D}}) where {D,T,AT,NC1,NC2,nw,Tx<:LatticeMatrix{D,T,AT,NC1,NC2,nw}}
    ensure_halo!(x)
    return x.data.shift
end

@inline get_shift(x::_LazyShifted_Lattice) = x.shift
@inline get_shift(x::Adjoint_Lattice{<:_LazyShifted_Lattice}) = x.data.shift




#function Shifted_Lattice(data::TD, shift::TS) where {D,T,AT,TD<:Lattice{D,T,AT},TS}
#    return Shifted_Lattice{typeof(data),D}(data, shift)
#end

function zero_halo_region! end
export zero_halo_region!
function zero_halo_dim! end
export zero_halo_dim!

function fold_halo_dim_to_core_grad! end
export fold_halo_dim_to_core_grad!


@inline function _as_shift_tuple(shift_in, ::Val{D}) where {D}
    if shift_in isa NTuple{D,Int}
        return shift_in
    elseif shift_in isa AbstractVector{<:Integer}
        len = length(shift_in)
        len > D && throw(ArgumentError("shift length must be <= $D"))
        return ntuple(i -> i <= len ? Int(shift_in[i]) : 0, D)
    elseif shift_in isa Tuple
        len = length(shift_in)
        len > D && throw(ArgumentError("shift length must be <= $D"))
        return ntuple(i -> i <= len ? Int(shift_in[i]) : 0, D)
    else
        error("Unsupported shift type: $(typeof(shift_in)). Provide NTuple{$D,Int} or Vector{Int}.")
    end
end

@inline make_step(i, r, ::Val{D}) where {D} =
    ntuple(j -> ifelse(j == i, r, 0), D)

Base.@noinline function Shifted_Lattice(data::TL, shift_in::TS) where {
    D,T,AT,NC1,NC2,nw,DI,
    TL<:LatticeMatrix{D,T,AT,NC1,NC2,nw,DI},TS
}
    return Shifted_Lattice_construct(data, shift_in)
end

@inline function _periodic_shift_index(i::Integer, shift::Integer, n::Integer)
    raw = i + shift
    return mod(raw - 1, n) + 1, fld(raw - 1, n)
end

@inline function _global_core_indices(local_indices::NTuple{D,<:Integer}, coords, local_size) where D
    return ntuple(d -> coords[d] * local_size[d] + local_indices[d], D)
end

"""
    global_site_coordinates(local_indices, mpi_coordinates, local_size)
    global_site_coordinates(lattice, local_indices)

Convert one-based core-local lattice coordinates to one-based global
coordinates. MPI Cartesian coordinates are expected to be zero-based, as in a
[`LatticeMatrix`](@ref).
"""
@inline function global_site_coordinates(
    local_indices::NTuple{D,<:Integer},
    coords::NTuple{D,<:Integer},
    local_size::NTuple{D,<:Integer},
) where D
    return _global_core_indices(local_indices, coords, local_size)
end

@inline function global_site_coordinates(
    lattice::LatticeMatrix{D},
    local_indices::NTuple{D,<:Integer},
) where D
    return global_site_coordinates(local_indices, lattice.coords, lattice.PN)
end

"""
    global_site_id(global_coordinates, global_size)
    global_site_id(lattice, local_indices)

Return the zero-based global linear site id in Julia column-major order.  The
result depends only on the global coordinates and lattice size, not on MPI rank
or process-grid decomposition.
"""
@inline function global_site_id(
    global_indices::NTuple{D,<:Integer},
    global_size::NTuple{D,<:Integer},
) where D
    site = UInt64(0)
    stride = UInt64(1)
    @inbounds for d in 1:D
        site += UInt64(global_indices[d] - 1) * stride
        stride *= UInt64(global_size[d])
    end
    return site
end

@inline function global_site_id(
    lattice::LatticeMatrix{D},
    local_indices::NTuple{D,<:Integer},
) where D
    global_indices = global_site_coordinates(lattice, local_indices)
    return global_site_id(global_indices, lattice.gsize)
end

export global_site_coordinates, global_site_id

@inline function _global_site_is_even(local_indices::NTuple{D,<:Integer}, coords, local_size) where D
    coordinate_sum = 0
    @inbounds for d in 1:D
        coordinate_sum += coords[d] * local_size[d] + local_indices[d]
    end
    return iszero(coordinate_sum & 1)
end

@inline function _shifted_global_indices_and_phase(indices::NTuple{D,<:Integer}, shift,
    global_size, phases, ::Type{T}) where {D,T}
    shifted_indices = ntuple(d -> begin
        shifted, _ = _periodic_shift_index(indices[d], shift[d], global_size[d])
        shifted
    end, D)

    factor = one(T)
    @inbounds for d in 1:D
        _, wraps = _periodic_shift_index(indices[d], shift[d], global_size[d])
        factor *= convert(T, phases[d])^wraps
    end
    return shifted_indices, factor
end

struct _DirectShiftSegment
    source_start::Int
    destination_start::Int
    length::Int
    destination_coord::Int
end

struct _DirectShiftFragment{D,T}
    source_start::NTuple{D,Int}
    destination_start::NTuple{D,Int}
    lengths::NTuple{D,Int}
    peer::Int
    factor::T
    buffer_offset::Int
end

struct _DirectShiftPlan{D,T}
    send_fragments::Vector{_DirectShiftFragment{D,T}}
    recv_fragments::Vector{_DirectShiftFragment{D,T}}
    send_counts::Vector{Cint}
    recv_counts::Vector{Cint}
    send_displacements::Vector{Cint}
    recv_displacements::Vector{Cint}
    element_count::Int
end

function _direct_shift_segments(source_coord, local_size, global_size, shift)
    first_destination_global = mod(source_coord * local_size - shift, global_size) + 1
    destination_coord = (first_destination_global - 1) ÷ local_size
    destination_start = mod(first_destination_global - 1, local_size) + 1
    first_length = min(local_size, local_size - destination_start + 1)

    segments = _DirectShiftSegment[
        _DirectShiftSegment(1, destination_start, first_length, destination_coord),
    ]
    remaining = local_size - first_length
    if !iszero(remaining)
        push!(segments, _DirectShiftSegment(
            first_length + 1,
            1,
            remaining,
            mod(destination_coord + 1, global_size ÷ local_size),
        ))
    end
    return segments
end

@inline _direct_fragment_element_count(fragment::_DirectShiftFragment, colors::Int) =
    colors * prod(fragment.lengths)

function _direct_outgoing_fragments(
    data::LatticeMatrix{D,T}, source_coords_in, shift::NTuple{D,Int}
) where {D,T}
    source_coords = ntuple(d -> Int(source_coords_in[d]), D)
    segments = ntuple(d -> _direct_shift_segments(
        source_coords[d], data.PN[d], data.gsize[d], shift[d]), D)
    fragments = _DirectShiftFragment{D,T}[]

    for selected in Iterators.product(segments...)
        source_start = ntuple(d -> selected[d].source_start, D)
        destination_start = ntuple(d -> selected[d].destination_start, D)
        lengths = ntuple(d -> selected[d].length, D)
        destination_coords = ntuple(d -> selected[d].destination_coord, D)
        destination_rank = MPI.Cart_rank(data.cart, destination_coords)

        factor = one(T)
        @inbounds for d in 1:D
            source_global = source_coords[d] * data.PN[d] + source_start[d]
            destination_global =
                destination_coords[d] * data.PN[d] + destination_start[d]
            wraps = div(destination_global + shift[d] - source_global, data.gsize[d])
            factor *= data.phases[d]^wraps
        end
        push!(fragments, _DirectShiftFragment(
            source_start, destination_start, lengths,
            destination_rank, factor, 0))
    end

    sort!(fragments; by=fragment ->
        (fragment.peer, fragment.source_start, fragment.destination_start))
    return fragments
end

function _direct_fragments_with_offsets(fragments, nranks, colors)
    counts_int = zeros(Int, nranks)
    @inbounds for fragment in fragments
        counts_int[fragment.peer + 1] +=
            _direct_fragment_element_count(fragment, colors)
    end
    total = sum(counts_int)
    total <= typemax(Cint) || throw(ArgumentError(
        "direct shift buffer contains $total elements, exceeding MPI Cint capacity"))
    counts = Cint.(counts_int)
    displacements = Vector{Cint}(undef, nranks)
    displacement = 0
    @inbounds for rank_index in 1:nranks
        displacements[rank_index] = Cint(displacement)
        displacement += counts_int[rank_index]
    end

    cursors = Int.(displacements) .+ 1
    with_offsets = similar(fragments, 0)
    @inbounds for fragment in fragments
        rank_index = fragment.peer + 1
        offset = cursors[rank_index]
        push!(with_offsets, _DirectShiftFragment(
            fragment.source_start,
            fragment.destination_start,
            fragment.lengths,
            fragment.peer,
            fragment.factor,
            offset,
        ))
        cursors[rank_index] += _direct_fragment_element_count(fragment, colors)
    end
    return with_offsets, counts, displacements, total
end

function _direct_shift_plan(
    data::LatticeMatrix{D,T,AT,NC1,NC2}, shift::NTuple{D,Int}
) where {D,T,AT,NC1,NC2}
    nranks = MPI.Comm_size(data.cart)
    colors = NC1 * NC2
    outgoing = _direct_outgoing_fragments(data, data.coords, shift)
    send_fragments, send_counts, send_displacements, send_total =
        _direct_fragments_with_offsets(outgoing, nranks, colors)

    incoming = _DirectShiftFragment{D,T}[]
    for source_rank in 0:(nranks-1)
        source_coords = MPI.Cart_coords(data.cart, source_rank)
        source_fragments = _direct_outgoing_fragments(data, source_coords, shift)
        @inbounds for fragment in source_fragments
            if fragment.peer == data.myrank
                push!(incoming, _DirectShiftFragment(
                    fragment.source_start,
                    fragment.destination_start,
                    fragment.lengths,
                    source_rank,
                    fragment.factor,
                    0,
                ))
            end
        end
    end
    sort!(incoming; by=fragment ->
        (fragment.peer, fragment.source_start, fragment.destination_start))
    recv_fragments, recv_counts, recv_displacements, recv_total =
        _direct_fragments_with_offsets(incoming, nranks, colors)

    expected = colors * prod(data.PN)
    send_total == expected || error(
        "internal direct-shift send plan covers $send_total of $expected elements")
    recv_total == expected || error(
        "internal direct-shift receive plan covers $recv_total of $expected elements")
    return _DirectShiftPlan(
        send_fragments, recv_fragments,
        send_counts, recv_counts,
        send_displacements, recv_displacements,
        expected,
    )
end

@inline function _kernel_pack_direct_shift_fragment!(
    site, packed, source, source_start, fragment_indexer,
    ::Val{NC1}, ::Val{NC2}, ::Val{nw}, buffer_offset,
) where {NC1,NC2,nw}
    relative = delinearize(fragment_indexer, site, 0)
    source_indices = ntuple(d -> source_start[d] + relative[d] - 1 + nw,
        length(source_start))
    color_offset = buffer_offset + (site - 1) * NC1 * NC2 - 1
    @inbounds for jc in 1:NC2
        for ic in 1:NC1
            packed[color_offset + (jc - 1) * NC1 + ic] =
                source[ic, jc, source_indices...]
        end
    end
    return nothing
end

@inline function _kernel_unpack_direct_shift_fragment!(
    site, destination, packed, destination_start, fragment_indexer,
    ::Val{NC1}, ::Val{NC2}, ::Val{nw}, buffer_offset, factor,
) where {NC1,NC2,nw}
    relative = delinearize(fragment_indexer, site, 0)
    destination_indices = ntuple(
        d -> destination_start[d] + relative[d] - 1 + nw,
        length(destination_start))
    color_offset = buffer_offset + (site - 1) * NC1 * NC2 - 1
    @inbounds for jc in 1:NC2
        for ic in 1:NC1
            destination[ic, jc, destination_indices...] = factor *
                packed[color_offset + (jc - 1) * NC1 + ic]
        end
    end
    return nothing
end

@inline function _kernel_direct_shift_local!(
    site, destination, source, indexer, shift, coords, local_size,
    global_size, phases, ::Val{NC1}, ::Val{NC2}, ::Val{nw},
) where {NC1,NC2,nw}
    destination_local = delinearize(indexer, site, 0)
    destination_global = _global_core_indices(destination_local, coords, local_size)
    source_global, factor = _shifted_global_indices_and_phase(
        destination_global, shift, global_size, phases, eltype(destination))
    source_local = ntuple(
        d -> mod(source_global[d] - 1, local_size[d]) + 1 + nw,
        length(local_size))
    destination_indices = ntuple(
        d -> destination_local[d] + nw, length(local_size))
    @inbounds for jc in 1:NC2
        for ic in 1:NC1
            destination[ic, jc, destination_indices...] =
                factor * source[ic, jc, source_local...]
        end
    end
    return nothing
end

function _pack_direct_shift!(
    packed,
    source::LatticeMatrix{D,T,AT,NC1,NC2,nw},
    fragments,
) where {D,T,AT,NC1,NC2,nw}
    for fragment in fragments
        fragment_sites = prod(fragment.lengths)
        JACC.parallel_for(
            fragment_sites,
            _kernel_pack_direct_shift_fragment!,
            packed,
            source.A,
            fragment.source_start,
            DIndexer(fragment.lengths),
            Val(NC1),
            Val(NC2),
            Val(nw),
            fragment.buffer_offset,
        )
    end
    return nothing
end

function _unpack_direct_shift!(
    destination::LatticeMatrix{D,T,AT,NC1,NC2,nw},
    packed,
    fragments,
) where {D,T,AT,NC1,NC2,nw}
    for fragment in fragments
        fragment_sites = prod(fragment.lengths)
        JACC.parallel_for(
            fragment_sites,
            _kernel_unpack_direct_shift_fragment!,
            destination.A,
            packed,
            fragment.destination_start,
            DIndexer(fragment.lengths),
            Val(NC1),
            Val(NC2),
            Val(nw),
            fragment.buffer_offset,
            fragment.factor,
        )
    end
    mark_halo_dirty!(destination)
    return nothing
end

function _exchange_direct_shift!(receive, send, data::LatticeMatrix, plan::_DirectShiftPlan)
    if send isa Array && receive isa Array
        send_buffer = reshape(send, :)
        receive_buffer = reshape(receive, :)
        MPI.Alltoallv!(
            MPI.VBuffer(send_buffer, plan.send_counts, plan.send_displacements),
            MPI.VBuffer(receive_buffer, plan.recv_counts, plan.recv_displacements),
            data.cart,
        )
    else
        buffers = _ensure_direct_shift_host_buffers!(
            data.shift_buf_host, plan.element_count)
        copyto!(buffers.send, 1, send, 1, plan.element_count)
        MPI.Alltoallv!(
            MPI.VBuffer(buffers.send, plan.send_counts, plan.send_displacements),
            MPI.VBuffer(buffers.recv, plan.recv_counts, plan.recv_displacements),
            data.cart,
        )
        copyto!(receive, 1, buffers.recv, 1, plan.element_count)
    end
    return nothing
end

function _direct_shift_local!(destination, source::LatticeMatrix{D,T,AT,NC1,NC2,nw},
    shift::NTuple{D,Int}) where {D,T,AT,NC1,NC2,nw}
    _parallel_for_mutating!(
        destination,
        prod(source.PN),
        _kernel_direct_shift_local!,
        destination.A,
        source.A,
        source.indexer,
        shift,
        source.coords,
        source.PN,
        source.gsize,
        source.phases,
        Val(NC1),
        Val(NC2),
        Val(nw),
    )
    return nothing
end

function _materialize_direct_shift(data::LatticeMatrix{D}, shift::NTuple{D,Int}) where {D}
    result_array, result_index = get_block(data.temps)
    result_lease = _new_shift_lease(data.temps, result_index)
    receive_lease = nothing
    try
        result = _lattice_alias_with_array(data, result_array)
        if MPI.Comm_size(data.cart) == 1
            _direct_shift_local!(result, data, shift)
        else
            receive_array, receive_index = get_block(data.temps)
            # This communication scratch never escapes this function and is
            # already covered by the catch path, so it needs no finalizer.
            receive_lease = ShiftLease(data.temps, receive_index, true)
            plan = _direct_shift_plan(data, shift)
            _pack_direct_shift!(result_array, data, plan.send_fragments)
            _exchange_direct_shift!(receive_array, result_array, data, plan)
            _unpack_direct_shift!(result, receive_array, plan.recv_fragments)
            _release_lease!(receive_lease)
        end
        zeroshift = ntuple(_ -> 0, D)
        return Shifted_Lattice(result, zeroshift, Val(D), result_lease)
    catch
        _release_lease!(result_lease)
        _release_lease!(receive_lease)
        rethrow()
    end
end

function _materialize_direct_shift_reusing_source(
    data::LatticeMatrix{D}, shift::NTuple{D,Int}, source_lease::ShiftLease,
) where {D}
    result_array, result_index = get_block(data.temps)
    result_lease = _new_shift_lease(data.temps, result_index)
    try
        result = _lattice_alias_with_array(data, result_array)
        if MPI.Comm_size(data.cart) == 1
            _direct_shift_local!(result, data, shift)
        else
            plan = _direct_shift_plan(data, shift)
            _pack_direct_shift!(result_array, data, plan.send_fragments)
            _exchange_direct_shift!(data.A, result_array, data, plan)
            _unpack_direct_shift!(result, data.A, plan.recv_fragments)
        end
        _release_lease!(source_lease)
        zeroshift = ntuple(_ -> 0, D)
        return Shifted_Lattice(result, zeroshift, Val(D), result_lease)
    catch
        _release_lease!(source_lease)
        _release_lease!(result_lease)
        rethrow()
    end
end

@inline function kernel_periodic_shift_nowing!(i, C, A, ::Val{NC1}, ::Val{NC2},
    dindexer, shift, coords, local_size, global_size, phases) where {NC1,NC2}
    local_indices = delinearize(dindexer, i, 0)
    global_indices = _global_core_indices(local_indices, coords, local_size)
    source_indices, factor = _shifted_global_indices_and_phase(
        global_indices, shift, global_size, phases, eltype(C))

    @inbounds for jc in 1:NC2
        for ic in 1:NC1
            C[ic, jc, local_indices...] = factor * A[ic, jc, source_indices...]
        end
    end
    return nothing
end

@inline _nowing_slab_indices(A, d, range) =
    ntuple(i -> i == d + 2 ? range : Colon(), ndims(A))

function _shift_one_dimension_host!(destination, source, data, d, direction)
    local_length = data.PN[d]
    if direction > 0
        destination_range = 1:(local_length-1)
        source_range = 2:local_length
        send_range = 1:1
        receive_range = local_length:local_length
        send_rank = data.nbr[d][1]
        receive_rank = data.nbr[d][2]
        crosses_global_boundary = data.coords[d] == data.dims[d] - 1
    else
        destination_range = 2:local_length
        source_range = 1:(local_length-1)
        send_range = local_length:local_length
        receive_range = 1:1
        send_rank = data.nbr[d][2]
        receive_rank = data.nbr[d][1]
        crosses_global_boundary = data.coords[d] == 0
    end

    destination_indices = _nowing_slab_indices(destination, d, destination_range)
    source_indices = _nowing_slab_indices(source, d, source_range)
    @views copyto!(destination[destination_indices...], source[source_indices...])

    send_indices = _nowing_slab_indices(source, d, send_range)
    send_buffer = Array(@view source[send_indices...])
    receive_buffer = similar(send_buffer)

    if send_rank == data.myrank && receive_rank == data.myrank
        copyto!(receive_buffer, send_buffer)
    else
        tag = 1200 + 2d + ifelse(direction > 0, 0, 1)
        requests = MPI.Request[]
        push!(requests, MPI.Irecv!(receive_buffer, receive_rank, tag, data.cart))
        push!(requests, MPI.Isend(send_buffer, send_rank, tag, data.cart))
        MPI.Waitall!(requests)
    end

    if crosses_global_boundary
        phase = direction > 0 ? data.phases[d] : inv(data.phases[d])
        _mul_phase!(receive_buffer, phase)
    end
    receive_indices = _nowing_slab_indices(destination, d, receive_range)
    @views copyto!(destination[receive_indices...], receive_buffer)
    return nothing
end

function _materialize_periodic_shift_mpi(data::TL, shift::NTuple{D,Int}) where {
    D,T,AT,NC1,NC2,DI,
    TL<:LatticeMatrix{D,T,AT,NC1,NC2,0,DI}
}
    current = Array(data.A)
    scratch = similar(current)

    for d in 1:D
        direction = sign(shift[d])
        for _ in 1:abs(shift[d])
            _shift_one_dimension_host!(scratch, current, data, d, direction)
            current, scratch = scratch, current
        end
    end

    shifted = similar(data)
    shifted.A .= JACC.array(current)
    mark_halo_dirty!(shifted)
    return shifted
end

function _materialize_periodic_shift(data::TL, shift::NTuple{D,Int}) where {
    D,T,AT,NC1,NC2,DI,
    TL<:LatticeMatrix{D,T,AT,NC1,NC2,0,DI}
}
    all(iszero, shift) && return data

    if MPI.Comm_size(data.cart) > 1
        return _materialize_periodic_shift_mpi(data, shift)
    end

    shifted = similar(data)
    _parallel_for_mutating!(shifted,
        prod(data.PN), kernel_periodic_shift_nowing!, shifted.A, data.A,
        Val(NC1), Val(NC2), data.indexer, shift, data.coords, data.PN,
        data.gsize, data.phases)
    return shifted
end

@inline function Shifted_Lattice_construct(data::TL, shift_in::TS) where {
    D,T,AT,NC1,NC2,DI,
    TL<:LatticeMatrix{D,T,AT,NC1,NC2,0,DI},TS
}
    shift = _as_shift_tuple(shift_in, Val(D))
    all(iszero, shift) && return Shifted_Lattice(data, shift, Val(D))
    return _materialize_direct_shift(data, shift)
end

@inline function _lazy_shift_nowing(data::TL, shift_in) where {
    D,T,AT,NC1,NC2,DI,
    TL<:LatticeMatrix{D,T,AT,NC1,NC2,0,DI}
}
    MPI.Comm_size(data.cart) == 1 || throw(ArgumentError(
        "lazy nw=0 shifts are only available on a single MPI rank"))
    shift = _as_shift_tuple(shift_in, Val(D))
    return _LazyShifted_Lattice{typeof(data),D}(data, shift)
end

Base.@noinline function Shifted_Lattice_construct(data::TL, shift_in::TS) where {
    D,T,AT,NC1,NC2,nw,DI,
    TL<:LatticeMatrix{D,T,AT,NC1,NC2,nw,DI},TS
}
    shift = _as_shift_tuple(shift_in, Val(D))

    @inbounds begin
        isinside = true
        for i in 1:D
            s = shift[i]
            if (s < -nw) | (s > nw)
                isinside = false
                break
            end
        end
        if isinside
            any(s -> !iszero(s), shift) && ensure_halo!(data)
            return Shifted_Lattice(data, shift, Val(D))
        end
    end

    return _materialize_direct_shift(data, shift)
end

#Base.@noinline function shift_L(A::LatticeMatrix, shift)
#    return Shifted_Lattice(A, shift)
#end

Base.@noinline function shift_L(B, sh::NTuple{Dim,Int}) where {Dim}
    #println("shift_L: Dim=$(Dim) length(sh)=$(length(sh)) sh=$(sh) typeof(B)=$(typeof(B))")
    return Shifted_Lattice(B, sh)
    #return Shifted_Lattice{typeof(B),Dim}(B, sh)
end

include("LinearAlgebras/mul_nowing.jl")

#=
function Shifted_Lattice(data::TL, shift) where {D,T,AT,NC1,NC2,nw,DI,TL<:LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}}
    #set_halo!(data)
    #error("dd")
    #nw = data.nw
    #println("shift")
    isinside = true
    for i in 1:D
        if shift[i] < -nw || shift[i] > nw
            isinside = false
            break
        end
    end
    #println("Shifted_Lattice: shift = ", shift, " isinside = ", isinside)
    if isinside
        sl = Shifted_Lattice{typeof(data),typeof(shift)}(data, Tuple(shift))
    else
        sl0 = similar(data)
        sl1 = similar(data)
        shift0 = zeros(Int64, D)
        substitute!(sl0, data)
        for i in 1:D
            if shift[i] > nw
                smallshift = shift[i] ÷ nw
                shift0 .= 0
                shift0[i] = nw
                for k = 1:smallshift
                    sls = Shifted_Lattice{typeof(data),typeof(shift0)}(sl0, Tuple(shift0))
                    substitute!(sl1, sls)
                    substitute!(sl0, sl1)
                end
                shift0 .= 0
                shift0[i] = shift[i] % nw
                sls = Shifted_Lattice{typeof(data),typeof(shift0)}(sl0, Tuple(shift0))
                substitute!(sl1, sls)
                substitute!(sl0, sl1)
            elseif shift[i] < -nw
                smallshift = abs(shift[i]) ÷ nw
                shift0 .= 0
                shift0[i] = -nw
                #println(shift0)
                for k = 1:smallshift
                    println(shift0)
                    sls = Shifted_Lattice{typeof(data),typeof(shift0)}(sl0, Tuple(shift0))
                    substitute!(sl1, sls)
                    substitute!(sl0, sl1)
                end
                shift0 .= 0
                shift0[i] = -(abs(shift[i]) % nw)
                #println(shift0)
                sls = Shifted_Lattice{typeof(data),typeof(shift0)}(sl0, Tuple(shift0))
                substitute!(sl1, sls)
                substitute!(sl0, sl1)
            else
                shift0 .= 0
                shift0[i] = shift[i]
                sls = Shifted_Lattice{typeof(data),typeof(shift0)}(sl0, Tuple(shift0))
                substitute!(sl1, sls)
                substitute!(sl0, sl1)
            end
        end
        zeroshift = ntuple(_ -> 0, D)
        sl = Shifted_Lattice{typeof(data),typeof(zeroshift)}(sl0, zeroshift)
    end
    return sl
end
=#

function get_matrix(a::T) where {T<:LatticeMatrix}
    return a.A
end

function get_matrix(a::T) where {T<:Shifted_Lattice}
    ensure_halo!(a)
    return a.data.A
end


function get_matrix(a::T) where {T<:Adjoint_Lattice}
    return a.data.A
end

function get_matrix(a::Adjoint_Lattice{T}) where {T<:Shifted_Lattice}
    ensure_halo!(a)
    return a.data.data.A
end

@inline function ensure_halo!(a::Shifted_Lattice)
    _assert_shift_open(a)
    shift = getfield(a, :shift)
    any(s -> !iszero(s), shift) && ensure_halo!(getfield(a, :data))
    return nothing
end

@inline function ensure_halo!(a::Adjoint_Lattice{<:Shifted_Lattice})
    ensure_halo!(getfield(a, :data))
    return nothing
end

@inline function Base.getproperty(a::Shifted_Lattice, name::Symbol)
    if name === :data
        ensure_halo!(a)
    end
    return getfield(a, name)
end

function JACC.parallel_for(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}, variables...) where {D,T1,AT1,NC1,NG,nw,DI}
    _parallel_for_mutating!(C,
        prod(C.PN), kernelfunction, C.A, variables..., Val(NC1), Val(NG), Val(nw), C.indexer
    )
end

function JACC.parallel_reduce(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}, variables...) where {D,T1,AT1,NC1,NG,nw,DI}
    s = JACC.parallel_reduce(
        prod(C.PN), +, kernelfunction, C.A, variables..., Val(NC1), Val(NG), Val(nw), C.indexer
        ; init=zero(eltype(C.A))
    )
    s = MPI.Allreduce(s, MPI.SUM, C.comm)
end

function JACC.parallel_for(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}) where {D,T1,AT1,NC1,NG,nw,DI}
    _parallel_for_mutating!(C,
        prod(C.PN), kernelfunction, C.A, Val(NC1), Val(NG), Val(nw), C.indexer
    )
end

function JACC.parallel_reduce(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}) where {D,T1,AT1,NC1,NG,nw,DI}
    s = JACC.parallel_reduce(
        prod(C.PN), +, kernelfunction, C.A, Val(NC1), Val(NG), Val(nw), C.indexer
        ; init=zero(eltype(C.A))
    )
    s = MPI.Allreduce(s, MPI.SUM, C.comm)
end

function JACC.parallel_for(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}, A::Lattice{D,T2,AT2,NC2,NG2,nw2}, variables...) where {D,T1,AT1,NC1,NG,nw,DI,
    T2,AT2,NC2,NG2,nw2}
    a = get_matrix(A)
    _parallel_for_mutating!(C,
        prod(C.PN), kernelfunction, C.A, a, variables..., Val(NC1), Val(NG), Val(nw), Val(NC2), Val(NG2), Val(nw2), C.indexer
    )

end

function JACC.parallel_reduce(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}, A::Lattice{D,T2,AT2,NC2,NG2,nw2}, variables...) where {D,T1,AT1,NC1,NG,nw,DI,
    T2,AT2,NC2,NG2,nw2}
    a = get_matrix(A)
    s = JACC.parallel_reduce(
        prod(C.PN), +, kernelfunction, C.A, a, variables..., Val(NC1), Val(NG), Val(nw), Val(NC2), Val(NG2), Val(nw2), C.indexer
        ; init=zero(eltype(C.A))
    )
    s = MPI.Allreduce(s, MPI.SUM, C.comm)
end

function JACC.parallel_for(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}, A::Lattice{D,T2,AT2,NC2,NG2,nw2}) where {D,T1,AT1,NC1,NG,nw,DI,
    T2,AT2,NC2,NG2,nw2}
    a = get_matrix(A)
    _parallel_for_mutating!(C,
        prod(C.PN), kernelfunction, C.A, a, Val(NC1), Val(NG), Val(nw), Val(NC2), Val(NG2), Val(nw2), C.indexer
    )

end

function JACC.parallel_reduce(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}, A::Lattice{D,T2,AT2,NC2,NG2,nw2}) where {D,T1,AT1,NC1,NG,nw,DI,
    T2,AT2,NC2,NG2,nw2}
    a = get_matrix(A)
    s = JACC.parallel_reduce(
        prod(C.PN), kernelfunction, C.A, a, Val(NC1), Val(NG), Val(nw), Val(NC2), Val(NG2), Val(nw2), C.indexer
        ; init=zero(eltype(C.A)), op=+
    )
    s = MPI.Allreduce(s, MPI.SUM, C.comm)
end

function JACC.parallel_for(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}, A::Lattice{D,T2,AT2,NC2,NG2,nw2},
    B::Lattice{D,T3,AT3,NC3,NG3,nw3},
    variables...) where {D,T1,AT1,NC1,NG,nw,DI,
    T2,AT2,NC2,NG2,nw2,
    T3,AT3,NC3,NG3,nw3}
    a = get_matrix(A)
    b = get_matrix(B)
    _parallel_for_mutating!(C,
        prod(C.PN), kernelfunction, C.A, a, b, variables..., Val(NC1), Val(NG), Val(nw), Val(NC2), Val(NG2), Val(nw2), Val(NC3), Val(NG3), Val(nw3), C.indexer
    )
end

function JACC.parallel_reduce(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}, A::Lattice{D,T2,AT2,NC2,NG2,nw2},
    B::Lattice{D,T3,AT3,NC3,NG3,nw3},
    variables...) where {D,T1,AT1,NC1,NG,nw,DI,
    T2,AT2,NC2,NG2,nw2,
    T3,AT3,NC3,NG3,nw3}
    a = get_matrix(A)
    b = get_matrix(B)
    s = JACC.parallel_reduce(
        prod(C.PN), kernelfunction, C.A, a, b, variables..., Val(NC1), Val(NG), Val(nw), Val(NC2), Val(NG2), Val(nw2), Val(NC3), Val(NG3), Val(nw3), C.indexer
        ; init=zero(eltype(C.A)), op=+
    )
    s = MPI.Allreduce(s, MPI.SUM, C.comm)
end

function JACC.parallel_for(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}, A::Lattice{D,T2,AT2,NC2,NG2,nw2},
    B::Lattice{D,T3,AT3,NC3,NG3,nw3},
) where {D,T1,AT1,NC1,NG,nw,DI,
    T2,AT2,NC2,NG2,nw2,
    T3,AT3,NC3,NG3,nw3}
    a = get_matrix(A)
    b = get_matrix(B)
    _parallel_for_mutating!(C,
        prod(C.PN), kernelfunction, C.A, a, b, Val(NC1), Val(NG), Val(nw), Val(NC2), Val(NG2), Val(nw2), Val(NC3), Val(NG3), Val(nw3), C.indexer
    )
end

function JACC.parallel_reduce(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}, A::Lattice{D,T2,AT2,NC2,NG2,nw2},
    B::Lattice{D,T3,AT3,NC3,NG3,nw3},
) where {D,T1,AT1,NC1,NG,nw,DI,
    T2,AT2,NC2,NG2,nw2,
    T3,AT3,NC3,NG3,nw3}
    a = get_matrix(A)
    b = get_matrix(B)
    s = JACC.parallel_reduce(
        prod(C.PN), kernelfunction, C.A, a, b, Val(NC1), Val(NG), Val(nw), Val(NC2), Val(NG2), Val(nw2), Val(NC3), Val(NG3), Val(nw3), C.indexer
        ; init=zero(eltype(C.A)), op=+
    )
    s = MPI.Allreduce(s, MPI.SUM, C.comm)
end


function get_PEs(ls::LatticeMatrix{D,T,AT,NC1,NC2}) where {D,T,AT,NC1,NC2}
    return ls.dims
end
export get_PEs

function Wirtinger! end
export Wirtinger!

"""
    Wirtinger!(gradient)

Convert an Enzyme complex gradient in place to the Wirtinger derivative
`(∂f/∂x - im*∂f/∂y)/2`. Only the local core is transformed; the halo is
marked stale and will be synchronized on the next shifted read.
"""
function Wirtinger!(gradient::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}) where {D,T<:Complex,AT,NC1,NC2,nw,DI}
    half = convert(T, 0.5)
    _parallel_for_mutating!(
        gradient,
        prod(gradient.PN),
        _wirtinger_kernel!,
        gradient.A,
        half,
        Val(NC1),
        Val(NC2),
        Val(nw),
        gradient.indexer,
    )
    return gradient
end

@inline function _wirtinger_kernel!(i, gradient, half, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, indexer) where {NC1,NC2,nw}
    indices = delinearize(indexer, i, nw)
    @inbounds for jc in 1:NC2, ic in 1:NC1
        gradient[ic, jc, indices...] = half * conj(gradient[ic, jc, indices...])
    end
    return nothing
end

function realtrace end
export realtrace
function Wirtinger_derivative! end
export Wirtinger_derivative!

# Compatibility with the misspelled pre-v1 API.
Base.@deprecate Wiltinger! Wirtinger!
Base.@deprecate Wiltinger_derivative! Wirtinger_derivative!
export Wiltinger!, Wiltinger_derivative!
function Enzyme_derivative! end
export Enzyme_derivative!
"""
    enzyme_duplicated(primal, shadow)

Construct the Enzyme annotation used for differentiable lattice arguments.
Loading Enzyme activates this method. On Julia 1.12 and later it selects the
mixed-activity ABI required by immutable lattice containers.
"""
function enzyme_duplicated end
export enzyme_duplicated
function fold_halo_to_core_grad! end

struct DiffArg{T}
    x::T
end
struct NoDiffArg{T}
    x::T
end
# User-facing helpers
diff(x) = DiffArg(x)      # argument should be differentiated
nodiff(x) = NoDiffArg(x)    # argument is treated as constant
export diff, nodiff
function toann end
export toann

export mul_AshiftB!
export mul_shiftAshiftB!
export mul_A_shiftBdag!

include("Operators/Operators.jl")
include("Operators/DiracOperators.jl")
include("Operators/DiracOperators_5D.jl")


end
