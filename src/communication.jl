"""A single-process communicator that does not require MPI.jl."""
struct SerialCommunicator end

const _SERIAL_COMMUNICATOR = SerialCommunicator()

"""
    _default_communicator()

Return `MPI.COMM_WORLD` when the MPI extension is loaded, otherwise return the
single-process communicator. This lookup only occurs during construction and
is deliberately kept out of computational kernels.
"""
function _default_communicator()
    extension = Base.get_extension(@__MODULE__, :LatticeMatricesMPIExt)
    return extension === nothing ? _SERIAL_COMMUNICATOR :
        extension.default_communicator()
end

@inline _resolve_communicator(comm) = comm
@inline _resolve_communicator(::Nothing) = _default_communicator()

@inline _communicator_ready(::SerialCommunicator) = true
@inline _is_mpi_communicator(::SerialCommunicator) = false
@inline _comm_size(::SerialCommunicator) = 1
@inline _comm_rank(::SerialCommunicator) = 0
@inline _barrier(::SerialCommunicator) = nothing

function _cart_create(comm::SerialCommunicator, dims; periodic)
    prod(dims) == 1 || throw(ArgumentError(
        "the serial communicator requires a process grid containing one rank, got $dims"))
    all(periodic) || throw(ArgumentError(
        "the serial lattice backend currently requires periodic dimensions"))
    return comm
end

@inline function _cart_coords(
    ::SerialCommunicator, rank::Integer, ::Val{D},
) where {D}
    iszero(rank) || throw(ArgumentError(
        "the serial communicator only contains rank 0, got rank $rank"))
    return ntuple(_ -> 0, D)
end

@inline function _cart_rank(::SerialCommunicator, coords)
    all(iszero, coords) || throw(ArgumentError(
        "the serial communicator only contains Cartesian coordinate zero, got $coords"))
    return 0
end

@inline _cart_shift(::SerialCommunicator, ::Integer, ::Integer) = (0, 0)
@inline _shared_rank_and_size(::SerialCommunicator) = (0, 1)

@inline _allreduce_sum(value, ::SerialCommunicator) = value
@inline _allreduce_min(value, ::SerialCommunicator) = value
@inline _allreduce_max(value, ::SerialCommunicator) = value

@inline function _reduce_sum(value, root::Integer, ::SerialCommunicator)
    iszero(root) || throw(ArgumentError(
        "the serial communicator only contains root rank 0, got root $root"))
    return value
end

@inline function _broadcast!(value, root::Integer, ::SerialCommunicator)
    iszero(root) || throw(ArgumentError(
        "the serial communicator only contains root rank 0, got root $root"))
    return value
end

@inline _mpi_library_info(::SerialCommunicator) = (library=nothing, version=nothing)

# Point-to-point and collective communication hooks. Their MPI methods live in
# LatticeMatricesMPIExt so loading the base package never loads MPI.jl.
function _request_vector end
function _irecv! end
function _isend end
function _waitall! end
function _send end
function _recv! end
function _alltoallv! end

@inline _request_vector(::SerialCommunicator) = Nothing[]
@inline _waitall!(::Vector{Nothing}) = nothing

@inline function _alltoallv!(
    receive, send, send_counts, recv_counts, send_displacements,
    recv_displacements, ::SerialCommunicator,
)
    length(receive) == length(send) || throw(DimensionMismatch(
        "serial all-to-all buffers have different lengths"))
    copyto!(receive, send)
    return receive
end

export SerialCommunicator
