module LatticeMatricesMPIExt

using LatticeMatrices
using MPI

import LatticeMatrices: _allreduce_max, _allreduce_min, _allreduce_sum,
    _alltoallv!, _barrier, _broadcast!, _cart_coords, _cart_create,
    _cart_rank, _cart_shift, _communicator_ready, _comm_rank, _comm_size,
    _irecv!, _isend, _is_mpi_communicator, _mpi_library_info, _recv!, _reduce_sum,
    _request_vector, _send, _shared_rank_and_size, _waitall!

default_communicator() = MPI.COMM_WORLD

@inline _communicator_ready(::MPI.Comm) =
    MPI.Initialized() && !MPI.Finalized()
@inline _is_mpi_communicator(::MPI.Comm) = true
@inline _comm_size(comm::MPI.Comm) = MPI.Comm_size(comm)
@inline _comm_rank(comm::MPI.Comm) = MPI.Comm_rank(comm)
@inline _barrier(comm::MPI.Comm) = MPI.Barrier(comm)

@inline _cart_create(comm::MPI.Comm, dims; periodic) =
    MPI.Cart_create(comm, dims; periodic)

@inline function _cart_coords(
    comm::MPI.Comm, rank::Integer, ::Val{D},
) where {D}
    coords = MPI.Cart_coords(comm, rank)
    return ntuple(d -> @inbounds(coords[d]), Val(D))
end

@inline _cart_rank(comm::MPI.Comm, coords) = MPI.Cart_rank(comm, coords)
@inline _cart_shift(comm::MPI.Comm, direction::Integer, displacement::Integer) =
    MPI.Cart_shift(comm, direction, displacement)

function _shared_rank_and_size(comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    local_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
    try
        return MPI.Comm_rank(local_comm), MPI.Comm_size(local_comm)
    finally
        MPI.free(local_comm)
    end
end

@inline _allreduce_sum(value, comm::MPI.Comm) =
    MPI.Allreduce(value, MPI.SUM, comm)
@inline _allreduce_min(value, comm::MPI.Comm) =
    MPI.Allreduce(value, MPI.MIN, comm)
@inline _allreduce_max(value, comm::MPI.Comm) =
    MPI.Allreduce(value, MPI.MAX, comm)
@inline _reduce_sum(value, root::Integer, comm::MPI.Comm) =
    MPI.Reduce(value, MPI.SUM, root, comm)
@inline _broadcast!(value, root::Integer, comm::MPI.Comm) =
    MPI.Bcast!(value, root, comm)

@inline _mpi_library_info(::MPI.Comm) = (
    library=MPI.MPI_LIBRARY,
    version=MPI.MPI_LIBRARY_VERSION,
)

@inline _request_vector(::MPI.Comm) = MPI.Request[]
@inline _irecv!(buffer, rank, tag, comm::MPI.Comm) =
    MPI.Irecv!(buffer, rank, tag, comm)
@inline _isend(buffer, rank, tag, comm::MPI.Comm) =
    MPI.Isend(buffer, rank, tag, comm)
@inline _waitall!(requests::Vector{MPI.Request}) = MPI.Waitall!(requests)
@inline _send(buffer, rank, tag, comm::MPI.Comm) =
    MPI.Send(buffer, rank, tag, comm)
@inline _recv!(buffer, rank, tag, comm::MPI.Comm) =
    MPI.Recv!(buffer, rank, tag, comm)

@inline function _alltoallv!(
    receive, send, send_counts, recv_counts, send_displacements,
    recv_displacements, comm::MPI.Comm,
)
    MPI.Alltoallv!(
        MPI.VBuffer(send, send_counts, send_displacements),
        MPI.VBuffer(receive, recv_counts, recv_displacements),
        comm,
    )
    return receive
end

end
