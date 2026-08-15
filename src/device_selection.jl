"""
    select_device_by_mpi_rank!([comm=MPI.COMM_WORLD])

Select one accelerator for the calling MPI process from its rank among the
processes that share the same node.  The selected JACC backend determines the
vendor-specific device API.  Device ordinals returned by this function are
one-based, independently of the backend's native indexing convention.

If the process sees exactly one accelerator, that accelerator is retained.
This supports launchers and schedulers that expose a different single device
to each MPI process.  If multiple accelerators are visible, node-local MPI
rank zero selects ordinal one, rank one selects ordinal two, and so on.

The CUDA, AMDGPU, or oneAPI package for the selected JACC backend must already
be loaded, normally by calling `JACC.@init_backend`.  Call this function before
allocating any JACC arrays.  `threads` requires no device selection.  Multi-rank
device selection for the Metal backend is not currently supported.
"""
function select_device_by_mpi_rank!(comm::MPI.Comm=MPI.COMM_WORLD)
    MPI.Initialized() || throw(ArgumentError(
        "MPI must be initialized before selecting a device by MPI rank"))

    rank = MPI.Comm_rank(comm)
    local_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
    local_rank, local_size = try
        MPI.Comm_rank(local_comm), MPI.Comm_size(local_comm)
    finally
        MPI.free(local_comm)
    end

    backend = Symbol(JACC.backend)
    if backend === :threads
        return (
            backend=backend,
            local_rank=local_rank,
            local_size=local_size,
            visible_devices=0,
            device_ordinal=nothing,
        )
    elseif backend === :metal
        local_size == 1 || throw(ArgumentError(
            "JACC Metal backend does not support automatic device mapping for " *
            "$local_size MPI ranks on one node"))
        return (
            backend=backend,
            local_rank=local_rank,
            local_size=local_size,
            visible_devices=1,
            device_ordinal=1,
        )
    end

    backend_value = Val(backend)
    visible_devices = _backend_device_count(backend_value)
    device_ordinal = _device_ordinal_for_local_rank(
        local_rank, local_size, visible_devices)
    _select_backend_device!(backend_value, device_ordinal)

    return (
        backend=backend,
        local_rank=local_rank,
        local_size=local_size,
        visible_devices=visible_devices,
        device_ordinal=device_ordinal,
    )
end

function _backend_device_count(::Val{backend}) where {backend}
    throw(ArgumentError(
        "device selection support for JACC backend '$backend' is not loaded; " *
        "call JACC.@init_backend before constructing a LatticeMatrix"))
end

function _select_backend_device!(::Val{backend}, ::Integer) where {backend}
    throw(ArgumentError(
        "device selection support for JACC backend '$backend' is not loaded"))
end

function _device_ordinal_for_local_rank(
    local_rank::Integer,
    local_size::Integer,
    visible_devices::Integer,
)
    local_rank >= 0 || throw(ArgumentError(
        "node-local MPI rank must be non-negative, got $local_rank"))
    local_rank < local_size || throw(ArgumentError(
        "node-local MPI rank $local_rank is outside communicator size $local_size"))
    visible_devices > 0 || throw(ArgumentError(
        "the selected JACC backend reports no visible accelerator devices"))

    # A scheduler may expose one distinct physical device to each process.  In
    # that case every process correctly selects its only visible ordinal.
    visible_devices == 1 && return 1

    local_size <= visible_devices || throw(ArgumentError(
        "$local_size MPI ranks share this node, but the selected JACC backend " *
        "reports only $visible_devices visible devices"))
    return local_rank + 1
end

function _prepare_backend_device!(comm::MPI.Comm, device_mapping::Symbol)
    device_mapping === :current && return nothing
    device_mapping === :auto || throw(ArgumentError(
        "device_mapping must be :auto or :current, got $device_mapping"))
    JACC.backend == "threads" && return nothing
    return select_device_by_mpi_rank!(comm)
end

export select_device_by_mpi_rank!
