using MPI

MPI.Initialized() || MPI.Init()

using CUDA
using Printf
using Statistics

function exchange!(recv, send, peer, comm)
    requests = MPI.Request[
        MPI.Irecv!(recv, peer, 91, comm),
        MPI.Isend(send, peer, 91, comm),
    ]
    MPI.Waitall!(requests)
    return nothing
end

function rank_max_time(operation, repetitions, samples, comm)
    timings = Float64[]
    for _ in 1:samples
        MPI.Barrier(comm)
        start = time_ns()
        for _ in 1:repetitions
            operation()
        end
        CUDA.synchronize()
        local_ms = (time_ns() - start) / 1e6 / repetitions
        push!(timings, MPI.Allreduce(local_ms, max, comm))
    end
    return timings
end

function main()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    nranks == 2 || error("this benchmark requires exactly two MPI ranks")
    select_device_by_rank = rank % length(CUDA.devices())
    CUDA.device!(select_device_by_rank)
    peer = 1 - rank

    repetitions = parse(Int, get(ENV, "MPI_GPU_BENCH_ITERS", "200"))
    samples = parse(Int, get(ENV, "MPI_GPU_BENCH_SAMPLES", "9"))
    sizes = parse.(Int, split(get(
        ENV, "MPI_GPU_BENCH_BYTES", "32768,131072,524288,2097152"), ','))

    for bytes in sizes
        elements = cld(bytes, sizeof(ComplexF64))
        device_send = CUDA.fill(ComplexF64(rank + 1), elements)
        device_recv = similar(device_send)
        host_send = Vector{ComplexF64}(undef, elements)
        host_recv = similar(host_send)
        pinned_send = CUDA.pin(Vector{ComplexF64}(undef, elements))
        pinned_recv = CUDA.pin(Vector{ComplexF64}(undef, elements))

        direct = () -> exchange!(device_recv, device_send, peer, comm)
        pageable = () -> begin
            copyto!(host_send, device_send)
            exchange!(host_recv, host_send, peer, comm)
            copyto!(device_recv, host_recv)
        end
        pinned = () -> begin
            copyto!(pinned_send, device_send)
            exchange!(pinned_recv, pinned_send, peer, comm)
            copyto!(device_recv, pinned_recv)
        end

        # Warm all registration, MPI protocol, and copy paths before timing.
        for operation in (direct, pageable, pinned)
            for _ in 1:10
                operation()
            end
            CUDA.synchronize()
        end

        for (name, operation) in
            (("device_direct", direct), ("host_pageable", pageable),
             ("host_pinned", pinned))
            timings = rank_max_time(operation, repetitions, samples, comm)
            if rank == 0
                @printf(
                    "RESULT operation=MPIBufferExchange path=%s bytes=%d iterations=%d samples=%d min_ms=%.9f median_ms=%.9f max_ms=%.9f all_ms=%s\n",
                    name, elements * sizeof(ComplexF64), repetitions, samples,
                    minimum(timings), median(timings), maximum(timings),
                    repr(timings))
            end
        end
    end
end

main()
