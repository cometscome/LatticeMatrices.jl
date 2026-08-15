using MPI
using CUDA
import JACC

JACC.@init_backend

using LatticeMatrices
using Test

const EXPECTED_RANKS = 4
const GLOBAL_SIZE = (12, 12, 12, 24)
const PROCESS_GRIDS = (
    (4, 1, 1, 1),
    (2, 2, 1, 1),
    (1, 1, 2, 2),
)
const PHASES = (cis(0.11), cis(-0.23), -1.0 + 0im, im)
const SHIFTS = (
    (2, 0, 0, 0),
    (0, -3, 0, 0),
    (0, 0, 5, 0),
    (0, 0, 0, -7),
    (5, -4, 7, -9),
    (12, -12, 25, -49),
    (37, -26, 31, -74),
)

function reference_direct_shift(A, shift)
    shifted = similar(A)
    for site in CartesianIndices(GLOBAL_SIZE)
        indices = Tuple(site)
        source, factor = LatticeMatrices._shifted_global_indices_and_phase(
            indices, shift, GLOBAL_SIZE, PHASES, eltype(A))
        @views shifted[:, :, indices...] .= factor .* A[:, :, source...]
    end
    return shifted
end

function physical_gpu_uuids(device, comm)
    uuid_bytes = collect(codeunits(string(CUDA.uuid(device))))
    all_uuid_bytes = MPI.Allgather(uuid_bytes, comm)
    width = length(uuid_bytes)
    return [String(all_uuid_bytes[(i - 1) * width + 1:i * width])
            for i in 1:MPI.Comm_size(comm)]
end

function synchronized_median_ms(f, iterations, comm)
    f()
    JACC.synchronize()
    samples = Vector{Float64}(undef, iterations)
    for i in eachindex(samples)
        MPI.Barrier(comm)
        start = time_ns()
        f()
        JACC.synchronize()
        samples[i] = 1e-6 * (time_ns() - start)
    end
    local_median = sort!(samples)[cld(iterations, 2)]
    return MPI.Allreduce(local_median, max, comm)
end

function run_direct_shift_4gpu_tests()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    nranks == EXPECTED_RANKS || error(
        "this test requires exactly $EXPECTED_RANKS MPI ranks, got $nranks")
    CUDA.functional() || error("CUDA is not functional on MPI rank $rank")

    devices = collect(CUDA.devices())
    length(devices) >= EXPECTED_RANKS || error(
        "this test requires at least $EXPECTED_RANKS visible GPUs, got $(length(devices))")
    selection = select_device_by_mpi_rank!(comm)
    device = CUDA.device()
    device_ids = MPI.Allgather(CUDA.deviceid(device), comm)
    device_uuids = physical_gpu_uuids(device, comm)

    @info "MPI rank $rank selected GPU $(CUDA.deviceid(device)): $(CUDA.name(device))" uuid=string(CUDA.uuid(device)) capability=CUDA.capability(device)

    values = reshape(
        ComplexF64.(1:(2 * prod(GLOBAL_SIZE))), 2, 1, GLOBAL_SIZE...)

    function run_all_cases()
        if rank == 0
            @test device_ids == collect(0:(EXPECTED_RANKS - 1))
            @test length(unique(device_uuids)) == EXPECTED_RANKS
            @test JACC.backend == "cuda"
            @test selection.backend === :cuda
        end

        for process_grid in PROCESS_GRIDS
            for nw in (0, 1, 2)
                lattice = LatticeMatrix(
                    values, 4, process_grid; nw, phases=PHASES, numtemps=2)
                result = similar(lattice)
                initial_pool_size = length(lattice.temps)

                for shift in SHIFTS
                    materialized =
                        LatticeMatrices._shift_requires_materialization(lattice, shift)
                    lease_ok = Ref(true)
                    with_shifted_lattice(lattice, shift) do shifted
                        lease_ok[] &= isopen(shifted)
                        expected_used = materialized ? 1 : 0
                        lease_ok[] &=
                            count(lattice.temps._flagusing) == expected_used
                        substitute!(result, shifted)
                    end
                    JACC.synchronize()
                    lease_ok[] &= count(lattice.temps._flagusing) == 0
                    lease_ok[] &= length(lattice.temps) == initial_pool_size
                    all_leases_ok = MPI.Allreduce(
                        lease_ok[] ? 1 : 0, MPI.MIN, comm) == 1
                    all_cuda_ok = MPI.Allreduce(
                        lattice.A isa CUDA.CuArray ? 1 : 0, MPI.MIN, comm) == 1

                    shifted = gather_matrix(result)
                    if rank == 0
                        @test all_cuda_ok
                        @test all_leases_ok
                        @test shifted ≈ reference_direct_shift(values, shift)
                    end
                end
            end

            benchmark_shift = SHIFTS[end]
            short_shift = (1, 0, 0, 0)
            benchmark_lattice = LatticeMatrix(
                values, 4, process_grid;
                nw=1, phases=PHASES, numtemps=2)
            short_milliseconds = synchronized_median_ms(11, comm) do
                with_shifted_lattice(benchmark_lattice, short_shift) do _
                end
            end
            long_milliseconds = synchronized_median_ms(11, comm) do
                with_shifted_lattice(benchmark_lattice, benchmark_shift) do _
                end
            end
            if rank == 0
                @info "four-GPU direct-shift timing" process_grid short_shift short_milliseconds benchmark_shift long_milliseconds slowdown=long_milliseconds / short_milliseconds
            end
        end
    end

    if rank == 0
        @testset "four-GPU direct shifts on 12x12x12x24" begin
            run_all_cases()
        end
    else
        run_all_cases()
    end
    return nothing
end

MPI.Init()
try
    run_direct_shift_4gpu_tests()
    MPI.Barrier(MPI.COMM_WORLD)
finally
    MPI.Finalize()
end
