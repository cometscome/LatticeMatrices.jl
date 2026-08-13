using MPI
using CUDA
import JACC

JACC.@init_backend

using LatticeMatrices
using LinearAlgebra
using Test

include("../staggered_dirac.jl")
include("../cg.jl")

const EXPECTED_RANKS = 2
const PROCESS_GRID = (2, 1, 1, 1)
const GLOBAL_SIZE = (16, 8, 8, 8)
const NC = 3
const HALO_WIDTH = 1

function global_input()
    values = ComplexF64.(1:(NC * NC * prod(GLOBAL_SIZE)))
    return reshape(values, NC, NC, GLOBAL_SIZE...)
end

function halo_reference(A, lattice)
    reference = similar(Array(lattice.A))
    local_shape = size(lattice.A)[3:end]

    for site in CartesianIndices(local_shape)
        local_indices = Tuple(site)
        global_indices = ntuple(d -> begin
            raw = lattice.coords[d] * lattice.PN[d] + local_indices[d] - lattice.nw
            mod(raw - 1, lattice.gsize[d]) + 1
        end, length(local_shape))
        @views reference[:, :, local_indices...] .= A[:, :, global_indices...]
    end

    return reference
end

function multiplication_reference(A)
    reference = similar(A)
    for site in CartesianIndices(GLOBAL_SIZE)
        indices = Tuple(site)
        @views reference[:, :, indices...] .=
            A[:, :, indices...] * A[:, :, indices...]
    end
    return reference
end

function shift_reference(A, shift)
    reference = similar(A)
    for site in CartesianIndices(GLOBAL_SIZE)
        indices = Tuple(site)
        source = ntuple(d -> mod(indices[d] + shift[d] - 1, GLOBAL_SIZE[d]) + 1,
            length(GLOBAL_SIZE))
        @views reference[:, :, indices...] .= A[:, :, source...]
    end
    return reference
end

function physical_gpu_uuids(device)
    uuid_bytes = collect(codeunits(string(CUDA.uuid(device))))
    all_uuid_bytes = MPI.Allgather(uuid_bytes, MPI.COMM_WORLD)
    width = length(uuid_bytes)
    return [String(all_uuid_bytes[(i - 1) * width + 1:i * width])
            for i in 1:MPI.Comm_size(MPI.COMM_WORLD)]
end

function run_tests()
    comm = MPI.COMM_WORLD
    nranks = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    require_h100 = lowercase(get(ENV, "LATTICEMATRICES_REQUIRE_H100", "true")) in
                   ("1", "true", "yes")

    nranks == EXPECTED_RANKS || error(
        "This test requires exactly $EXPECTED_RANKS MPI ranks, got $nranks")
    CUDA.functional() || error("CUDA is not functional on MPI rank $rank")

    devices = collect(CUDA.devices())
    length(devices) >= EXPECTED_RANKS || error(
        "This test requires at least $EXPECTED_RANKS visible GPUs, got $(length(devices))")

    CUDA.device!(rank)
    device = CUDA.device()
    device_name = CUDA.name(device)
    device_ids = MPI.Allgather(CUDA.deviceid(device), comm)
    device_uuids = physical_gpu_uuids(device)

    @info "MPI rank $rank is using GPU $(CUDA.deviceid(device)): $device_name" uuid =
        string(CUDA.uuid(device)) capability = CUDA.capability(device)

    @testset "two-GPU rank-to-device mapping" begin
        if require_h100
            @test occursin("H100", uppercase(device_name))
            @test CUDA.capability(device) == v"9.0"
        elseif rank == 0
            @info "H100 model check disabled by LATTICEMATRICES_REQUIRE_H100"
        end
        @test device_ids == [0, 1]
        @test length(unique(device_uuids)) == EXPECTED_RANKS
        @test JACC.backend == "cuda"
    end

    A = global_input()
    lattice = LatticeMatrix(A, 4, PROCESS_GRID; nw=HALO_WIDTH)

    @testset "CUDA allocation and two-rank halo exchange" begin
        @test lattice.A isa CUDA.CuArray
        @test lattice.dims == PROCESS_GRID
        @test Array(lattice.A) == halo_reference(A, lattice)

        gathered = gather_matrix(lattice)
        if rank == 0
            @test gathered == A
        end
    end

    result = similar(lattice)
    iterations = parse(Int, get(ENV, "LATTICEMATRICES_GPU_TEST_ITERS", "3"))
    iterations > 0 || error("LATTICEMATRICES_GPU_TEST_ITERS must be positive")

    # Warm up compilation before measuring the repeated multi-GPU operation.
    mul!(result, lattice, lattice)
    JACC.synchronize()
    MPI.Barrier(comm)
    elapsed = @elapsed begin
        for _ in 1:iterations
            mul!(result, lattice, lattice)
        end
        JACC.synchronize()
    end
    max_elapsed = MPI.Allreduce(elapsed, max, comm)

    @testset "distributed GPU kernels and collectives" begin
        product = gather_matrix(result)
        if rank == 0
            @test product ≈ multiplication_reference(A)
        end

        shift = (1, 0, 0, 0)
        substitute!(result, Shifted_Lattice(lattice, shift))
        shifted = gather_matrix(result)
        if rank == 0
            @test shifted == shift_reference(A, shift)
        end

        global_sum = allsum(lattice)
        if rank == 0
            @test global_sum ≈ sum(A)
            @info "two-GPU test completed" iterations max_elapsed_seconds = max_elapsed
        end
    end

    @testset "two-GPU staggered Dirac operator" begin
        # Each rank owns an odd x extent.  This detects rank-local eta phases
        # while exercising the same CUDA halo path as a production Dslash.
        staggered_size = (6, 4, 4, 4)
        staggered_phases = (1, 1, 1, -1)
        staggered_mass = 0.17f0
        staggered_links = _staggered_test_links(
            staggered_size, NC; elementtype=ComplexF32)
        staggered_psi_host = _staggered_test_fermion(
            staggered_size, NC; elementtype=ComplexF32)
        staggered_reference = _staggered_test_reference(
            staggered_links, staggered_psi_host,
            staggered_mass, staggered_phases)
        staggered_reference_dag = _staggered_test_reference(
            staggered_links, staggered_psi_host,
            staggered_mass, staggered_phases; adjoint_operator=true)

        staggered_U = [LatticeMatrix(
            link, 4, PROCESS_GRID; nw=HALO_WIDTH) for link in staggered_links]
        staggered_psi = LatticeMatrix(
            staggered_psi_host, 4, PROCESS_GRID;
            nw=HALO_WIDTH, phases=staggered_phases)
        staggered_result = similar(staggered_psi)
        staggered_operator = StaggeredDiracOperator4D(
            staggered_U, staggered_mass)
        @test staggered_result.A isa CUDA.CuArray

        mul!(staggered_result, staggered_operator, staggered_psi)
        JACC.synchronize()
        staggered_global = gather_matrix(staggered_result)
        if rank == 0
            @test staggered_global ≈ staggered_reference atol=4f-5 rtol=4f-5
        end

        MPI.Barrier(comm)
        staggered_elapsed = @elapsed begin
            for _ in 1:iterations
                mul!(staggered_result, staggered_operator, staggered_psi)
            end
            JACC.synchronize()
        end
        staggered_max_elapsed = MPI.Allreduce(staggered_elapsed, max, comm)

        mul!(staggered_result, adjoint(staggered_operator), staggered_psi)
        JACC.synchronize()
        staggered_global_dag = gather_matrix(staggered_result)
        if rank == 0
            @test isapprox(
                staggered_global_dag, staggered_reference_dag;
                atol=4f-5, rtol=4f-5)
            @info "two-GPU staggered test completed" iterations max_elapsed_seconds=staggered_max_elapsed
        end
    end

    # The same backend-independent solver tests exercise CUDA kernels for
    # axpby!, mul!, dot/Allreduce, explicit DdagD storage, and the legacy pool
    # adapter once each MPI rank has selected its GPU above.
    cg_tests()

    return nothing
end

MPI.Init()
try
    run_tests()
finally
    MPI.Finalize()
end
