using CUDA
using JACC
JACC.@init_backend
using LatticeMatrices
using MPI
using Test

MPI.Initialized() || MPI.Init()

const TRANSPORT_COMM = MPI.COMM_WORLD
const TRANSPORT_RANK = MPI.Comm_rank(TRANSPORT_COMM)
const TRANSPORT_NRANKS = MPI.Comm_size(TRANSPORT_COMM)

TRANSPORT_NRANKS == 2 || error(
    "mpi_transport.jl requires exactly two MPI ranks, got $TRANSPORT_NRANKS")
CUDA.functional() || error("CUDA is not functional on MPI rank $TRANSPORT_RANK")

const TRANSPORT_GRID = (TRANSPORT_NRANKS, 1, 1, 1)
const TRANSPORT_SIZE = (4 * TRANSPORT_NRANKS, 4, 4, 4)

function transport_input()
    values = reshape(
        Float32.(1:(9 * prod(TRANSPORT_SIZE))),
        3, 3, TRANSPORT_SIZE...,
    )
    return complex.(values, reverse(values; dims=1))
end

function materialized_shift(field, shift)
    destination = similar(field)
    with_shifted_lattice(field, shift) do shifted
        substitute!(destination, shifted)
    end
    return gather_matrix(destination)
end

@testset "CUDA-aware MPI transport" begin
    input = transport_input()
    staged = LatticeMatrix(
        input, 4, TRANSPORT_GRID;
        nw=1, numtemps=2, mpi_transport=:host_staged)
    direct = LatticeMatrix(
        input, 4, TRANSPORT_GRID;
        nw=1, numtemps=2, device_mapping=:current,
        mpi_transport=:device_direct)
    automatic = LatticeMatrix(
        input, 4, TRANSPORT_GRID;
        nw=1, numtemps=2, device_mapping=:current, mpi_transport=:auto)

    @test mpi_transport_info(staged).resolved === :host_staged
    @test mpi_transport_info(direct).resolved === :device_direct
    @test mpi_transport_info(automatic).resolved === :device_direct
    @test Array(staged.A) == Array(direct.A) == Array(automatic.A)

    # A displacement larger than nw exercises the Alltoallv direct-shift path
    # in addition to the nearest-neighbor halo exchange above.
    shift = (2, -1, 0, 0)
    staged_shift = materialized_shift(staged, shift)
    direct_shift = materialized_shift(direct, shift)
    if TRANSPORT_RANK == 0
        @test staged_shift == direct_shift
        staged_info = mpi_transport_info(staged)
        direct_info = mpi_transport_info(direct)
        automatic_info = mpi_transport_info(automatic)
        @info "CUDA-aware MPI transport test completed" staged_info direct_info automatic_info
    end
end

MPI.Barrier(TRANSPORT_COMM)
