import JACC
JACC.@init_backend

using LatticeMatrices
using MPI
using Test

MPI.Initialized() || MPI.Init()

include("random_fill.jl")

@testset "global-site random fills on accelerator" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    process_grid = (nprocs, 1)
    global_size = (8, 3)

    for algorithm in (PCG32(), Xoshiro256PlusPlus(), Philox4x32())
        uniform_key = RNGStreamKey(0x243f6a88, 9, 3, 1, 0x71)
        uniform_field = LatticeMatrix(
            3,
            2,
            2,
            global_size,
            process_grid;
            nw=1,
            elementtype=ComplexF64,
        )
        randomize_matrix!(uniform_field, uniform_key; rng_algorithm=algorithm)
        uniform_global = gather_and_bcast_matrix(uniform_field)
        uniform_reference = _uniform_fill_reference(
            global_size,
            3,
            2,
            uniform_key,
            algorithm,
            ComplexF64,
        )
        @test uniform_global == uniform_reference

        gaussian_key = RNGStreamKey(0x13198a2e, 13, 2, 0, 0x72)
        gaussian_field = LatticeMatrix(
            5,
            1,
            2,
            global_size,
            process_grid;
            nw=1,
            elementtype=Float64,
        )
        randomize_gaussian_matrix!(
            gaussian_field,
            gaussian_key;
            sigma=1.25,
            rng_algorithm=algorithm,
        )
        gaussian_global = gather_and_bcast_matrix(gaussian_field)
        gaussian_reference = _gaussian_fill_reference(
            global_size,
            5,
            1,
            gaussian_key,
            algorithm,
            1.25,
            Float64,
        )
        @test gaussian_global ≈ gaussian_reference rtol = 2e-12 atol = 2e-12
    end

    float32_key = RNGStreamKey(0xa4093822, 4, 1, 0, 0x73)
    uniform_float32 = LatticeMatrix(
        2,
        2,
        2,
        global_size,
        process_grid;
        nw=1,
        elementtype=ComplexF32,
    )
    randomize_matrix!(uniform_float32, float32_key; rng_algorithm=Philox4x32())
    @test gather_and_bcast_matrix(uniform_float32) == _uniform_fill_reference(
        global_size,
        2,
        2,
        float32_key,
        Philox4x32(),
        ComplexF32,
    )

    gaussian_float32 = LatticeMatrix(
        3,
        1,
        2,
        global_size,
        process_grid;
        nw=1,
        elementtype=Float32,
    )
    randomize_gaussian_matrix!(
        gaussian_float32,
        float32_key;
        sigma=0.75f0,
        rng_algorithm=Philox4x32(),
    )
    @test gather_and_bcast_matrix(gaussian_float32) ≈ _gaussian_fill_reference(
        global_size,
        3,
        1,
        float32_key,
        Philox4x32(),
        0.75f0,
        Float32,
    ) rtol = 3f-6 atol = 3f-6
end
