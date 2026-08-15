using MPI
using CUDA
import JACC

JACC.@init_backend

using LatticeMatrices
using LinearAlgebra
using Test

include("../staggered_dirac.jl")

function _elapsed_staggered_ms(f, iterations)
    f()
    JACC.synchronize()
    elapsed = @elapsed begin
        for _ in 1:iterations
            f()
        end
        JACC.synchronize()
    end
    return 1_000 * elapsed / iterations
end

function _median_staggered(values)
    sort!(values)
    return values[cld(length(values), 2)]
end

function _staggered_gpu_correctness(::Type{T}) where {T<:Complex}
    real_type = typeof(real(zero(T)))
    lattice_size = (4, 4, 4, 4)
    process_grid = (1, 1, 1, 1)
    phases = (1, 1, 1, -1)
    mass = real_type(0.17)

    links = _staggered_test_links(lattice_size, 3; elementtype=T)
    psi_host = _staggered_test_fermion(lattice_size, 3; elementtype=T)
    reference = _staggered_test_reference(links, psi_host, mass, phases)
    reference_dag = _staggered_test_reference(
        links, psi_host, mass, phases; adjoint_operator=true)

    U = [LatticeMatrix(link, 4, process_grid; nw=1) for link in links]
    psi = LatticeMatrix(psi_host, 4, process_grid; nw=1, phases)
    result = similar(psi)
    operator = StaggeredDiracOperator4D(U, mass)
    @test result.A isa CUDA.CuArray

    tolerance = T === ComplexF32 ? 3f-5 : 4e-12
    mul!(result, operator, psi)
    JACC.synchronize()
    @test gather_matrix(result) ≈ reference atol=tolerance rtol=tolerance

    mul!(result, adjoint(operator), psi)
    JACC.synchronize()
    @test gather_matrix(result) ≈ reference_dag atol=tolerance rtol=tolerance
    return nothing
end

function _staggered_gpu_benchmark(::Type{T}) where {T<:Complex}
    real_type = typeof(real(zero(T)))
    linear_size = parse(Int,
        get(ENV, "LATTICEMATRICES_STAGGERED_BENCH_L", "24"))
    iterations = parse(Int,
        get(ENV, "LATTICEMATRICES_STAGGERED_BENCH_ITERS", "30"))
    samples = parse(Int,
        get(ENV, "LATTICEMATRICES_STAGGERED_BENCH_SAMPLES", "5"))
    linear_size > 2 || error("benchmark linear size must be greater than two")
    iterations > 0 || error("benchmark iterations must be positive")
    samples > 0 || error("benchmark samples must be positive")

    lattice_size = ntuple(_ -> linear_size, 4)
    process_grid = (1, 1, 1, 1)
    NC = 3
    unit_link = zeros(T, NC, NC, lattice_size...)
    for color in 1:NC
        selectdim(selectdim(unit_link, 1, color), 1, color) .= one(T)
    end
    psi_host = fill(T(real_type(0.25), real_type(-0.5)),
        NC, 1, lattice_size...)

    U = [LatticeMatrix(unit_link, 4, process_grid; nw=1) for _ in 1:4]
    psi = LatticeMatrix(psi_host, 4, process_grid;
        nw=1, phases=(1, 1, 1, -1))
    result = similar(psi)
    operator = StaggeredDiracOperator4D(U, real_type(0.01))
    JACC.synchronize()

    forward_samples = [_elapsed_staggered_ms(
        () -> mul!(result, operator, psi), iterations) for _ in 1:samples]
    adjoint_samples = [_elapsed_staggered_ms(
        () -> mul!(result, adjoint(operator), psi), iterations)
                       for _ in 1:samples]
    forward_ms = _median_staggered(forward_samples)
    adjoint_ms = _median_staggered(adjoint_samples)

    sites = prod(lattice_size)
    # Bridge++ Fopr_Staggered::flop_count gives 594 real flops/site for Nc=3.
    flops_per_site = 594
    # Lower bound: eight gauge matrices, eight neighboring color vectors,
    # one local vector, and one output vector (102 complex numbers/site).
    minimum_bytes_per_site = 102 * sizeof(T)
    msites_per_second = sites / (forward_ms * 1_000)
    gflops_per_second = flops_per_site * sites / (forward_ms * 1.0e6)
    minimum_bandwidth_gbs = minimum_bytes_per_site * sites /
                            (forward_ms * 1.0e6)

    println("GPU=", CUDA.name(CUDA.device()))
    println("lattice=", join(lattice_size, "x"), " NC=", NC,
        " precision=", T)
    println("staggered_apply_ms=", forward_ms)
    println("staggered_adjoint_ms=", adjoint_ms)
    println("staggered_msites_per_second=", msites_per_second)
    println("staggered_bridge_convention_gflops=", gflops_per_second)
    println("staggered_minimum_traffic_bandwidth_GBs=", minimum_bandwidth_gbs)
    return nothing
end

function run_staggered_gpu_benchmark()
    MPI.Comm_size(MPI.COMM_WORLD) == 1 || error(
        "This benchmark currently expects one MPI rank and one GPU")
    CUDA.functional() || error("CUDA is not functional")
    JACC.backend == "cuda" || error("JACC CUDA backend is required")

    precision = lowercase(get(
        ENV, "LATTICEMATRICES_STAGGERED_BENCH_PRECISION", "float32"))
    elementtype = precision == "float32" ? ComplexF32 :
                  precision == "float64" ? ComplexF64 :
                  error("precision must be Float32 or Float64")
    _staggered_gpu_correctness(elementtype)
    GC.gc(true)
    CUDA.reclaim()
    _staggered_gpu_benchmark(elementtype)
    return nothing
end

MPI.Init()
try
    run_staggered_gpu_benchmark()
finally
    MPI.Finalize()
end
