using MPI
using CUDA
import JACC

JACC.@init_backend

using LatticeMatrices
using LinearAlgebra
using Test

function elapsed_per_call_ms(f, iterations)
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

function median_value(values)
    sort!(values)
    return values[cld(length(values), 2)]
end

function run_wilson_clover_benchmark()
    MPI.Comm_size(MPI.COMM_WORLD) == 1 || error(
        "This benchmark currently expects one MPI rank and one GPU")
    CUDA.functional() || error("CUDA is not functional")
    JACC.backend == "cuda" || error("JACC CUDA backend is required")

    linear_size = parse(Int, get(ENV, "LATTICEMATRICES_CLOVER_BENCH_L", "24"))
    iterations = parse(Int, get(ENV, "LATTICEMATRICES_CLOVER_BENCH_ITERS", "20"))
    samples = parse(Int, get(ENV, "LATTICEMATRICES_CLOVER_BENCH_SAMPLES", "5"))
    lattice_size = ntuple(_ -> linear_size, 4)
    process_grid = (1, 1, 1, 1)
    NC = 3

    unit_link = zeros(ComplexF64, NC, NC, lattice_size...)
    for color in 1:NC
        selectdim(selectdim(unit_link, 1, color), 1, color) .= 1
    end
    psi_array = fill(0.25 - 0.5im, NC, 4, lattice_size...)

    U = [LatticeMatrix(unit_link, 4, process_grid; nw=1) for _ in 1:4]
    psi = LatticeMatrix(psi_array, 4, process_grid; nw=1)
    result_wilson = similar(psi)
    result_clover = similar(psi)
    wilson = WilsonDiracOperator4D(U, 0.12)
    clover = WilsonDiracCloverOperator4D(U, 0.12, 1.0)
    JACC.synchronize()

    update_samples = [elapsed_per_call_ms(
        () -> update_clover!(clover), max(1, iterations ÷ 4)) for _ in 1:samples]
    wilson_samples = Float64[]
    clover_samples = Float64[]
    ratios = Float64[]
    for sample in 1:samples
        if isodd(sample)
            wilson_ms = elapsed_per_call_ms(
                () -> mul!(result_wilson, wilson, psi), iterations)
            clover_ms = elapsed_per_call_ms(
                () -> mul!(result_clover, clover, psi), iterations)
        else
            clover_ms = elapsed_per_call_ms(
                () -> mul!(result_clover, clover, psi), iterations)
            wilson_ms = elapsed_per_call_ms(
                () -> mul!(result_wilson, wilson, psi), iterations)
        end
        push!(wilson_samples, wilson_ms)
        push!(clover_samples, clover_ms)
        push!(ratios, clover_ms / wilson_ms)
    end

    update_ms = median_value(update_samples)
    wilson_ms = median_value(wilson_samples)
    clover_ms = median_value(clover_samples)
    overhead_percent = 100 * (median_value(ratios) - 1)

    mul!(result_wilson, wilson, psi)
    mul!(result_clover, clover, psi)
    JACC.synchronize()
    wilson_host = gather_matrix(result_wilson)
    clover_host = gather_matrix(result_clover)
    @test clover_host ≈ wilson_host atol=2e-12 rtol=2e-12

    println("GPU=", CUDA.name(CUDA.device()))
    println("lattice=", join(lattice_size, "x"), " NC=", NC,
        " precision=ComplexF64")
    println("clover_update_ms=", update_ms)
    println("wilson_apply_ms=", wilson_ms)
    println("clover_apply_ms=", clover_ms)
    println("clover_overhead_percent=", overhead_percent)
    return nothing
end

MPI.Init()
try
    run_wilson_clover_benchmark()
finally
    MPI.Finalize()
end
