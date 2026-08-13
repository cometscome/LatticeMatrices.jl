using MPI
using CUDA
import JACC

JACC.@init_backend

using LatticeMatrices
using LinearAlgebra
using Enzyme
using Test

function _domainwall_bench_loss(operator, psi, left, result)
    mul!(result, operator, psi)
    return real(dot(left, result))
end

function _domainwall_elapsed_ms(f, iterations)
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

function _domainwall_median(values)
    sort!(values)
    return values[cld(length(values), 2)]
end

function run_domainwall_pullback_benchmark()
    MPI.Comm_size(MPI.COMM_WORLD) == 1 || error(
        "This benchmark currently expects one MPI rank and one GPU")
    CUDA.functional() || error("CUDA is not functional")
    JACC.backend == "cuda" || error("JACC CUDA backend is required")

    linear_size = parse(Int,
        get(ENV, "LATTICEMATRICES_DOMAINWALL_BENCH_L", "16"))
    L5 = parse(Int,
        get(ENV, "LATTICEMATRICES_DOMAINWALL_BENCH_L5", "12"))
    iterations = parse(Int,
        get(ENV, "LATTICEMATRICES_DOMAINWALL_BENCH_ITERS", "20"))
    samples = parse(Int,
        get(ENV, "LATTICEMATRICES_DOMAINWALL_BENCH_SAMPLES", "5"))
    precision = lowercase(get(
        ENV, "LATTICEMATRICES_DOMAINWALL_BENCH_PRECISION", "float32"))
    elementtype = precision == "float32" ? ComplexF32 :
                  precision == "float64" ? ComplexF64 :
                  error("precision must be Float32 or Float64")
    realtype = typeof(real(zero(elementtype)))
    linear_size > 2 || error("benchmark linear size must be greater than two")
    L5 > 0 || error("benchmark L5 must be positive")
    iterations > 0 || error("benchmark iterations must be positive")
    samples > 0 || error("benchmark samples must be positive")

    lattice_size = ntuple(_ -> linear_size, 4)
    fermion_size = (lattice_size..., L5)
    process_grid = (1, 1, 1, 1)
    process_grid5 = (process_grid..., 1)
    NC = 3
    unit_link = zeros(elementtype, NC, NC, lattice_size...)
    for color in 1:NC
        selectdim(selectdim(unit_link, 1, color), 1, color) .= one(elementtype)
    end
    psi_host = fill(elementtype(realtype(0.25), realtype(-0.5)),
        NC, 4, fermion_size...)
    left_host = fill(elementtype(realtype(-0.125), realtype(0.375)),
        NC, 4, fermion_size...)

    U = [LatticeMatrix(unit_link, 4, process_grid; nw=1) for _ in 1:4]
    dU = [similar(link) for link in U]
    psi = LatticeMatrix(psi_host, 5, process_grid5;
        nw=1, phases=(1, 1, 1, -1, 1))
    left = LatticeMatrix(left_host, 5, process_grid5;
        nw=1, phases=(1, 1, 1, -1, 1))
    dpsi = similar(psi)
    result = similar(psi)
    dresult = similar(psi)
    expected_dpsi = similar(psi)
    clear_matrix!.(dU)
    clear_matrix!.((dpsi, result, dresult, expected_dpsi))

    operator = D5DW_MobiusDomainwallOperator5D(
        U, L5, realtype(0.01), realtype(-1), realtype(2), realtype(1))
    shadow_operator = D5DW_MobiusDomainwallOperator5D(
        dU, L5, realtype(0.01), realtype(-1), realtype(2), realtype(1))

    forward!() = mul!(result, operator, psi)
    function pullback!()
        Enzyme.autodiff(
            Enzyme.Reverse,
            Enzyme.Const(_domainwall_bench_loss),
            Enzyme.Active,
            Enzyme.Duplicated(operator, shadow_operator),
            Enzyme.Duplicated(psi, dpsi),
            Enzyme.Const(left),
            Enzyme.Duplicated(result, dresult),
        )
        return nothing
    end

    Enzyme.API.strictAliasing!(false)
    # Compile the CUDA kernels and check the spinor part of the pullback once.
    pullback!()
    JACC.synchronize()
    mul!(expected_dpsi, adjoint(operator), left)
    JACC.synchronize()
    tolerance = elementtype === ComplexF32 ? 8f-5 : 2e-11
    @test gather_matrix(dpsi) ≈ gather_matrix(expected_dpsi) atol=tolerance rtol=tolerance
    @test all(iszero, Array(dresult.A))
    @test sum(sum(abs2, link.A) for link in dU) > 0

    clear_matrix!.(dU)
    clear_matrix!.((dpsi, result, dresult))
    JACC.synchronize()
    forward_samples = Float64[]
    pullback_samples = Float64[]
    for sample in 1:samples
        if isodd(sample)
            push!(forward_samples,
                _domainwall_elapsed_ms(forward!, iterations))
            push!(pullback_samples,
                _domainwall_elapsed_ms(pullback!, iterations))
        else
            push!(pullback_samples,
                _domainwall_elapsed_ms(pullback!, iterations))
            push!(forward_samples,
                _domainwall_elapsed_ms(forward!, iterations))
        end
    end

    forward_ms = _domainwall_median(forward_samples)
    pullback_ms = _domainwall_median(pullback_samples)
    five_dimensional_sites = prod(lattice_size) * L5
    println("GPU=", CUDA.name(CUDA.device()))
    println("lattice=", join(lattice_size, "x"), "x", L5,
        " NC=", NC, " precision=", elementtype)
    println("domainwall_apply_ms=", forward_ms)
    println("domainwall_loss_and_pullback_ms=", pullback_ms)
    println("pullback_over_apply_ratio=", pullback_ms / forward_ms)
    println("apply_5d_Msites_per_second=",
        five_dimensional_sites / (forward_ms * 1_000))
    return nothing
end

MPI.Init()
try
    run_domainwall_pullback_benchmark()
finally
    MPI.Finalize()
end
