using LatticeMatrices, LinearAlgebra, Random
import JACC
JACC.@init_backend
include("../test/reference/scaled_identity_prelinear.jl")

function compare_us(functions; samples=51)
    for f in functions, _ in 1:4
        f()
    end
    JACC.synchronize()
    GC.gc()
    times = [Float64[] for _ in functions]
    rng = MersenneTwister(92026)
    for _ in 1:samples, k in randperm(rng, length(functions))
        start = time_ns()
        functions[k]()
        JACC.synchronize()
        push!(times[k], (time_ns() - start) / 1e3)
    end
    return map(t -> sort(t)[cld(samples, 2)], times)
end

function benchmark_scaled_identity()
    Random.seed!(92026)
    n = isempty(ARGS) ? 16 : parse(Int, ARGS[1])
    lattice, grid = (n,n,n,n), (1,1,1,1)
    nc = 3
    A = LatticeMatrix(randn(ComplexF64, nc, nc, lattice...), 4, grid; nw=1)
    scalar = LatticeMatrix(exp.(im .* randn(1,1,lattice...)), 4, grid; nw=1)
    B = ScaledIdentityLattice(nc, scalar)
    dense_B, C = similar(A), similar(A)
    substitute!(dense_B, B)
    for shift in ((0,0,0,0), (1,0,0,0))
        with_shifted_lattice(B, shift) do sb
            with_shifted_lattice(dense_B, shift) do db
                mul!(C, A, db)
                reference = gather_and_bcast_matrix(C)
                PrelinearScaledIdentity.mul_scalar!(C, A, sb.scalar)
                previous = gather_and_bcast_matrix(C)
                mul!(C, A, sb)
                result = gather_and_bcast_matrix(C)
                difference = maximum(abs, result - reference)
                previous_difference = maximum(abs, result - previous)
                @assert difference < 1e-12
                @assert isequal(result, previous)
                functions = (() -> mul!(C, A, db),
                    () -> PrelinearScaledIdentity.mul_scalar!(C, A, sb.scalar),
                    () -> mul!(C, A, sb))
                full_us, previous_us, compact_us = compare_us(functions)
                bytes = map(f -> @allocated(f()), functions)
                println((n=n, shift=shift, full_us=full_us, previous_us=previous_us,
                    compact_us=compact_us, full_speedup=full_us/compact_us,
                    previous_speedup=previous_us/compact_us, maxdiff=difference,
                    previous_maxdiff=previous_difference, allocated_bytes=bytes))
            end
        end
    end
    println((dense_elements=length(dense_B.A), scalar_elements=length(scalar.A),
        storage_ratio=length(dense_B.A)/length(scalar.A)))
end

benchmark_scaled_identity()
