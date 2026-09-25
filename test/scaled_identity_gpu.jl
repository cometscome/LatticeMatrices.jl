# Opt-in: run in an environment with CUDA and JACC's cuda backend enabled.
using CUDA, LatticeMatrices, LinearAlgebra, Test
import JACC
JACC.@init_backend

CUDA.functional() || error("A functional CUDA device is required")
CUDA.allowscalar(false)
println((device=CUDA.name(CUDA.device()), julia=VERSION,
    cuda=Base.pkgversion(CUDA), jacc=Base.pkgversion(JACC)))

@testset "scaled identity executes on CUDA storage" begin
    probe = LatticeMatrix(1, 1, 1, (2,), (1,); nw=1)
    @test probe.A isa CUDA.CuArray
    @test similar(probe).A isa CUDA.CuArray
    @test LatticeMatrices._center_component_threads(probe.A)
    @test !LatticeMatrices._center_component_threads(zeros(ComplexF64, 1, 1, 2))
end

include("communication_helpers.jl")
include("scaled_identity.jl")

@testset "CUDA component threads across partial blocks" begin
    lattice, grid = (5, 7, 3, 2), (1, 1, 1, 1)
    rng = MersenneTwister(92426)
    ashift, bshift = (1, -1, 0, 1), (-1, 0, 1, 0)
    for nc in (2, 3, 4), T in (ComplexF32, ComplexF64)
        tol = T == ComplexF32 ? 3e-6 : 3e-13
        adata = randn(rng, T, nc, nc, lattice...)
        bdata = randn(rng, T, 1, 1, lattice...)
        cdata = randn(rng, T, nc, nc, lattice...)
        A = LatticeMatrix(adata, 4, grid; nw=1)
        scalar = LatticeMatrix(bdata, 4, grid; nw=1)
        B = ScaledIdentityLattice(nc, scalar)
        C = LatticeMatrix(cdata, 4, grid; nw=1)
        reference = LatticeMatrix(cdata, 4, grid; nw=1)
        alpha, beta = T(0.3 + 0.2im), T(-0.1)
        aview = conj.(permutedims(circshift(adata, (0, 0, (-).(ashift)...)),
            (2, 1, 3, 4, 5, 6)))
        with_shifted_lattice(A, ashift) do sa
            with_shifted_lattice(B, bshift) do sb
                bview = conj.(circshift(bdata, (0, 0, (-).(bshift)...)))
                mul!(C, sa', sb', alpha, beta)
                PrelinearScaledIdentity.mul_scalar!(reference, sa', sb.scalar', alpha, beta)
                result = gather_and_bcast_matrix(C)
                @test result ≈ alpha .* (aview .* bview) .+ beta .* cdata atol=tol rtol=tol
                @test isequal(result, gather_and_bcast_matrix(reference))
                # Reuse a halo-backed view after changing the live scalar field.
                bdata .*= T(0.5 + 0.3im)
                substitute!(scalar, LatticeMatrix(bdata, 4, grid; nw=1))
                mul!(C, sa', sb')
                PrelinearScaledIdentity.mul_scalar!(reference, sa', sb.scalar')
                result = gather_and_bcast_matrix(C)
                bview = conj.(circshift(bdata, (0, 0, (-).(bshift)...)))
                @test result ≈ aview .* bview atol=tol rtol=tol
                @test isequal(result, gather_and_bcast_matrix(reference))
            end
        end
    end
end
CUDA.synchronize()
