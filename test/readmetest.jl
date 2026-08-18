using LatticeMatrices
using Test
using MPI
using LinearAlgebra
using Random
import JACC

JACC.@init_backend

initialized_here = !MPI.Initialized()
initialized_here && MPI.Init()

try
    @testset "README quick tour" begin
        d = DIndexer((16, 16, 16, 16))
        @test linearize(d, (1, 1, 1, 1)) == 1
        @test Tuple(delinearize(d, 4)) == (4, 1, 1, 1)
        @test shiftindices((4, 1, 1, 1), (1, 0, 0, 0)) ==
            (5, 1, 1, 1)

        nprocs = MPI.Comm_size(MPI.COMM_WORLD)
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        dim = 4
        gsize = (4 * nprocs, 4, 4, 4)
        PEs = (nprocs, 1, 1, 1)
        nw = 1
        NC = 3

        Random.seed!(1234)
        A2 = rand(ComplexF64, NC, NC, gsize...)
        A3 = rand(ComplexF64, NC, NC, gsize...)
        M = LatticeMatrix(
            NC, NC, dim, gsize, PEs; nw, elementtype=ComplexF64)
        M2 = LatticeMatrix(A2, dim, PEs; nw, numtemps=2)
        M3 = LatticeMatrix(A3, dim, PEs; nw)

        set_halo!(M)
        gathered = gather_matrix(M; root=0)
        rank == 0 && @test size(gathered) == (NC, NC, gsize...)
        @test size(gather_and_bcast_matrix(M; root=0)) ==
            (NC, NC, gsize...)

        add_matrix!(M, M2)
        @test halo_is_dirty(M)
        shifted = Shifted_Lattice(M, (1, 0, 0, 0))
        @test !halo_is_dirty(M)
        release!(shifted)

        M1 = similar(M)
        mul!(M1, M2, M3)
        product = gather_matrix(M1; root=0)
        if rank == 0
            for site in CartesianIndices(gsize)
                indices = Tuple(site)
                @test product[:, :, indices...] ≈
                    A2[:, :, indices...] * A3[:, :, indices...]
            end
        end

        expt!(M1, M2, 1)
        exponential = gather_matrix(M1; root=0)
        if rank == 0
            indices = ntuple(_ -> 1, dim)
            @test exponential[:, :, indices...] ≈ exp(A2[:, :, indices...])
        end

        M2p = Shifted_Lattice(M2, (1, 0, 0, 0))
        M3p = Shifted_Lattice(M3, (0, 1, 0, 0))
        mul!(M1, M2p, M3)
        mul!(M1, M2', M3p)
        mul!(M1, M2', M3')
        release!(M2p)
        release!(M3p)

        long_shift = Shifted_Lattice(M2, (nw + 2, 0, 0, 0))
        @test isopen(long_shift)
        try
            mul!(M1, long_shift, M3)
        finally
            release!(long_shift)
        end
        @test !isopen(long_shift)
        @test release!(long_shift) === nothing

        scoped_called = Ref(false)
        with_shifted_lattice(M2, (nw + 2, 0, 0, 0)) do shifted
            mul!(M1, shifted, M3)
            scoped_called[] = true
        end
        @test scoped_called[]

        @test isfinite(real(tr(M1)))
        reduced_sum = allsum(M1)
        if rank == 0
            @test isfinite(real(reduced_sum))
        else
            @test reduced_sum === nothing
        end

        unit_link = zeros(ComplexF64, NC, NC, gsize...)
        for site in CartesianIndices(gsize), color in 1:NC
            unit_link[color, color, Tuple(site)...] = 1
        end
        U = [LatticeMatrix(unit_link, 4, PEs; nw) for _ in 1:4]

        @testset "README Wilson, clover, and staggered" begin
            Random.seed!(1234)
            psi_host = randn(ComplexF64, NC, 4, gsize...)
            psi = LatticeMatrix(
                psi_host, 4, PEs; nw, phases=(1, 1, 1, -1))
            out = similar(psi)
            wilson = WilsonDiracOperator4D(U, 0.12)
            mul!(out, wilson, psi)
            mul!(out, adjoint(wilson), psi)
            clover = WilsonDiracCloverOperator4D(U, 0.12, 1.0)
            mul!(out, clover, psi)
            mul!(out, adjoint(clover), psi)
            hopping = WilsonDiracOperator4D_Donly(U)
            mul!(out, hopping, psi)
            field_strength = CloverFieldStrength4D(U)
            @test length(field_strength) == 6
            @test update_clover!(clover) === clover
            clover_gradient = [similar(link) for link in U]
            clear_matrix!.(clover_gradient)
            wilson_clover_link_pullback!(
                clover_gradient, clover, U, psi, psi)
            @test all(field -> all(isfinite, field.A), clover_gradient)

            staggered_host = randn(ComplexF64, NC, 1, gsize...)
            staggered = LatticeMatrix(
                staggered_host, 4, PEs;
                nw, phases=(1, 1, 1, -1))
            staggered_out = similar(staggered)
            staggered_operator = StaggeredDiracOperator4D(U, 0.01)
            mul!(staggered_out, staggered_operator, staggered)
            mul!(staggered_out, adjoint(staggered_operator), staggered)
        end

        @testset "README HISQ" begin
            nw_hisq = 3
            U_hisq = [
                LatticeMatrix(unit_link, 4, PEs; nw=nw_hisq)
                for _ in 1:4
            ]
            level1 = hisq_fat7_level1(U_hisq)
            level1_preallocated = [similar(link) for link in U_hisq]
            hisq_fat7_level1!(level1_preallocated, U_hisq)
            @test all(
                level1_preallocated[mu].A ≈ level1[mu].A for mu in 1:4)

            naik_epsilon = -0.083
            links = hisq_links_from_thin(
                U_hisq; naik_epsilon)
            hisq = HISQDiracOperator4D(
                links, 0.01; naik_epsilon)
            hisq_from_thin = HISQDiracOperator4D(
                U_hisq, 0.01; naik_epsilon)
            staggered_host = randn(ComplexF64, NC, 1, gsize...)
            staggered = LatticeMatrix(
                staggered_host, 4, PEs;
                nw=nw_hisq, phases=(1, 1, 1, -1))
            hisq_out = similar(staggered)
            mul!(hisq_out, hisq, staggered)
            mul!(hisq_out, adjoint(hisq_from_thin), staggered)

            V = [similar(link) for link in U_hisq]
            W = [similar(link) for link in U_hisq]
            X = [similar(link) for link in U_hisq]
            L = [similar(link) for link in U_hisq]
            hisq_links_from_thin!(
                X, L, V, W, U_hisq; naik_epsilon)
            cache = HISQDiracCache4D(
                U_hisq, 0.01; naik_epsilon)
            mul_cached_hisq!(
                hisq_out, cache,
                U_hisq[1], U_hisq[2], U_hisq[3], U_hisq[4], staggered)
            mul_cached_hisq_adjoint!(
                hisq_out, cache,
                U_hisq[1], U_hisq[2], U_hisq[3], U_hisq[4], staggered)
        end

        @testset "README domain wall" begin
            L5 = 4
            gsize5 = (gsize..., L5)
            PEs5 = (PEs..., 1)
            psi5 = LatticeMatrix(
                randn(ComplexF64, NC, 4, gsize5...), 5, PEs5;
                nw=1, phases=(1, 1, 1, -1, 1))
            out5 = similar(psi5)
            domainwall = D5DW_MobiusDomainwallOperator5D(
                U, L5, 0.01, -1.0, 2.0, 1.0)
            mul!(out5, domainwall, psi5)
            mul!(out5, adjoint(domainwall), psi5)

            a5 = 1 .+ 0.05 .* sin.(2pi .* (0:L5-1) ./ L5)
            b5 = 1.5 .+ 0.10 .* cos.(2pi .* (0:L5-1) ./ L5)
            c5 = 0.5 .+ 0.08 .* sin.(2pi .* (0:L5-1) ./ L5)
            generalized = D5DW_GeneralizedDomainwallOperator5D(
                U, L5, 0.01, -1.0, a5, b5, c5)
            mul!(out5, generalized, psi5)
            mul!(out5, adjoint(generalized), psi5)
            @test adjoint(adjoint(generalized)) === generalized
        end
    end
    MPI.Barrier(MPI.COMM_WORLD)
finally
    initialized_here && MPI.Finalize()
end
