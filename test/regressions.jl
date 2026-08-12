function _reduce_three_lattices_with_scale(i, C, A, B, scale,
    ::Val{NC1}, ::Val{NG}, ::Val{nw},
    ::Val{NC2}, ::Val{NG2}, ::Val{nw2},
    ::Val{NC3}, ::Val{NG3}, ::Val{nw3}, dindexer) where {
    NC1,NG,nw,NC2,NG2,nw2,NC3,NG3,nw3}
    indices = delinearize(dindexer, i, nw)
    return scale * (C[1, 1, indices...] + A[1, 1, indices...] + B[1, 1, indices...])
end

function regressiontests()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)

    @testset "5D nw=0 safety" begin
        L5 = 2
        gauge_size = (2 * nprocs, 2, 2, 2)
        fermion_size = (gauge_size..., L5)
        gauge_grid = (nprocs, 1, 1, 1)
        fermion_grid = (gauge_grid..., 1)

        gauge = ones(ComplexF64, 1, 1, gauge_size...)
        fermion = ones(ComplexF64, 1, 4, fermion_size...)
        U = [LatticeMatrix(gauge, 4, gauge_grid; nw=0) for _ in 1:4]
        ψ = LatticeMatrix(fermion, 5, fermion_grid; nw=0)
        C = similar(ψ)
        D = D5DW_MobiusDomainwallOperator5D(U, L5, 0.1, -1.0, 1.0, 1.0)

        @test_throws ArgumentError mul!(C, D, ψ)
        @test_throws ArgumentError mul!(C, adjoint(D), ψ)
        @test_throws ArgumentError LatticeMatrices.D4x_5D!(C, U, ψ, 1.0)
        @test_throws ArgumentError LatticeMatrices.apply_F_5D!(C, 0.1, L5, ψ)
        @test_throws ArgumentError LatticeMatrices.apply_δF_5D!(C, 0.1, L5, ψ)
    end

    @testset "long shifts with halos" begin
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        global_size = (4 * nprocs,)
        process_grid = (nprocs,)
        A = reshape(ComplexF64.(1:prod(global_size)), 1, 1, global_size...)
        M = LatticeMatrix(A, 1, process_grid; nw=1)
        C = similar(M)

        for amount in (2, -2, global_size[1] + 1, -global_size[1] - 1)
            substitute!(C, Shifted_Lattice(M, (amount,)))
            result = gather_matrix(C)
            if rank == 0
                expected = circshift(A, (0, 0, -amount))
                @test result == expected
            end
        end
    end

    @testset "three-lattice parallel_reduce with arguments" begin
        global_size = (4 * nprocs,)
        process_grid = (nprocs,)
        A = fill(1.0 + 0im, 1, 1, global_size...)
        B = fill(2.0 + 0im, 1, 1, global_size...)
        C = fill(3.0 + 0im, 1, 1, global_size...)
        MA = LatticeMatrix(A, 1, process_grid; nw=0)
        MB = LatticeMatrix(B, 1, process_grid; nw=0)
        MC = LatticeMatrix(C, 1, process_grid; nw=0)
        scale = 2.0 + 0im

        result = JACC.parallel_reduce(
            _reduce_three_lattices_with_scale, MA, MB, MC, scale)
        @test result == scale * sum(A + B + C)
    end

    @testset "complex boundary phases are reversible" begin
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        global_size = (4 * nprocs,)
        process_grid = (nprocs,)
        A = reshape(ComplexF64.(1:prod(global_size)), 1, 1, global_size...)

        for nw in (0, 1)
            M = LatticeMatrix(A, 1, process_grid; nw, phases=(im,))
            shifted = similar(M)
            restored = similar(M)
            substitute!(shifted, Shifted_Lattice(M, (1,)))
            set_halo!(shifted)
            substitute!(restored, Shifted_Lattice(shifted, (-1,)))
            result = gather_matrix(restored)
            if rank == 0
                @test result == A
            end
        end
    end

    @testset "constructor validation" begin
        @test_throws ArgumentError LatticeMatrix(
            1, 1, 1, (4 * nprocs,), (nprocs + 1,); nw=0)
        @test_throws ArgumentError LatticeMatrix(
            1, 1, 1, (4 * nprocs,), (nprocs,); nw=0, phases=(0,))
        @test_throws ArgumentError LatticeMatrix(
            1, 1, 2, (4 * nprocs,), (nprocs, 1); nw=0)
        if nprocs > 1
            @test_throws ArgumentError LatticeMatrix(
                1, 1, 1, (4 * nprocs + 1,), (nprocs,); nw=0)
        end
    end

    @testset "similar owns communication buffers" begin
        global_size = (4 * nprocs,)
        process_grid = (nprocs,)
        M = LatticeMatrix(1, 1, 1, global_size, process_grid; nw=1)
        S = similar(M)
        @test M.buf !== S.buf
        @test M.buf_host !== S.buf_host
        @test all(M.buf[i] !== S.buf[i] for i in eachindex(M.buf))
        @test all(M.buf_host[i] !== S.buf_host[i] for i in eachindex(M.buf_host))
    end
end
