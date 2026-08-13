struct _CGScaledOperator{T}
    scale::T
end

Base.adjoint(A::_CGScaledOperator) = _CGScaledOperator(conj(A.scale))

function LinearAlgebra.mul!(y, A::_CGScaledOperator, x)
    mul!(y, A.scale, x)
    return y
end

function cg_tests()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    global_size = (4 * nprocs,)
    process_grid = (nprocs,)
    right_hand_side_array = reshape(
        ComplexF64.(1:prod(global_size)), 1, 1, global_size...)
    b = LatticeMatrix(right_hand_side_array, 1, process_grid;
        nw=1, numtemps=3)

    @testset "explicit-workspace CG" begin
        x = similar(b)
        r = similar(b)
        p = similar(b)
        Ap = similar(b)
        clear_matrix!(x)

        A = _CGScaledOperator(4.0)
        result = cg!(x, A, b, r, p, Ap;
            rtol=1e-13, atol=0, maxiter=10)
        @test result isa CGResult
        @test result.converged
        @test result.iterations == 1
        @test result.reason == :converged
        solution = gather_matrix(x)
        if rank == 0
            @test solution ≈ right_hand_side_array ./ 4 atol=2e-13 rtol=2e-13
        end

        clear_matrix!(x)
        zero_b = similar(b)
        clear_matrix!(zero_b)
        zero_result = cg!(x, A, zero_b, r, p, Ap;
            rtol=1e-13, atol=0, maxiter=10)
        @test zero_result.converged
        @test zero_result.iterations == 0
        @test iszero(zero_result.residual_norm)
        @test iszero(zero_result.relative_residual)

        clear_matrix!(x)
        stopped = cg!(x, A, b, r, p, Ap; maxiter=0)
        @test !stopped.converged
        @test stopped.reason == :maximum_iterations

        clear_matrix!(x)
        indefinite = cg!(x, _CGScaledOperator(-1.0), b, r, p, Ap;
            maxiter=10)
        @test !indefinite.converged
        @test indefinite.reason == :nonpositive_curvature

        @test_throws ArgumentError cg!(x, A, b, x, p, Ap)
        @test_throws ArgumentError cg!(x, A, b, r, p, Ap; rtol=-1)
        @test_throws ArgumentError cg!(x, A, b, r, p, Ap; maxiter=-1)
    end

    @testset "explicit DdagD temporary" begin
        x = similar(b)
        r = similar(b)
        p = similar(b)
        Ap = similar(b)
        Dtemp = similar(b)
        output = similar(b)
        clear_matrix!(x)

        D = _CGScaledOperator(2 + im)
        normal_operator = DdagDOp(D, Dtemp)
        @test adjoint(normal_operator) === normal_operator
        mul!(output, normal_operator, b)
        normal_result = gather_matrix(output)
        if rank == 0
            @test normal_result ≈ 5 .* right_hand_side_array
        end

        status = solve!(x, normal_operator, b, r, p, Ap;
            rtol=1e-13, maxiter=10)
        @test status.converged
        solution = gather_matrix(x)
        if rank == 0
            @test solution ≈ right_hand_side_array ./ 5 atol=2e-13 rtol=2e-13
        end

        aliased_operator = DdagDOp(D, b)
        @test_throws ArgumentError mul!(output, aliased_operator, b)
    end

    @testset "pool-based CG compatibility" begin
        x = similar(b)
        clear_matrix!(x)
        temps = LatticeMatrices.PreallocatedArray(
            b; num=3, haslabel=false)
        result = LatticeMatrices.cg(x, _CGScaledOperator(4.0), b, temps;
            eps=1e-13, maxsteps=10, verboselevel=2)
        @test result === nothing
        solution = gather_matrix(x)
        if rank == 0
            @test solution ≈ right_hand_side_array ./ 4 atol=2e-13 rtol=2e-13
        end
    end

    return nothing
end
