struct _CGScaledOperator{T}
    scale::T
end

Base.adjoint(A::_CGScaledOperator) = _CGScaledOperator(conj(A.scale))

function LinearAlgebra.mul!(y, A::_CGScaledOperator, x)
    mul!(y, A.scale, x)
    return y
end

function cg_tests()
    nprocs = test_comm_size()
    rank = test_comm_rank()
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

    @testset "pre-v1 Dirac solver compatibility" begin
        scale = 2 + im
        apply_scale!(y, U1, U2, U3, U4, x, coefficient, phitemp, temp) =
            mul!(y, coefficient, x)
        apply_scale_adjoint!(y, U1, U2, U3, U4, x, coefficient, phitemp, temp) =
            mul!(y, conj(coefficient), x)

        U = [b for _ in 1:4]
        D = DiracOp(U, apply_scale!, apply_scale_adjoint!, scale, b;
            numtemp=1, numphitemp=4)
        normal_operator = DdagDOp(D)
        output = similar(b)
        mul!(output, normal_operator, b)
        normal_result = gather_matrix(output)
        if rank == 0
            @test normal_result ≈ abs2(scale) .* right_hand_side_array
        end

        x = similar(b)
        clear_matrix!(x)
        @test solve!(x, normal_operator, b; verboselevel=2) === nothing
        solution = gather_matrix(x)
        if rank == 0
            @test solution ≈ right_hand_side_array ./ abs2(scale) atol=2e-13 rtol=2e-13
        end

        expected_action = real(dot(b, b)) / abs2(scale)
        @test pseudofermion_action(D, b) ≈ expected_action atol=2e-13 rtol=2e-13
        @test count(D.phitemps._flagusing) == 0
    end

    return nothing
end
