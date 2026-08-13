function _enzyme_epoch_shift_loss(U1, U2, U3, U4, temp)
    C = temp[1]
    clear_matrix!(C)
    mul_AshiftB!(C, U1, U2, (1, 0, 0, 0))
    return realtrace(C)
end

function enzyme_gradient_tests()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (4 * nprocs, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)
    NC = 2
    nw = 1

    nvalues = NC * NC * prod(global_size)
    components = reshape(Float64.(1:nvalues), NC, NC, global_size...)
    values1 = complex.(components ./ nvalues, reverse(components; dims=2) ./ (2nvalues))
    values2 = reverse(values1; dims=1)
    direction1_values = complex.(components ./ (3nvalues), -components ./ (5nvalues))
    direction2_values = reverse(direction1_values; dims=2)

    U1 = LatticeMatrix(values1, 4, process_grid; nw)
    U2 = LatticeMatrix(values2, 4, process_grid; nw)
    U3 = deepcopy(U1)
    U4 = deepcopy(U2)
    direction1 = LatticeMatrix(direction1_values, 4, process_grid; nw)
    direction2 = LatticeMatrix(direction2_values, 4, process_grid; nw)
    set_halo!.((U1, U2, U3, U4, direction1, direction2))

    dU = [similar(U1) for _ in 1:4]
    temp = [similar(U1)]
    dtemp = [similar(U1)]
    clear_matrix!.(dU)
    clear_matrix!.(temp)
    clear_matrix!.(dtemp)

    # Exercise the epoch-aware explicit-shift path inside Enzyme's primal rule.
    add_matrix!(U2, direction2, 0.25)

    @testset "Enzyme gradient with halo epochs" begin
        @test Base.get_extension(LatticeMatrices, :LatticeMatricesEnzymeExt) !== nothing
        @test halo_is_dirty(U2)

        Enzyme_derivative!(
            _enzyme_epoch_shift_loss,
            U1, U2, U3, U4,
            dU[1], dU[2], dU[3], dU[4];
            temp,
            dtemp,
        )

        @test !halo_is_dirty(U2)
        @test all(iszero, dU[3].A)
        @test all(iszero, dU[4].A)

        epsilon = 1e-6
        U1_plus, U2_plus = deepcopy(U1), deepcopy(U2)
        U1_minus, U2_minus = deepcopy(U1), deepcopy(U2)
        add_matrix!(U1_plus, direction1, epsilon)
        add_matrix!(U2_plus, direction2, epsilon)
        add_matrix!(U1_minus, direction1, -epsilon)
        add_matrix!(U2_minus, direction2, -epsilon)

        plus = _enzyme_epoch_shift_loss(U1_plus, U2_plus, U3, U4, temp)
        minus = _enzyme_epoch_shift_loss(U1_minus, U2_minus, U3, U4, temp)
        finite_difference = (plus - minus) / (2epsilon)
        enzyme_directional = real(dot(dU[1], direction1) + dot(dU[2], direction2))

        @test enzyme_directional ≈ finite_difference atol=5e-6 rtol=5e-7

        enzyme_gradient = copy(dU[1].A)
        legacy_gradient = deepcopy(dU[1])
        Wirtinger!(dU[1])
        @test halo_is_dirty(dU[1])
        @test dU[1].A ≈ 0.5 .* conj.(enzyme_gradient)

        @test_deprecated LatticeMatrices.Wiltinger!(legacy_gradient)
        @test legacy_gradient.A ≈ dU[1].A
        @test isdefined(LatticeMatrices, :Wiltinger_derivative!)
        @test isdefined(LatticeMatrices, :Wiltinger_numerical_derivative)
    end
end
