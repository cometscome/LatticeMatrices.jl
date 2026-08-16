function _enzyme_epoch_shift_loss(U1, U2, U3, U4, temp)
    C = temp[1]
    clear_matrix!(C)
    mul_AshiftB!(C, U1, U2, (1, 0, 0, 0))
    return realtrace(C)
end

function _enzyme_long_shift_loss(U1, U2, U3, U4, temp)
    C = temp[1]
    clear_matrix!(C)
    shifted = shift_L(U2, (6, -3, 5, -4))
    substitute!(C, shifted)
    release!(shifted)
    return realtrace(C)
end

function _enzyme_expt_ta_loss(U1, U2, U3, U4, temp)
    expt_TA!(temp[1], U1, 0.3)
    return realtrace(temp[1])
end

function _enzyme_expt_ta_wrapper_loss(U1, U2, U3, U4, temp)
    expt!(temp[1], Traceless_AntiHermitian(U1), 0.3)
    return realtrace(temp[1])
end

function _enzyme_expt_ta_weighted_loss(U1, U2, U3, U4, temp)
    expt_TA!(temp[1], U1, 0.3)
    return real(dot(temp[1], U2))
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


    @testset "Enzyme gradient through a pooled long shift" begin
        phases = (cis(0.37), cis(-0.21), -1.0 + 0im, im)
        V1 = LatticeMatrix(values1, 4, process_grid; nw, phases, numtemps=2)
        V2 = LatticeMatrix(values2, 4, process_grid; nw, phases, numtemps=2)
        V3 = deepcopy(V1)
        V4 = deepcopy(V2)
        direction = LatticeMatrix(direction2_values, 4, process_grid; nw, phases)
        dV = [similar(V1) for _ in 1:4]
        long_temp = [similar(V1)]
        dlong_temp = [similar(V1)]
        clear_matrix!.(dV)
        clear_matrix!.(long_temp)
        clear_matrix!.(dlong_temp)

        Enzyme_derivative!(
            _enzyme_long_shift_loss,
            V1, V2, V3, V4,
            dV[1], dV[2], dV[3], dV[4];
            temp=long_temp,
            dtemp=dlong_temp,
        )

        @test count(V2.temps._flagusing) == 0
        @test count(dV[2].temps._flagusing) == 0
        @test length(V2.temps) == 2
        @test length(dV[2].temps) == 2
        @test all(iszero, dV[1].A)
        @test all(iszero, dV[3].A)
        @test all(iszero, dV[4].A)

        epsilon = 1e-6
        V2_plus = deepcopy(V2)
        V2_minus = deepcopy(V2)
        add_matrix!(V2_plus, direction, epsilon)
        add_matrix!(V2_minus, direction, -epsilon)
        plus = _enzyme_long_shift_loss(V1, V2_plus, V3, V4, long_temp)
        minus = _enzyme_long_shift_loss(V1, V2_minus, V3, V4, long_temp)
        finite_difference = (plus - minus) / (2epsilon)
        enzyme_directional = real(dot(dV[2], direction))
        @test enzyme_directional ≈ finite_difference atol=5e-6 rtol=5e-7
    end


    @testset "Enzyme gradient through fused SU(3) exponential" begin
        exp_global_size = (2 * nprocs, 2, 1, 1)
        exp_process_grid = (nprocs, 1, 1, 1)
        nvalues3 = 3 * 3 * prod(exp_global_size)
        components3 = reshape(Float64.(1:nvalues3), 3, 3, exp_global_size...)
        values3 = complex.(
            components3 ./ (2nvalues3),
            reverse(components3; dims=1) ./ (3nvalues3),
        )
        direction_values3 = complex.(
            reverse(components3; dims=2) ./ (5nvalues3),
            -components3 ./ (7nvalues3),
        )

        W1 = LatticeMatrix(values3, 4, exp_process_grid; nw)
        W2, W3, W4 = deepcopy(W1), deepcopy(W1), deepcopy(W1)
        direction3 = LatticeMatrix(direction_values3, 4, exp_process_grid; nw)
        dW = [similar(W1) for _ in 1:4]
        exp_temp = [similar(W1)]
        dexp_temp = [similar(W1)]

        for (label, loss, u2_is_constant) in (
            ("direct", _enzyme_expt_ta_loss, true),
            ("wrapper", _enzyme_expt_ta_wrapper_loss, true),
            ("weighted upstream", _enzyme_expt_ta_weighted_loss, false),
        )
            clear_matrix!.(dW)
            clear_matrix!.(exp_temp)
            clear_matrix!.(dexp_temp)
            Enzyme_derivative!(
                loss,
                W1, W2, W3, W4,
                dW[1], dW[2], dW[3], dW[4];
                temp=exp_temp,
                dtemp=dexp_temp,
            )

            epsilon = 1e-6
            W1_plus, W1_minus = deepcopy(W1), deepcopy(W1)
            add_matrix!(W1_plus, direction3, epsilon)
            add_matrix!(W1_minus, direction3, -epsilon)
            plus = loss(W1_plus, W2, W3, W4, exp_temp)
            minus = loss(W1_minus, W2, W3, W4, exp_temp)
            finite_difference = (plus - minus) / (2epsilon)
            enzyme_directional = real(dot(dW[1], direction3))

            @testset "$label" begin
                @test enzyme_directional ≈ finite_difference atol=2e-6 rtol=2e-6
                if u2_is_constant
                    @test all(iszero, dW[2].A)
                end
                @test all(iszero, dW[3].A)
                @test all(iszero, dW[4].A)
            end
        end
    end

    @testset "Enzyme TA exponential at small and degenerate fields" begin
        stable_global_size = (2 * nprocs, 2, 1, 1)
        stable_process_grid = (nprocs, 1, 1, 1)
        for (label, matrix_A) in (
            ("SU(2) small", ComplexF64[1e-10im 2e-10; -2e-10 -1e-10im]),
            ("SU(3) c0 positive", Matrix(Diagonal(ComplexF64[-im, -im, 2im]))),
            ("SU(3) c0 negative", Matrix(Diagonal(ComplexF64[im, im, -2im]))),
        )
            NC_stable = size(matrix_A, 1)
            values_A = Array{ComplexF64}(
                undef, NC_stable, NC_stable, stable_global_size...,
            )
            values_weight = similar(values_A)
            values_direction = similar(values_A)
            for site in CartesianIndices(stable_global_size)
                coordinates = Tuple(site)
                @views values_A[:, :, coordinates...] .= matrix_A
                for jc = 1:NC_stable, ic = 1:NC_stable
                    values_weight[ic, jc, coordinates...] = complex(
                        (7ic + 3jc + sum(coordinates)) / 31,
                        (2ic - 5jc + coordinates[1]) / 29,
                    )
                    values_direction[ic, jc, coordinates...] = complex(
                        (3ic - 2jc + coordinates[2]) / 37,
                        (5ic + jc - coordinates[1]) / 41,
                    )
                end
            end

            A_stable = LatticeMatrix(
                values_A, 4, stable_process_grid; nw,
            )
            weight_stable = LatticeMatrix(
                values_weight, 4, stable_process_grid; nw,
            )
            direction_stable = LatticeMatrix(
                values_direction, 4, stable_process_grid; nw,
            )
            constants = [deepcopy(A_stable) for _ in 1:2]
            dinputs = [similar(A_stable) for _ in 1:4]
            stable_temp = [similar(A_stable)]
            stable_dtemp = [similar(A_stable)]
            clear_matrix!.(dinputs)
            clear_matrix!.(stable_temp)
            clear_matrix!.(stable_dtemp)

            Enzyme_derivative!(
                _enzyme_expt_ta_weighted_loss,
                A_stable, weight_stable, constants[1], constants[2],
                dinputs[1], dinputs[2], dinputs[3], dinputs[4];
                temp=stable_temp,
                dtemp=stable_dtemp,
            )

            epsilon = 1e-6
            plus_A, minus_A = deepcopy(A_stable), deepcopy(A_stable)
            add_matrix!(plus_A, direction_stable, epsilon)
            add_matrix!(minus_A, direction_stable, -epsilon)
            plus = _enzyme_expt_ta_weighted_loss(
                plus_A, weight_stable, constants[1], constants[2], stable_temp,
            )
            minus = _enzyme_expt_ta_weighted_loss(
                minus_A, weight_stable, constants[1], constants[2], stable_temp,
            )
            finite_difference = (plus - minus) / (2epsilon)
            enzyme_directional = real(dot(dinputs[1], direction_stable))
            @testset "$label" begin
                @test enzyme_directional ≈ finite_difference atol=3e-6 rtol=3e-6
                @test all(isfinite, dinputs[1].A)
            end
        end
    end
end
