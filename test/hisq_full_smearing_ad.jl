function _hisq_projection_ad_values(
    ::Val{NC}, global_size, offset; identity_shift=0.0,
) where NC
    count = NC * NC * prod(global_size)
    values = reshape(Float64.(1:count), NC, NC, global_size...)
    output = complex.(
        sin.((values .+ offset) ./ 11) ./ 5,
        cos.((2values .+ offset) ./ 13) ./ 7,
    )
    if !iszero(identity_shift)
        for site_index in CartesianIndices(global_size), color in 1:NC
            output[color, color, Tuple(site_index)...] += identity_shift
        end
    end
    return output
end

function _hisq_projection_ad_loss_from_links(input, output, left)
    hisq_project_u3!(output, input)
    return real(
        dot(left[1], output[1]) + dot(left[2], output[2]) +
        dot(left[3], output[3]) + dot(left[4], output[4]))
end

function _hisq_projection_ad_loss(input, left)
    output = [similar(link) for link in input]
    return _hisq_projection_ad_loss_from_links(input, output, left)
end

function _hisq_level2_ad_values(::Val{NC}, global_size, offset) where NC
    count = NC * NC * prod(global_size)
    values = reshape(Float64.(1:count), NC, NC, global_size...)
    return complex.(
        sin.((values .+ offset) ./ 17) ./ 4,
        cos.((3values .+ offset) ./ 19) ./ 6,
    )
end

function _hisq_level2_ad_loss_from_links(
    input, output, left, naik_epsilon,
)
    hisq_fat7_level2!(output, input, naik_epsilon)
    return real(
        dot(left[1], output[1]) + dot(left[2], output[2]) +
        dot(left[3], output[3]) + dot(left[4], output[4]))
end

function _hisq_level2_ad_loss(input, left, naik_epsilon)
    output = [similar(link) for link in input]
    return _hisq_level2_ad_loss_from_links(
        input, output, left, naik_epsilon)
end

function _hisq_naik_ad_loss_from_links(input, output, left)
    hisq_naik_links!(output, input)
    return real(
        dot(left[1], output[1]) + dot(left[2], output[2]) +
        dot(left[3], output[3]) + dot(left[4], output[4]))
end

function _hisq_naik_ad_loss(input, left)
    output = [similar(link) for link in input]
    return _hisq_naik_ad_loss_from_links(input, output, left)
end

function _hisq_full_chain_ad_loss_from_links(
    thin, level1, reunitarized, fat, long,
    left_fat, left_long, naik_epsilon,
)
    hisq_links_from_thin!(
        fat, long, level1, reunitarized, thin, naik_epsilon)
    return real(
        dot(left_fat[1], fat[1]) + dot(left_fat[2], fat[2]) +
        dot(left_fat[3], fat[3]) + dot(left_fat[4], fat[4]) +
        dot(left_long[1], long[1]) + dot(left_long[2], long[2]) +
        dot(left_long[3], long[3]) + dot(left_long[4], long[4]))
end

function _hisq_full_chain_ad_loss(
    thin, left_fat, left_long, naik_epsilon,
)
    level1 = [similar(link) for link in thin]
    reunitarized = [similar(link) for link in thin]
    fat = [similar(link) for link in thin]
    long = [similar(link) for link in thin]
    return _hisq_full_chain_ad_loss_from_links(
        thin, level1, reunitarized, fat, long,
        left_fat, left_long, naik_epsilon)
end

function hisq_full_smearing_ad_tests()
    nprocs = test_comm_size()
    process_grid = (nprocs, 1, 1, 1)
    global_size = (3 * nprocs, 3, 3, 3)
    nw = 1
    input = [
        LatticeMatrix(
            _hisq_projection_ad_values(
                Val(3), global_size, 7mu; identity_shift=1.25),
            4, process_grid; nw,
        ) for mu in 1:4
    ]
    direction = [
        LatticeMatrix(
            _hisq_projection_ad_values(Val(3), global_size, 37 + 5mu),
            4, process_grid; nw,
        ) for mu in 1:4
    ]
    left = [
        LatticeMatrix(
            _hisq_projection_ad_values(Val(3), global_size, 71 + 11mu),
            4, process_grid; nw,
        ) for mu in 1:4
    ]
    set_halo!.(input)
    set_halo!.(direction)
    set_halo!.(left)

    dinput = [similar(link) for link in input]
    output = [similar(link) for link in input]
    doutput = [similar(link) for link in input]
    clear_matrix!.(dinput)
    clear_matrix!.(output)
    clear_matrix!.(doutput)

    @testset "HISQ U(3) projection Enzyme pullback" begin
        Enzyme.API.strictAliasing!(false)
        Enzyme.autodiff(
            Enzyme.Reverse,
            Enzyme.Const(_hisq_projection_ad_loss_from_links),
            Enzyme.Active,
            enzyme_duplicated(input, dinput),
            enzyme_duplicated(output, doutput),
            Enzyme.Const(Tuple(left)),
        )

        epsilon = 1e-6
        input_plus = deepcopy(input)
        input_minus = deepcopy(input)
        for mu in 1:4
            add_matrix!(input_plus[mu], direction[mu], epsilon)
            add_matrix!(input_minus[mu], direction[mu], -epsilon)
        end
        finite_difference = (
            _hisq_projection_ad_loss(input_plus, left) -
            _hisq_projection_ad_loss(input_minus, left)
        ) / (2epsilon)
        enzyme_directional = real(sum(
            dot(dinput[mu], direction[mu]) for mu in 1:4))
        @test isapprox(
            enzyme_directional, finite_difference;
            atol=5e-6, rtol=5e-7)
        @test all(link -> all(iszero, link.A), doutput)
    end

    @testset "HISQ generic U(N) projection Enzyme pullback" begin
        for NC in (2, 4)
            generic_input = [
                LatticeMatrix(
                    _hisq_projection_ad_values(
                        Val(NC), global_size, 13mu; identity_shift=1.25),
                    4, process_grid; nw,
                ) for mu in 1:4
            ]
            generic_direction = [
                LatticeMatrix(
                    _hisq_projection_ad_values(
                        Val(NC), global_size, 43 + 7mu),
                    4, process_grid; nw,
                ) for mu in 1:4
            ]
            generic_left = [
                LatticeMatrix(
                    _hisq_projection_ad_values(
                        Val(NC), global_size, 83 + 11mu),
                    4, process_grid; nw,
                ) for mu in 1:4
            ]
            set_halo!.(generic_input)
            set_halo!.(generic_direction)
            set_halo!.(generic_left)
            generic_dinput = [similar(link) for link in generic_input]
            generic_output = [similar(link) for link in generic_input]
            generic_doutput = [similar(link) for link in generic_input]
            clear_matrix!.(generic_dinput)
            clear_matrix!.(generic_output)
            clear_matrix!.(generic_doutput)

            Enzyme.autodiff(
                Enzyme.Reverse,
                Enzyme.Const(_hisq_projection_ad_loss_from_links),
                Enzyme.Active,
                enzyme_duplicated(generic_input, generic_dinput),
                enzyme_duplicated(generic_output, generic_doutput),
                Enzyme.Const(Tuple(generic_left)),
            )

            epsilon = 1e-6
            input_plus = deepcopy(generic_input)
            input_minus = deepcopy(generic_input)
            for mu in 1:4
                add_matrix!(input_plus[mu], generic_direction[mu], epsilon)
                add_matrix!(input_minus[mu], generic_direction[mu], -epsilon)
            end
            finite_difference = (
                _hisq_projection_ad_loss(input_plus, generic_left) -
                _hisq_projection_ad_loss(input_minus, generic_left)
            ) / (2epsilon)
            enzyme_directional = real(sum(
                dot(generic_dinput[mu], generic_direction[mu])
                for mu in 1:4))
            @test isapprox(
                enzyme_directional, finite_difference;
                atol=8e-6, rtol=8e-7)
            @test all(link -> all(iszero, link.A), generic_doutput)
        end
    end


    @testset "HISQ level-2 Fat7 and Lepage Enzyme pullback" begin
        level2_size = (3 * nprocs, 2, 2, 2)
        NC = 2
        level2_input = [
            LatticeMatrix(
                _hisq_level2_ad_values(
                    Val(NC), level2_size, 13mu),
                4, process_grid; nw=2,
            ) for mu in 1:4
        ]
        level2_direction = [
            LatticeMatrix(
                _hisq_level2_ad_values(
                    Val(NC), level2_size, 43 + 7mu),
                4, process_grid; nw=2,
            ) for mu in 1:4
        ]
        level2_left = [
            LatticeMatrix(
                _hisq_level2_ad_values(
                    Val(NC), level2_size, 79 + 9mu),
                4, process_grid; nw=2,
            ) for mu in 1:4
        ]
        set_halo!.(level2_input)
        set_halo!.(level2_direction)
        set_halo!.(level2_left)
        dlevel2_input = [similar(link) for link in level2_input]
        level2_output = [similar(link) for link in level2_input]
        dlevel2_output = [similar(link) for link in level2_input]
        clear_matrix!.(dlevel2_input)
        clear_matrix!.(level2_output)
        clear_matrix!.(dlevel2_output)
        naik_epsilon = -0.071

        Enzyme.autodiff(
            Enzyme.Reverse,
            Enzyme.Const(_hisq_level2_ad_loss_from_links),
            Enzyme.Active,
            enzyme_duplicated(level2_input, dlevel2_input),
            enzyme_duplicated(level2_output, dlevel2_output),
            Enzyme.Const(Tuple(level2_left)),
            Enzyme.Const(naik_epsilon),
        )

        epsilon = 1e-6
        input_plus = deepcopy(level2_input)
        input_minus = deepcopy(level2_input)
        for mu in 1:4
            add_matrix!(
                input_plus[mu], level2_direction[mu], epsilon)
            add_matrix!(
                input_minus[mu], level2_direction[mu], -epsilon)
        end
        set_halo!.(input_plus)
        set_halo!.(input_minus)
        finite_difference = (
            _hisq_level2_ad_loss(
                input_plus, level2_left, naik_epsilon) -
            _hisq_level2_ad_loss(
                input_minus, level2_left, naik_epsilon)
        ) / (2epsilon)
        enzyme_directional = real(sum(
            dot(dlevel2_input[mu], level2_direction[mu]) for mu in 1:4))
        @test isapprox(
            enzyme_directional, finite_difference;
            atol=8e-6, rtol=8e-7)
        @test all(link -> all(iszero, link.A), dlevel2_output)

        clear_matrix!.(dlevel2_output)
        epsilon_result = Enzyme.autodiff(
            Enzyme.Reverse,
            Enzyme.Const(_hisq_level2_ad_loss_from_links),
            Enzyme.Active,
            Enzyme.Const(level2_input),
            enzyme_duplicated(level2_output, dlevel2_output),
            Enzyme.Const(Tuple(level2_left)),
            Enzyme.Active(naik_epsilon),
        )
        epsilon_result isa Tuple && length(epsilon_result) == 1 &&
            (epsilon_result = epsilon_result[1])
        epsilon_gradient = epsilon_result isa Tuple ?
            epsilon_result[4] : epsilon_result
        epsilon_finite_difference = (
            _hisq_level2_ad_loss(
                level2_input, level2_left, naik_epsilon + epsilon) -
            _hisq_level2_ad_loss(
                level2_input, level2_left, naik_epsilon - epsilon)
        ) / (2epsilon)
        @test isapprox(
            epsilon_gradient, epsilon_finite_difference;
            atol=5e-7, rtol=5e-8)
        @test all(link -> all(iszero, link.A), dlevel2_output)
    end

    @testset "factorized U(N) Fat7 pullback matches direct paths" begin
        pullback_size = (3 * nprocs, 3, 3, 3)
        coefficients = (1.0 - 0.071 / 8, 1 / 16, 1 / 64, 1 / 384, -1 / 8)
        for NC in (2, 3, 4)
            pullback_input = [
                LatticeMatrix(
                    _hisq_projection_ad_values(
                        Val(NC), pullback_size, 17mu; identity_shift=0.75),
                    4, process_grid; nw=3,
                ) for mu in 1:4
            ]
            pullback_left = [
                LatticeMatrix(
                    _hisq_projection_ad_values(
                        Val(NC), pullback_size, 71 + 11mu),
                    4, process_grid; nw=3,
                ) for mu in 1:4
            ]
            direct_output = [similar(link) for link in pullback_input]
            factorized_output = [similar(link) for link in pullback_input]
            forward_workspace = HISQFat7Workspace(pullback_input[1])
            LatticeMatrices._hisq_fat7!(
                direct_output, pullback_input, coefficients)
            LatticeMatrices._hisq_fat7!(
                factorized_output, pullback_input, coefficients,
                forward_workspace)
            direct_gradient = [similar(link) for link in pullback_input]
            factorized_gradient = [similar(link) for link in pullback_input]
            clear_matrix!.(direct_gradient)
            clear_matrix!.(factorized_gradient)

            LatticeMatrices._hisq_fat7_pullback_accumulate!(
                direct_gradient, pullback_left, pullback_input, coefficients)
            workspace = HISQFat7PullbackWorkspace(pullback_input[1])
            LatticeMatrices._hisq_fat7_pullback_accumulate!(
                factorized_gradient, pullback_left, pullback_input,
                coefficients, workspace)

            gathered_direct = gather_matrix.(direct_gradient)
            gathered_factorized = gather_matrix.(factorized_gradient)
            gathered_direct_output = gather_matrix.(direct_output)
            gathered_factorized_output = gather_matrix.(factorized_output)
            if test_comm_rank() == 0
                for mu in 1:4
                    @test isapprox(
                        gathered_factorized_output[mu],
                        gathered_direct_output[mu]; atol=2e-11, rtol=2e-11)
                    @test isapprox(
                        gathered_factorized[mu], gathered_direct[mu];
                        atol=2e-11, rtol=2e-11)
                end
            end
        end
    end

    @testset "HISQ Naik-link Enzyme pullback" begin
        naik_size = (3 * nprocs, 2, 2, 2)
        NC = 2
        naik_input = [
            LatticeMatrix(
                _hisq_level2_ad_values(Val(NC), naik_size, 17mu),
                4, process_grid; nw=2,
            ) for mu in 1:4
        ]
        naik_direction = [
            LatticeMatrix(
                _hisq_level2_ad_values(
                    Val(NC), naik_size, 53 + 7mu),
                4, process_grid; nw=2,
            ) for mu in 1:4
        ]
        naik_left = [
            LatticeMatrix(
                _hisq_level2_ad_values(
                    Val(NC), naik_size, 91 + 11mu),
                4, process_grid; nw=2,
            ) for mu in 1:4
        ]
        set_halo!.(naik_input)
        set_halo!.(naik_direction)
        set_halo!.(naik_left)
        dnaik_input = [similar(link) for link in naik_input]
        naik_output = [similar(link) for link in naik_input]
        dnaik_output = [similar(link) for link in naik_input]
        clear_matrix!.(dnaik_input)
        clear_matrix!.(naik_output)
        clear_matrix!.(dnaik_output)

        Enzyme.autodiff(
            Enzyme.Reverse,
            Enzyme.Const(_hisq_naik_ad_loss_from_links),
            Enzyme.Active,
            enzyme_duplicated(naik_input, dnaik_input),
            enzyme_duplicated(naik_output, dnaik_output),
            Enzyme.Const(Tuple(naik_left)),
        )

        epsilon = 1e-6
        input_plus = deepcopy(naik_input)
        input_minus = deepcopy(naik_input)
        for mu in 1:4
            add_matrix!(input_plus[mu], naik_direction[mu], epsilon)
            add_matrix!(input_minus[mu], naik_direction[mu], -epsilon)
        end
        set_halo!.(input_plus)
        set_halo!.(input_minus)
        finite_difference = (
            _hisq_naik_ad_loss(input_plus, naik_left) -
            _hisq_naik_ad_loss(input_minus, naik_left)
        ) / (2epsilon)
        enzyme_directional = real(sum(
            dot(dnaik_input[mu], naik_direction[mu]) for mu in 1:4))
        @test isapprox(
            enzyme_directional, finite_difference;
            atol=5e-6, rtol=5e-7)
        @test all(link -> all(iszero, link.A), dnaik_output)
    end

    @testset "HISQ complete smearing-chain Enzyme pullback" begin
        chain_size = (3 * nprocs, 3, 3, 3)
        thin = [
            LatticeMatrix(
                _hisq_projection_ad_values(
                    Val(3), chain_size, 19mu; identity_shift=1.2),
                4, process_grid; nw=3,
            ) for mu in 1:4
        ]
        thin_direction = [
            LatticeMatrix(
                _hisq_projection_ad_values(
                    Val(3), chain_size, 61 + 7mu),
                4, process_grid; nw=3,
            ) for mu in 1:4
        ]
        left_fat = [
            LatticeMatrix(
                _hisq_projection_ad_values(
                    Val(3), chain_size, 103 + 11mu),
                4, process_grid; nw=3,
            ) for mu in 1:4
        ]
        left_long = [
            LatticeMatrix(
                _hisq_projection_ad_values(
                    Val(3), chain_size, 151 + 13mu),
                4, process_grid; nw=3,
            ) for mu in 1:4
        ]
        set_halo!.(thin)
        set_halo!.(thin_direction)
        set_halo!.(left_fat)
        set_halo!.(left_long)

        level1 = [similar(link) for link in thin]
        reunitarized = [similar(link) for link in thin]
        fat = [similar(link) for link in thin]
        long = [similar(link) for link in thin]
        dthin = [similar(link) for link in thin]
        dlevel1 = [similar(link) for link in thin]
        dreunitarized = [similar(link) for link in thin]
        dfat = [similar(link) for link in thin]
        dlong = [similar(link) for link in thin]
        clear_matrix!.(dthin)
        clear_matrix!.(dlevel1)
        clear_matrix!.(dreunitarized)
        clear_matrix!.(dfat)
        clear_matrix!.(dlong)
        naik_epsilon = -0.083

        Enzyme.autodiff(
            Enzyme.Reverse,
            Enzyme.Const(_hisq_full_chain_ad_loss_from_links),
            Enzyme.Active,
            enzyme_duplicated(thin, dthin),
            enzyme_duplicated(level1, dlevel1),
            enzyme_duplicated(reunitarized, dreunitarized),
            enzyme_duplicated(fat, dfat),
            enzyme_duplicated(long, dlong),
            Enzyme.Const(Tuple(left_fat)),
            Enzyme.Const(Tuple(left_long)),
            Enzyme.Const(naik_epsilon),
        )

        epsilon = 1e-6
        thin_plus = deepcopy(thin)
        thin_minus = deepcopy(thin)
        for mu in 1:4
            add_matrix!(thin_plus[mu], thin_direction[mu], epsilon)
            add_matrix!(thin_minus[mu], thin_direction[mu], -epsilon)
        end
        set_halo!.(thin_plus)
        set_halo!.(thin_minus)
        finite_difference = (
            _hisq_full_chain_ad_loss(
                thin_plus, left_fat, left_long, naik_epsilon) -
            _hisq_full_chain_ad_loss(
                thin_minus, left_fat, left_long, naik_epsilon)
        ) / (2epsilon)
        enzyme_directional = real(sum(
            dot(dthin[mu], thin_direction[mu]) for mu in 1:4))
        @test isapprox(
            enzyme_directional, finite_difference;
            atol=3e-5, rtol=2e-6)
        @test all(link -> all(iszero, link.A), dlevel1)
        @test all(link -> all(iszero, link.A), dreunitarized)
        @test all(link -> all(iszero, link.A), dfat)
        @test all(link -> all(iszero, link.A), dlong)
    end
end
