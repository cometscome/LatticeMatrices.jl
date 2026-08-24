function _domainwall_ad_loss(operator, psi, left, result)
    mul!(result, operator, psi)
    return real(dot(left, result))
end

function _domainwall_ad_core(field)
    ranges = ntuple(
        d -> (field.nw + 1):(field.nw + field.PN[d]), length(field.PN))
    return @view field.A[:, :, ranges...]
end

function _domainwall_ad_shifted_links(links, directions, epsilon)
    shifted = deepcopy(links)
    for mu in 1:4
        add_matrix!(shifted[mu], directions[mu], epsilon)
    end
    set_halo!.(shifted)
    return shifted
end

function domainwall_ad_tests(NC=2)
    nprocs = test_comm_size()
    global_size = (2 * nprocs, 2, 2, 2)
    fermion_size = (global_size..., 3)
    process_grid = (nprocs, 1, 1, 1)
    process_grid5 = (process_grid..., 1)
    L5 = 3
    nw = 1
    mass = 0.13
    M = -1.0
    b, c = 2.0, 1.0
    phases = (cis(0.13), cis(-0.21), cis(0.34), cis(pi - 0.17), 1.0)

    links = [
        LatticeMatrix(
            _domainwall_test_values(Val(NC), Val(NC), global_size, 7mu),
            4, process_grid; nw)
        for mu in 1:4
    ]
    directions = [
        LatticeMatrix(
            _domainwall_test_values(Val(NC), Val(NC), global_size, 13mu + 5),
            4, process_grid; nw)
        for mu in 1:4
    ]
    psi = LatticeMatrix(
        _domainwall_test_values(Val(NC), Val(4), fermion_size, 11),
        5, process_grid5; nw, phases)
    psi_direction = LatticeMatrix(
        _domainwall_test_values(Val(NC), Val(4), fermion_size, 19),
        5, process_grid5; nw, phases)
    left = LatticeMatrix(
        _domainwall_test_values(Val(NC), Val(4), fermion_size, 29),
        5, process_grid5; nw, phases)
    set_halo!.(links)
    set_halo!.(directions)
    set_halo!.((psi, psi_direction, left))

    operator = D5DW_MobiusDomainwallOperator5D(
        links, L5, mass, M, b, c)

    @testset "Möbius/domain-wall Enzyme pullback NC=$NC" begin
        @test Base.get_extension(
            LatticeMatrices, :LatticeMatricesEnzymeExt) !== nothing
        Enzyme.API.strictAliasing!(false)

        for adjoint_mode in (false, true)
            dlinks = [similar(link) for link in links]
            dpsi = similar(psi)
            result = similar(psi)
            dresult = similar(psi)
            clear_matrix!.(dlinks)
            clear_matrix!.((dpsi, result, dresult))

            shadow_parent = D5DW_MobiusDomainwallOperator5D(
                dlinks, L5, mass, M, b, c)
            applied = adjoint_mode ? adjoint(operator) : operator
            shadow_applied = adjoint_mode ?
                adjoint(shadow_parent) : shadow_parent
            Enzyme.autodiff(
                Enzyme.Reverse,
                Enzyme.Const(_domainwall_ad_loss),
                Enzyme.Active,
                enzyme_duplicated(applied, shadow_applied),
                enzyme_duplicated(psi, dpsi),
                Enzyme.Const(left),
                enzyme_duplicated(result, dresult),
            )

            expected_dpsi = similar(psi)
            expected_operator = adjoint_mode ? operator : adjoint(operator)
            mul!(expected_dpsi, expected_operator, left)
            @test _domainwall_ad_core(dpsi) ≈
                  _domainwall_ad_core(expected_dpsi) atol=8e-12 rtol=8e-12

            epsilon = 1e-6
            links_plus = _domainwall_ad_shifted_links(
                links, directions, epsilon)
            links_minus = _domainwall_ad_shifted_links(
                links, directions, -epsilon)
            operator_plus = D5DW_MobiusDomainwallOperator5D(
                links_plus, L5, mass, M, b, c)
            operator_minus = D5DW_MobiusDomainwallOperator5D(
                links_minus, L5, mass, M, b, c)
            applied_plus = adjoint_mode ? adjoint(operator_plus) : operator_plus
            applied_minus = adjoint_mode ? adjoint(operator_minus) : operator_minus
            loss_plus = _domainwall_ad_loss(
                applied_plus, psi, left, similar(psi))
            loss_minus = _domainwall_ad_loss(
                applied_minus, psi, left, similar(psi))
            finite_difference = (loss_plus - loss_minus) / (2epsilon)
            enzyme_directional = real(sum(
                dot(dlinks[mu], directions[mu]) for mu in 1:4))
            @test isapprox(
                enzyme_directional, finite_difference;
                atol=8e-6, rtol=8e-7)

            psi_plus = deepcopy(psi)
            psi_minus = deepcopy(psi)
            add_matrix!(psi_plus, psi_direction, epsilon)
            add_matrix!(psi_minus, psi_direction, -epsilon)
            set_halo!.((psi_plus, psi_minus))
            psi_loss_plus = _domainwall_ad_loss(
                applied, psi_plus, left, similar(psi))
            psi_loss_minus = _domainwall_ad_loss(
                applied, psi_minus, left, similar(psi))
            psi_finite_difference =
                (psi_loss_plus - psi_loss_minus) / (2epsilon)
            psi_enzyme_directional = real(dot(dpsi, psi_direction))
            @test isapprox(
                psi_enzyme_directional, psi_finite_difference;
                atol=8e-6, rtol=8e-7)
            @test all(iszero, dresult.A)
        end
    end


    @testset "generalized domain-wall Enzyme pullback NC=$NC" begin
        a = [0.83, 1.17, 1.31]
        b5 = [1.2, 0.91, 1.47]
        c5 = [-0.18, 0.37, 0.22]
        coefficient_directions = (
            [0.21, -0.13, 0.34],
            [-0.17, 0.29, 0.11],
            [0.25, 0.07, -0.19],
        )
        generalized = D5DW_GeneralizedDomainwallOperator5D(
            links, L5, mass, M, a, b5, c5)

        for adjoint_mode in (false, true)
            dlinks = [similar(link) for link in links]
            dpsi = similar(psi)
            result = similar(psi)
            dresult = similar(psi)
            clear_matrix!.(dlinks)
            clear_matrix!.((dpsi, result, dresult))
            shadow_parent = D5DW_GeneralizedDomainwallOperator5D(
                dlinks, L5, mass, M, zeros(L5), zeros(L5), zeros(L5))
            applied = adjoint_mode ? adjoint(generalized) : generalized
            shadow_applied = adjoint_mode ?
                adjoint(shadow_parent) : shadow_parent

            Enzyme.autodiff(
                Enzyme.Reverse,
                Enzyme.Const(_domainwall_ad_loss),
                Enzyme.Active,
                enzyme_duplicated(applied, shadow_applied),
                enzyme_duplicated(psi, dpsi),
                Enzyme.Const(left),
                enzyme_duplicated(result, dresult),
            )

            expected_dpsi = similar(psi)
            expected_operator = adjoint_mode ?
                generalized : adjoint(generalized)
            mul!(expected_dpsi, expected_operator, left)
            @test _domainwall_ad_core(dpsi) ≈
                _domainwall_ad_core(expected_dpsi) atol=8e-12 rtol=8e-12

            epsilon = 1e-6
            links_plus = _domainwall_ad_shifted_links(
                links, directions, epsilon)
            links_minus = _domainwall_ad_shifted_links(
                links, directions, -epsilon)
            operator_plus = D5DW_GeneralizedDomainwallOperator5D(
                links_plus, L5, mass, M, a, b5, c5)
            operator_minus = D5DW_GeneralizedDomainwallOperator5D(
                links_minus, L5, mass, M, a, b5, c5)
            applied_plus = adjoint_mode ? adjoint(operator_plus) : operator_plus
            applied_minus = adjoint_mode ? adjoint(operator_minus) : operator_minus
            finite_difference = (
                _domainwall_ad_loss(applied_plus, psi, left, similar(psi)) -
                _domainwall_ad_loss(applied_minus, psi, left, similar(psi))) /
                (2epsilon)
            enzyme_directional = real(sum(
                dot(dlinks[mu], directions[mu]) for mu in 1:4))
            @test isapprox(
                enzyme_directional, finite_difference;
                atol=8e-6, rtol=8e-7)

            psi_plus = deepcopy(psi)
            psi_minus = deepcopy(psi)
            add_matrix!(psi_plus, psi_direction, epsilon)
            add_matrix!(psi_minus, psi_direction, -epsilon)
            set_halo!.((psi_plus, psi_minus))
            psi_finite_difference = (
                _domainwall_ad_loss(applied, psi_plus, left, similar(psi)) -
                _domainwall_ad_loss(applied, psi_minus, left, similar(psi))) /
                (2epsilon)
            @test isapprox(
                real(dot(dpsi, psi_direction)), psi_finite_difference;
                atol=8e-6, rtol=8e-7)

            coefficient_gradients =
                (Array(shadow_parent.a), Array(shadow_parent.b),
                 Array(shadow_parent.c))
            coefficients = (a, b5, c5)
            for parameter in 1:3
                plus = ntuple(i -> i == parameter ?
                    coefficients[i] .+ epsilon .* coefficient_directions[i] :
                    coefficients[i], 3)
                minus = ntuple(i -> i == parameter ?
                    coefficients[i] .- epsilon .* coefficient_directions[i] :
                    coefficients[i], 3)
                plus_operator = D5DW_GeneralizedDomainwallOperator5D(
                    links, L5, mass, M, plus...)
                minus_operator = D5DW_GeneralizedDomainwallOperator5D(
                    links, L5, mass, M, minus...)
                plus_applied = adjoint_mode ?
                    adjoint(plus_operator) : plus_operator
                minus_applied = adjoint_mode ?
                    adjoint(minus_operator) : minus_operator
                parameter_finite_difference = (
                    _domainwall_ad_loss(
                        plus_applied, psi, left, similar(psi)) -
                    _domainwall_ad_loss(
                        minus_applied, psi, left, similar(psi))) / (2epsilon)
                parameter_enzyme = dot(
                    coefficient_gradients[parameter],
                    coefficient_directions[parameter])
                @test isapprox(
                    parameter_enzyme, parameter_finite_difference;
                    atol=8e-6, rtol=8e-7)
            end
            @test all(iszero, dresult.A)
        end
    end
end
