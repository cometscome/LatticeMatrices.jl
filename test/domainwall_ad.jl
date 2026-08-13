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

function domainwall_ad_tests()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (2 * nprocs, 2, 2, 2)
    fermion_size = (global_size..., 3)
    process_grid = (nprocs, 1, 1, 1)
    process_grid5 = (process_grid..., 1)
    NC = 2
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

    @testset "Möbius/domain-wall Enzyme pullback" begin
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
                Enzyme.Duplicated(applied, shadow_applied),
                Enzyme.Duplicated(psi, dpsi),
                Enzyme.Const(left),
                Enzyme.Duplicated(result, dresult),
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
end
