function _hisq_dirac_ad_loss(operator, psi, left, result)
    mul!(result, operator, psi)
    return real(dot(left, result))
end

function _hisq_dirac_ad_loss_from_links(
    X1, X2, X3, X4, L1, L2, L3, L4,
    mass, naik_epsilon, psi, left, result,
)
    operator = HISQDiracOperator4D(
        [X1, X2, X3, X4], [L1, L2, L3, L4], mass; naik_epsilon)
    return _hisq_dirac_ad_loss(operator, psi, left, result)
end

function _hisq_dirac_ad_values(
    ::Val{N1}, ::Val{N2}, global_size, offset,
) where {N1,N2}
    count = N1 * N2 * prod(global_size)
    values = reshape(Float64.(1:count), N1, N2, global_size...)
    scale = count + offset
    return complex.(
        (values .+ offset) ./ scale,
        reverse(values; dims=1) ./ (2scale),
    )
end

function _hisq_dirac_ad_core(field)
    ranges = ntuple(
        d -> (field.nw + 1):(field.nw + field.PN[d]), length(field.PN))
    return @view field.A[:, :, ranges...]
end

function _hisq_dirac_ad_shifted_links(links, directions, epsilon)
    shifted = deepcopy(links)
    for mu in 1:4
        add_matrix!(shifted[mu], directions[mu], epsilon)
    end
    set_halo!.(shifted)
    return shifted
end

function hisq_dirac_ad_tests()
    nprocs = test_comm_size()
    local_x = iseven(nprocs) ? 3 : 4
    global_size = (local_x * nprocs, 3, 3, 3)
    process_grid = (nprocs, 1, 1, 1)
    NC = 2
    nw = 3
    mass = 0.137
    naik_epsilon = -0.083
    fermion_phases = (
        cis(0.13), cis(-0.21), cis(0.34), cis(pi - 0.17))

    X = [
        LatticeMatrix(
            _hisq_dirac_ad_values(
                Val(NC), Val(NC), global_size, 7mu),
            4, process_grid; nw,
        ) for mu in 1:4
    ]
    L = [
        LatticeMatrix(
            _hisq_dirac_ad_values(
                Val(NC), Val(NC), global_size, 31 + 9mu),
            4, process_grid; nw,
        ) for mu in 1:4
    ]
    X_direction = [
        LatticeMatrix(
            _hisq_dirac_ad_values(
                Val(NC), Val(NC), global_size, 53 + 11mu),
            4, process_grid; nw,
        ) for mu in 1:4
    ]
    L_direction = [
        LatticeMatrix(
            _hisq_dirac_ad_values(
                Val(NC), Val(NC), global_size, 79 + 13mu),
            4, process_grid; nw,
        ) for mu in 1:4
    ]
    psi = LatticeMatrix(
        _hisq_dirac_ad_values(Val(NC), Val(1), global_size, 5),
        4, process_grid; nw, phases=fermion_phases,
    )
    psi_direction = LatticeMatrix(
        _hisq_dirac_ad_values(Val(NC), Val(1), global_size, 19),
        4, process_grid; nw, phases=fermion_phases,
    )
    left = LatticeMatrix(
        _hisq_dirac_ad_values(Val(NC), Val(1), global_size, 29),
        4, process_grid; nw, phases=fermion_phases,
    )
    set_halo!.(X)
    set_halo!.(L)
    set_halo!.((psi, psi_direction, left))

    dX = [similar(link) for link in X]
    dL = [similar(link) for link in L]
    dpsi = similar(psi)
    result = similar(psi)
    dresult = similar(psi)
    clear_matrix!.(dX)
    clear_matrix!.(dL)
    clear_matrix!.((dpsi, result, dresult))

    operator = HISQDiracOperator4D(X, L, mass; naik_epsilon)
    shadow_operator = HISQDiracOperator4D(
        dX, dL, mass; naik_epsilon)

    @testset "HISQDiracOperator4D Enzyme pullback" begin
        @test Base.get_extension(
            LatticeMatrices, :LatticeMatricesEnzymeExt) !== nothing
        Enzyme.API.strictAliasing!(false)
        Enzyme.autodiff(
            Enzyme.Reverse,
            Enzyme.Const(_hisq_dirac_ad_loss),
            Enzyme.Active,
            enzyme_duplicated(operator, shadow_operator),
            enzyme_duplicated(psi, dpsi),
            Enzyme.Const(left),
            enzyme_duplicated(result, dresult),
        )

        expected_dpsi = similar(psi)
        mul!(expected_dpsi, operator', left)
        @test _hisq_dirac_ad_core(dpsi) ≈
              _hisq_dirac_ad_core(expected_dpsi) atol=3e-12 rtol=3e-12

        epsilon = 1e-6
        X_plus = _hisq_dirac_ad_shifted_links(X, X_direction, epsilon)
        X_minus = _hisq_dirac_ad_shifted_links(X, X_direction, -epsilon)
        result_plus = similar(psi)
        result_minus = similar(psi)
        X_loss_plus = _hisq_dirac_ad_loss(
            HISQDiracOperator4D(X_plus, L, mass; naik_epsilon),
            psi, left, result_plus)
        X_loss_minus = _hisq_dirac_ad_loss(
            HISQDiracOperator4D(X_minus, L, mass; naik_epsilon),
            psi, left, result_minus)
        X_finite_difference = (X_loss_plus - X_loss_minus) / (2epsilon)
        X_enzyme_directional = real(sum(
            dot(dX[mu], X_direction[mu]) for mu in 1:4))
        @test isapprox(
            X_enzyme_directional, X_finite_difference;
            atol=3e-6, rtol=3e-7)

        L_plus = _hisq_dirac_ad_shifted_links(L, L_direction, epsilon)
        L_minus = _hisq_dirac_ad_shifted_links(L, L_direction, -epsilon)
        L_loss_plus = _hisq_dirac_ad_loss(
            HISQDiracOperator4D(X, L_plus, mass; naik_epsilon),
            psi, left, result_plus)
        L_loss_minus = _hisq_dirac_ad_loss(
            HISQDiracOperator4D(X, L_minus, mass; naik_epsilon),
            psi, left, result_minus)
        L_finite_difference = (L_loss_plus - L_loss_minus) / (2epsilon)
        L_enzyme_directional = real(sum(
            dot(dL[mu], L_direction[mu]) for mu in 1:4))
        @test isapprox(
            L_enzyme_directional, L_finite_difference;
            atol=3e-6, rtol=3e-7)

        psi_plus = deepcopy(psi)
        psi_minus = deepcopy(psi)
        add_matrix!(psi_plus, psi_direction, epsilon)
        add_matrix!(psi_minus, psi_direction, -epsilon)
        set_halo!.((psi_plus, psi_minus))
        psi_loss_plus = _hisq_dirac_ad_loss(
            operator, psi_plus, left, result_plus)
        psi_loss_minus = _hisq_dirac_ad_loss(
            operator, psi_minus, left, result_minus)
        psi_finite_difference =
            (psi_loss_plus - psi_loss_minus) / (2epsilon)
        psi_enzyme_directional = real(dot(dpsi, psi_direction))
        @test isapprox(
            psi_enzyme_directional, psi_finite_difference;
            atol=3e-6, rtol=3e-7)

        @test all(iszero, dresult.A)
    end
end
