function _wilson_clover_ad_loss(operator, psi, left, result)
    mul!(result, operator, psi)
    return real(dot(left, result))
end

function wilson_clover_ad_tests()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (2 * nprocs, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)
    NC = 2
    nw = 1
    kappa = 0.117
    cSW = 1.21
    fermion_phases = (1.0, 1.0, 1.0, -1.0)

    links = [
        LatticeMatrix(
            _wilson_dirac_ad_values(Val(NC), Val(NC), global_size, 7mu + 2),
            4, process_grid; nw,
        ) for mu in 1:4
    ]
    directions = [
        LatticeMatrix(
            _wilson_dirac_ad_values(Val(NC), Val(NC), global_size, 13mu + 5),
            4, process_grid; nw,
        ) for mu in 1:4
    ]
    psi = LatticeMatrix(
        _wilson_dirac_ad_values(Val(NC), Val(4), global_size, 11),
        4, process_grid; nw, phases=fermion_phases,
    )
    left = LatticeMatrix(
        _wilson_dirac_ad_values(Val(NC), Val(4), global_size, 17),
        4, process_grid; nw, phases=fermion_phases,
    )

    dlinks = [similar(link) for link in links]
    dpsi = similar(psi)
    result = similar(psi)
    dresult = similar(psi)
    clear_matrix!.(dlinks)
    clear_matrix!.((dpsi, result, dresult))

    operator = WilsonDiracCloverOperator4D(links, kappa, cSW)
    shadow_operator = WilsonDiracCloverOperator4D(dlinks, 0.0, 0.0)

    @testset "WilsonDiracCloverOperator4D Enzyme pullback" begin
        @test Base.get_extension(LatticeMatrices, :LatticeMatricesEnzymeExt) !== nothing
        Enzyme.API.strictAliasing!(false)
        Enzyme.autodiff(
            Enzyme.Reverse,
            Enzyme.Const(_wilson_clover_ad_loss),
            Enzyme.Active,
            Enzyme.Duplicated(operator, shadow_operator),
            Enzyme.Duplicated(psi, dpsi),
            Enzyme.Const(left),
            Enzyme.Duplicated(result, dresult),
        )

        expected_dpsi = similar(psi)
        mul!(expected_dpsi, operator', left)
        @test _wilson_dirac_ad_core(dpsi) ≈
              _wilson_dirac_ad_core(expected_dpsi) atol=3e-12 rtol=3e-12

        epsilon = 1e-6
        links_plus = deepcopy(links)
        links_minus = deepcopy(links)
        for mu in 1:4
            add_matrix!(links_plus[mu], directions[mu], epsilon)
            add_matrix!(links_minus[mu], directions[mu], -epsilon)
        end

        loss_plus = _wilson_clover_ad_loss(
            WilsonDiracCloverOperator4D(links_plus, kappa, cSW),
            psi, left, similar(psi))
        loss_minus = _wilson_clover_ad_loss(
            WilsonDiracCloverOperator4D(links_minus, kappa, cSW),
            psi, left, similar(psi))
        finite_difference = (loss_plus - loss_minus) / (2epsilon)
        enzyme_directional =
            real(sum(dot(dlinks[mu], directions[mu]) for mu in 1:4))

        @test enzyme_directional ≈ finite_difference atol=3e-6 rtol=3e-7
        @test all(iszero, dresult.A)
    end
end
