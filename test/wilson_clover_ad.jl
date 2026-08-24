function _wilson_clover_ad_loss(operator, psi, left, result)
    mul!(result, operator, psi)
    return real(dot(left, result))
end

function _wilson_clover_cached_ad_loss(
    U1, U2, U3, U4, cache, psi, left, result,
)
    mul_cached_clover!(result, cache, U1, U2, U3, U4, psi)
    return real(dot(left, result))
end

function wilson_clover_ad_tests()
    nprocs = test_comm_size()
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
            enzyme_duplicated(operator, shadow_operator),
            enzyme_duplicated(psi, dpsi),
            Enzyme.Const(left),
            enzyme_duplicated(result, dresult),
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

    @testset "cached Wilson-clover explicit-link pullback" begin
        cached_result = similar(psi)
        direct_result = similar(psi)
        mul_cached_clover!(
            cached_result, operator, links[1], links[2], links[3], links[4], psi)
        mul!(direct_result, operator, psi)
        @test _wilson_dirac_ad_core(cached_result) ≈
              _wilson_dirac_ad_core(direct_result) atol=2e-12 rtol=2e-12

        cached_dlinks = [similar(link) for link in links]
        cached_dpsi = similar(psi)
        cached_dresult = similar(psi)
        clear_matrix!.(cached_dlinks)
        clear_matrix!.((cached_dpsi, cached_result, cached_dresult))

        Enzyme.autodiff(
            Enzyme.Reverse,
            Enzyme.Const(_wilson_clover_cached_ad_loss),
            Enzyme.Active,
            enzyme_duplicated(links[1], cached_dlinks[1]),
            enzyme_duplicated(links[2], cached_dlinks[2]),
            enzyme_duplicated(links[3], cached_dlinks[3]),
            enzyme_duplicated(links[4], cached_dlinks[4]),
            Enzyme.Const(operator),
            enzyme_duplicated(psi, cached_dpsi),
            Enzyme.Const(left),
            enzyme_duplicated(cached_result, cached_dresult),
        )

        expected_dpsi = similar(psi)
        mul!(expected_dpsi, operator', left)
        @test _wilson_dirac_ad_core(cached_dpsi) ≈
              _wilson_dirac_ad_core(expected_dpsi) atol=3e-12 rtol=3e-12

        epsilon = 1e-6
        links_plus = deepcopy(links)
        links_minus = deepcopy(links)
        for mu in 1:4
            add_matrix!(links_plus[mu], directions[mu], epsilon)
            add_matrix!(links_minus[mu], directions[mu], -epsilon)
        end
        cache_plus = WilsonDiracCloverOperator4D(links_plus, kappa, cSW)
        cache_minus = WilsonDiracCloverOperator4D(links_minus, kappa, cSW)
        loss_plus = _wilson_clover_cached_ad_loss(
            links_plus[1], links_plus[2], links_plus[3], links_plus[4],
            cache_plus, psi, left, similar(psi))
        loss_minus = _wilson_clover_cached_ad_loss(
            links_minus[1], links_minus[2], links_minus[3], links_minus[4],
            cache_minus, psi, left, similar(psi))
        finite_difference = (loss_plus - loss_minus) / (2epsilon)
        enzyme_directional = real(sum(
            dot(cached_dlinks[mu], directions[mu]) for mu in 1:4))

        @test enzyme_directional ≈ finite_difference atol=3e-6 rtol=3e-7
        @test all(iszero, cached_dresult.A)

        cache_epoch_after_ad = halo_epochs(operator.clover[1]).core
        mul_cached_clover!(
            cached_result, operator,
            links[1], links[2], links[3], links[4], psi)
        @test halo_epochs(operator.clover[1]).core == cache_epoch_after_ad

        cache_epoch_before = halo_epochs(operator.clover[1]).core
        updated_result = similar(psi)
        mul_cached_clover!(
            updated_result, operator,
            links_plus[1], links_plus[2], links_plus[3], links_plus[4], psi)
        cache_epoch_after = halo_epochs(operator.clover[1]).core
        direct_updated_result = similar(psi)
        mul!(direct_updated_result, cache_plus, psi)
        @test cache_epoch_after > cache_epoch_before
        @test _wilson_dirac_ad_core(updated_result) ≈
              _wilson_dirac_ad_core(direct_updated_result) atol=2e-12 rtol=2e-12

        mul_cached_clover!(
            updated_result, operator,
            links_plus[1], links_plus[2], links_plus[3], links_plus[4], psi)
        @test halo_epochs(operator.clover[1]).core == cache_epoch_after

        add_matrix!(links_plus[1], directions[1], epsilon)
        mul_cached_clover!(
            updated_result, operator,
            links_plus[1], links_plus[2], links_plus[3], links_plus[4], psi)
        cache_epoch_after_mutation = halo_epochs(operator.clover[1]).core
        direct_mutated_operator =
            WilsonDiracCloverOperator4D(links_plus, kappa, cSW)
        mul!(direct_updated_result, direct_mutated_operator, psi)
        @test cache_epoch_after_mutation > cache_epoch_after
        @test _wilson_dirac_ad_core(updated_result) ≈
              _wilson_dirac_ad_core(direct_updated_result) atol=2e-12 rtol=2e-12

        adjoint_cache = WilsonDiracCloverOperator4D(links, kappa, cSW)
        adjoint_epoch_before = halo_epochs(adjoint_cache.clover[1]).core
        cached_adjoint_result = similar(psi)
        direct_adjoint_result = similar(psi)
        mul_cached_clover_adjoint!(
            cached_adjoint_result, adjoint_cache,
            links_minus[1], links_minus[2], links_minus[3], links_minus[4], psi)
        mul!(direct_adjoint_result, cache_minus', psi)
        @test halo_epochs(adjoint_cache.clover[1]).core > adjoint_epoch_before
        @test _wilson_dirac_ad_core(cached_adjoint_result) ≈
              _wilson_dirac_ad_core(direct_adjoint_result) atol=2e-12 rtol=2e-12
    end
end
