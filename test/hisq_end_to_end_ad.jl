function _hisq_end_to_end_fermion_values(
    global_size, offset,
)
    count = 3 * prod(global_size)
    values = reshape(Float64.(1:count), 3, 1, global_size...)
    return complex.(
        sin.((values .+ offset) ./ 23) ./ 3,
        cos.((2values .+ offset) ./ 29) ./ 5,
    )
end

function _hisq_end_to_end_loss_from_thin(
    thin, level1, reunitarized, operator, psi, left, result,
)
    hisq_links_from_thin!(
        operator.links.fat_links, operator.links.long_links,
        level1, reunitarized, thin, operator.naik_epsilon)
    mul!(result, operator, psi)
    return real(dot(left, result))
end

function _hisq_end_to_end_loss(
    thin, mass, naik_epsilon, psi, left,
)
    links = hisq_links_from_thin(thin; naik_epsilon)
    operator = HISQDiracOperator4D(
        links, mass; naik_epsilon)
    result = similar(psi)
    mul!(result, operator, psi)
    return real(dot(left, result))
end

function _hisq_cached_end_to_end_loss(
    U1, U2, U3, U4, cache, psi, left, result,
)
    mul_cached_hisq!(result, cache, U1, U2, U3, U4, psi)
    return real(dot(left, result))
end

function _hisq_end_to_end_core(field)
    ranges = ntuple(
        d -> (field.nw + 1):(field.nw + field.PN[d]), 4)
    return @view field.A[:, :, ranges...]
end

function hisq_end_to_end_ad_tests()
    nprocs = test_comm_size()
    process_grid = (nprocs, 1, 1, 1)
    global_size = (3 * nprocs, 3, 3, 3)
    nw = 3
    mass = 0.137
    naik_epsilon = -0.083

    thin = [
        LatticeMatrix(
            _hisq_projection_ad_values(
                Val(3), global_size, 23mu; identity_shift=1.15),
            4, process_grid; nw,
        ) for mu in 1:4
    ]
    thin_direction = [
        LatticeMatrix(
            _hisq_projection_ad_values(Val(3), global_size, 73 + 7mu),
            4, process_grid; nw,
        ) for mu in 1:4
    ]
    phases = (cis(0.11), cis(-0.19), cis(0.31), cis(pi - 0.13))
    psi = LatticeMatrix(
        _hisq_end_to_end_fermion_values(global_size, 17),
        4, process_grid; nw, phases)
    left = LatticeMatrix(
        _hisq_end_to_end_fermion_values(global_size, 47),
        4, process_grid; nw, phases)
    set_halo!.(thin)
    set_halo!.(thin_direction)
    set_halo!.((psi, left))

    level1 = [similar(link) for link in thin]
    reunitarized = [similar(link) for link in thin]
    fat = [similar(link) for link in thin]
    long = [similar(link) for link in thin]
    dthin = [similar(link) for link in thin]
    dlevel1 = [similar(link) for link in thin]
    dreunitarized = [similar(link) for link in thin]
    dfat = [similar(link) for link in thin]
    dlong = [similar(link) for link in thin]
    dpsi = similar(psi)
    result = similar(psi)
    dresult = similar(psi)
    for fields in (
        dthin, dlevel1, dreunitarized, dfat, dlong)
        clear_matrix!.(fields)
    end
    clear_matrix!.((dpsi, result, dresult))
    operator = HISQDiracOperator4D(
        Tuple(fat), Tuple(long), mass; naik_epsilon)
    shadow_operator = HISQDiracOperator4D(
        Tuple(dfat), Tuple(dlong), mass; naik_epsilon)

    # Julia 1.12 cannot type-analyze constructing/composing the immutable
    # HISQ operator inside one generic Enzyme invocation.  The static cached
    # rule below covers the same complete thin-link-to-action pullback.
    if VERSION < v"1.12"
      @testset "HISQ thin-link to Dirac-action Enzyme pullback" begin
        Enzyme.API.strictAliasing!(false)
        Enzyme.autodiff(
            Enzyme.Reverse,
            Enzyme.Const(_hisq_end_to_end_loss_from_thin),
            Enzyme.Active,
            enzyme_duplicated(thin, dthin),
            enzyme_duplicated(level1, dlevel1),
            enzyme_duplicated(reunitarized, dreunitarized),
            enzyme_duplicated(operator, shadow_operator),
            enzyme_duplicated(psi, dpsi),
            Enzyme.Const(left),
            enzyme_duplicated(result, dresult),
        )

        expected_dpsi = similar(psi)
        mul!(expected_dpsi, operator', left)
        @test _hisq_end_to_end_core(dpsi) ≈
            _hisq_end_to_end_core(expected_dpsi) atol=8e-11 rtol=8e-11

        epsilon = 2e-7
        thin_plus = deepcopy(thin)
        thin_minus = deepcopy(thin)
        for mu in 1:4
            add_matrix!(thin_plus[mu], thin_direction[mu], epsilon)
            add_matrix!(thin_minus[mu], thin_direction[mu], -epsilon)
        end
        set_halo!.(thin_plus)
        set_halo!.(thin_minus)
        finite_difference = (
            _hisq_end_to_end_loss(
                thin_plus, mass, naik_epsilon, psi, left) -
            _hisq_end_to_end_loss(
                thin_minus, mass, naik_epsilon, psi, left)
        ) / (2epsilon)
        enzyme_directional = real(sum(
            dot(dthin[mu], thin_direction[mu]) for mu in 1:4))
        @test isapprox(
            enzyme_directional, finite_difference;
            atol=8e-5, rtol=5e-6)

        @test all(link -> all(iszero, link.A), dlevel1)
        @test all(link -> all(iszero, link.A), dreunitarized)
        @test all(link -> all(iszero, link.A), dfat)
        @test all(link -> all(iszero, link.A), dlong)
        @test all(iszero, dresult.A)
      end
    end

    @testset "transparent cached HISQ Enzyme pullback" begin
        cache = HISQDiracCache4D(
            thin, mass; naik_epsilon)
        cached_result = similar(psi)
        cached_dresult = similar(psi)
        cached_dpsi = similar(psi)
        cached_dthin = [similar(link) for link in thin]
        clear_matrix!.((cached_result, cached_dresult, cached_dpsi))
        clear_matrix!.(cached_dthin)

        initial_loss = _hisq_cached_end_to_end_loss(
            thin[1], thin[2], thin[3], thin[4],
            cache, psi, left, cached_result)
        @test initial_loss ≈ _hisq_end_to_end_loss(
            thin, mass, naik_epsilon, psi, left) atol=2e-12 rtol=2e-12

        Enzyme.autodiff(
            Enzyme.Reverse,
            Enzyme.Const(_hisq_cached_end_to_end_loss),
            Enzyme.Active,
            enzyme_duplicated(thin[1], cached_dthin[1]),
            enzyme_duplicated(thin[2], cached_dthin[2]),
            enzyme_duplicated(thin[3], cached_dthin[3]),
            enzyme_duplicated(thin[4], cached_dthin[4]),
            Enzyme.Const(cache),
            enzyme_duplicated(psi, cached_dpsi),
            Enzyme.Const(left),
            enzyme_duplicated(cached_result, cached_dresult),
        )

        analytic_dthin = [similar(link) for link in thin]
        clear_matrix!.(analytic_dthin)
        hisq_link_pullback!(
            analytic_dthin, cache, thin, left, psi)
        for mu in 1:4
            @test _hisq_end_to_end_core(analytic_dthin[mu]) ≈
                _hisq_end_to_end_core(cached_dthin[mu]) atol=8e-11 rtol=8e-11
        end

        expected_dpsi = similar(psi)
        mul!(expected_dpsi, cache.operator', left)
        @test _hisq_end_to_end_core(cached_dpsi) ≈
            _hisq_end_to_end_core(expected_dpsi) atol=8e-11 rtol=8e-11

        epsilon = 2e-7
        thin_plus = deepcopy(thin)
        thin_minus = deepcopy(thin)
        for mu in 1:4
            add_matrix!(thin_plus[mu], thin_direction[mu], epsilon)
            add_matrix!(thin_minus[mu], thin_direction[mu], -epsilon)
        end
        plus_cache = HISQDiracCache4D(
            thin_plus, mass; naik_epsilon)
        minus_cache = HISQDiracCache4D(
            thin_minus, mass; naik_epsilon)
        loss_plus = _hisq_cached_end_to_end_loss(
            thin_plus[1], thin_plus[2], thin_plus[3], thin_plus[4],
            plus_cache, psi, left, similar(psi))
        loss_minus = _hisq_cached_end_to_end_loss(
            thin_minus[1], thin_minus[2], thin_minus[3], thin_minus[4],
            minus_cache, psi, left, similar(psi))
        finite_difference = (loss_plus - loss_minus) / (2epsilon)
        enzyme_directional = real(sum(
            dot(cached_dthin[mu], thin_direction[mu]) for mu in 1:4))
        @test enzyme_directional ≈ finite_difference atol=8e-5 rtol=5e-6
        @test all(iszero, cached_dresult.A)

        cache_epoch_after_ad = halo_epochs(cache.fat_links[1]).core
        mul_cached_hisq!(
            cached_result, cache,
            thin[1], thin[2], thin[3], thin[4], psi)
        @test halo_epochs(cache.fat_links[1]).core == cache_epoch_after_ad

        cache_epoch_before_replacement =
            halo_epochs(cache.fat_links[1]).core
        updated_result = similar(psi)
        mul_cached_hisq!(
            updated_result, cache,
            thin_plus[1], thin_plus[2], thin_plus[3], thin_plus[4], psi)
        cache_epoch_after_replacement =
            halo_epochs(cache.fat_links[1]).core
        direct_updated_result = similar(psi)
        mul!(direct_updated_result, plus_cache.operator, psi)
        @test cache_epoch_after_replacement > cache_epoch_before_replacement
        @test _hisq_end_to_end_core(updated_result) ≈
            _hisq_end_to_end_core(direct_updated_result) atol=2e-12 rtol=2e-12

        mul_cached_hisq!(
            updated_result, cache,
            thin_plus[1], thin_plus[2], thin_plus[3], thin_plus[4], psi)
        @test halo_epochs(cache.fat_links[1]).core ==
            cache_epoch_after_replacement

        add_matrix!(thin_plus[1], thin_direction[1], epsilon)
        mul_cached_hisq!(
            updated_result, cache,
            thin_plus[1], thin_plus[2], thin_plus[3], thin_plus[4], psi)
        cache_epoch_after_mutation = halo_epochs(cache.fat_links[1]).core
        mutated_cache = HISQDiracCache4D(
            thin_plus, mass; naik_epsilon)
        mul!(direct_updated_result, mutated_cache.operator, psi)
        @test cache_epoch_after_mutation > cache_epoch_after_replacement
        @test _hisq_end_to_end_core(updated_result) ≈
            _hisq_end_to_end_core(direct_updated_result) atol=2e-12 rtol=2e-12

        adjoint_cache = HISQDiracCache4D(
            thin, mass; naik_epsilon)
        adjoint_epoch_before = halo_epochs(adjoint_cache.fat_links[1]).core
        cached_adjoint_result = similar(psi)
        direct_adjoint_result = similar(psi)
        mul_cached_hisq_adjoint!(
            cached_adjoint_result, adjoint_cache,
            thin_minus[1], thin_minus[2], thin_minus[3], thin_minus[4], psi)
        mul!(direct_adjoint_result, minus_cache.operator', psi)
        @test halo_epochs(adjoint_cache.fat_links[1]).core >
            adjoint_epoch_before
        @test _hisq_end_to_end_core(cached_adjoint_result) ≈
            _hisq_end_to_end_core(direct_adjoint_result) atol=2e-12 rtol=2e-12
    end
end
