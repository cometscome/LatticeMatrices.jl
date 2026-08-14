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
    thin, level1, reunitarized, fat, long,
    mass, naik_epsilon, psi, left, result,
)
    links = hisq_links_from_thin!(
        fat, long, level1, reunitarized, thin, naik_epsilon)
    operator = HISQDiracOperator4D(
        links, mass; naik_epsilon)
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

function _hisq_end_to_end_core(field)
    ranges = ntuple(
        d -> (field.nw + 1):(field.nw + field.PN[d]), 4)
    return @view field.A[:, :, ranges...]
end

function hisq_end_to_end_ad_tests()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    process_grid = (nprocs, 1, 1, 1)
    global_size = (3 * nprocs, 3, 3, 3)
    nw = 3
    mass = 0.137
    naik_epsilon = -0.083

    thin = [
        LatticeMatrix(
            _hisq_projection_ad_values(
                global_size, 23mu; identity_shift=1.15),
            4, process_grid; nw,
        ) for mu in 1:4
    ]
    thin_direction = [
        LatticeMatrix(
            _hisq_projection_ad_values(global_size, 73 + 7mu),
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

    @testset "HISQ thin-link to Dirac-action Enzyme pullback" begin
        Enzyme.API.strictAliasing!(false)
        Enzyme.autodiff(
            Enzyme.Reverse,
            Enzyme.Const(_hisq_end_to_end_loss_from_thin),
            Enzyme.Active,
            Enzyme.Duplicated(thin, dthin),
            Enzyme.Duplicated(level1, dlevel1),
            Enzyme.Duplicated(reunitarized, dreunitarized),
            Enzyme.Duplicated(fat, dfat),
            Enzyme.Duplicated(long, dlong),
            Enzyme.Const(mass),
            Enzyme.Const(naik_epsilon),
            Enzyme.Duplicated(psi, dpsi),
            Enzyme.Const(left),
            Enzyme.Duplicated(result, dresult),
        )

        operator = HISQDiracOperator4D(
            fat, long, mass; naik_epsilon)
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
