function _staggered_dirac_ad_loss(operator, psi, left, result)
    mul!(result, operator, psi)
    return real(dot(left, result))
end

function _staggered_dirac_ad_loss_from_links(
    U1, U2, U3, U4, mass, psi, left, result,
)
    operator = StaggeredDiracOperator4D([U1, U2, U3, U4], mass)
    return _staggered_dirac_ad_loss(operator, psi, left, result)
end

function _staggered_dirac_ad_values(
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

function _staggered_dirac_ad_core(field)
    ranges = ntuple(
        d -> (field.nw + 1):(field.nw + field.PN[d]), length(field.PN))
    return @view field.A[:, :, ranges...]
end

function staggered_dirac_ad_tests()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    # An odd local x extent exercises global staggered signs across MPI ranks.
    global_size = (3 * nprocs, 3, 3, 3)
    process_grid = (nprocs, 1, 1, 1)
    NC = 2
    nw = 1
    mass = 0.137
    fermion_phases = (
        cis(0.13), cis(-0.21), cis(0.34), cis(pi - 0.17))

    links = [
        LatticeMatrix(
            _staggered_dirac_ad_values(
                Val(NC), Val(NC), global_size, 7mu),
            4, process_grid; nw,
        ) for mu in 1:4
    ]
    directions = [
        LatticeMatrix(
            _staggered_dirac_ad_values(
                Val(NC), Val(NC), global_size, 11mu + 3),
            4, process_grid; nw,
        ) for mu in 1:4
    ]
    psi = LatticeMatrix(
        _staggered_dirac_ad_values(Val(NC), Val(1), global_size, 5),
        4, process_grid; nw, phases=fermion_phases,
    )
    psi_direction = LatticeMatrix(
        _staggered_dirac_ad_values(Val(NC), Val(1), global_size, 19),
        4, process_grid; nw, phases=fermion_phases,
    )
    left = LatticeMatrix(
        _staggered_dirac_ad_values(Val(NC), Val(1), global_size, 13),
        4, process_grid; nw, phases=fermion_phases,
    )
    set_halo!.(links)
    set_halo!.(directions)
    set_halo!.((psi, psi_direction, left))

    dlinks = [similar(link) for link in links]
    dpsi = similar(psi)
    result = similar(psi)
    dresult = similar(psi)
    clear_matrix!.(dlinks)
    clear_matrix!.((dpsi, result, dresult))

    operator = StaggeredDiracOperator4D(links, mass)

    @testset "StaggeredDiracOperator4D Enzyme pullback" begin
        @test Base.get_extension(
            LatticeMatrices, :LatticeMatricesEnzymeExt) !== nothing
        Enzyme.API.strictAliasing!(false)
        Enzyme.autodiff(
            Enzyme.Reverse,
            Enzyme.Const(_staggered_dirac_ad_loss_from_links),
            Enzyme.Active,
            Enzyme.Duplicated(links[1], dlinks[1]),
            Enzyme.Duplicated(links[2], dlinks[2]),
            Enzyme.Duplicated(links[3], dlinks[3]),
            Enzyme.Duplicated(links[4], dlinks[4]),
            Enzyme.Const(mass),
            Enzyme.Duplicated(psi, dpsi),
            Enzyme.Const(left),
            Enzyme.Duplicated(result, dresult),
        )

        expected_dpsi = similar(psi)
        mul!(expected_dpsi, operator', left)
        @test _staggered_dirac_ad_core(dpsi) ≈
              _staggered_dirac_ad_core(expected_dpsi) atol=2e-12 rtol=2e-12

        epsilon = 1e-6
        links_plus = deepcopy(links)
        links_minus = deepcopy(links)
        for mu in 1:4
            add_matrix!(links_plus[mu], directions[mu], epsilon)
            add_matrix!(links_minus[mu], directions[mu], -epsilon)
        end
        set_halo!.(links_plus)
        set_halo!.(links_minus)

        result_plus = similar(psi)
        result_minus = similar(psi)
        loss_plus = _staggered_dirac_ad_loss(
            StaggeredDiracOperator4D(links_plus, mass),
            psi, left, result_plus)
        loss_minus = _staggered_dirac_ad_loss(
            StaggeredDiracOperator4D(links_minus, mass),
            psi, left, result_minus)
        finite_difference = (loss_plus - loss_minus) / (2epsilon)
        enzyme_directional = real(sum(
            dot(dlinks[mu], directions[mu]) for mu in 1:4))
        @test enzyme_directional ≈ finite_difference atol=2e-6 rtol=2e-7

        psi_plus = deepcopy(psi)
        psi_minus = deepcopy(psi)
        add_matrix!(psi_plus, psi_direction, epsilon)
        add_matrix!(psi_minus, psi_direction, -epsilon)
        set_halo!.((psi_plus, psi_minus))
        psi_loss_plus = _staggered_dirac_ad_loss(
            operator, psi_plus, left, result_plus)
        psi_loss_minus = _staggered_dirac_ad_loss(
            operator, psi_minus, left, result_minus)
        psi_finite_difference =
            (psi_loss_plus - psi_loss_minus) / (2epsilon)
        psi_enzyme_directional = real(dot(dpsi, psi_direction))
        @test isapprox(
            psi_enzyme_directional, psi_finite_difference;
            atol=2e-6, rtol=2e-7)

        @test all(iszero, dresult.A)
    end
end
