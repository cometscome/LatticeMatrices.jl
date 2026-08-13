function _wilson_dirac_ad_loss(operator, psi, left, result)
    mul!(result, operator, psi)
    return real(dot(left, result))
end

function _wilson_dirac_ad_loss_from_links(
    U1, U2, U3, U4, kappa, psi, left, result,
)
    operator = WilsonDiracOperator4D([U1, U2, U3, U4], kappa)
    return _wilson_dirac_ad_loss(operator, psi, left, result)
end

function _wilson_dirac_ad_values(::Val{N1}, ::Val{N2}, global_size, offset) where {N1,N2}
    count = N1 * N2 * prod(global_size)
    values = reshape(Float64.(1:count), N1, N2, global_size...)
    scale = count + offset
    return complex.((values .+ offset) ./ scale, reverse(values; dims=2) ./ (2scale))
end

function _wilson_dirac_ad_core(field)
    ranges = ntuple(d -> (field.nw + 1):(field.nw + field.PN[d]), length(field.PN))
    return @view field.A[:, :, ranges...]
end

function wilson_dirac_ad_tests()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (2 * nprocs, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)
    NC = 2
    nw = 1
    kappa = 0.117
    fermion_phases = (1.0, 1.0, 1.0, -1.0)

    links = [
        LatticeMatrix(
            _wilson_dirac_ad_values(Val(NC), Val(NC), global_size, 7mu),
            4, process_grid; nw,
        ) for mu in 1:4
    ]
    directions = [
        LatticeMatrix(
            _wilson_dirac_ad_values(Val(NC), Val(NC), global_size, 11mu + 3),
            4, process_grid; nw,
        ) for mu in 1:4
    ]
    psi = LatticeMatrix(
        _wilson_dirac_ad_values(Val(NC), Val(4), global_size, 5),
        4, process_grid; nw, phases=fermion_phases,
    )
    left = LatticeMatrix(
        _wilson_dirac_ad_values(Val(NC), Val(4), global_size, 13),
        4, process_grid; nw, phases=fermion_phases,
    )
    set_halo!.(links)
    set_halo!.(directions)
    set_halo!.((psi, left))

    dlinks = [similar(link) for link in links]
    dpsi = similar(psi)
    result = similar(psi)
    dresult = similar(psi)
    clear_matrix!.(dlinks)
    clear_matrix!.((dpsi, result, dresult))

    operator = WilsonDiracOperator4D(links, kappa)

    @testset "WilsonDiracOperator4D Enzyme pullback" begin
        @test Base.get_extension(LatticeMatrices, :LatticeMatricesEnzymeExt) !== nothing
        Enzyme.API.strictAliasing!(false)
        Enzyme.autodiff(
            Enzyme.Reverse,
            Enzyme.Const(_wilson_dirac_ad_loss_from_links),
            Enzyme.Active,
            Enzyme.Duplicated(links[1], dlinks[1]),
            Enzyme.Duplicated(links[2], dlinks[2]),
            Enzyme.Duplicated(links[3], dlinks[3]),
            Enzyme.Duplicated(links[4], dlinks[4]),
            Enzyme.Const(kappa),
            Enzyme.Duplicated(psi, dpsi),
            Enzyme.Const(left),
            Enzyme.Duplicated(result, dresult),
        )

        expected_dpsi = similar(psi)
        mul!(expected_dpsi, operator', left)
        @test _wilson_dirac_ad_core(dpsi) ≈
              _wilson_dirac_ad_core(expected_dpsi) atol=2e-12 rtol=2e-12

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
        loss_plus = _wilson_dirac_ad_loss(
            WilsonDiracOperator4D(links_plus, kappa), psi, left, result_plus)
        loss_minus = _wilson_dirac_ad_loss(
            WilsonDiracOperator4D(links_minus, kappa), psi, left, result_minus)
        finite_difference = (loss_plus - loss_minus) / (2epsilon)
        enzyme_directional = real(sum(dot(dlinks[mu], directions[mu]) for mu in 1:4))

        @test enzyme_directional ≈ finite_difference atol=2e-6 rtol=2e-7
        @test all(iszero, dresult.A)
    end
end
