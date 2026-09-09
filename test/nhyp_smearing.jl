function _nhyp_test_shift(site, axis, amount, lattice_size)
    return ntuple(Val(4)) do direction
        mod1(site[direction] + ifelse(direction == axis, amount, 0),
             lattice_size[direction])
    end
end

function _nhyp_test_sym_staple(side, middle, side_axis, middle_axis)
    lattice_size = size(side)[3:6]
    staple = similar(side)
    for index in CartesianIndices(lattice_size)
        origin = Tuple(index)
        plus_side = _nhyp_test_shift(origin, side_axis, 1, lattice_size)
        plus_middle = _nhyp_test_shift(origin, middle_axis, 1, lattice_size)
        minus_side = _nhyp_test_shift(origin, side_axis, -1, lattice_size)
        minus_side_plus_middle = _nhyp_test_shift(
            minus_side, middle_axis, 1, lattice_size)
        @views staple[:, :, origin...] .=
            side[:, :, origin...] * middle[:, :, plus_side...] *
            side[:, :, plus_middle...]' +
            side[:, :, minus_side...]' * middle[:, :, minus_side...] *
            side[:, :, minus_side_plus_middle...]
    end
    return staple
end

function _nhyp_test_project(input)
    lattice_size = size(input)[3:6]
    output = similar(input)
    for index in CartesianIndices(lattice_size)
        site = Tuple(index)
        decomposition = svd(Matrix(@view input[:, :, site...]))
        @views output[:, :, site...] .= decomposition.U * decomposition.Vt
    end
    return output
end

function _nhyp_test_reference(thin_links, parameters)
    inner = Dict{Tuple{Int,Int},typeof(thin_links[1])}()
    for (mu, nu) in LatticeMatrices._nhyp_direction_pairs
        candidate = (1 - parameters.alpha_inner) .* thin_links[mu]
        candidate .+= (parameters.alpha_inner / 2) .*
            _nhyp_test_sym_staple(
                thin_links[nu], thin_links[mu], nu, mu)
        inner[(mu, nu)] = _nhyp_test_project(candidate)
    end

    middle = Dict{Tuple{Int,Int},typeof(thin_links[1])}()
    for (mu, nu) in LatticeMatrices._nhyp_direction_pairs
        candidate = (1 - parameters.alpha_middle) .* thin_links[mu]
        for side_axis in 1:4
            (side_axis == mu || side_axis == nu) && continue
            excluded_axis = 10 - mu - nu - side_axis
            candidate .+= (parameters.alpha_middle / 4) .*
                _nhyp_test_sym_staple(
                    inner[(side_axis, excluded_axis)],
                    inner[(mu, excluded_axis)], side_axis, mu)
        end
        middle[(mu, nu)] = _nhyp_test_project(candidate)
    end

    return [begin
        candidate = (1 - parameters.alpha_outer) .* thin_links[mu]
        for nu in 1:4
            nu == mu && continue
            candidate .+= (parameters.alpha_outer / 6) .*
                _nhyp_test_sym_staple(
                    middle[(nu, mu)], middle[(mu, nu)], nu, mu)
        end
        _nhyp_test_project(candidate)
    end for mu in 1:4]
end

function _nhyp_test_values(::Val{NC}, lattice_size, offset; diagonal=0) where NC
    count = NC * NC * prod(lattice_size)
    values = reshape(Float64.(1:count), NC, NC, lattice_size...)
    output = complex.(
        sin.((values .+ offset) ./ 17) ./ 12,
        cos.((2values .+ offset) ./ 19) ./ 15,
    )
    if !iszero(diagonal)
        for index in CartesianIndices(lattice_size), color in 1:NC
            output[color, color, Tuple(index)...] += diagonal
        end
    end
    return output
end

function _nhyp_test_loss(thin_links, left, parameters)
    smeared, _ = nhyp_smear(thin_links, parameters)
    return real(sum(dot(left[mu], smeared[mu]) for mu in 1:4))
end

function nhyp_smearing_tests()
    nprocs = test_comm_size()
    process_grid = (nprocs, 1, 1, 1)
    lattice_size = (2nprocs, 2, 2, 2)
    NC = 3
    parameters = NHYPParameters(
        alpha_outer=0.5, alpha_middle=0.5, alpha_inner=0.4)
    thin_arrays = [
        _nhyp_test_values(Val(NC), lattice_size, 7mu; diagonal=1.1)
        for mu in 1:4
    ]
    thin_links = [
        LatticeMatrix(thin_arrays[mu], 4, process_grid; nw=1)
        for mu in 1:4
    ]
    direction_links = [
        LatticeMatrix(
            _nhyp_test_values(Val(NC), lattice_size, 41 + 5mu),
            4, process_grid; nw=1)
        for mu in 1:4
    ]
    left_links = [
        LatticeMatrix(
            _nhyp_test_values(Val(NC), lattice_size, 83 + 11mu),
            4, process_grid; nw=1)
        for mu in 1:4
    ]

    @testset "nHYP parameters and QEX convention" begin
        @test parameters.alpha_inner == 0.4
        @test parameters.alpha_middle == 0.5
        @test parameters.alpha_outer == 0.5
        @test_throws ArgumentError NHYPParameters(alpha_outer=Inf)
        float32_links = [
            LatticeMatrix(
                ComplexF32.(thin_arrays[mu]), 4, process_grid; nw=1)
            for mu in 1:4
        ]
        float32_cache = NHYPSmearingCache4D(float32_links, parameters)
        @test float32_cache.parameters isa NHYPParameters{Float32}
    end

    @testset "nHYP forward reference" begin
        expected = _nhyp_test_reference(thin_arrays, parameters)
        smeared, cache = nhyp_smear(thin_links, parameters)
        @test cache isa NHYPSmearingCache4D
        for mu in 1:4
            gathered = gather_matrix(smeared[mu])
            if test_comm_rank() == 0
                @test isapprox(
                    gathered, expected[mu]; atol=2e-11, rtol=2e-11)
                for index in CartesianIndices(lattice_size)
                    matrix = @view gathered[:, :, Tuple(index)...]
                    @test isapprox(
                        matrix' * matrix, I; atol=3e-11, rtol=3e-11)
                end
            end
        end

        inplace = [similar(link) for link in thin_links]
        inplace_cache = NHYPSmearingCache4D(thin_links, parameters)
        @test nhyp_smear!(inplace, thin_links, inplace_cache) === inplace
        @test_throws ArgumentError nhyp_smear!(
            thin_links, thin_links, inplace_cache)
    end

    @testset "nHYP analytic pullback" begin
        smeared, cache = nhyp_smear(thin_links, parameters)
        dthin = [similar(link) for link in thin_links]
        @test nhyp_pullback!(
            dthin, left_links, thin_links, cache) === dthin

        epsilon = 2e-6
        plus_links = deepcopy(thin_links)
        minus_links = deepcopy(thin_links)
        for mu in 1:4
            add_matrix!(plus_links[mu], direction_links[mu], epsilon)
            add_matrix!(minus_links[mu], direction_links[mu], -epsilon)
        end
        finite_difference = (
            _nhyp_test_loss(plus_links, left_links, parameters) -
            _nhyp_test_loss(minus_links, left_links, parameters)
        ) / (2epsilon)
        pullback_directional = real(sum(
            dot(dthin[mu], direction_links[mu]) for mu in 1:4))
        @test isapprox(
            pullback_directional, finite_difference;
            atol=4e-5, rtol=4e-6)

        add_matrix!(thin_links[1], direction_links[1], 1e-7)
        @test_throws ArgumentError nhyp_pullback!(
            dthin, left_links, thin_links, cache)
    end

    @testset "nHYP halo validation" begin
        no_halo_links = [
            LatticeMatrix(thin_arrays[mu], 4, process_grid; nw=0)
            for mu in 1:4
        ]
        @test_throws ArgumentError NHYPSmearingCache4D(
            no_halo_links, parameters)
    end
    return nothing
end
