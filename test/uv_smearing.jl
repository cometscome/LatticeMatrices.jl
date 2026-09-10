function _uv_test_links(global_size, process_grid; normalized=true)
    links = [
        LatticeMatrix(
            _nhyp_test_values(Val(3), global_size, 13mu; diagonal=1.2),
            4, process_grid; nw=1)
        for mu in 1:4
    ]
    if normalized
        normalize_matrix!.(links)
    end
    return links
end

function _uv_test_directions(global_size, process_grid, offset)
    return [
        LatticeMatrix(
            _nhyp_test_values(Val(3), global_size, offset + 7mu),
            4, process_grid; nw=1)
        for mu in 1:4
    ]
end

function _uv_smearing_loss(links, left, parameters)
    output, _ = smear_links(links, parameters)
    return real(sum(dot(left[mu], output[mu]) for mu in 1:4))
end

function _uv_check_pullback(parameters, thin, direction, left; tolerance)
    output, cache = smear_links(thin, parameters)
    dthin = [similar(link) for link in thin]
    smear_links_pullback!(dthin, left, thin, cache)

    epsilon = 2e-6
    plus = deepcopy(thin)
    minus = deepcopy(thin)
    for mu in 1:4
        add_matrix!(plus[mu], direction[mu], epsilon)
        add_matrix!(minus[mu], direction[mu], -epsilon)
    end
    finite_difference = (
        _uv_smearing_loss(plus, left, parameters) -
        _uv_smearing_loss(minus, left, parameters)
    ) / (2epsilon)
    analytic = real(sum(dot(dthin[mu], direction[mu]) for mu in 1:4))
    @test isapprox(analytic, finite_difference; atol=tolerance, rtol=tolerance)
    return output, cache
end

function _uv_check_special_unitary(links; tolerance=5e-11)
    for link in links
        gathered = gather_and_bcast_matrix(link)
        if test_comm_rank() == 0
            for site in CartesianIndices(size(gathered)[3:6])
                matrix = Matrix(@view gathered[:, :, Tuple(site)...])
                @test isapprox(matrix' * matrix, I;
                    atol=tolerance, rtol=tolerance)
                @test isapprox(det(matrix), 1;
                    atol=tolerance, rtol=tolerance)
            end
        end
    end
    return nothing
end

function uv_smearing_tests()
    nprocs = test_comm_size()
    process_grid = (nprocs, 1, 1, 1)
    global_size = (2nprocs, 2, 2, 2)
    thin = _uv_test_links(global_size, process_grid)
    direction = _uv_test_directions(global_size, process_grid, 47)
    left = _uv_test_directions(global_size, process_grid, 89)

    @testset "UV-smearing parameters and common protocol" begin
        @test APEParameters().alpha == 0.6
        @test APEParameters().projection == :max_retr
        @test StoutParameters().rho == 0.1
        @test HYPParameters() == HYPParameters(0.75, 0.6, 0.3)
        @test HYPParameters().projection == :max_retr
        @test HEXParameters() == HEXParameters(0.125, 0.15, 0.15)
        @test_throws ArgumentError APEParameters(Inf)
        @test_throws ArgumentError StoutParameters(NaN)
        @test_throws ArgumentError HYPParameters(alpha_inner=Inf)
        @test_throws ArgumentError APEParameters(projection=:unknown)
        @test_throws ArgumentError HYPParameters(max_retr_iterations=0)
        @test_throws ArgumentError HEXParameters(alpha_outer=NaN)
        @test_throws ArgumentError IteratedSmearing(StoutParameters(), 0)

        float32_links = [
            LatticeMatrix(
                ComplexF32.(_nhyp_test_values(
                    Val(3), global_size, 121 + mu; diagonal=1.1)),
                4, process_grid; nw=1)
            for mu in 1:4
        ]
        @test smearing_cache(float32_links, StoutParameters()).parameters isa
            StoutParameters{Float32}
        @test smearing_cache(float32_links, HEXParameters()).parameters isa
            HEXParameters{Float32}
        for parameters in (APEParameters(), HYPParameters())
            output, cache = smear_links(float32_links, parameters)
            @test cache.parameters isa Union{
                APEParameters{Float32},HYPParameters{Float32}}
            @test all(link -> all(isfinite, link.A), output)
        end
    end

    @testset "stout and HEX forward and pullback" begin
        stout_output, stout_cache = _uv_check_pullback(
            StoutParameters(), thin, direction, left; tolerance=7e-6)
        @test stout_cache isa StoutSmearingCache4D
        _uv_check_special_unitary(stout_output)

        hex_output, hex_cache = _uv_check_pullback(
            HEXParameters(), thin, direction, left; tolerance=8e-6)
        @test hex_cache isa HEXSmearingCache4D
        _uv_check_special_unitary(hex_output; tolerance=8e-11)

        add_matrix!(thin[1], direction[1], 1e-8)
        @test_throws ArgumentError stout_pullback!(
            [similar(link) for link in thin], left, thin, stout_cache)
        @test_throws ArgumentError hex_pullback!(
            [similar(link) for link in thin], left, thin, hex_cache)
        add_matrix!(thin[1], direction[1], -1e-8)
    end

    @testset "APE and HYP principal-polar forward and pullback" begin
        ape_output, ape_cache = _uv_check_pullback(
            APEParameters(projection=:polar), thin, direction, left;
            tolerance=8e-6)
        hyp_output, hyp_cache = _uv_check_pullback(
            HYPParameters(projection=:polar), thin, direction, left;
            tolerance=1e-5)
        @test ape_cache isa APESmearingCache4D
        @test hyp_cache isa HYPSmearingCache4D
        _uv_check_special_unitary(ape_output; tolerance=2e-10)
        _uv_check_special_unitary(hyp_output; tolerance=2e-10)
        add_matrix!(thin[1], direction[1], 1e-8)
        @test_throws ArgumentError ape_pullback!(
            [similar(link) for link in thin], left, thin, ape_cache)
        @test_throws ArgumentError hyp_pullback!(
            [similar(link) for link in thin], left, thin, hyp_cache)
        add_matrix!(thin[1], direction[1], -1e-8)
    end

    @testset "APE and HYP MaxReTr forward" begin
        for parameters in (APEParameters(), HYPParameters())
            output, cache = smear_links(thin, parameters)
            _uv_check_special_unitary(output; tolerance=3e-10)
            exception = try
                smear_links_pullback!(
                    [similar(link) for link in thin], left, thin, cache)
                nothing
            catch caught
                caught
            end
            @test exception isa ArgumentError
            @test occursin(
                "projection=:polar", sprint(showerror, exception))
        end

        su2_thin = [
            LatticeMatrix(
                _nhyp_test_values(
                    Val(2), global_size, 211 + mu; diagonal=1.1),
                4, process_grid; nw=1)
            for mu in 1:4
        ]
        normalize_matrix!.(su2_thin)
        for parameters in (APEParameters(), HYPParameters())
            output, _ = smear_links(su2_thin, parameters)
            unitarity_error = 0.0
            determinant_error = 0.0
            for link in output
                gathered = gather_and_bcast_matrix(link)
                for site in CartesianIndices(size(gathered)[3:6])
                    matrix = Matrix(@view gathered[:, :, Tuple(site)...])
                    unitarity_error = max(
                        unitarity_error, maximum(abs, matrix' * matrix - I))
                    determinant_error = max(
                        determinant_error, abs(det(matrix) - 1))
                end
            end
            @test unitarity_error < 3e-10
            @test determinant_error < 3e-10
        end
    end

    @testset "zero coefficients and iteration" begin
        zero_specs = (
            APEParameters(0; projection=:polar),
            StoutParameters(0),
            HYPParameters(0, 0, 0; projection=:polar),
            HEXParameters(0, 0, 0),
        )
        for specification in zero_specs
            output, _ = smear_links(thin, specification)
            for mu in 1:4
                @test gather_and_bcast_matrix(output[mu]) ≈
                    gather_and_bcast_matrix(thin[mu]) atol=7e-11 rtol=7e-11
            end
        end

        iterated = IteratedSmearing(StoutParameters(0.07), 2)
        output, cache = _uv_check_pullback(
            iterated, thin, direction, left; tolerance=1e-5)
        @test cache isa IteratedSmearingCache4D
        @test length(cache.stage_caches) == 2
        _uv_check_special_unitary(output; tolerance=8e-11)
    end
    return nothing
end
