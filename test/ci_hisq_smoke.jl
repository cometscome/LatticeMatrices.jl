function ci_hisq_smoke_tests()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    lattice_size = (3 * nprocs, 3, 3, 3)
    process_grid = (nprocs, 1, 1, 1)
    NC = 3
    nw = 3
    naik_epsilon = -0.083

    unit_link = zeros(ComplexF64, NC, NC, lattice_size...)
    for site in CartesianIndices(lattice_size), color in 1:NC
        unit_link[color, color, Tuple(site)...] = 1
    end
    thin_links = [
        LatticeMatrix(unit_link, 4, process_grid; nw) for _ in 1:4
    ]

    @testset "minimal complete HISQ smoke" begin
        links = hisq_links_from_thin(thin_links; naik_epsilon)
        @test length(links.fat_links) == 4
        @test length(links.long_links) == 4
        for link in (links.fat_links..., links.long_links...)
            @test all(isfinite, Array(link.A))
        end

        site_values = reshape(
            ComplexF64.(1:(NC * prod(lattice_size))),
            NC, 1, lattice_size...,
        )
        psi = LatticeMatrix(site_values, 4, process_grid;
            nw, phases=(1, 1, 1, -1))
        result = similar(psi)
        operator = HISQDiracOperator4D(
            links, 0.13; naik_epsilon)
        mul!(result, operator, psi)

        gathered = gather_matrix(result)
        if MPI.Comm_rank(MPI.COMM_WORLD) == 0
            @test all(isfinite, gathered)
            @test !iszero(norm(gathered))
        end

        left_values = complex.(
            sin.(real.(site_values) ./ 13),
            cos.(real.(site_values) ./ 17),
        )
        left = LatticeMatrix(left_values, 4, process_grid;
            nw, phases=(1, 1, 1, -1))
        direction_links = [
            LatticeMatrix(
                complex.(
                    sin.((reshape(
                        Float64.(1:(NC * NC * prod(lattice_size))),
                        NC, NC, lattice_size...) .+ 7mu) ./ 19) ./ 20,
                    cos.((reshape(
                        Float64.(1:(NC * NC * prod(lattice_size))),
                        NC, NC, lattice_size...) .+ 11mu) ./ 23) ./ 20,
                ),
                4, process_grid; nw,
            ) for mu in 1:4
        ]
        force_links = deepcopy(thin_links)
        for mu in 1:4
            add_matrix!(force_links[mu], direction_links[mu], 0.1)
        end
        gradient = [similar(link) for link in force_links]
        clear_matrix!.(gradient)
        cache = HISQDiracCache4D(force_links, 0.13; naik_epsilon)

        @test hisq_link_pullback!(
            gradient, cache, force_links, left, psi) === gradient

        step = 1e-6
        plus_links = deepcopy(force_links)
        minus_links = deepcopy(force_links)
        for mu in 1:4
            add_matrix!(plus_links[mu], direction_links[mu], step)
            add_matrix!(minus_links[mu], direction_links[mu], -step)
        end
        function contraction(links)
            local_cache = HISQDiracCache4D(
                links, 0.13; naik_epsilon)
            local_result = similar(psi)
            mul_cached_hisq!(
                local_result, local_cache,
                links[1], links[2], links[3], links[4], psi)
            return real(dot(left, local_result))
        end
        finite_difference =
            (contraction(plus_links) - contraction(minus_links)) / (2step)
        pullback_directional = real(sum(
            dot(gradient[mu], direction_links[mu]) for mu in 1:4))
        @test isapprox(
            pullback_directional, finite_difference;
            atol=2e-5, rtol=2e-6)
    end

    return nothing
end
