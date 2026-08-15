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
    end

    return nothing
end
