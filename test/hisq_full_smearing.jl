function _hisq_full_test_projection_reference(fat_links)
    projected = [similar(link) for link in fat_links]
    lattice_size = size(fat_links[1])[3:end]
    for mu in 1:4, site_index in CartesianIndices(lattice_size)
        site = Tuple(site_index)
        @views factorization = svd(fat_links[mu][:, :, site...])
        @views projected[mu][:, :, site...] .=
            factorization.U * factorization.Vt
    end
    return projected
end

function _hisq_full_test_level2_reference(reunitarized_links, naik_epsilon)
    fat_links = _hisq_smearing_test_level1_reference(reunitarized_links)
    lattice_size = size(reunitarized_links[1])[3:end]
    real_type = typeof(real(zero(eltype(reunitarized_links[1]))))
    one_link_correction =
        one(real_type) + naik_epsilon / 8 - one(real_type) / 8
    lepage_coefficient = -one(real_type) / 8
    for mu in 1:4, site_index in CartesianIndices(lattice_size)
        site = Tuple(site_index)
        @views fat_links[mu][:, :, site...] .+=
            one_link_correction .* reunitarized_links[mu][:, :, site...]
        for nu in 1:4
            nu == mu && continue
            for sign_nu in (-1, 1)
                signed_nu = sign_nu * nu
                @views fat_links[mu][:, :, site...] .+=
                    lepage_coefficient .* _hisq_smearing_test_path(
                        reunitarized_links, site,
                        (signed_nu, signed_nu, mu,
                         -signed_nu, -signed_nu))
            end
        end
    end
    return fat_links
end

function _hisq_full_test_naik_reference(reunitarized_links)
    lattice_size = size(reunitarized_links[1])[3:end]
    element_type = eltype(reunitarized_links[1])
    long_links = [zeros(
        element_type, 3, 3, lattice_size...) for _ in 1:4]
    for mu in 1:4, site_index in CartesianIndices(lattice_size)
        site = Tuple(site_index)
        @views long_links[mu][:, :, site...] .=
            _hisq_smearing_test_path(
                reunitarized_links, site, (mu, mu, mu))
    end
    return long_links
end

function hisq_full_smearing_tests()
    nprocs = test_comm_size()
    rank = test_comm_rank()
    process_grid = (nprocs, 1, 1, 1)
    lattice_size = (3 * nprocs, 3, 3, 3)
    NC = 3
    thin_arrays = _staggered_test_links(lattice_size, NC)
    for mu in 1:4, site_index in CartesianIndices(lattice_size)
        site = Tuple(site_index)
        @views thin_arrays[mu][:, :, site...] .*= 0.05
        @views thin_arrays[mu][:, :, site...] .+=
            Matrix{ComplexF64}(I, NC, NC)
    end
    U = [LatticeMatrix(link, 4, process_grid; nw=3)
         for link in thin_arrays]
    level1 = hisq_fat7_level1(U)

    @testset "HISQ U(3) projection" begin
        global_level1 = _hisq_smearing_test_gather(level1)
        reference = rank == 0 ?
            _hisq_full_test_projection_reference(global_level1) : nothing
        projected = hisq_project_u3(level1)
        global_projected = _hisq_smearing_test_gather(projected)
        if rank == 0
            for mu in 1:4
                @test global_projected[mu] ≈ reference[mu] atol=2e-11 rtol=2e-11
                maximum_unitarity_error = 0.0
                for site_index in CartesianIndices(lattice_size)
                    site = Tuple(site_index)
                    @views unitary_check =
                        global_projected[mu][:, :, site...]' *
                        global_projected[mu][:, :, site...]
                    maximum_unitarity_error = max(
                        maximum_unitarity_error,
                        norm(unitary_check - Matrix{ComplexF64}(I, 3, 3)))
                end
                @test maximum_unitarity_error < 3e-11
            end
        end
        @test all(halo_is_dirty, projected)

        projected_inplace = [similar(link) for link in level1]
        @test hisq_project_u3!(projected_inplace, level1) === projected_inplace
        global_inplace = _hisq_smearing_test_gather(projected_inplace)
        if rank == 0
            for mu in 1:4
                @test global_inplace[mu] ≈ global_projected[mu]
            end
        end
    end

    @testset "HISQ level-2 Fat7 and Lepage" begin
        naik_epsilon = -0.083
        reunitarized = hisq_project_u3(level1)
        fat_links = hisq_fat7_level2(
            reunitarized; naik_epsilon)
        global_reunitarized = _hisq_smearing_test_gather(reunitarized)
        global_fat_links = _hisq_smearing_test_gather(fat_links)
        if rank == 0
            reference = _hisq_full_test_level2_reference(
                global_reunitarized, naik_epsilon)
            for mu in 1:4
                @test global_fat_links[mu] ≈ reference[mu] atol=8e-11 rtol=8e-11
            end
        end
        @test all(halo_is_dirty, fat_links)

        inplace = [similar(link) for link in reunitarized]
        @test hisq_fat7_level2!(
            inplace, reunitarized; naik_epsilon) === inplace
        global_inplace = _hisq_smearing_test_gather(inplace)
        if rank == 0
            for mu in 1:4
                @test global_inplace[mu] ≈ global_fat_links[mu]
            end
        end

        unit_array = _staggered_test_unit_link(lattice_size, NC)
        unit_links = [LatticeMatrix(
            unit_array, 4, process_grid; nw=2) for _ in 1:4]
        global_unit_fat = _hisq_smearing_test_gather(
            hisq_fat7_level2(unit_links; naik_epsilon))
        if rank == 0
            expected_scale = (9 + naik_epsilon) / 8
            for mu in 1:4
                @test global_unit_fat[mu] ≈
                    expected_scale .* unit_array atol=5e-15 rtol=5e-15
            end
        end

        short_halo_links = [LatticeMatrix(
            thin_arrays[mu], 4, process_grid; nw=1) for mu in 1:4]
        @test_throws ArgumentError hisq_fat7_level2(short_halo_links)

        small_size = (2 * nprocs, 2, 2, 2)
        small_arrays = _staggered_test_links(small_size, 3)
        small_links = [LatticeMatrix(
            link, 4, process_grid; nw=0) for link in small_arrays]
        small_fat = _hisq_smearing_test_gather(
            hisq_fat7_level2(small_links; naik_epsilon))
        if rank == 0
            small_reference = _hisq_full_test_level2_reference(
                small_arrays, naik_epsilon)
            for mu in 1:4
                @test small_fat[mu] ≈ small_reference[mu] atol=8e-11 rtol=8e-11
            end
        end
    end

    @testset "HISQ forward-anchored Naik links" begin
        reunitarized = hisq_project_u3(level1)
        long_links = hisq_naik_links(reunitarized)
        global_reunitarized = _hisq_smearing_test_gather(reunitarized)
        global_long_links = _hisq_smearing_test_gather(long_links)
        if rank == 0
            reference = _hisq_full_test_naik_reference(
                global_reunitarized)
            for mu in 1:4
                @test global_long_links[mu] ≈ reference[mu] atol=3e-11 rtol=3e-11
            end
        end
        @test all(halo_is_dirty, long_links)

        inplace = [similar(link) for link in reunitarized]
        @test hisq_naik_links!(inplace, reunitarized) === inplace
        global_inplace = _hisq_smearing_test_gather(inplace)
        if rank == 0
            for mu in 1:4
                @test global_inplace[mu] ≈ global_long_links[mu]
            end
        end

        unit_array = _staggered_test_unit_link(lattice_size, NC)
        unit_links = [LatticeMatrix(
            unit_array, 4, process_grid; nw=2) for _ in 1:4]
        global_unit_long = _hisq_smearing_test_gather(
            hisq_naik_links(unit_links))
        if rank == 0
            for mu in 1:4
                @test global_unit_long[mu] ≈ unit_array
            end
        end


        small_size = (2 * nprocs, 2, 2, 2)
        small_arrays = _staggered_test_links(small_size, 3)
        small_links = [LatticeMatrix(
            link, 4, process_grid; nw=0) for link in small_arrays]
        small_long = _hisq_smearing_test_gather(
            hisq_naik_links(small_links))
        if rank == 0
            small_reference = _hisq_full_test_naik_reference(small_arrays)
            for mu in 1:4
                @test small_long[mu] ≈ small_reference[mu] atol=2e-12 rtol=2e-12
            end
        end
    end

    @testset "HISQ thin-link integrated builder" begin
        naik_epsilon = -0.083
        level1_workspace = [similar(link) for link in U]
        reunitarized_workspace = [similar(link) for link in U]
        fat_links = [similar(link) for link in U]
        long_links = [similar(link) for link in U]
        links = hisq_links_from_thin!(
            fat_links, long_links, level1_workspace,
            reunitarized_workspace, U; naik_epsilon)
        @test links.fat_links === fat_links
        @test links.long_links === long_links

        staged_level1 = hisq_fat7_level1(U)
        staged_reunitarized = hisq_project_u3(staged_level1)
        staged_fat = hisq_fat7_level2(
            staged_reunitarized; naik_epsilon)
        staged_long = hisq_naik_links(staged_reunitarized)
        global_fat = _hisq_smearing_test_gather(fat_links)
        global_long = _hisq_smearing_test_gather(long_links)
        global_staged_fat = _hisq_smearing_test_gather(staged_fat)
        global_staged_long = _hisq_smearing_test_gather(staged_long)
        if rank == 0
            for mu in 1:4
                @test global_fat[mu] ≈ global_staged_fat[mu]
                @test global_long[mu] ≈ global_staged_long[mu]
            end
        end

        factorized_level1 = [similar(link) for link in U]
        factorized_reunitarized = [similar(link) for link in U]
        factorized_fat = [similar(link) for link in U]
        factorized_long = [similar(link) for link in U]
        fat7_workspace = HISQFat7Workspace(U[1])
        factorized_links = hisq_links_from_thin!(
            factorized_fat, factorized_long, factorized_level1,
            factorized_reunitarized, U;
            naik_epsilon, fat7_workspace)
        @test factorized_links.fat_links === factorized_fat
        @test factorized_links.long_links === factorized_long
        global_factorized_fat = _hisq_smearing_test_gather(factorized_fat)
        global_factorized_long = _hisq_smearing_test_gather(factorized_long)
        if rank == 0
            for mu in 1:4
                @test global_factorized_fat[mu] ≈ global_fat[mu]
                @test global_factorized_long[mu] ≈ global_long[mu]
            end
        end

        workspace_fields = (
            fat7_workspace.first_stage..., fat7_workspace.second_stage...)
        @test length(workspace_fields) == 6
        @test all(field -> field !== U[1] && field.A !== U[1].A,
            workspace_fields)
        @test all(i -> all(j -> workspace_fields[i].A !==
            workspace_fields[j].A, (i + 1):length(workspace_fields)),
            eachindex(workspace_fields))

        allocating_links = hisq_links_from_thin(U; naik_epsilon)
        global_allocating_fat = _hisq_smearing_test_gather(
            allocating_links.fat_links)
        global_allocating_long = _hisq_smearing_test_gather(
            allocating_links.long_links)
        if rank == 0
            for mu in 1:4
                @test global_allocating_fat[mu] ≈ global_fat[mu]
                @test global_allocating_long[mu] ≈ global_long[mu]
            end
        end

        operator = HISQDiracOperator4D(
            U, 0.13; naik_epsilon)
        @test operator.naik_epsilon == naik_epsilon

        cache = HISQDiracCache4D(U, 0.13; naik_epsilon)
        cached_workspace = cache.fat7_workspace
        cached_fields = (
            cached_workspace.first_stage..., cached_workspace.second_stage...)
        @test update_hisq_cache!(cache, U) === cache
        @test cache.fat7_workspace === cached_workspace
        @test all(i -> cache.fat7_workspace.first_stage[i] ===
            cached_workspace.first_stage[i], 1:3)
        @test all(field -> all(output -> field.A !== output.A,
            (cache.level1_links..., cache.reunitarized_links...,
             cache.fat_links..., cache.long_links...)), cached_fields)

        @test_throws ArgumentError hisq_links_from_thin!(
            fat_links, long_links, level1_workspace,
            reunitarized_workspace, reunitarized_workspace,
            naik_epsilon)
    end


    @testset "SIMULATeQCD complete HISQ numerical fingerprints" begin
        # Generated with test/reference/simulateqcd/hisq_full_reference.cpp
        # against unmodified SIMULATeQCD commit 767a1b1.  The external code
        # runs SmearAll(..., false); its centered Naik storage is shifted by
        # +mu while fingerprinting to match our forward-anchored convention.
        if nprocs == 1
            oracle_size = (4, 4, 4, 4)
            oracle_arrays = _staggered_test_links(oracle_size, 3)
            for mu in 1:4, site_index in CartesianIndices(oracle_size)
                site = Tuple(site_index)
                @views oracle_arrays[mu][:, :, site...] .*= 0.05
                @views oracle_arrays[mu][:, :, site...] .+=
                    Matrix{ComplexF64}(I, 3, 3)
            end
            oracle_thin = [LatticeMatrix(
                link, 4, process_grid; nw=3) for link in oracle_arrays]
            oracle_links = hisq_links_from_thin(
                oracle_thin; naik_epsilon=-0.083)
            oracle_fat = _hisq_smearing_test_gather(
                oracle_links.fat_links)
            oracle_long = _hisq_smearing_test_gather(
                oracle_links.long_links)
            expected_fat = (
                (853.41183333640481, -66.39794398689061,
                 983245.21666339913, -80498.631526783254,
                 954.15509307762159),
                (853.58749744956594, -63.995520058562001,
                 983448.67161160929, -77868.557850150959,
                 954.16187675216861),
                (853.75144888633542, -61.543730591729862,
                 983584.74607885501, -75956.094795296769,
                 954.17512186157603),
                (853.90291229981756, -59.042186117717975,
                 983434.34811777505, -78085.285368423953,
                 954.1945225327936),
            )
            expected_long = (
                (746.94343196422199, -177.21277936443548,
                 858538.53621536866, -213969.95381318283,
                 767.99999999999943),
                (748.4229872091264, -170.90286991902855,
                 860369.19904656592, -206524.7738402886,
                 768.00000000000102),
                (749.8837430292117, -164.45467537282036,
                 862363.75391496124, -198084.52122487241,
                 767.99999999999989),
                (751.32073692799725, -157.86624416021118,
                 865254.3048196903, -185197.79308273623,
                 768.00000000000136),
            )
            for mu in 1:4
                fat_fingerprint = _staggered_test_fingerprint(
                    oracle_fat[mu])
                long_fingerprint = _staggered_test_fingerprint(
                    oracle_long[mu])
                @test isapprox(
                    collect(fat_fingerprint), collect(expected_fat[mu]);
                    atol=2e-8, rtol=3e-14)
                @test isapprox(
                    collect(long_fingerprint), collect(expected_long[mu]);
                    atol=2e-8, rtol=3e-14)
            end
        end
    end
end
