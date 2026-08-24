function _hisq_smearing_test_path(thin_links, origin, path)
    lattice_size = size(thin_links[1])[3:end]
    NC = size(thin_links[1], 1)
    elementtype = eltype(thin_links[1])
    transporter = Matrix{elementtype}(I, NC, NC)
    site = origin

    for direction in path
        axis = abs(direction)
        if direction > 0
            @views transporter =
                transporter * thin_links[axis][:, :, site...]
            site = _staggered_test_shift(
                site, axis, 1, lattice_size)
        else
            site = _staggered_test_shift(
                site, axis, -1, lattice_size)
            @views transporter =
                transporter * thin_links[axis][:, :, site...]'
        end
    end
    return transporter
end

function _hisq_smearing_test_level1_reference(thin_links)
    lattice_size = size(thin_links[1])[3:end]
    NC = size(thin_links[1], 1)
    elementtype = eltype(thin_links[1])
    real_type = typeof(real(zero(elementtype)))
    coefficients = (
        one(real_type) / 8,
        one(real_type) / 16,
        one(real_type) / 64,
        one(real_type) / 384,
    )
    coefficient_1, coefficient_3, coefficient_5, coefficient_7 = coefficients
    fat_links = [zeros(elementtype, NC, NC, lattice_size...) for _ in 1:4]

    for site_index in CartesianIndices(lattice_size)
        site = Tuple(site_index)
        for mu in 1:4
            @views value = coefficient_1 .* thin_links[mu][:, :, site...]
            for nu in 1:4
                nu == mu && continue
                for sign_nu in (-1, 1)
                    signed_nu = sign_nu * nu
                    value .+= coefficient_3 .* _hisq_smearing_test_path(
                        thin_links, site, (signed_nu, mu, -signed_nu))
                end

                for rho in 1:4
                    (rho == mu || rho == nu) && continue
                    for sign_nu in (-1, 1), sign_rho in (-1, 1)
                        signed_nu = sign_nu * nu
                        signed_rho = sign_rho * rho
                        value .+= coefficient_5 .* _hisq_smearing_test_path(
                            thin_links, site,
                            (signed_nu, signed_rho, mu,
                             -signed_rho, -signed_nu))
                    end

                    for sigma in 1:4
                        (sigma == mu || sigma == nu || sigma == rho) &&
                            continue
                        for sign_nu in (-1, 1), sign_rho in (-1, 1),
                            sign_sigma in (-1, 1)
                            signed_nu = sign_nu * nu
                            signed_rho = sign_rho * rho
                            signed_sigma = sign_sigma * sigma
                            value .+= coefficient_7 .*
                                _hisq_smearing_test_path(
                                    thin_links, site,
                                    (signed_nu, signed_rho, signed_sigma, mu,
                                     -signed_sigma, -signed_rho, -signed_nu))
                        end
                    end
                end
            end
            @views fat_links[mu][:, :, site...] .= value
        end
    end
    return fat_links
end

function _hisq_smearing_test_transform_links(thin_links, transformations)
    lattice_size = size(thin_links[1])[3:end]
    transformed = [similar(link) for link in thin_links]
    for site_index in CartesianIndices(lattice_size)
        site = Tuple(site_index)
        for mu in 1:4
            site_plus = _staggered_test_shift(site, mu, 1, lattice_size)
            @views transformed[mu][:, :, site...] .=
                transformations[:, :, site...] *
                thin_links[mu][:, :, site...] *
                transformations[:, :, site_plus...]'
        end
    end
    return transformed
end

function _hisq_smearing_test_expected_transform(fat_links, transformations)
    lattice_size = size(fat_links[1])[3:end]
    expected = [similar(link) for link in fat_links]
    for site_index in CartesianIndices(lattice_size)
        site = Tuple(site_index)
        for mu in 1:4
            site_plus = _staggered_test_shift(site, mu, 1, lattice_size)
            @views expected[mu][:, :, site...] .=
                transformations[:, :, site...] *
                fat_links[mu][:, :, site...] *
                transformations[:, :, site_plus...]'
        end
    end
    return expected
end

function _hisq_smearing_test_gather(links)
    return [gather_matrix(link) for link in links]
end

function hisq_smearing_tests()
    nprocs = test_comm_size()
    rank = test_comm_rank()
    process_grid = (nprocs, 1, 1, 1)
    local_x = iseven(nprocs) ? 3 : 4
    lattice_size = (local_x * nprocs, 3, 3, 3)
    NC = 3

    thin_arrays = _staggered_test_links(lattice_size, NC)
    reference = _hisq_smearing_test_level1_reference(thin_arrays)
    U = [LatticeMatrix(link, 4, process_grid; nw=1)
         for link in thin_arrays]

    @testset "HISQ level-1 Fat7 dense SIMULATeQCD convention" begin
        V = hisq_fat7_level1(U)
        global_V = _hisq_smearing_test_gather(V)
        if rank == 0
            for mu in 1:4
                @test global_V[mu] ≈ reference[mu] atol=2e-11 rtol=2e-11
            end
        end
        @test all(halo_is_dirty, V)

        V_inplace = [similar(link) for link in U]
        returned = hisq_fat7_level1!(V_inplace, U)
        @test returned === V_inplace
        global_inplace = _hisq_smearing_test_gather(V_inplace)
        if rank == 0
            for mu in 1:4
                @test global_inplace[mu] ≈ reference[mu]
            end
        end
    end

    @testset "SIMULATeQCD level-1 Fat7 numerical fingerprints" begin
        # Generated on an H100 by the external driver in
        # test/reference/simulateqcd, compiled against an unmodified
        # SIMULATeQCD commit 767a1b1.  The explicit weighted sums use Julia's
        # column-major order, so they also check site, color, and direction
        # layout conventions rather than only global norms.
        if 4 % nprocs == 0
            simulate_size = (4, 4, 4, 4)
            simulate_arrays = _staggered_test_links(simulate_size, NC)
            simulate_U = [LatticeMatrix(
                link, 4, process_grid; nw=1) for link in simulate_arrays]
            simulate_V = hisq_fat7_level1(simulate_U)
            global_simulate_V = _hisq_smearing_test_gather(simulate_V)
            expected = (
                (114531.3933907014, -113676.94385310293,
                 157887254.55079165, -155410539.92300072,
                 14529410.344861686),
                (115419.01644607708, -105133.77020051634,
                 159713642.80883369, -143984662.85621107,
                 14168242.841688637),
                (113113.92391727255, -94687.902697536047,
                 157584705.84684169, -130843680.65564176,
                 13660180.153343556),
                (107767.61936873377, -82915.12943421246,
                 152643046.05234742, -120030977.8007717,
                 13044245.423952268),
            )
            if rank == 0
                for mu in 1:4
                    fingerprint = _staggered_test_fingerprint(
                        global_simulate_V[mu])
                    @test isapprox(
                        collect(fingerprint), collect(expected[mu]);
                        atol=5e-7, rtol=2e-14)
                end
            end
        end
    end

    @testset "HISQ level-1 Fat7 free field and path normalization" begin
        unit_array = _staggered_test_unit_link(lattice_size, NC)
        unit_U = [LatticeMatrix(unit_array, 4, process_grid; nw=1)
                  for _ in 1:4]
        unit_V = hisq_fat7_level1(unit_U)
        global_unit_V = _hisq_smearing_test_gather(unit_V)
        if rank == 0
            @test 1//8 + 6//16 + 24//64 + 48//384 == 1
            for mu in 1:4
                @test global_unit_V[mu] ≈ unit_array atol=3e-15 rtol=3e-15
            end
        end
    end

    @testset "HISQ level-1 Fat7 gauge covariance" begin
        transformations = _staggered_test_transformations(lattice_size, NC)
        transformed_arrays = _hisq_smearing_test_transform_links(
            thin_arrays, transformations)
        transformed_U = [LatticeMatrix(link, 4, process_grid; nw=1)
                         for link in transformed_arrays]
        transformed_V = hisq_fat7_level1(transformed_U)
        global_transformed_V = _hisq_smearing_test_gather(transformed_V)
        expected = _hisq_smearing_test_expected_transform(
            reference, transformations)
        if rank == 0
            for mu in 1:4
                @test isapprox(
                    global_transformed_V[mu], expected[mu];
                    atol=4e-11, rtol=4e-11)
            end
        end
    end

    @testset "HISQ level-1 Fat7 halo refresh" begin
        increment_arrays = [fill(
            ComplexF64(0.002mu - 0.001im * mu),
            NC, NC, lattice_size...) for mu in 1:4]
        increments = [LatticeMatrix(link, 4, process_grid; nw=1)
                      for link in increment_arrays]
        updated_arrays = [thin_arrays[mu] .+ increment_arrays[mu]
                          for mu in 1:4]
        for mu in 1:4
            add_matrix!(U[mu], increments[mu])
            @test halo_is_dirty(U[mu])
        end
        updated_reference = _hisq_smearing_test_level1_reference(
            updated_arrays)
        updated_V = hisq_fat7_level1(U)
        @test all(link -> !halo_is_dirty(link), U)
        global_updated_V = _hisq_smearing_test_gather(updated_V)
        if rank == 0
            for mu in 1:4
                @test isapprox(
                    global_updated_V[mu], updated_reference[mu];
                    atol=2e-11, rtol=2e-11)
            end
        end
    end

    @testset "HISQ level-1 Fat7 nw=0 fallback" begin
        small_size = (2 * nprocs, 2, 2, 2)
        small_arrays = _staggered_test_links(small_size, 1)
        small_reference =
            _hisq_smearing_test_level1_reference(small_arrays)
        U0 = [LatticeMatrix(link, 4, process_grid; nw=0)
              for link in small_arrays]
        V0 = hisq_fat7_level1(U0)
        global_V0 = _hisq_smearing_test_gather(V0)
        if rank == 0
            for mu in 1:4
                @test isapprox(
                    global_V0[mu], small_reference[mu];
                    atol=2e-12, rtol=2e-12)
            end
        end
    end

    @testset "HISQ level-1 Fat7 Float32 and validation" begin
        small_size = (2 * nprocs, 2, 2, 2)
        arrays32 = _staggered_test_links(
            small_size, 2; elementtype=ComplexF32)
        reference32 = _hisq_smearing_test_level1_reference(arrays32)
        U32 = [LatticeMatrix(link, 4, process_grid; nw=1)
               for link in arrays32]
        V32 = hisq_fat7_level1(U32)
        global_V32 = _hisq_smearing_test_gather(V32)
        if rank == 0
            for mu in 1:4
                @test isapprox(
                    global_V32[mu], reference32[mu];
                    atol=8f-5, rtol=8f-5)
            end
        end

        @test_throws ArgumentError hisq_fat7_level1!(U, U)
        @test_throws ArgumentError hisq_fat7_level1(U[1:3])
        bad_phase_U = [LatticeMatrix(
            thin_arrays[mu], 4, process_grid; nw=1,
            phases=(1, 1, 1, mu == 4 ? -1 : 1)) for mu in 1:4]
        @test_throws ArgumentError hisq_fat7_level1(bad_phase_U)

        mismatched_output = LatticeMatrix{4}[
            similar(link) for link in U]
        mismatched_output[4] = LatticeMatrix(
            zeros(ComplexF64, NC, NC, lattice_size...),
            4, process_grid; nw=0)
        @test_throws ArgumentError hisq_fat7_level1!(mismatched_output, U)
    end
end
