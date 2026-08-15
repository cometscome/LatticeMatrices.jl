function _hisq_test_links(lattice_size, NC=3; elementtype=ComplexF64)
    fat_links = [zeros(elementtype, NC, NC, lattice_size...) for _ in 1:4]
    long_links = [zeros(elementtype, NC, NC, lattice_size...) for _ in 1:4]
    real_type = typeof(real(zero(elementtype)))

    for site in CartesianIndices(lattice_size)
        x = Tuple(site)
        coordinate = x[1] + 3x[2] + 5x[3] + 7x[4]
        for mu in 1:4, col in 1:NC, row in 1:NC
            fat_real = real_type(0.011 *
                (2row - col + coordinate + 3mu))
            fat_imaginary = real_type(0.014 *
                (row + 2col - coordinate + mu))
            long_real = real_type(0.007 *
                (row - 3col + 2coordinate + mu))
            long_imaginary = real_type(0.009 *
                (2row + col - coordinate - 2mu))
            fat_links[mu][row, col, x...] =
                elementtype(fat_real + im * fat_imaginary)
            long_links[mu][row, col, x...] =
                elementtype(long_real + im * long_imaginary)
        end
    end
    return fat_links, long_links
end

# Independent dense oracle for the convention used by SIMULATeQCD's
# HisqDSlash: +1/2 for X and -(1+epsilon)/48 for the forward-anchored
# three-hop transporter L. Links in this test do not contain eta or boundary
# phases.
function _hisq_test_reference(
    fat_links, long_links, psi, mass, naik_epsilon, phases;
    adjoint_operator=false,
)
    lattice_size = size(psi)[3:end]
    result = similar(psi)
    direction_sign = adjoint_operator ? -one(mass) : one(mass)
    fat_coefficient = direction_sign / 2
    long_coefficient =
        -direction_sign * (one(naik_epsilon) + naik_epsilon) / 48

    for site in CartesianIndices(lattice_size)
        x = Tuple(site)
        value = mass .* copy(@view psi[:, 1, x...])
        for mu in 1:4
            eta = _staggered_test_eta(x, mu)
            for (links, distance, coefficient) in (
                (fat_links, 1, fat_coefficient),
                (long_links, 3, long_coefficient),
            )
                xplus, phase_plus = _staggered_test_fermion_shift(
                    x, mu, distance, lattice_size, phases)
                xminus, phase_minus = _staggered_test_fermion_shift(
                    x, mu, -distance, lattice_size, phases)
                value .+= coefficient * eta .* (
                    links[mu][:, :, x...] *
                    (phase_plus .* psi[:, 1, xplus...]) -
                    links[mu][:, :, xminus...]' *
                    (phase_minus .* psi[:, 1, xminus...]))
            end
        end
        @views result[:, 1, x...] .= value
    end
    return result
end

function _hisq_test_gauge_transform(fat_links, long_links, psi)
    lattice_size = size(psi)[3:end]
    NC = size(psi, 1)
    transformations = _staggered_test_transformations(lattice_size, NC)
    transformed_fat = [similar(link) for link in fat_links]
    transformed_long = [similar(link) for link in long_links]
    transformed_psi = similar(psi)

    for site in CartesianIndices(lattice_size)
        x = Tuple(site)
        @views transformed_psi[:, 1, x...] .=
            transformations[:, :, x...] * psi[:, 1, x...]
        for mu in 1:4
            xplus1 = _staggered_test_shift(x, mu, 1, lattice_size)
            xplus3 = _staggered_test_shift(x, mu, 3, lattice_size)
            @views transformed_fat[mu][:, :, x...] .=
                transformations[:, :, x...] * fat_links[mu][:, :, x...] *
                transformations[:, :, xplus1...]'
            @views transformed_long[mu][:, :, x...] .=
                transformations[:, :, x...] * long_links[mu][:, :, x...] *
                transformations[:, :, xplus3...]'
        end
    end
    return transformed_fat, transformed_long, transformed_psi,
        transformations
end

function _hisq_test_scaled_unit_link(
    lattice_size, scale, NC=1; elementtype=ComplexF64,
)
    link = _staggered_test_unit_link(
        lattice_size, NC; elementtype)
    link .*= elementtype(scale)
    return link
end

function hisq_dirac_tests()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    process_grid = (nprocs, 1, 1, 1)
    # The fused three-hop path needs three local sites in every direction.
    # For an even rank count, an odd local x extent also exercises global eta.
    local_x = iseven(nprocs) ? 3 : 4
    lattice_size = (local_x * nprocs, 4, 4, 4)
    phases = (1, 1, 1, -1)
    twisted_phases = (
        cis(0.13), cis(-0.21), cis(0.34), cis(pi - 0.17))
    mass = 0.17
    naik_epsilon = -0.083
    NC = 3

    fat_links, long_links = _hisq_test_links(lattice_size, NC)
    psi_array = _staggered_test_fermion(lattice_size, NC)
    chi_array = _staggered_test_fermion(
        lattice_size, NC; offset=0.29)

    X3 = [LatticeMatrix(link, 4, process_grid; nw=3)
          for link in fat_links]
    L3 = [LatticeMatrix(link, 4, process_grid; nw=3)
          for link in long_links]
    links3 = HISQLinks4D(X3, L3)
    psi3 = LatticeMatrix(psi_array, 4, process_grid; nw=3, phases)
    chi3 = LatticeMatrix(chi_array, 4, process_grid; nw=3, phases)
    result3 = similar(psi3)
    operator3 = HISQDiracOperator4D(
        links3, mass; naik_epsilon)

    @testset "HISQ dense SIMULATeQCD stencil convention" begin
        twisted_psi = LatticeMatrix(
            psi_array, 4, process_grid; nw=3, phases=twisted_phases)
        twisted_result = similar(twisted_psi)
        reference = _hisq_test_reference(
            fat_links, long_links, psi_array, mass, naik_epsilon,
            twisted_phases)
        reference_dag = _hisq_test_reference(
            fat_links, long_links, psi_array, mass, naik_epsilon,
            twisted_phases; adjoint_operator=true)

        mul!(twisted_result, operator3, twisted_psi)
        global_result = gather_matrix(twisted_result)
        if rank == 0
            @test global_result ≈ reference atol=8e-12 rtol=8e-12
        end

        mul!(twisted_result, adjoint(operator3), twisted_psi)
        global_adjoint = gather_matrix(twisted_result)
        if rank == 0
            @test global_adjoint ≈ reference_dag atol=8e-12 rtol=8e-12
        end
    end

    @testset "HISQ adjoint and epsilon Hermiticity" begin
        Dpsi = similar(psi3)
        Ddag_chi = similar(chi3)
        Ddag_psi = similar(psi3)
        mul!(Dpsi, operator3, psi3)
        mul!(Ddag_chi, adjoint(operator3), chi3)
        mul!(Ddag_psi, adjoint(operator3), psi3)
        Dpsi_global = gather_matrix(Dpsi)
        Ddag_chi_global = gather_matrix(Ddag_chi)
        Ddag_psi_global = gather_matrix(Ddag_psi)
        if rank == 0
            @test isapprox(
                dot(vec(chi_array), vec(Dpsi_global)),
                dot(vec(Ddag_chi_global), vec(psi_array));
                atol=3e-11, rtol=8e-12)
        end

        epsilon_psi_array = _staggered_test_epsilon(psi_array)
        epsilon_psi = LatticeMatrix(
            epsilon_psi_array, 4, process_grid; nw=3, phases)
        D_epsilon_psi = similar(epsilon_psi)
        mul!(D_epsilon_psi, operator3, epsilon_psi)
        epsilon_D_epsilon = _staggered_test_epsilon(
            gather_and_bcast_matrix(D_epsilon_psi))
        if rank == 0
            @test isapprox(
                Ddag_psi_global, epsilon_D_epsilon;
                atol=9e-12, rtol=9e-12)
        end
    end

    @testset "HISQ gauge covariance" begin
        transformed_fat, transformed_long, transformed_psi_array,
            transformations = _hisq_test_gauge_transform(
                fat_links, long_links, psi_array)
        transformed_X = [LatticeMatrix(link, 4, process_grid; nw=3)
                         for link in transformed_fat]
        transformed_L = [LatticeMatrix(link, 4, process_grid; nw=3)
                         for link in transformed_long]
        transformed_psi = LatticeMatrix(
            transformed_psi_array, 4, process_grid; nw=3, phases)
        transformed_result = similar(transformed_psi)
        transformed_operator = HISQDiracOperator4D(
            transformed_X, transformed_L, mass; naik_epsilon)

        mul!(transformed_result, transformed_operator, transformed_psi)
        transformed_global = gather_matrix(transformed_result)
        mul!(result3, operator3, psi3)
        original_global = gather_matrix(result3)
        if rank == 0
            expected = similar(original_global)
            for site in CartesianIndices(lattice_size)
                x = Tuple(site)
                @views expected[:, 1, x...] .=
                    transformations[:, :, x...] *
                    original_global[:, 1, x...]
            end
            @test transformed_global ≈ expected atol=2e-11 rtol=2e-11
        end
    end

    @testset "HISQ nw=0 and nw=3" begin
        X0 = [LatticeMatrix(link, 4, process_grid; nw=0)
              for link in fat_links]
        L0 = [LatticeMatrix(link, 4, process_grid; nw=0)
              for link in long_links]
        psi0 = LatticeMatrix(psi_array, 4, process_grid; nw=0, phases)
        result0 = similar(psi0)
        operator0 = HISQDiracOperator4D(
            X0, L0, mass; naik_epsilon)

        mul!(result0, operator0, psi0)
        mul!(result3, operator3, psi3)
        global0 = gather_matrix(result0)
        global3 = gather_matrix(result3)
        if rank == 0
            @test global0 ≈ global3 atol=9e-12 rtol=9e-12
        end

        mul!(result0, adjoint(operator0), psi0)
        mul!(result3, adjoint(operator3), psi3)
        global0_dag = gather_matrix(result0)
        global3_dag = gather_matrix(result3)
        if rank == 0
            @test global0_dag ≈ global3_dag atol=9e-12 rtol=9e-12
        end
    end

    @testset "free HISQ DdagD spectrum" begin
        even_size = (4 * nprocs, 4, 4, 4)
        # Level-2 HISQ smearing gives (9+epsilon)/8 on a free field.
        fat_scale = (9 + naik_epsilon) / 8
        unit_fat = _hisq_test_scaled_unit_link(even_size, fat_scale)
        unit_long = _hisq_test_scaled_unit_link(even_size, 1)
        free_X = [LatticeMatrix(unit_fat, 4, process_grid; nw=3)
                  for _ in 1:4]
        free_L = [LatticeMatrix(unit_long, 4, process_grid; nw=3)
                  for _ in 1:4]
        momentum = (2pi / even_size[1], 0.0, 0.0,
                    pi / even_size[4])
        plane_wave = zeros(ComplexF64, 1, 1, even_size...)
        for site in CartesianIndices(even_size)
            x = Tuple(site)
            phase = sum(
                momentum[mu] * (x[mu] - 1) for mu in 1:4)
            plane_wave[1, 1, x...] = cis(phase)
        end
        wave = LatticeMatrix(
            plane_wave, 4, process_grid; nw=3, phases)
        temp = similar(wave)
        output = similar(wave)
        free_operator = HISQDiracOperator4D(
            free_X, free_L, mass; naik_epsilon)
        mul!(temp, free_operator, wave)
        mul!(output, adjoint(free_operator), temp)
        output_global = gather_matrix(output)
        if rank == 0
            momentum_function(p) =
                fat_scale * sin(p) -
                (1 + naik_epsilon) * sin(3p) / 24
            eigenvalue = mass^2 +
                sum(momentum_function(p)^2 for p in momentum)
            @test isapprox(
                output_global, eigenvalue .* plane_wave;
                atol=2e-11, rtol=2e-11)
        end
    end

    @testset "HISQ validation and generic color path" begin
        @test adjoint(adjoint(operator3)) === operator3
        @test operator3.mass isa Float64
        @test operator3.naik_epsilon isa Float64
        @test_throws ArgumentError mul!(psi3, operator3, psi3)
        @test_throws ArgumentError HISQDiracOperator4D(
            X3, L3, NaN; naik_epsilon)
        @test_throws ArgumentError HISQDiracOperator4D(
            X3, L3, mass; naik_epsilon=Inf)

        X1 = [LatticeMatrix(link, 4, process_grid; nw=1)
              for link in fat_links]
        L1 = [LatticeMatrix(link, 4, process_grid; nw=1)
              for link in long_links]
        @test_throws ArgumentError HISQLinks4D(X1, L1)
        @test_throws ArgumentError HISQLinks4D(X3[1:3], L3)

        bad_phase_psi = LatticeMatrix(
            psi_array, 4, process_grid; nw=3, phases=(1, 1, 1, 2))
        @test_throws ArgumentError mul!(
            similar(bad_phase_psi), operator3, bad_phase_psi)

        small_size = (3 * nprocs, 3, 3, 3)
        fat32, long32 = _hisq_test_links(
            small_size, 2; elementtype=ComplexF32)
        psi32_array = _staggered_test_fermion(
            small_size, 2; elementtype=ComplexF32)
        reference32 = _hisq_test_reference(
            fat32, long32, psi32_array, Float32(mass),
            Float32(naik_epsilon), phases)
        X32 = [LatticeMatrix(link, 4, process_grid; nw=3)
               for link in fat32]
        L32 = [LatticeMatrix(link, 4, process_grid; nw=3)
               for link in long32]
        psi32 = LatticeMatrix(
            psi32_array, 4, process_grid; nw=3, phases)
        result32 = similar(psi32)
        operator32 = HISQDiracOperator4D(
            X32, L32, mass; naik_epsilon)
        @test operator32.mass isa Float32
        @test operator32.naik_epsilon isa Float32
        mul!(result32, operator32, psi32)
        global32 = gather_matrix(result32)
        if rank == 0
            @test global32 ≈ reference32 atol=5f-5 rtol=5f-5
        end
    end
end
