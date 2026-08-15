function _staggered_test_shift(site, direction, amount, lattice_size)
    return ntuple(d -> d == direction ?
        mod1(site[d] + amount, lattice_size[d]) : site[d], 4)
end

function _staggered_test_fermion_shift(
    site, direction, amount, lattice_size, phases)
    shifted = _staggered_test_shift(site, direction, amount, lattice_size)
    wraps = fld(site[direction] + amount - 1, lattice_size[direction])
    return shifted, phases[direction]^wraps
end

function _staggered_test_eta(site, direction)
    direction == 1 && return 1
    exponent = sum(site[nu] - 1 for nu in 1:(direction-1))
    return iseven(exponent) ? 1 : -1
end

function _staggered_test_links(lattice_size, NC=3; elementtype=ComplexF64)
    links = [zeros(elementtype, NC, NC, lattice_size...) for _ in 1:4]
    real_type = typeof(real(zero(elementtype)))
    for site in CartesianIndices(lattice_size)
        x = Tuple(site)
        coordinate = x[1] + 3x[2] + 5x[3] + 7x[4]
        for mu in 1:4, col in 1:NC, row in 1:NC
            real_part = real_type(0.013 *
                (2row - col + coordinate + 3mu))
            imaginary_part = real_type(0.017 *
                (row + 2col - coordinate + mu))
            links[mu][row, col, x...] =
                elementtype(real_part + im * imaginary_part)
        end
    end
    return links
end

function _staggered_test_fermion(lattice_size, NC=3;
    elementtype=ComplexF64, offset=0.0)
    psi = zeros(elementtype, NC, 1, lattice_size...)
    real_type = typeof(real(zero(elementtype)))
    for site in CartesianIndices(lattice_size)
        x = Tuple(site)
        coordinate = x[1] + 2x[2] + 4x[3] + 8x[4]
        for color in 1:NC
            real_part = real_type(offset + 0.019 * (color + coordinate))
            imaginary_part = real_type(0.023 * (2color - coordinate))
            psi[color, 1, x...] = elementtype(real_part + im * imaginary_part)
        end
    end
    return psi
end

function _staggered_test_reference(
    links, psi, mass, phases; adjoint_operator=false)
    lattice_size = size(psi)[3:end]
    result = similar(psi)
    hopping_sign = adjoint_operator ? -0.5 : 0.5

    for site in CartesianIndices(lattice_size)
        x = Tuple(site)
        value = mass .* copy(@view psi[:, 1, x...])
        for mu in 1:4
            xplus, phase_plus = _staggered_test_fermion_shift(
                x, mu, 1, lattice_size, phases)
            xminus, phase_minus = _staggered_test_fermion_shift(
                x, mu, -1, lattice_size, phases)
            eta = _staggered_test_eta(x, mu)
            value .+= hopping_sign * eta .* (
                links[mu][:, :, x...] *
                (phase_plus .* psi[:, 1, xplus...]) -
                links[mu][:, :, xminus...]' *
                (phase_minus .* psi[:, 1, xminus...]))
        end
        @views result[:, 1, x...] .= value
    end
    return result
end

function _staggered_test_transformations(lattice_size, NC=3)
    transformations = zeros(ComplexF64, NC, NC, lattice_size...)
    for site in CartesianIndices(lattice_size)
        x = Tuple(site)
        coordinate = x[1] + 3x[2] + 5x[3] + 7x[4]
        angles = if NC == 3
            (0.031 * coordinate, -0.047 * (coordinate + 1),
             0.047 * (coordinate + 1) - 0.031 * coordinate)
        else
            ntuple(color -> (-1)^color * 0.037 * coordinate, NC)
        end
        for color in 1:NC
            transformations[color, color, x...] = cis(angles[color])
        end
    end
    return transformations
end

function _staggered_test_gauge_transform(links, psi)
    lattice_size = size(psi)[3:end]
    NC = size(psi, 1)
    transformations = _staggered_test_transformations(lattice_size, NC)
    transformed_links = [similar(link) for link in links]
    transformed_psi = similar(psi)

    for site in CartesianIndices(lattice_size)
        x = Tuple(site)
        @views transformed_psi[:, 1, x...] .=
            transformations[:, :, x...] * psi[:, 1, x...]
        for mu in 1:4
            xplus = _staggered_test_shift(x, mu, 1, lattice_size)
            @views transformed_links[mu][:, :, x...] .=
                transformations[:, :, x...] * links[mu][:, :, x...] *
                transformations[:, :, xplus...]'
        end
    end
    return transformed_links, transformed_psi, transformations
end

function _staggered_test_epsilon(psi)
    lattice_size = size(psi)[3:end]
    result = similar(psi)
    for site in CartesianIndices(lattice_size)
        x = Tuple(site)
        epsilon = iseven(sum(x[d] - 1 for d in 1:4)) ? 1 : -1
        @views result[:, :, x...] .= epsilon .* psi[:, :, x...]
    end
    return result
end

function _staggered_test_unit_link(lattice_size, NC=1;
    elementtype=ComplexF64)
    link = zeros(elementtype, NC, NC, lattice_size...)
    for site in CartesianIndices(lattice_size), color in 1:NC
        link[color, color, Tuple(site)...] = one(elementtype)
    end
    return link
end

function _staggered_test_fingerprint(field)
    values = vec(field)
    return (
        sum(real, values),
        sum(imag, values),
        sum(i * real(values[i]) for i in eachindex(values)),
        sum(i * imag(values[i]) for i in eachindex(values)),
        sum(abs2, values),
    )
end

function staggered_dirac_tests()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    process_grid = (nprocs, 1, 1, 1)
    # With an even number of ranks the local x extent is deliberately odd.
    # Single-rank and odd-rank runs keep the global lattice bipartite so that
    # epsilon Hermiticity remains a valid check.
    local_x = iseven(nprocs) ? 3 : 4
    lattice_size = (local_x * nprocs, 2, 2, 2)
    phases = (1, 1, 1, -1)
    mass = 0.17
    NC = 3

    links = _staggered_test_links(lattice_size, NC)
    psi_array = _staggered_test_fermion(lattice_size, NC)
    chi_array = _staggered_test_fermion(lattice_size, NC; offset=0.29)
    reference = _staggered_test_reference(
        links, psi_array, mass, phases)
    reference_dag = _staggered_test_reference(
        links, psi_array, mass, phases; adjoint_operator=true)

    U1 = [LatticeMatrix(link, 4, process_grid; nw=1) for link in links]
    psi1 = LatticeMatrix(psi_array, 4, process_grid;
        nw=1, phases)
    chi1 = LatticeMatrix(chi_array, 4, process_grid;
        nw=1, phases)
    result1 = similar(psi1)
    operator1 = StaggeredDiracOperator4D(U1, mass)

    @testset "staggered dense Bridge++ convention" begin
        mul!(result1, operator1, psi1)
        global_result = gather_matrix(result1)
        if rank == 0
            @test global_result ≈ reference atol=3e-12 rtol=3e-12
        end

        mul!(result1, adjoint(operator1), psi1)
        global_adjoint = gather_matrix(result1)
        if rank == 0
            @test global_adjoint ≈ reference_dag atol=3e-12 rtol=3e-12
        end
    end

    @testset "Bridge++ 2.1.3 numerical fingerprints" begin
        # Generated by test/reference/bridgepp_staggered_reference.cpp using
        # an unmodified Bridge++ 2.1.3 Fopr_Staggered.  The small reference
        # lattice can also be decomposed over two or four ranks.
        if 4 % nprocs == 0
            bridge_size = (4, 2, 2, 2)
            bridge_links = _staggered_test_links(bridge_size, NC)
            bridge_psi_array = _staggered_test_fermion(bridge_size, NC)
            bridge_U = [LatticeMatrix(link, 4, process_grid; nw=1)
                        for link in bridge_links]
            bridge_psi = LatticeMatrix(
                bridge_psi_array, 4, process_grid; nw=1, phases)
            bridge_result = similar(bridge_psi)
            bridge_operator = StaggeredDiracOperator4D(bridge_U, mass)

            expected = (
                D=(-32.41850400000002, -50.125175999999996,
                   -1997.541064, -2931.7741839999999,
                   227.22356287198397),
                Ddag=(48.232584000000003, 35.486136000000002,
                      2845.4031440000003, 2125.3757840000003,
                      227.22356287198406),
            )
            for (mode, applied_operator) in
                ((:D, bridge_operator), (:Ddag, adjoint(bridge_operator)))
                mul!(bridge_result, applied_operator, bridge_psi)
                global_result = gather_matrix(bridge_result)
                if rank == 0
                    fingerprint = _staggered_test_fingerprint(global_result)
                    @test isapprox(
                        collect(fingerprint), collect(expected[mode]);
                        atol=8e-10, rtol=2e-13)
                end
            end
        end
    end

    @testset "staggered adjoint and epsilon Hermiticity" begin
        Dpsi = similar(psi1)
        Ddag_chi = similar(chi1)
        Ddag_psi = similar(psi1)
        mul!(Dpsi, operator1, psi1)
        mul!(Ddag_chi, adjoint(operator1), chi1)
        mul!(Ddag_psi, adjoint(operator1), psi1)
        Dpsi_global = gather_matrix(Dpsi)
        Ddag_chi_global = gather_matrix(Ddag_chi)
        Ddag_psi_global = gather_matrix(Ddag_psi)
        if rank == 0
            @test dot(vec(chi_array), vec(Dpsi_global)) ≈
                dot(vec(Ddag_chi_global), vec(psi_array)) atol=2e-11 rtol=5e-12
        end

        epsilon_psi_array = _staggered_test_epsilon(psi_array)
        epsilon_psi = LatticeMatrix(epsilon_psi_array, 4, process_grid;
            nw=1, phases)
        D_epsilon_psi = similar(epsilon_psi)
        mul!(D_epsilon_psi, operator1, epsilon_psi)
        epsilon_D_epsilon = _staggered_test_epsilon(
            gather_and_bcast_matrix(D_epsilon_psi))
        if rank == 0
            @test Ddag_psi_global ≈ epsilon_D_epsilon atol=4e-12 rtol=4e-12
        end
    end

    @testset "staggered complex boundary phases and validation" begin
        twisted_phases = (
            cis(0.13), cis(-0.21), cis(0.34), cis(pi - 0.17))
        twisted_psi = LatticeMatrix(psi_array, 4, process_grid;
            nw=1, phases=twisted_phases)
        twisted_result = similar(twisted_psi)
        twisted_reference = _staggered_test_reference(
            links, psi_array, mass, twisted_phases)
        twisted_reference_dag = _staggered_test_reference(
            links, psi_array, mass, twisted_phases;
            adjoint_operator=true)

        mul!(twisted_result, operator1, twisted_psi)
        twisted_global = gather_matrix(twisted_result)
        if rank == 0
            @test twisted_global ≈ twisted_reference atol=4e-12 rtol=4e-12
        end
        mul!(twisted_result, adjoint(operator1), twisted_psi)
        twisted_global_dag = gather_matrix(twisted_result)
        if rank == 0
            @test isapprox(
                twisted_global_dag, twisted_reference_dag;
                atol=4e-12, rtol=4e-12)
        end

        @test_throws ArgumentError mul!(psi1, operator1, psi1)
        bad_phase_psi = LatticeMatrix(psi_array, 4, process_grid;
            nw=1, phases=(1, 1, 1, 2))
        @test_throws ArgumentError mul!(
            similar(bad_phase_psi), operator1, bad_phase_psi)
        @test_throws ArgumentError StaggeredDiracOperator4D(U1, NaN)
    end

    @testset "staggered gauge covariance" begin
        transformed_links, transformed_psi_array, transformations =
            _staggered_test_gauge_transform(links, psi_array)
        transformed_U = [LatticeMatrix(link, 4, process_grid; nw=1)
                         for link in transformed_links]
        transformed_psi = LatticeMatrix(
            transformed_psi_array, 4, process_grid; nw=1, phases)
        transformed_result = similar(transformed_psi)
        transformed_operator = StaggeredDiracOperator4D(
            transformed_U, mass)
        mul!(transformed_result, transformed_operator, transformed_psi)
        transformed_global = gather_matrix(transformed_result)

        mul!(result1, operator1, psi1)
        original_global = gather_matrix(result1)
        if rank == 0
            expected = similar(original_global)
            for site in CartesianIndices(lattice_size)
                x = Tuple(site)
                @views expected[:, 1, x...] .=
                    transformations[:, :, x...] * original_global[:, 1, x...]
            end
            @test transformed_global ≈ expected atol=6e-12 rtol=6e-12
        end
    end

    @testset "staggered nw=0 and nw=1" begin
        U0 = [LatticeMatrix(link, 4, process_grid; nw=0) for link in links]
        psi0 = LatticeMatrix(psi_array, 4, process_grid;
            nw=0, phases)
        result0 = similar(psi0)
        operator0 = StaggeredDiracOperator4D(U0, mass)
        mul!(result0, operator0, psi0)
        mul!(result1, operator1, psi1)
        global0 = gather_matrix(result0)
        global1 = gather_matrix(result1)
        if rank == 0
            @test global0 ≈ global1 atol=4e-12 rtol=4e-12
        end

        mul!(result0, adjoint(operator0), psi0)
        mul!(result1, adjoint(operator1), psi1)
        global0_dag = gather_matrix(result0)
        global1_dag = gather_matrix(result1)
        if rank == 0
            @test global0_dag ≈ global1_dag atol=4e-12 rtol=4e-12
        end
    end

    @testset "global staggered eta on odd local extents" begin
        eta_lattice_size = (3 * nprocs, 2, 2, 2)
        scalar = ones(ComplexF64, 1, 1, eta_lattice_size...)
        A = LatticeMatrix(scalar, 4, process_grid; nw=1)
        B = LatticeMatrix(scalar, 4, process_grid; nw=1)
        C = similar(A)
        for mu in 1:4
            mul!(C, Staggered_Lattice(A, mu), B)
            eta_global = gather_matrix(C)
            if rank == 0
                expected = similar(scalar)
                for site in CartesianIndices(eta_lattice_size)
                    expected[1, 1, Tuple(site)...] =
                        _staggered_test_eta(Tuple(site), mu)
                end
                @test eta_global == expected
            end
        end
    end

    @testset "free staggered DdagD spectrum" begin
        even_size = (4 * nprocs, 4, 4, 4)
        unit_link = _staggered_test_unit_link(even_size)
        unit_U = [LatticeMatrix(unit_link, 4, process_grid; nw=1)
                  for _ in 1:4]
        momentum = (2pi / even_size[1], 0.0, 0.0, pi / even_size[4])
        plane_wave = zeros(ComplexF64, 1, 1, even_size...)
        for site in CartesianIndices(even_size)
            x = Tuple(site)
            phase = sum(momentum[mu] * (x[mu] - 1) for mu in 1:4)
            plane_wave[1, 1, x...] = cis(phase)
        end
        wave = LatticeMatrix(plane_wave, 4, process_grid;
            nw=1, phases)
        temp = similar(wave)
        output = similar(wave)
        free_operator = StaggeredDiracOperator4D(unit_U, mass)
        mul!(temp, free_operator, wave)
        mul!(output, adjoint(free_operator), temp)
        output_global = gather_matrix(output)
        if rank == 0
            eigenvalue = mass^2 + sum(sin(p)^2 for p in momentum)
            @test output_global ≈ eigenvalue .* plane_wave atol=8e-12 rtol=8e-12
        end
    end

    @testset "staggered Float32 generic color path" begin
        small_size = (2 * nprocs, 2, 2, 2)
        links32 = _staggered_test_links(
            small_size, 2; elementtype=ComplexF32)
        psi32_array = _staggered_test_fermion(
            small_size, 2; elementtype=ComplexF32)
        reference32 = _staggered_test_reference(
            links32, psi32_array, Float32(mass), phases)
        U32 = [LatticeMatrix(link, 4, process_grid; nw=1)
               for link in links32]
        psi32 = LatticeMatrix(psi32_array, 4, process_grid;
            nw=1, phases)
        result32 = similar(psi32)
        operator32 = StaggeredDiracOperator4D(U32, mass)
        @test operator32.mass isa Float32
        mul!(result32, operator32, psi32)
        global32 = gather_matrix(result32)
        if rank == 0
            @test global32 ≈ reference32 atol=2f-5 rtol=2f-5
        end
    end
end
