function _clover_test_shift(x, direction, amount, lattice_size)
    return ntuple(d -> d == direction ? mod1(x[d] + amount, lattice_size[d]) : x[d], 4)
end

function _clover_test_fermion_shift(x, direction, amount, lattice_size, phases)
    shifted = _clover_test_shift(x, direction, amount, lattice_size)
    raw = x[direction] + amount
    wraps = fld(raw - 1, lattice_size[direction])
    return shifted, phases[direction]^wraps
end

function _clover_test_su2(angle, phase)
    c = cos(angle)
    s = sin(angle)
    z = cis(phase)
    return ComplexF64[c s*z; -s*conj(z) c]
end

function _clover_test_links(lattice_size)
    links = [zeros(ComplexF64, 2, 2, lattice_size...) for _ in 1:4]
    for site in CartesianIndices(lattice_size)
        x = Tuple(site)
        coordinate = x[1] + 2x[2] + 3x[3] + 5x[4]
        for mu in 1:4
            angle = 0.071 * (coordinate + 2mu)
            phase = 0.113 * (2coordinate - mu)
            @views links[mu][:, :, x...] .= _clover_test_su2(angle, phase)
        end
    end
    return links
end

function _clover_test_spinor(lattice_size; offset=0.0)
    psi = zeros(ComplexF64, 2, 4, lattice_size...)
    for site in CartesianIndices(lattice_size)
        x = Tuple(site)
        coordinate = x[1] + 2x[2] + 3x[3] + 5x[4]
        for spin in 1:4, color in 1:2
            psi[color, spin, x...] =
                offset + 0.031 * (color + 2spin + coordinate) +
                im * 0.047 * (2color - spin + coordinate)
        end
    end
    return psi
end

function _clover_test_field_strength(links)
    lattice_size = size(links[1])[3:end]
    NC = size(links[1], 1)
    fields = [zeros(eltype(links[1]), NC, NC, lattice_size...) for _ in 1:6]
    plane_pairs = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))

    for (plane, (mu, nu)) in enumerate(plane_pairs)
        for site in CartesianIndices(lattice_size)
            x = Tuple(site)
            xpmu = _clover_test_shift(x, mu, 1, lattice_size)
            xpnu = _clover_test_shift(x, nu, 1, lattice_size)
            xmmu = _clover_test_shift(x, mu, -1, lattice_size)
            xmnu = _clover_test_shift(x, nu, -1, lattice_size)
            xpnu_mmu = _clover_test_shift(xpnu, mu, -1, lattice_size)
            xmmu_mnu = _clover_test_shift(xmmu, nu, -1, lattice_size)
            xpmu_mnu = _clover_test_shift(xmnu, mu, 1, lattice_size)

            Q =
                links[mu][:, :, x...] * links[nu][:, :, xpmu...] *
                links[mu][:, :, xpnu...]' * links[nu][:, :, x...]' +
                links[nu][:, :, x...] * links[mu][:, :, xpnu_mmu...]' *
                links[nu][:, :, xmmu...]' * links[mu][:, :, xmmu...] +
                links[mu][:, :, xmmu...]' * links[nu][:, :, xmmu_mnu...]' *
                links[mu][:, :, xmmu_mnu...] * links[nu][:, :, xmnu...] +
                links[nu][:, :, xmnu...]' * links[mu][:, :, xmnu...] *
                links[nu][:, :, xpmu_mnu...] * links[mu][:, :, x...]'

            @views fields[plane][:, :, x...] .= 0.125 .* (Q .- Q')
        end
    end
    return fields
end

function _clover_test_reference(links, psi, kappa, cSW, phases)
    lattice_size = size(psi)[3:end]
    fields = _clover_test_field_strength(links)
    result = similar(psi)
    gamma_products = ntuple(6) do plane
        mu, nu = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))[plane]
        Matrix(γs[mu] * γs[nu])
    end
    identity_spin = Matrix{ComplexF64}(I, 4, 4)

    for site in CartesianIndices(lattice_size)
        x = Tuple(site)
        value = copy(psi[:, :, x...])
        for mu in 1:4
            xplus, phase_plus = _clover_test_fermion_shift(
                x, mu, 1, lattice_size, phases)
            xminus, phase_minus = _clover_test_fermion_shift(
                x, mu, -1, lattice_size, phases)
            value .-= kappa .* links[mu][:, :, x...] *
                (phase_plus .* psi[:, :, xplus...]) *
                transpose(identity_spin - γs[mu])
            value .-= kappa .* links[mu][:, :, xminus...]' *
                (phase_minus .* psi[:, :, xminus...]) *
                transpose(identity_spin + γs[mu])
        end
        for plane in 1:6
            value .-= kappa * cSW .* fields[plane][:, :, x...] *
                psi[:, :, x...] * transpose(gamma_products[plane])
        end
        @views result[:, :, x...] .= value
    end
    return result, fields
end

function _clover_test_apply_gamma5(psi)
    gamma5 = Matrix(γ1 * γ2 * γ3 * γ4)
    result = similar(psi)
    lattice_size = size(psi)[3:end]
    for site in CartesianIndices(lattice_size)
        x = Tuple(site)
        @views result[:, :, x...] .= psi[:, :, x...] * transpose(gamma5)
    end
    return result
end

function _clover_test_gauge_transform(links, psi)
    lattice_size = size(psi)[3:end]
    transformations = zeros(ComplexF64, 2, 2, lattice_size...)
    transformed_links = [similar(link) for link in links]
    transformed_psi = similar(psi)

    for site in CartesianIndices(lattice_size)
        x = Tuple(site)
        coordinate = x[1] + 3x[2] + 4x[3] + 7x[4]
        @views transformations[:, :, x...] .=
            _clover_test_su2(0.053 * coordinate, 0.097 * (coordinate + 1))
    end

    for site in CartesianIndices(lattice_size)
        x = Tuple(site)
        @views transformed_psi[:, :, x...] .=
            transformations[:, :, x...] * psi[:, :, x...]
        for mu in 1:4
            xplus = _clover_test_shift(x, mu, 1, lattice_size)
            @views transformed_links[mu][:, :, x...] .=
                transformations[:, :, x...] * links[mu][:, :, x...] *
                transformations[:, :, xplus...]'
        end
    end
    return transformed_links, transformed_psi, transformations
end

function wilson_clover_tests()
    nprocs = test_comm_size()
    rank = test_comm_rank()
    lattice_size = (2 * nprocs, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)
    phases = (1, 1, 1, -1)
    kappa = 0.12
    cSW = 1.17

    links = _clover_test_links(lattice_size)
    psi_array = _clover_test_spinor(lattice_size)
    chi_array = _clover_test_spinor(lattice_size; offset=0.19)
    reference, reference_fields =
        _clover_test_reference(links, psi_array, kappa, cSW, phases)

    U1 = [LatticeMatrix(link, 4, process_grid; nw=1) for link in links]
    psi1 = LatticeMatrix(psi_array, 4, process_grid; nw=1, phases)
    chi1 = LatticeMatrix(chi_array, 4, process_grid; nw=1, phases)
    result1 = similar(psi1)
    operator1 = WilsonDiracCloverOperator4D(U1, kappa, cSW)

    @testset "Wilson--clover dense reference" begin
        for plane in 1:6
            field = gather_matrix(operator1.clover[plane])
            if rank == 0
                @test field ≈ reference_fields[plane] atol=2e-12 rtol=2e-12
                for site in CartesianIndices(lattice_size)
                    x = Tuple(site)
                    @test field[:, :, x...]' ≈ -field[:, :, x...] atol=2e-12
                end
            end
        end

        mul!(result1, operator1, psi1)
        result = gather_matrix(result1)
        if rank == 0
            @test result ≈ reference atol=3e-12 rtol=3e-12
        end
    end

    @testset "analytic Wilson--clover link pullback" begin
        direction_arrays = _clover_test_links(lattice_size)
        directions = [
            LatticeMatrix(
                direction_arrays[mu], 4, process_grid; nw=1,
            ) for mu in 1:4
        ]
        dlinks = [similar(link) for link in U1]
        clear_matrix!.(dlinks)
        wilson_clover_link_pullback!(
            dlinks, operator1, U1, chi1, psi1)

        epsilon = 1e-6
        links_plus = deepcopy(links)
        links_minus = deepcopy(links)
        for mu in 1:4
            links_plus[mu] .+= epsilon .* direction_arrays[mu]
            links_minus[mu] .-= epsilon .* direction_arrays[mu]
        end
        Uplus = [LatticeMatrix(link, 4, process_grid; nw=1) for link in links_plus]
        Uminus = [LatticeMatrix(link, 4, process_grid; nw=1) for link in links_minus]
        result_plus = similar(psi1)
        result_minus = similar(psi1)
        mul!(result_plus, WilsonDiracCloverOperator4D(Uplus, kappa, cSW), psi1)
        mul!(result_minus, WilsonDiracCloverOperator4D(Uminus, kappa, cSW), psi1)
        finite_difference = (
            real(dot(chi1, result_plus)) - real(dot(chi1, result_minus))
        ) / (2epsilon)
        analytic_directional = real(sum(
            dot(dlinks[mu], directions[mu]) for mu in 1:4))
        @test analytic_directional ≈ finite_difference atol=3e-6 rtol=3e-7
    end

    @testset "Wilson and unit-gauge limits" begin
        clover_zero = WilsonDiracCloverOperator4D(U1, kappa, 0.0)
        wilson = WilsonDiracOperator4D(U1, kappa)
        wilson_result = similar(psi1)
        clover_result = similar(psi1)
        mul!(wilson_result, wilson, psi1)
        mul!(clover_result, clover_zero, psi1)
        wilson_global = gather_matrix(wilson_result)
        clover_global = gather_matrix(clover_result)
        if rank == 0
            @test clover_global ≈ wilson_global atol=2e-12 rtol=2e-12
        end

        unit_link = zeros(ComplexF64, 2, 2, lattice_size...)
        for site in CartesianIndices(lattice_size), color in 1:2
            unit_link[color, color, Tuple(site)...] = 1
        end
        unit_U = [LatticeMatrix(unit_link, 4, process_grid; nw=1) for _ in 1:4]
        unit_operator = WilsonDiracCloverOperator4D(unit_U, kappa, cSW)
        for field in unit_operator.clover.components
            field_global = gather_matrix(field)
            if rank == 0
                @test iszero(maximum(abs, field_global))
            end
        end
    end

    @testset "adjoint and gamma5 Hermiticity" begin
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
                dot(vec(Ddag_chi_global), vec(psi_array)) atol=5e-11 rtol=5e-12
        end

        gamma5_psi_array = _clover_test_apply_gamma5(psi_array)
        gamma5_psi = LatticeMatrix(gamma5_psi_array, 4, process_grid; nw=1, phases)
        D_gamma5_psi = similar(gamma5_psi)
        mul!(D_gamma5_psi, operator1, gamma5_psi)
        D_gamma5_global = gather_matrix(D_gamma5_psi)
        if rank == 0
            expected_gamma5 = _clover_test_apply_gamma5(D_gamma5_global)
            @test Ddag_psi_global ≈ expected_gamma5 atol=5e-12 rtol=5e-12
        end
    end

    @testset "gauge covariance" begin
        transformed_links, transformed_psi_array, transformations =
            _clover_test_gauge_transform(links, psi_array)
        transformed_U = [LatticeMatrix(link, 4, process_grid; nw=1)
                         for link in transformed_links]
        transformed_psi = LatticeMatrix(
            transformed_psi_array, 4, process_grid; nw=1, phases)
        transformed_result = similar(transformed_psi)
        transformed_operator = WilsonDiracCloverOperator4D(
            transformed_U, kappa, cSW)
        mul!(transformed_result, transformed_operator, transformed_psi)
        transformed_global = gather_matrix(transformed_result)

        mul!(result1, operator1, psi1)
        original_global = gather_matrix(result1)
        if rank == 0
            expected = similar(original_global)
            for site in CartesianIndices(lattice_size)
                x = Tuple(site)
                @views expected[:, :, x...] .=
                    transformations[:, :, x...] * original_global[:, :, x...]
            end
            @test transformed_global ≈ expected atol=8e-12 rtol=8e-12
        end
    end

    @testset "nw=0 and nw=1" begin
        U0 = [LatticeMatrix(link, 4, process_grid; nw=0) for link in links]
        psi0 = LatticeMatrix(psi_array, 4, process_grid; nw=0, phases)
        result0 = similar(psi0)
        operator0 = WilsonDiracCloverOperator4D(U0, kappa, cSW)
        mul!(result0, operator0, psi0)
        mul!(result1, operator1, psi1)
        global0 = gather_matrix(result0)
        global1 = gather_matrix(result1)
        if rank == 0
            @test global0 ≈ global1 atol=5e-12 rtol=5e-12
        end
    end

    @testset "halo refresh after spinor mutation" begin
        modified_psi_array = _clover_test_spinor(lattice_size; offset=0.37)
        modified_psi = LatticeMatrix(
            modified_psi_array, 4, process_grid; nw=1, phases)
        substitute!(psi1, modified_psi)
        mul!(result1, operator1, psi1)
        refreshed_result = gather_matrix(result1)
        refreshed_reference, _ = _clover_test_reference(
            links, modified_psi_array, kappa, cSW, phases)
        if rank == 0
            @test refreshed_result ≈ refreshed_reference atol=3e-12 rtol=3e-12
        end
    end

    @testset "explicit clover cache refresh" begin
        modified_links = deepcopy(links)
        modified_links[1][1, 1, 1, 1, 1, 1] += 0.07 + 0.02im
        modified_U1 = LatticeMatrix(
            modified_links[1], 4, process_grid; nw=1)
        substitute!(U1[1], modified_U1)
        update_clover!(operator1)
        modified_reference_fields = _clover_test_field_strength(modified_links)

        for plane in 1:6
            refreshed = gather_matrix(operator1.clover[plane])
            if rank == 0
                @test refreshed ≈ modified_reference_fields[plane] atol=2e-12 rtol=2e-12
            end
        end
        if rank == 0
            @test maximum(abs,
                modified_reference_fields[1] - reference_fields[1]) > 1e-6
        end
    end
end
