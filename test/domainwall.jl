function _domainwall_test_shift4(x, direction, amount, lattice_size)
    return ntuple(d -> d == direction ?
        mod1(x[d] + amount, lattice_size[d]) : x[d], 4)
end

function _domainwall_test_core(field)
    ranges = ntuple(
        d -> (field.nw + 1):(field.nw + field.PN[d]), length(field.PN))
    return @view field.A[:, :, ranges...]
end

function _domainwall_test_fermion_shift4(
    x, direction, amount, lattice_size, phases,
)
    shifted = _domainwall_test_shift4(x, direction, amount, lattice_size)
    wraps = fld(x[direction] + amount - 1, lattice_size[direction])
    return shifted, phases[direction]^wraps
end

function _domainwall_test_values(::Val{N1}, ::Val{N2}, lattice_size, offset) where {N1,N2}
    result = zeros(ComplexF64, N1, N2, lattice_size...)
    for site in CartesianIndices(lattice_size)
        x = Tuple(site)
        coordinate = sum((2d - 1) * x[d] for d in eachindex(x))
        for second in 1:N2, first in 1:N1
            result[first, second, x...] =
                0.013 * (first + 2second + coordinate + offset) +
                im * 0.017 * (2first - second - coordinate + offset)
        end
    end
    return result
end

function _domainwall_test_F(psi, mass; adjoint_operator=false)
    L5 = size(psi, 7)
    result = similar(psi)
    for s in 1:L5
        if adjoint_operator
            source_low = mod1(s + 1, L5)
            source_high = mod1(s - 1, L5)
            low_coefficient = s == L5 ? -mass : one(mass)
            high_coefficient = s == 1 ? -mass : one(mass)
        else
            source_low = mod1(s - 1, L5)
            source_high = mod1(s + 1, L5)
            low_coefficient = s == 1 ? -mass : one(mass)
            high_coefficient = s == L5 ? -mass : one(mass)
        end
        @views result[:, 1:2, :, :, :, :, s] .=
            low_coefficient .* psi[:, 1:2, :, :, :, :, source_low]
        @views result[:, 3:4, :, :, :, :, s] .=
            high_coefficient .* psi[:, 3:4, :, :, :, :, source_high]
    end
    return result
end

function _domainwall_test_DW(
    links, psi, M, phases; adjoint_operator=false,
)
    lattice_size = size(psi)[3:6]
    L5 = size(psi, 7)
    result = similar(psi)
    identity_spin = Matrix{ComplexF64}(I, 4, 4)
    for s in 1:L5, site in CartesianIndices(lattice_size)
        x = Tuple(site)
        value = (4 + M) .* copy(@view psi[:, :, x..., s])
        for mu in 1:4
            xplus, phase_plus = _domainwall_test_fermion_shift4(
                x, mu, 1, lattice_size, phases)
            xminus, phase_minus = _domainwall_test_fermion_shift4(
                x, mu, -1, lattice_size, phases)
            gamma_forward = adjoint_operator ?
                identity_spin + γs[mu] : identity_spin - γs[mu]
            gamma_backward = adjoint_operator ?
                identity_spin - γs[mu] : identity_spin + γs[mu]
            value .-= 0.5 .* links[mu][:, :, x...] *
                (phase_plus .* psi[:, :, xplus..., s]) *
                transpose(gamma_forward)
            value .-= 0.5 .* links[mu][:, :, xminus...]' *
                (phase_minus .* psi[:, :, xminus..., s]) *
                transpose(gamma_backward)
        end
        @views result[:, :, x..., s] .= value
    end
    return result
end

function _domainwall_test_reference(
    links, psi, mass, M, b, c, phases; adjoint_operator=false,
)
    a = (b + c) / 2
    e = (b - c) / 2
    if adjoint_operator
        Fdag_psi = _domainwall_test_F(psi, mass; adjoint_operator=true)
        DWdag_psi = _domainwall_test_DW(
            links, psi, M, phases; adjoint_operator=true)
        Fdag_DWdag_psi = _domainwall_test_F(
            DWdag_psi, mass; adjoint_operator=true)
        return psi .- Fdag_psi .+ a .* DWdag_psi .+ e .* Fdag_DWdag_psi
    end

    Fpsi = _domainwall_test_F(psi, mass)
    effective = a .* psi .+ e .* Fpsi
    return psi .- Fpsi .+ _domainwall_test_DW(links, effective, M, phases)
end

function _domainwall_test_generalized_reference(
    links, psi, mass, M, a, b, c, phases; adjoint_operator=false,
)
    L5 = size(psi, 7)
    reshape5(coefficients) = reshape(coefficients, 1, 1, 1, 1, 1, 1, L5)
    A, B, C = reshape5(a), reshape5(b), reshape5(c)
    if adjoint_operator
        q = A .* psi
        Fdag_q = _domainwall_test_F(q, mass; adjoint_operator=true)
        DWdag_q = _domainwall_test_DW(
            links, q, M, phases; adjoint_operator=true)
        Fdag_C_DWdag_q = _domainwall_test_F(
            C .* DWdag_q, mass; adjoint_operator=true)
        return q .- Fdag_q .+ B .* DWdag_q .+ Fdag_C_DWdag_q
    end

    Fpsi = _domainwall_test_F(psi, mass)
    effective = B .* psi .+ C .* Fpsi
    return A .* (psi .- Fpsi .+
        _domainwall_test_DW(links, effective, M, phases))
end

function _domainwall_nc3_fastpath_tests()
    nprocs = test_comm_size()
    rank = test_comm_rank()
    lattice_size = (2 * nprocs, 2, 2, 2)
    L5 = 3
    process_grid = (nprocs, 1, 1, 1)
    process_grid5 = (process_grid..., 1)
    phases = (cis(0.13), cis(-0.21), cis(0.34), cis(pi - 0.17), 1.0)
    NC = 3
    mass = 0.13
    M = -1.0
    link_arrays = [
        _domainwall_test_values(Val(NC), Val(NC), lattice_size, 7mu)
        for mu in 1:4
    ]
    fermion_size = (lattice_size..., L5)
    psi_array = _domainwall_test_values(Val(NC), Val(4), fermion_size, 11)
    left_array = _domainwall_test_values(Val(NC), Val(4), fermion_size, 23)
    links = [LatticeMatrix(link, 4, process_grid; nw=1) for link in link_arrays]
    psi = LatticeMatrix(psi_array, 5, process_grid5; nw=1, phases)
    left = LatticeMatrix(left_array, 5, process_grid5; nw=1, phases)
    result = similar(psi)
    adjoint_result = similar(left)
    adjoint_scratch_slots = length(left.temps)

    @testset "NC=3 half-spin domain-wall dense reference" begin
        for (b, c) in ((1.0, 1.0), (2.0, 0.0), (2.0, 1.0))
            operator = D5DW_MobiusDomainwallOperator5D(
                links, L5, mass, M, b, c)
            mul!(result, operator, psi)
            mul!(adjoint_result, adjoint(operator), left)
            global_result = gather_matrix(result)
            global_adjoint = gather_matrix(adjoint_result)
            if rank == 0
                reference = _domainwall_test_reference(
                    link_arrays, psi_array, mass, M, b, c, phases)
                reference_dag = _domainwall_test_reference(
                    link_arrays, left_array, mass, M, b, c, phases;
                    adjoint_operator=true)
                @test global_result ≈ reference atol=8e-12 rtol=8e-12
                @test global_adjoint ≈ reference_dag atol=8e-12 rtol=8e-12
                @test isapprox(
                    dot(vec(left_array), vec(global_result)),
                    dot(vec(global_adjoint), vec(psi_array));
                    atol=2e-10, rtol=2e-11)
            end
        end

        a = [0.83, 1.17, 1.31]
        b5 = [1.2, 0.91, 1.47]
        c5 = [-0.18, 0.37, 0.22]
        operator = D5DW_GeneralizedDomainwallOperator5D(
            links, L5, mass, M, a, b5, c5)
        mul!(result, operator, psi)
        mul!(adjoint_result, adjoint(operator), left)
        global_result = gather_matrix(result)
        global_adjoint = gather_matrix(adjoint_result)
        if rank == 0
            reference = _domainwall_test_generalized_reference(
                link_arrays, psi_array, mass, M, a, b5, c5, phases)
            reference_dag = _domainwall_test_generalized_reference(
                link_arrays, left_array, mass, M, a, b5, c5, phases;
                adjoint_operator=true)
            @test global_result ≈ reference atol=8e-12 rtol=8e-12
            @test global_adjoint ≈ reference_dag atol=8e-12 rtol=8e-12
            @test isapprox(
                dot(vec(left_array), vec(global_result)),
                dot(vec(global_adjoint), vec(psi_array));
                atol=2e-10, rtol=2e-11)
        end
        @test length(left.temps) == adjoint_scratch_slots
        @test !any(left.temps._flagusing)
    end
    return nothing
end

function domainwall_tests()
    nprocs = test_comm_size()
    rank = test_comm_rank()
    lattice_size = (2 * nprocs, 2, 2, 2)
    L5 = 3
    process_grid = (nprocs, 1, 1, 1)
    process_grid5 = (process_grid..., 1)
    phases = (cis(0.13), cis(-0.21), cis(0.34), cis(pi - 0.17), 1.0)
    NC = 2
    mass = 0.13
    M = -1.0

    link_arrays = [
        _domainwall_test_values(Val(NC), Val(NC), lattice_size, 7mu)
        for mu in 1:4
    ]
    fermion_size = (lattice_size..., L5)
    psi_array = _domainwall_test_values(
        Val(NC), Val(4), fermion_size, 11)
    left_array = _domainwall_test_values(
        Val(NC), Val(4), fermion_size, 23)
    links = [LatticeMatrix(link, 4, process_grid; nw=1) for link in link_arrays]
    psi = LatticeMatrix(psi_array, 5, process_grid5; nw=1, phases)
    left = LatticeMatrix(left_array, 5, process_grid5; nw=1, phases)
    result = similar(psi)
    adjoint_result = similar(left)

    @testset "Möbius/domain-wall dense reference" begin
        for (b, c) in ((1.0, 1.0), (2.0, 0.0), (2.0, 1.0))
            operator = D5DW_MobiusDomainwallOperator5D(
                links, L5, mass, M, b, c)
            reference = _domainwall_test_reference(
                link_arrays, psi_array, mass, M, b, c, phases)
            reference_dag = _domainwall_test_reference(
                link_arrays, left_array, mass, M, b, c, phases;
                adjoint_operator=true)

            mul!(result, operator, psi)
            mul!(adjoint_result, adjoint(operator), left)
            global_result = gather_matrix(result)
            global_adjoint = gather_matrix(adjoint_result)
            if rank == 0
                @test global_result ≈ reference atol=8e-12 rtol=8e-12
                @test global_adjoint ≈ reference_dag atol=8e-12 rtol=8e-12
                @test dot(vec(left_array), vec(global_result)) ≈
                    dot(vec(global_adjoint), vec(psi_array)) atol=2e-10 rtol=2e-11
            end
        end
    end


    @testset "generalized domain-wall dense reference" begin
        a = [0.83, 1.17, 1.31]
        b5 = [1.2, 0.91, 1.47]
        c5 = [-0.18, 0.37, 0.22]
        operator = D5DW_GeneralizedDomainwallOperator5D(
            links, L5, mass, M, a, b5, c5)
        reference = _domainwall_test_generalized_reference(
            link_arrays, psi_array, mass, M, a, b5, c5, phases)
        reference_dag = _domainwall_test_generalized_reference(
            link_arrays, left_array, mass, M, a, b5, c5, phases;
            adjoint_operator=true)

        mul!(result, operator, psi)
        mul!(adjoint_result, adjoint(operator), left)
        global_result = gather_matrix(result)
        global_adjoint = gather_matrix(adjoint_result)
        if rank == 0
            @test global_result ≈ reference atol=8e-12 rtol=8e-12
            @test global_adjoint ≈ reference_dag atol=8e-12 rtol=8e-12
            @test dot(vec(left_array), vec(global_result)) ≈
                dot(vec(global_adjoint), vec(psi_array)) atol=2e-10 rtol=2e-11
        end
        @test adjoint(adjoint(operator)) === operator
    end

    @testset "generalized domain-wall Möbius compatibility" begin
        for (legacy_b, legacy_c) in ((1.0, 1.0), (2.0, 0.0), (2.0, 1.0))
            b5 = fill((legacy_b + legacy_c) / 2, L5)
            c5 = fill((legacy_b - legacy_c) / 2, L5)
            generalized = D5DW_GeneralizedDomainwallOperator5D(
                links, L5, mass, M, ones(L5), b5, c5)
            mobius = D5DW_MobiusDomainwallOperator5D(
                links, L5, mass, M, legacy_b, legacy_c)
            generalized_result = similar(psi)
            mobius_result = similar(psi)
            mul!(generalized_result, generalized, psi)
            mul!(mobius_result, mobius, psi)
            @test _domainwall_test_core(generalized_result) ≈
                _domainwall_test_core(mobius_result) atol=8e-12 rtol=8e-12
            mul!(generalized_result, adjoint(generalized), psi)
            mul!(mobius_result, adjoint(mobius), psi)
            @test _domainwall_test_core(generalized_result) ≈
                _domainwall_test_core(mobius_result) atol=8e-12 rtol=8e-12
        end
    end

    @testset "domain-wall parameter precision and validation" begin
        operator32 = D5DW_MobiusDomainwallOperator5D(
            links, L5, 0.13f0, -1.0f0, 2.0f0, 1.0f0)
        @test operator32.mass isa Float32
        @test operator32.b isa Float32
        @test operator32.c isa Float32
        @test operator32.wilson_params.κ_wilson isa Float32
        @test_throws ArgumentError D5DW_MobiusDomainwallOperator5D(
            links[1:3], L5, mass, M, 1.0, 1.0)
        @test_throws ArgumentError D5DW_MobiusDomainwallOperator5D(
            links, 0, mass, M, 1.0, 1.0)
        @test_throws ArgumentError D5DW_MobiusDomainwallOperator5D(
            links, 3.0, mass, M, 1.0, 1.0)

        generalized32 = D5DW_GeneralizedDomainwallOperator5D(
            links, L5, 0.13f0, -1.0f0,
            ones(Float32, L5), fill(1.5f0, L5), fill(0.5f0, L5))
        @test generalized32.mass isa Float32
        @test eltype(generalized32.a) === Float32
        @test Array(generalized32.b) == fill(1.5f0, L5)
        @test_throws DimensionMismatch D5DW_GeneralizedDomainwallOperator5D(
            links, L5, mass, M, ones(L5 - 1), ones(L5), zeros(L5))
        @test_throws ArgumentError D5DW_GeneralizedDomainwallOperator5D(
            links, L5, mass, M, ones(L5), fill(Inf, L5), zeros(L5))
    end
    _domainwall_nc3_fastpath_tests()
end
