function _wilson_halfspin_shift(x, direction, amount, lattice_size, phases)
    raw = x[direction] + amount
    shifted = Base.setindex(x, mod1(raw, lattice_size[direction]), direction)
    wraps = fld(raw - 1, lattice_size[direction])
    return shifted, phases[direction]^wraps
end

function _wilson_halfspin_reference(
    links, psi, kappa, phases; adjoint_operator=false, donly=false,
)
    lattice_size = size(psi)[3:end]
    result = similar(psi)
    element_type = eltype(psi)
    identity_spin = Matrix{element_type}(I, 4, 4)
    gammas = ntuple(mu -> Matrix{element_type}(γs[mu]), 4)
    for site in CartesianIndices(lattice_size)
        x = Tuple(site)
        value = donly ? zeros(element_type, 3, 4) : copy(psi[:, :, x...])
        for mu in 1:4
            xplus, phase_plus = _wilson_halfspin_shift(
                x, mu, 1, lattice_size, phases)
            xminus, phase_minus = _wilson_halfspin_shift(
                x, mu, -1, lattice_size, phases)
            plus_sign = adjoint_operator ? 1 : -1
            plus_projector = identity_spin + plus_sign * gammas[mu]
            minus_projector = identity_spin - plus_sign * gammas[mu]
            hopping = links[mu][:, :, x...] *
                      (phase_plus * psi[:, :, xplus...]) *
                      transpose(plus_projector) +
                      links[mu][:, :, xminus...]' *
                      (phase_minus * psi[:, :, xminus...]) *
                      transpose(minus_projector)
            if donly
                value .+= element_type(0.5) .* hopping
            else
                value .-= element_type(kappa) .* hopping
            end
        end
        @views result[:, :, x...] .= value
    end
    return result
end

function _wilson_halfspin_check_precision(::Type{T}) where T
    nprocs = test_comm_size()
    rank = test_comm_rank()
    lattice_size = (2 * nprocs, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)
    phases = (1, 1, 1, -1)
    links_array = [rand(T, 3, 3, lattice_size...) for _ in 1:4]
    psi_array = rand(T, 3, 4, lattice_size...)
    chi_array = rand(T, 3, 4, lattice_size...)
    links = [LatticeMatrix(link, 4, process_grid; nw=1) for link in links_array]
    links_nw0 = [LatticeMatrix(link, 4, process_grid; nw=0) for link in links_array]
    psi = LatticeMatrix(psi_array, 4, process_grid; nw=1, phases)
    chi = LatticeMatrix(chi_array, 4, process_grid; nw=1, phases)
    result = similar(psi)
    adjoint_result = similar(psi)
    donly_result = similar(psi)
    adjoint_donly_result = similar(psi)
    kappa = 0.117
    operator = WilsonDiracOperator4D(links, kappa)
    donly_operator = WilsonDiracOperator4D_Donly(links)
    clover_zero_operator = WilsonDiracCloverOperator4D(links, kappa, 0.0)
    clover_operator = WilsonDiracCloverOperator4D(links, kappa, 1.17)
    clover_operator_nw0 = WilsonDiracCloverOperator4D(links_nw0, kappa, 1.17)

    mul!(result, operator, psi)
    mul!(adjoint_result, adjoint(operator), psi)
    mul!(donly_result, donly_operator, psi)
    mul!(adjoint_donly_result, adjoint(donly_operator), psi)
    clover_zero_result = similar(psi)
    mul!(clover_zero_result, clover_zero_operator, psi)
    clover_result = similar(psi)
    clover_split_result = similar(psi)
    mul!(clover_result, clover_operator, psi)
    mul!(clover_split_result, operator, psi)
    LatticeMatrices._add_clover_term!(clover_split_result, clover_operator, psi)

    actual = gather_matrix(result)
    actual_adjoint = gather_matrix(adjoint_result)
    actual_donly = gather_matrix(donly_result)
    actual_adjoint_donly = gather_matrix(adjoint_donly_result)
    actual_clover_zero = gather_matrix(clover_zero_result)
    actual_clover = gather_matrix(clover_result)
    actual_clover_split = gather_matrix(clover_split_result)
    actual_clover_fields = ntuple(
        plane -> gather_matrix(clover_operator.clover[plane]), Val(6))
    actual_clover_fields_nw0 = ntuple(
        plane -> gather_matrix(clover_operator_nw0.clover[plane]), Val(6))
    Ddag_chi_lattice = similar(chi)
    mul!(Ddag_chi_lattice, adjoint(operator), chi)
    Ddag_chi = gather_matrix(Ddag_chi_lattice)

    if rank == 0
        reference = _wilson_halfspin_reference(
            links_array, psi_array, kappa, phases)
        reference_adjoint = _wilson_halfspin_reference(
            links_array, psi_array, kappa, phases; adjoint_operator=true)
        reference_donly = _wilson_halfspin_reference(
            links_array, psi_array, kappa, phases; donly=true)
        reference_adjoint_donly = _wilson_halfspin_reference(
            links_array, psi_array, kappa, phases;
            adjoint_operator=true, donly=true)
        tolerance = T === ComplexF32 ? 2f-5 : 2e-12
        reference_clover_fields = _clover_test_field_strength(links_array)
        @test actual ≈ reference atol=tolerance rtol=tolerance
        @test actual_adjoint ≈ reference_adjoint atol=tolerance rtol=tolerance
        @test actual_donly ≈ reference_donly atol=tolerance rtol=tolerance
        @test actual_adjoint_donly ≈
              reference_adjoint_donly atol=tolerance rtol=tolerance
        @test actual_clover_zero ≈ actual atol=tolerance rtol=tolerance
        @test actual_clover ≈ actual_clover_split atol=tolerance rtol=tolerance
        for plane in 1:6
            @test actual_clover_fields[plane] ≈
                  reference_clover_fields[plane] atol=tolerance rtol=tolerance
            @test actual_clover_fields_nw0[plane] ≈
                  reference_clover_fields[plane] atol=tolerance rtol=tolerance
        end
        @test dot(vec(chi_array), vec(actual)) ≈
              dot(vec(Ddag_chi), vec(psi_array)) atol=tolerance rtol=tolerance
    end
    return nothing
end

function wilson_halfspin_tests()
    @testset "NC=3 half-spin Wilson kernels" begin
        _wilson_halfspin_check_precision(ComplexF64)
        _wilson_halfspin_check_precision(ComplexF32)
    end
    return nothing
end
