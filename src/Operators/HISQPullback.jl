@inline function _hisq_pullback_path_link_offset(path, occurrence)
    offset = (0, 0, 0, 0)
    link_offset = offset
    @inbounds for path_index in eachindex(path)
        direction = path[path_index]
        direction < 0 && (offset = _hisq_shift_site(offset, direction))
        path_index == occurrence && (link_offset = offset)
        direction > 0 && (offset = _hisq_shift_site(offset, direction))
    end
    return link_offset
end

@inline function _hisq_pullback_basis_vector(index, ::Type{T}, ::Val{NC}) where {T,NC}
    return SVector{NC}(ntuple(
        component -> ifelse(component == index, one(T), zero(T)),
        Val(NC)))
end

@inline function _hisq_pullback_path_site_before(origin, path, path_index)
    site = origin
    @inbounds for index in 1:(path_index - 1)
        site = _hisq_shift_site(site, path[index])
    end
    return site
end

@inline function _hisq_pullback_oriented_link_times_column(
    U1, U2, U3, U4, site, direction, column_values, ::Val{NC},
) where NC
    matrix_site = ifelse(
        direction > 0, site, _hisq_shift_site(site, direction))
    axis = abs(direction)
    return SVector{NC}(ntuple(Val(NC)) do row
        value = zero(eltype(column_values))
        @inbounds for contracted in 1:NC
            link_element = if direction > 0
                _hisq_link_element(
                    U1, U2, U3, U4, axis,
                    row, contracted, matrix_site)
            else
                conj(_hisq_link_element(
                    U1, U2, U3, U4, axis,
                    contracted, row, matrix_site))
            end
            value += link_element * column_values[contracted]
        end
        value
    end)
end

@inline function _hisq_pullback_path_left_column(
    U1, U2, U3, U4, origin, path, occurrence, column,
    ::Type{T}, ::Val{NC},
) where {T,NC}
    values = _hisq_pullback_basis_vector(column, T, Val(NC))
    @inbounds for path_index in (occurrence - 1):-1:1
        site = _hisq_pullback_path_site_before(origin, path, path_index)
        values = _hisq_pullback_oriented_link_times_column(
            U1, U2, U3, U4, site, path[path_index], values, Val(NC))
    end
    return values
end

@inline function _hisq_pullback_path_right_row(
    U1, U2, U3, U4, origin, path, occurrence, row,
    ::Type{T}, ::Val{NC},
) where {T,NC}
    values = _hisq_pullback_basis_vector(row, T, Val(NC))
    site = _hisq_pullback_path_site_before(origin, path, occurrence + 1)
    @inbounds for path_index in (occurrence + 1):length(path)
        values, site = _hisq_row_times_oriented_link(
            values, U1, U2, U3, U4, site,
            path[path_index], Val(NC))
    end
    return values
end

@inline function _hisq_pullback_path_element(
    dV1, dV2, dV3, dV4, U1, U2, U3, U4,
    target, path, coefficient, output_direction, axis, row, column,
    ::Type{T}, ::Val{NC},
) where {T,NC}
    gradient = zero(T)
    @inbounds for occurrence in eachindex(path)
        direction = path[occurrence]
        abs(direction) == axis || continue

        link_offset = _hisq_pullback_path_link_offset(path, occurrence)
        origin = ntuple(d -> target[d] - link_offset[d], 4)
        hrow = ifelse(direction > 0, row, column)
        hcolumn = ifelse(direction > 0, column, row)
        left_column = _hisq_pullback_path_left_column(
            U1, U2, U3, U4, origin, path, occurrence, hrow,
            T, Val(NC))
        right_row = _hisq_pullback_path_right_row(
            U1, U2, U3, U4, origin, path, occurrence, hcolumn,
            T, Val(NC))

        value = zero(T)
        for output_column in 1:NC
            left_times_output = zero(T)
            for output_row in 1:NC
                left_times_output += conj(left_column[output_row]) *
                    _hisq_link_element(
                        dV1, dV2, dV3, dV4, output_direction,
                        output_row, output_column, origin)
            end
            value += left_times_output * conj(right_row[output_column])
        end
        gradient += coefficient * ifelse(
            direction > 0, value, conj(value))
    end
    return gradient
end

@inline function _hisq_pullback_element_indices(
    combined_index, volume, ::Val{NC},
) where NC
    zero_based = combined_index - 1
    site_index = mod(zero_based, volume) + 1
    matrix_element_and_axis = div(zero_based, volume)
    row = mod(matrix_element_and_axis, NC) + 1
    column_and_axis = div(matrix_element_and_axis, NC)
    column = mod(column_and_axis, NC) + 1
    axis = div(column_and_axis, NC) + 1
    return site_index, row, column, axis
end

@inline function _hisq_pullback_add_element!(
    dU1, dU2, dU3, dU4, axis, row, column, target, value,
)
    if axis == 1
        dU1[row, column, target...] += value
    elseif axis == 2
        dU2[row, column, target...] += value
    elseif axis == 3
        dU3[row, column, target...] += value
    else
        dU4[row, column, target...] += value
    end
    return nothing
end

@inline function _kernel_hisq_pullback_one_link!(
    combined_index, dU1, dU2, dU3, dU4,
    dV1, dV2, dV3, dV4, U1, U2, U3, U4,
    coefficient, volume, ::Val{NC}, ::Val{nw}, indexer,
) where {NC,nw}
    site_index, row, column, axis = _hisq_pullback_element_indices(
        combined_index, volume, Val(NC))
    target = delinearize(indexer, site_index, nw)
    T = eltype(dU1)
    gradient = zero(T)
    @inbounds for output_direction in 1:4
        gradient += _hisq_pullback_path_element(
            dV1, dV2, dV3, dV4, U1, U2, U3, U4,
            target, (output_direction,), coefficient,
            output_direction, axis, row, column, T, Val(NC))
    end
    _hisq_pullback_add_element!(
        dU1, dU2, dU3, dU4, axis, row, column, target, gradient)
    return nothing
end

@inline function _kernel_hisq_pullback_staple3!(
    combined_index, dU1, dU2, dU3, dU4,
    dV1, dV2, dV3, dV4, U1, U2, U3, U4,
    coefficient, volume, ::Val{NC}, ::Val{nw}, indexer,
) where {NC,nw}
    site_index, row, column, axis = _hisq_pullback_element_indices(
        combined_index, volume, Val(NC))
    target = delinearize(indexer, site_index, nw)
    T = eltype(dU1)
    gradient = zero(T)
    @inbounds for output_direction in 1:4, nu in 1:4
        nu == output_direction && continue
        gradient += _hisq_pullback_path_element(
            dV1, dV2, dV3, dV4, U1, U2, U3, U4,
            target, (nu, output_direction, -nu), coefficient,
            output_direction, axis, row, column, T, Val(NC))
        gradient += _hisq_pullback_path_element(
            dV1, dV2, dV3, dV4, U1, U2, U3, U4,
            target, (-nu, output_direction, nu), coefficient,
            output_direction, axis, row, column, T, Val(NC))
    end
    _hisq_pullback_add_element!(
        dU1, dU2, dU3, dU4, axis, row, column, target, gradient)
    return nothing
end

@inline function _kernel_hisq_pullback_staple5!(
    combined_index, dU1, dU2, dU3, dU4,
    dV1, dV2, dV3, dV4, U1, U2, U3, U4,
    coefficient, volume, part::Val{P}, ::Val{NC}, ::Val{nw}, indexer,
) where {P,NC,nw}
    site_index, row, column, axis = _hisq_pullback_element_indices(
        combined_index, volume, Val(NC))
    target = delinearize(indexer, site_index, nw)
    T = eltype(dU1)
    gradient = zero(T)
    sign_nu, sign_rho = _hisq_five_signs(part)
    @inbounds for output_direction in 1:4, nu in 1:4
        nu == output_direction && continue
        for rho in 1:4
            (rho == output_direction || rho == nu) && continue
            signed_nu = sign_nu * nu
            signed_rho = sign_rho * rho
            path = (signed_nu, signed_rho, output_direction,
                    -signed_rho, -signed_nu)
            gradient += _hisq_pullback_path_element(
                dV1, dV2, dV3, dV4, U1, U2, U3, U4,
                target, path, coefficient, output_direction,
                axis, row, column, T, Val(NC))
        end
    end
    _hisq_pullback_add_element!(
        dU1, dU2, dU3, dU4, axis, row, column, target, gradient)
    return nothing
end

@inline function _kernel_hisq_pullback_staple7!(
    combined_index, dU1, dU2, dU3, dU4,
    dV1, dV2, dV3, dV4, U1, U2, U3, U4,
    coefficient, volume, part::Val{P}, ::Val{NC}, ::Val{nw}, indexer,
) where {P,NC,nw}
    site_index, row, column, axis = _hisq_pullback_element_indices(
        combined_index, volume, Val(NC))
    target = delinearize(indexer, site_index, nw)
    T = eltype(dU1)
    gradient = zero(T)
    sign_nu, sign_rho, sign_sigma = _hisq_seven_signs(part)
    @inbounds for output_direction in 1:4, nu in 1:4
        nu == output_direction && continue
        for rho in 1:4
            (rho == output_direction || rho == nu) && continue
            sigma = 10 - output_direction - nu - rho
            signed_nu = sign_nu * nu
            signed_rho = sign_rho * rho
            signed_sigma = sign_sigma * sigma
            path = (signed_nu, signed_rho, signed_sigma, output_direction,
                    -signed_sigma, -signed_rho, -signed_nu)
            gradient += _hisq_pullback_path_element(
                dV1, dV2, dV3, dV4, U1, U2, U3, U4,
                target, path, coefficient, output_direction,
                axis, row, column, T, Val(NC))
        end
    end
    _hisq_pullback_add_element!(
        dU1, dU2, dU3, dU4, axis, row, column, target, gradient)
    return nothing
end

@inline function _kernel_hisq_pullback_lepage!(
    combined_index, dU1, dU2, dU3, dU4,
    dV1, dV2, dV3, dV4, U1, U2, U3, U4,
    coefficient, volume, ::Val{NC}, ::Val{nw}, indexer,
) where {NC,nw}
    site_index, row, column, axis = _hisq_pullback_element_indices(
        combined_index, volume, Val(NC))
    target = delinearize(indexer, site_index, nw)
    T = eltype(dU1)
    gradient = zero(T)
    @inbounds for output_direction in 1:4, nu in 1:4
        nu == output_direction && continue
        gradient += _hisq_pullback_path_element(
            dV1, dV2, dV3, dV4, U1, U2, U3, U4,
            target, (nu, nu, output_direction, -nu, -nu), coefficient,
            output_direction, axis, row, column, T, Val(NC))
        gradient += _hisq_pullback_path_element(
            dV1, dV2, dV3, dV4, U1, U2, U3, U4,
            target, (-nu, -nu, output_direction, nu, nu), coefficient,
            output_direction, axis, row, column, T, Val(NC))
    end
    _hisq_pullback_add_element!(
        dU1, dU2, dU3, dU4, axis, row, column, target, gradient)
    return nothing
end

function _hisq_fat7_pullback_accumulate!(dU, dV, U, coefficients)
    ensure_halo!.(dV)
    ensure_halo!.(U)
    coefficient_1, coefficient_3, coefficient_5, coefficient_7,
        coefficient_lepage = coefficients
    volume = prod(U[1].PN)
    NC = U[1].NC1
    combined_volume = 4 * NC * NC * volume
    common_arguments = (
        dU[1].A, dU[2].A, dU[3].A, dU[4].A,
        dV[1].A, dV[2].A, dV[3].A, dV[4].A,
        U[1].A, U[2].A, U[3].A, U[4].A)
    geometry_arguments = (volume, Val(NC), Val(U[1].nw), U[1].indexer)
    _hisq_parallel_for(
        combined_volume, _kernel_hisq_pullback_one_link!,
        common_arguments..., coefficient_1, geometry_arguments...)
    _hisq_parallel_for(
        combined_volume, _kernel_hisq_pullback_staple3!,
        common_arguments..., coefficient_3, geometry_arguments...)
    for part in 1:4
        _hisq_parallel_for(
            combined_volume, _kernel_hisq_pullback_staple5!,
            common_arguments..., coefficient_5, volume, Val(part),
            Val(NC), Val(U[1].nw), U[1].indexer)
    end
    for part in 1:8
        _hisq_parallel_for(
            combined_volume, _kernel_hisq_pullback_staple7!,
            common_arguments..., coefficient_7, volume, Val(part),
            Val(NC), Val(U[1].nw), U[1].indexer)
    end
    if !iszero(coefficient_lepage)
        _hisq_parallel_for(
            combined_volume, _kernel_hisq_pullback_lepage!,
            common_arguments..., coefficient_lepage, geometry_arguments...)
    end
    mark_halo_dirty!.(dU)
    return dU
end

@inline function _hisq_pullback_lu_solve_vector!(
    LU::MMatrix{N,N,T}, piv::MVector{N,Int},
    right_hand_side::MVector{N,T},
) where {N,T}
    @inbounds for k in 1:N
        pivot = piv[k]
        if pivot != k
            right_hand_side[k], right_hand_side[pivot] =
                right_hand_side[pivot], right_hand_side[k]
        end
        for row in (k + 1):N
            right_hand_side[row] -= LU[row, k] * right_hand_side[k]
        end
    end
    @inbounds for row in N:-1:1
        value = right_hand_side[row]
        for column in (row + 1):N
            value -= LU[row, column] * right_hand_side[column]
        end
        right_hand_side[row] = value / LU[row, row]
    end
    return nothing
end

@inline function _hisq_pullback_solve_sylvester_3x3!(
    solution, hermitian, rhs, system, vector, pivots,
)
    element_type = eltype(solution)
    @inbounds for column in 1:9, row in 1:9
        system[row, column] = zero(element_type)
    end
    @inbounds for column in 1:3, row in 1:3
        equation = row + 3 * (column - 1)
        vector[equation] = rhs[row, column]
        for contracted in 1:3
            unknown = contracted + 3 * (column - 1)
            system[equation, unknown] += hermitian[row, contracted]
            unknown = row + 3 * (contracted - 1)
            system[equation, unknown] += hermitian[contracted, column]
        end
    end
    lu_factor!(system, pivots)
    _hisq_pullback_lu_solve_vector!(system, pivots, vector)
    @inbounds for column in 1:3, row in 1:3
        solution[row, column] = vector[row + 3 * (column - 1)]
    end
    return nothing
end

@inline function _kernel_hisq_project_u3_pullback_core!(
    site_index, dinput, doutput, input, ::Val{nw}, indexer,
) where nw
    site = delinearize(indexer, site_index, nw)
    element_type = eltype(input)
    V = MMatrix{3,3,element_type}(undef)
    Q = MMatrix{3,3,element_type}(undef)
    Q2 = MMatrix{3,3,element_type}(undef)
    inverse_sqrt = MMatrix{3,3,element_type}(undef)
    hermitian = MMatrix{3,3,element_type}(undef)
    projected = MMatrix{3,3,element_type}(undef)
    output_cotangent = MMatrix{3,3,element_type}(undef)
    skew_rhs = MMatrix{3,3,element_type}(undef)
    sylvester_solution = MMatrix{3,3,element_type}(undef)
    gradient = MMatrix{3,3,element_type}(undef)
    system = MMatrix{9,9,element_type}(undef)
    vector = MVector{9,element_type}(undef)
    pivots = MVector{9,Int}(undef)

    @inbounds for column in 1:3, row in 1:3
        V[row, column] = input[row, column, site...]
        output_cotangent[row, column] = doutput[row, column, site...]
    end
    _hisq_u3_project_matrix!(
        projected, V, Q, Q2, inverse_sqrt, hermitian)

    @inbounds for column in 1:3, row in 1:3
        value = zero(element_type)
        adjoint_value = zero(element_type)
        for contracted in 1:3
            value += conj(projected[contracted, row]) *
                output_cotangent[contracted, column]
            adjoint_value += conj(output_cotangent[contracted, row]) *
                projected[contracted, column]
        end
        skew_rhs[row, column] = value - adjoint_value
    end
    _hisq_pullback_solve_sylvester_3x3!(
        sylvester_solution, hermitian, skew_rhs, system, vector, pivots)
    gemm!(gradient, projected, sylvester_solution)
    @inbounds for column in 1:3, row in 1:3
        dinput[row, column, site...] += gradient[row, column]
    end
    return nothing
end

function _hisq_project_u3_pullback_accumulate!(dfat, dprojected, fat_links)
    for mu in 1:4
        JACC.parallel_for(
            prod(fat_links[mu].PN),
            _kernel_hisq_project_u3_pullback_core!,
            dfat[mu].A, dprojected[mu].A, fat_links[mu].A,
            Val(fat_links[mu].nw), fat_links[mu].indexer)
        mark_halo_dirty!(dfat[mu])
    end
    return dfat
end

@inline function _kernel_hisq_naik_pullback_core!(
    combined_index, dW1, dW2, dW3, dW4,
    dL1, dL2, dL3, dL4, W1, W2, W3, W4,
    volume, ::Val{NC}, ::Val{nw}, indexer,
) where {NC,nw}
    site_index, row, column, axis = _hisq_pullback_element_indices(
        combined_index, volume, Val(NC))
    target = delinearize(indexer, site_index, nw)
    T = eltype(dW1)
    gradient = _hisq_pullback_path_element(
        dL1, dL2, dL3, dL4, W1, W2, W3, W4,
        target, (axis, axis, axis), one(T), axis,
        axis, row, column, T, Val(NC))
    _hisq_pullback_add_element!(
        dW1, dW2, dW3, dW4, axis, row, column, target, gradient)
    return nothing
end

function _hisq_naik_pullback_accumulate!(dinput, dlong, reunitarized_links)
    ensure_halo!.(dlong)
    ensure_halo!.(reunitarized_links)
    volume = prod(reunitarized_links[1].PN)
    NC = reunitarized_links[1].NC1
    _hisq_parallel_for(
        4 * NC * NC * volume, _kernel_hisq_naik_pullback_core!,
        dinput[1].A, dinput[2].A, dinput[3].A, dinput[4].A,
        dlong[1].A, dlong[2].A, dlong[3].A, dlong[4].A,
        reunitarized_links[1].A, reunitarized_links[2].A,
        reunitarized_links[3].A, reunitarized_links[4].A,
        volume, Val(NC), Val(reunitarized_links[1].nw),
        reunitarized_links[1].indexer)
    mark_halo_dirty!.(dinput)
    return dinput
end

@inline function _kernel_hisq_staggered_link_pullback_direction!(
    dU, dresult, psi, x, xplus, coefficient, eta, ::Val{NC},
) where NC
    @inbounds for row in 1:NC, column in 1:NC
        value =
            dresult[row, 1, x...] * conj(psi[column, 1, xplus...]) -
            psi[row, 1, x...] * conj(dresult[column, 1, xplus...])
        dU[row, column, x...] += coefficient * eta * value
    end
    return nothing
end

@inline function _kernel_hisq_cached_link_pullback!(
    site,
    dX1, dX2, dX3, dX4, dL1, dL2, dL3, dL4,
    dresult, psi, fat_coefficient, long_coefficient,
    ::Val{NC}, ::Val{nw}, indexer, mpi_coordinates, local_size,
) where {NC,nw}
    x = delinearize(indexer, site, nw)
    eta2 = staggered_eta_global_halo(
        x, 2, nw, mpi_coordinates, local_size)
    eta3 = staggered_eta_global_halo(
        x, 3, nw, mpi_coordinates, local_size)
    eta4 = staggered_eta_global_halo(
        x, 4, nw, mpi_coordinates, local_size)

    _kernel_hisq_staggered_link_pullback_direction!(
        dX1, dresult, psi, x, shiftindices(x, shift_1p),
        fat_coefficient, 1, Val(NC))
    _kernel_hisq_staggered_link_pullback_direction!(
        dX2, dresult, psi, x, shiftindices(x, shift_2p),
        fat_coefficient, eta2, Val(NC))
    _kernel_hisq_staggered_link_pullback_direction!(
        dX3, dresult, psi, x, shiftindices(x, shift_3p),
        fat_coefficient, eta3, Val(NC))
    _kernel_hisq_staggered_link_pullback_direction!(
        dX4, dresult, psi, x, shiftindices(x, shift_4p),
        fat_coefficient, eta4, Val(NC))

    _kernel_hisq_staggered_link_pullback_direction!(
        dL1, dresult, psi, x, shiftindices(x, hisq_long_shifts_p[1]),
        long_coefficient, 1, Val(NC))
    _kernel_hisq_staggered_link_pullback_direction!(
        dL2, dresult, psi, x, shiftindices(x, hisq_long_shifts_p[2]),
        long_coefficient, eta2, Val(NC))
    _kernel_hisq_staggered_link_pullback_direction!(
        dL3, dresult, psi, x, shiftindices(x, hisq_long_shifts_p[3]),
        long_coefficient, eta3, Val(NC))
    _kernel_hisq_staggered_link_pullback_direction!(
        dL4, dresult, psi, x, shiftindices(x, hisq_long_shifts_p[4]),
        long_coefficient, eta4, Val(NC))
    return nothing
end

@inline function _hisq_pullback_scratch_lattice(
    array, primal::T,
) where {T<:LatticeMatrix}
    return T(
        primal.nw, primal.phases, primal.NC1, primal.NC2, primal.gsize,
        primal.cart, primal.coords, primal.dims, primal.nbr,
        array, primal.buf, primal.buf_host, primal.shift_buf_host,
        primal.mpi_transport,
        primal.myrank, primal.PN, primal.comm, primal.indexer,
        primal.temps, HaloEpoch(),
    )
end

function _reserve_hisq_pullback_vector_scratch(fields)
    blocks = ntuple(mu -> get_block(fields[mu].temps), Val(4))
    scratch = ntuple(
        mu -> _hisq_pullback_scratch_lattice(blocks[mu][1], fields[mu]),
        Val(4))
    indices = ntuple(mu -> blocks[mu][2], Val(4))
    return scratch, indices
end

function _reserve_hisq_pullback_scratch(cache::HISQDiracCache4D)
    dFat, fat_indices =
        _reserve_hisq_pullback_vector_scratch(cache.fat_links)
    dLong, long_indices =
        _reserve_hisq_pullback_vector_scratch(cache.long_links)
    dReunit, reunit_indices =
        _reserve_hisq_pullback_vector_scratch(cache.reunitarized_links)
    return (dFat, dLong, dReunit),
           (fat_indices, long_indices, reunit_indices)
end

function _release_hisq_pullback_scratch!(cache::HISQDiracCache4D, indices)
    collections = (
        cache.fat_links, cache.long_links, cache.reunitarized_links)
    for collection_index in 1:3, mu in 1:4
        unused!(
            collections[collection_index][mu].temps,
            indices[collection_index][mu])
    end
    return nothing
end

function _validate_hisq_link_pullback(
    dthin_links, cache, thin_links, result_cotangent, psi,
)
    length(dthin_links) == 4 || throw(ArgumentError(
        "HISQ link pullback requires four thin-link cotangents"))
    all(link -> link isa LatticeMatrix{4}, dthin_links) ||
        throw(ArgumentError(
            "HISQ thin-link cotangents must be four-dimensional lattice matrices"))
    _validate_hisq_smearing_output(dthin_links, thin_links)

    reference = thin_links[1]
    reference.NC1 == 3 && reference.NC2 == 3 || throw(ArgumentError(
        "HISQ link pullback currently requires three-color thin links"))
    reference.nw >= 3 || throw(ArgumentError(
        "HISQ link pullback requires halo width nw >= 3"))
    result_cotangent.NC1 == 3 && result_cotangent.NC2 == 1 &&
        psi.NC1 == 3 && psi.NC2 == 1 || throw(ArgumentError(
            "HISQ link pullback requires three-color one-column fermion fields"))
    result_cotangent.gsize == psi.gsize &&
        result_cotangent.PN == psi.PN &&
        result_cotangent.dims == psi.dims &&
        result_cotangent.nw == psi.nw &&
        result_cotangent.phases == psi.phases || throw(ArgumentError(
            "HISQ result cotangent and source must share geometry and boundary phases"))
    reference.gsize == psi.gsize && reference.PN == psi.PN &&
        reference.dims == psi.dims && reference.nw == psi.nw ||
        throw(ArgumentError(
            "HISQ thin links and fermion fields must share one lattice geometry"))
    eltype(reference.A) == eltype(psi.A) ==
        eltype(result_cotangent.A) || throw(ArgumentError(
            "HISQ links and fermion fields must share an element type"))

    for mu in 1:4
        for nu in 1:4
            (dthin_links[mu] === thin_links[nu] ||
             dthin_links[mu].A === thin_links[nu].A) &&
                throw(ArgumentError(
                    "HISQ thin-link cotangents must not alias primal links"))
        end
    end
    cache.operator.links.fat_links === cache.fat_links || throw(ArgumentError(
        "HISQ cache contains inconsistent fat-link storage"))
    return nothing
end

"""
    hisq_link_pullback!(
        dthin_links, cache, thin_links, result_cotangent, psi;
        coefficient=1)

Accumulate the thin-link pullback of
`coefficient * real(dot(result_cotangent,
mul_cached_hisq!(..., cache, thin_links..., psi)))` into `dthin_links`.

This is the analytic HISQ force path through the one-link and Naik terms,
level-2 Fat7 smearing, U(3) projection, and level-1 Fat7 smearing. It does not
depend on Enzyme. `cache` supplies reusable temporary storage, and is refreshed
from `thin_links` when necessary. The four destination fields are not cleared,
so repeated calls accumulate. HISQ force evaluation requires `NC=3` and
`nw >= 3`.
"""
function hisq_link_pullback!(
    dthin_links::Union{AbstractVector,Tuple},
    cache::HISQDiracCache4D{T},
    thin_links::Union{AbstractVector,Tuple},
    result_cotangent::F,
    psi::F;
    coefficient=1,
) where {T<:LatticeMatrix{4},F<:LatticeMatrix{4}}
    length(thin_links) == 4 || throw(ArgumentError(
        "HISQ link pullback requires four thin links"))
    all(link -> link isa T, thin_links) || throw(ArgumentError(
        "HISQ thin links must match the cache lattice type"))
    U = (thin_links[1], thin_links[2], thin_links[3], thin_links[4])
    _ensure_hisq_cache_current!(cache, U...)
    _validate_hisq_link_pullback(
        dthin_links, cache, U, result_cotangent, psi)
    ensure_halo!(result_cotangent)
    ensure_halo!(psi)

    scratch, scratch_indices = _reserve_hisq_pullback_scratch(cache)
    try
        dFat, dLong, dReunit = scratch
        for fields in scratch, field in fields
            clear_matrix!(field)
        end

        operator = cache.operator
        real_type = typeof(operator.mass)
        scale = convert(real_type, coefficient)
        fat_coefficient = scale / 2
        long_coefficient =
            -scale * (one(operator.naik_epsilon) +
                      operator.naik_epsilon) / 48
        JACC.parallel_for(
            prod(result_cotangent.PN),
            _kernel_hisq_cached_link_pullback!,
            dFat[1].A, dFat[2].A, dFat[3].A, dFat[4].A,
            dLong[1].A, dLong[2].A, dLong[3].A, dLong[4].A,
            result_cotangent.A, psi.A,
            fat_coefficient, long_coefficient,
            Val(result_cotangent.NC1), Val(result_cotangent.nw),
            result_cotangent.indexer, result_cotangent.coords,
            result_cotangent.PN,
        )
        mark_halo_dirty!.(dFat)
        mark_halo_dirty!.(dLong)

        _hisq_naik_pullback_accumulate!(
            dReunit, dLong, cache.reunitarized_links)

        epsilon = convert(real_type, operator.naik_epsilon)
        level2_coefficients = (
            one(real_type) + epsilon / 8,
            one(real_type) / 16,
            one(real_type) / 64,
            one(real_type) / 384,
            -one(real_type) / 8,
        )
        _hisq_fat7_pullback_accumulate!(
            dReunit, dFat, cache.reunitarized_links,
            level2_coefficients)

        clear_matrix!.(dFat)
        _hisq_project_u3_pullback_accumulate!(
            dFat, dReunit, cache.level1_links)

        level1_coefficients = (
            one(real_type) / 8,
            one(real_type) / 16,
            one(real_type) / 64,
            one(real_type) / 384,
            zero(real_type),
        )
        _hisq_fat7_pullback_accumulate!(
            dthin_links, dFat, U, level1_coefficients)
    finally
        _release_hisq_pullback_scratch!(cache, scratch_indices)
    end
    return dthin_links
end

function hisq_link_pullback!(
    dU1::LatticeMatrix{4}, dU2::LatticeMatrix{4},
    dU3::LatticeMatrix{4}, dU4::LatticeMatrix{4},
    cache::HISQDiracCache4D,
    U1::LatticeMatrix{4}, U2::LatticeMatrix{4},
    U3::LatticeMatrix{4}, U4::LatticeMatrix{4},
    result_cotangent::LatticeMatrix{4}, psi::LatticeMatrix{4};
    coefficient=1,
)
    return hisq_link_pullback!(
        (dU1, dU2, dU3, dU4), cache, (U1, U2, U3, U4),
        result_cotangent, psi; coefficient)
end

export hisq_link_pullback!
