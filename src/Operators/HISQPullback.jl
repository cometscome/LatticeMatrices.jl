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
    if !iszero(coefficient_1)
        _hisq_parallel_for(
            combined_volume, _kernel_hisq_pullback_one_link!,
            common_arguments..., coefficient_1, geometry_arguments...)
    end
    if !iszero(coefficient_3)
        _hisq_parallel_for(
            combined_volume, _kernel_hisq_pullback_staple3!,
            common_arguments..., coefficient_3, geometry_arguments...)
    end
    if !iszero(coefficient_5)
        for part in 1:4
            _hisq_parallel_for(
                combined_volume, _kernel_hisq_pullback_staple5!,
                common_arguments..., coefficient_5, volume, Val(part),
                Val(NC), Val(U[1].nw), U[1].indexer)
        end
    end
    if !iszero(coefficient_7)
        for part in 1:8
            _hisq_parallel_for(
                combined_volume, _kernel_hisq_pullback_staple7!,
                common_arguments..., coefficient_7, volume, Val(part),
                Val(NC), Val(U[1].nw), U[1].indexer)
        end
    end
    if !iszero(coefficient_lepage)
        _hisq_parallel_for(
            combined_volume, _kernel_hisq_pullback_lepage!,
            common_arguments..., coefficient_lepage, geometry_arguments...)
    end
    mark_halo_dirty!.(dU)
    return dU
end

struct HISQFat7PullbackWorkspace{P,C}
    primal::P
    cotangent::C
end

"""
    HISQFat7PullbackWorkspace(reference_link)

Allocate reusable primal and cotangent intermediates for the factorized
Fat7 pullback. The workspace is color-generic and follows the geometry and
backend of `reference_link`.
"""
function HISQFat7PullbackWorkspace(reference_link::LatticeMatrix{4})
    primal = HISQFat7Workspace(reference_link)
    cotangent = HISQFat7Workspace(reference_link)
    return HISQFat7PullbackWorkspace(primal, cotangent)
end

export HISQFat7PullbackWorkspace

@inline function _hisq_load_site_matrix(
    field, site, ::Val{NC},
) where NC
    T = eltype(field)
    matrix = MMatrix{NC,NC,T}(undef)
    @inbounds for column in 1:NC, row in 1:NC
        matrix[row, column] = field[row, column, site...]
    end
    return matrix
end

@inline function _hisq_matrix_multiply!(output, left, right, ::Val{NC}) where NC
    @inbounds for column in 1:NC, row in 1:NC
        value = zero(eltype(output))
        for contracted in 1:NC
            value += left[row, contracted] * right[contracted, column]
        end
        output[row, column] = value
    end
    return output
end

@inline function _hisq_adjoint_left_multiply!(
    output, left, right, ::Val{NC},
) where NC
    @inbounds for column in 1:NC, row in 1:NC
        value = zero(eltype(output))
        for contracted in 1:NC
            value += conj(left[contracted, row]) * right[contracted, column]
        end
        output[row, column] = value
    end
    return output
end

@inline function _hisq_adjoint_right_multiply!(
    output, left, right, ::Val{NC},
) where NC
    @inbounds for column in 1:NC, row in 1:NC
        value = zero(eltype(output))
        for contracted in 1:NC
            value += left[row, contracted] * conj(right[column, contracted])
        end
        output[row, column] = value
    end
    return output
end

@inline function _hisq_add_matrix!(destination, source, ::Val{NC}) where NC
    @inbounds for column in 1:NC, row in 1:NC
        destination[row, column] += source[row, column]
    end
    return destination
end

@inline function _kernel_hisq_reverse_staple_transport!(
    site_index, dfield, daxis, chain, field, axis_link,
    coefficient, ::Val{NC}, ::Val{nw}, indexer,
    ::Val{mu}, ::Val{axis}, ::Val{orientation},
) where {NC,nw,mu,axis,orientation}
    target = delinearize(indexer, site_index, nw)
    target_minus_axis = _hisq_shift_site(target, -axis)
    target_plus_axis = _hisq_shift_site(target, axis)
    target_minus_mu = _hisq_shift_site(target, -mu)
    target_plus_mu = _hisq_shift_site(target, mu)
    target_minus_axis_plus_mu =
        _hisq_shift_site(target_minus_axis, mu)
    target_minus_mu_plus_axis =
        _hisq_shift_site(target_minus_mu, axis)
    target_plus_axis_minus_mu =
        _hisq_shift_site(target_plus_axis, -mu)

    U = _hisq_load_site_matrix(axis_link, target, Val(NC))
    U_minus_axis = _hisq_load_site_matrix(
        axis_link, target_minus_axis, Val(NC))
    U_plus_mu = _hisq_load_site_matrix(
        axis_link, target_plus_mu, Val(NC))
    U_minus_axis_plus_mu = _hisq_load_site_matrix(
        axis_link, target_minus_axis_plus_mu, Val(NC))
    U_minus_mu = _hisq_load_site_matrix(
        axis_link, target_minus_mu, Val(NC))

    C = _hisq_load_site_matrix(chain, target, Val(NC))
    C_minus_axis = _hisq_load_site_matrix(
        chain, target_minus_axis, Val(NC))
    C_plus_axis = _hisq_load_site_matrix(
        chain, target_plus_axis, Val(NC))
    C_minus_mu = _hisq_load_site_matrix(
        chain, target_minus_mu, Val(NC))
    C_plus_axis_minus_mu = _hisq_load_site_matrix(
        chain, target_plus_axis_minus_mu, Val(NC))

    F = _hisq_load_site_matrix(field, target, Val(NC))
    F_plus_axis = _hisq_load_site_matrix(
        field, target_plus_axis, Val(NC))
    F_minus_mu = _hisq_load_site_matrix(
        field, target_minus_mu, Val(NC))
    F_minus_mu_plus_axis = _hisq_load_site_matrix(
        field, target_minus_mu_plus_axis, Val(NC))

    T = eltype(dfield)
    temporary = MMatrix{NC,NC,T}(undef)
    contribution = MMatrix{NC,NC,T}(undef)
    field_gradient = MMatrix{NC,NC,T}(undef)
    axis_gradient = MMatrix{NC,NC,T}(undef)
    fill!(field_gradient, zero(T))
    fill!(axis_gradient, zero(T))

    if orientation >= 0
        # dF(z): U_a(z-a)' C(z-a) U_a(z-a+mu)
        _hisq_adjoint_left_multiply!(
            temporary, U_minus_axis, C_minus_axis, Val(NC))
        _hisq_matrix_multiply!(
            contribution, temporary, U_minus_axis_plus_mu, Val(NC))
        _hisq_add_matrix!(field_gradient, contribution, Val(NC))

        # dU_a(z), positive leading link: C(z) U_a(z+mu) F(z+a)'
        _hisq_matrix_multiply!(temporary, C, U_plus_mu, Val(NC))
        _hisq_adjoint_right_multiply!(
            contribution, temporary, F_plus_axis, Val(NC))
        _hisq_add_matrix!(axis_gradient, contribution, Val(NC))
        # dU_a(z), positive trailing adjoint link:
        # C(z-mu)' U_a(z-mu) F(z-mu+a)
        _hisq_adjoint_left_multiply!(
            temporary, C_minus_mu, U_minus_mu, Val(NC))
        _hisq_matrix_multiply!(
            contribution, temporary, F_minus_mu_plus_axis, Val(NC))
        _hisq_add_matrix!(axis_gradient, contribution, Val(NC))
    end

    if orientation <= 0
        # dF(z): U_a(z) C(z+a) U_a(z+mu)'
        _hisq_matrix_multiply!(temporary, U, C_plus_axis, Val(NC))
        _hisq_adjoint_right_multiply!(
            contribution, temporary, U_plus_mu, Val(NC))
        _hisq_add_matrix!(field_gradient, contribution, Val(NC))

        # dU_a(z), negative leading adjoint link:
        # F(z) U_a(z+mu) C(z+a)'
        _hisq_matrix_multiply!(temporary, F, U_plus_mu, Val(NC))
        _hisq_adjoint_right_multiply!(
            contribution, temporary, C_plus_axis, Val(NC))
        _hisq_add_matrix!(axis_gradient, contribution, Val(NC))
        # dU_a(z), negative trailing link:
        # F(z-mu)' U_a(z-mu) C(z+a-mu)
        _hisq_adjoint_left_multiply!(
            temporary, F_minus_mu, U_minus_mu, Val(NC))
        _hisq_matrix_multiply!(
            contribution, temporary, C_plus_axis_minus_mu, Val(NC))
        _hisq_add_matrix!(axis_gradient, contribution, Val(NC))
    end

    @inbounds for column in 1:NC, row in 1:NC
        dfield[row, column, target...] +=
            coefficient * field_gradient[row, column]
        daxis[row, column, target...] +=
            coefficient * axis_gradient[row, column]
    end
    return nothing
end

@inline function _kernel_hisq_factorized_pullback_initialize!(
    site_index, dUmu, dfirst_a, dfirst_b, dfirst_c,
    dsecond_a, dsecond_b, dsecond_c, chain,
    coefficient_1, coefficient_3, coefficient_5,
    ::Val{NC}, ::Val{nw}, indexer,
) where {NC,nw}
    target = delinearize(indexer, site_index, nw)
    @inbounds for column in 1:NC, row in 1:NC
        value = chain[row, column, target...]
        dUmu[row, column, target...] += coefficient_1 * value
        dfirst_a[row, column, target...] = coefficient_3 * value
        dfirst_b[row, column, target...] = coefficient_3 * value
        dfirst_c[row, column, target...] = coefficient_3 * value
        dsecond_a[row, column, target...] = coefficient_5 * value
        dsecond_b[row, column, target...] = coefficient_5 * value
        dsecond_c[row, column, target...] = coefficient_5 * value
    end
    return nothing
end

function _hisq_prepare_factorized_pullback_direction!(
    U, workspace::HISQFat7Workspace, mu,
)
    U1, U2, U3, U4 = U
    first = workspace.first_stage
    second = workspace.second_stage
    NC = U1.NC1
    volume = prod(U1.PN)
    starts, domain_indexer, domain_volume =
        _hisq_factorized_domain(U1.PN, U1.nw, ())
    axes = _hisq_transverse_axes(mu)
    arrays_U = (U1.A, U2.A, U3.A, U4.A)
    for slot in 1:3
        _hisq_parallel_for(
            NC * domain_volume, kernel_hisq_fat7_factorized_first!,
            first[slot].A, arrays_U..., domain_volume, starts,
            domain_indexer, Val(NC), Val(mu), Val(axes[slot]))
        mark_halo_dirty!(first[slot])
    end
    ensure_halo!.(first)
    for slot in 1:3
        b = axes[slot == 1 ? 2 : 1]
        c = axes[slot == 3 ? 2 : 3]
        _hisq_parallel_for(
            NC * volume, kernel_hisq_fat7_factorized_second!,
            second[slot].A,
            first[slot == 1 ? 2 : 1].A,
            first[slot == 3 ? 2 : 3].A,
            arrays_U..., volume, starts, domain_indexer,
            Val(NC), Val(mu), Val(b), Val(c))
        mark_halo_dirty!(second[slot])
    end
    ensure_halo!.(second)
    return axes
end

function _hisq_reverse_staple_transport!(
    dfield, daxis, chain, field, axis_link,
    coefficient, reference, mu, axis, orientation::Val=Val(0),
)
    _hisq_parallel_for(
        prod(reference.PN), _kernel_hisq_reverse_staple_transport!,
        dfield.A, daxis.A, chain.A, field.A, axis_link.A,
        coefficient, Val(reference.NC1), Val(reference.nw),
        reference.indexer, Val(mu), Val(axis), orientation)
    mark_halo_dirty!(dfield)
    mark_halo_dirty!(daxis)
    return nothing
end

@inline function _kernel_hisq_oriented_staple_transport!(
    combined_index, output, field, U1, U2, U3, U4,
    volume, ::Val{NC}, ::Val{nw}, indexer,
    ::Val{mu}, ::Val{axis}, ::Val{orientation},
) where {NC,nw,mu,axis,orientation}
    zero_based = combined_index - 1
    row = mod(zero_based, NC) + 1
    site_index = div(zero_based, NC) + 1
    origin = delinearize(indexer, site_index, nw)
    transported = _hisq_oriented_staple_transport_row(
        field, U1, U2, U3, U4, origin, mu, axis, row,
        Val(NC), Val(orientation))
    _hisq_store_single_row!(output, origin, row, transported)
    return nothing
end

function _hisq_oriented_staple_transport!(
    output, field, U, reference, mu, axis, orientation::Val,
)
    _hisq_parallel_for(
        reference.NC1 * prod(reference.PN),
        _kernel_hisq_oriented_staple_transport!,
        output.A, field.A, U[1].A, U[2].A, U[3].A, U[4].A,
        prod(reference.PN), Val(reference.NC1), Val(reference.nw),
        reference.indexer, Val(mu), Val(axis), orientation)
    mark_halo_dirty!(output)
    return output
end

function _hisq_lepage_factorized_pullback_accumulate!(
    dU, dV, U, coefficient, workspace::HISQFat7PullbackWorkspace,
)
    reference = U[1]
    positive = workspace.primal.first_stage[1]
    negative = workspace.primal.first_stage[2]
    dpositive = workspace.cotangent.first_stage[1]
    dnegative = workspace.cotangent.first_stage[2]

    for mu in 1:4
        for axis in _hisq_transverse_axes(mu)
            _hisq_oriented_staple_transport!(
                positive, U[mu], U, reference, mu, axis, Val(1))
            _hisq_oriented_staple_transport!(
                negative, U[mu], U, reference, mu, axis, Val(-1))
            ensure_halo!(positive)
            ensure_halo!(negative)
            clear_matrix!(dpositive)
            clear_matrix!(dnegative)

            _hisq_reverse_staple_transport!(
                dpositive, dU[axis], dV[mu], positive, U[axis],
                coefficient, reference, mu, axis, Val(1))
            _hisq_reverse_staple_transport!(
                dnegative, dU[axis], dV[mu], negative, U[axis],
                coefficient, reference, mu, axis, Val(-1))
            ensure_halo!(dpositive)
            ensure_halo!(dnegative)

            _hisq_reverse_staple_transport!(
                dU[mu], dU[axis], dpositive, U[mu], U[axis],
                one(coefficient), reference, mu, axis, Val(1))
            _hisq_reverse_staple_transport!(
                dU[mu], dU[axis], dnegative, U[mu], U[axis],
                one(coefficient), reference, mu, axis, Val(-1))
        end
    end
    return dU
end

function _hisq_fat7_factorized_pullback_accumulate!(
    dU, dV, U, coefficients,
    workspace::HISQFat7PullbackWorkspace,
)
    _validate_hisq_fat7_workspace(workspace.primal, dV, U)
    _validate_hisq_fat7_workspace(workspace.cotangent, dV, U)
    ensure_halo!.(dV)
    ensure_halo!.(U)
    coefficient_1, coefficient_3, coefficient_5, coefficient_7,
        coefficient_lepage = coefficients
    first = workspace.primal.first_stage
    second = workspace.primal.second_stage
    dfirst = workspace.cotangent.first_stage
    dsecond = workspace.cotangent.second_stage
    reference = U[1]
    volume = prod(reference.PN)
    NC = reference.NC1

    for mu in 1:4
        axes = _hisq_prepare_factorized_pullback_direction!(
            U, workspace.primal, mu)
        clear_matrix!.(dfirst)
        clear_matrix!.(dsecond)
        _hisq_parallel_for(
            volume, _kernel_hisq_factorized_pullback_initialize!,
            dU[mu].A, dfirst[1].A, dfirst[2].A, dfirst[3].A,
            dsecond[1].A, dsecond[2].A, dsecond[3].A, dV[mu].A,
            coefficient_1, coefficient_3, coefficient_5,
            Val(NC), Val(reference.nw), reference.indexer)
        mark_halo_dirty!(dU[mu])
        mark_halo_dirty!.(dfirst)
        mark_halo_dirty!.(dsecond)

        for slot in 1:3
            axis = axes[slot]
            _hisq_reverse_staple_transport!(
                dsecond[slot], dU[axis], dV[mu], second[slot], U[axis],
                coefficient_7, reference, mu, axis)
        end
        ensure_halo!.(dsecond)

        for slot in 1:3
            b_slot = slot == 1 ? 2 : 1
            c_slot = slot == 3 ? 2 : 3
            b = axes[b_slot]
            c = axes[c_slot]
            _hisq_reverse_staple_transport!(
                dfirst[c_slot], dU[b], dsecond[slot],
                first[c_slot], U[b], one(coefficient_5),
                reference, mu, b)
            _hisq_reverse_staple_transport!(
                dfirst[b_slot], dU[c], dsecond[slot],
                first[b_slot], U[c], one(coefficient_5),
                reference, mu, c)
        end
        ensure_halo!.(dfirst)

        for slot in 1:3
            axis = axes[slot]
            _hisq_reverse_staple_transport!(
                dU[mu], dU[axis], dfirst[slot], U[mu], U[axis],
                one(coefficient_3), reference, mu, axis)
        end
    end

    if !iszero(coefficient_lepage)
        _hisq_lepage_factorized_pullback_accumulate!(
            dU, dV, U, coefficient_lepage, workspace)
    end
    mark_halo_dirty!.(dU)
    return dU
end

function _hisq_fat7_pullback_accumulate!(
    dU, dV, U, coefficients,
    workspace::HISQFat7PullbackWorkspace,
)
    iszero(U[1].nw) && return _hisq_fat7_pullback_accumulate!(
        dU, dV, U, coefficients)
    return _hisq_fat7_factorized_pullback_accumulate!(
        dU, dV, U, coefficients, workspace)
end

_hisq_fat7_pullback_accumulate!(
    dU, dV, U, coefficients, ::Nothing,
) = _hisq_fat7_pullback_accumulate!(dU, dV, U, coefficients)

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

@inline function _hisq_pullback_solve_sylvester!(
    solution, hermitian, rhs, system, vector, ::Val{NC},
) where NC
    element_type = eltype(solution)
    number_of_unknowns = NC * NC
    @inbounds for column in 1:number_of_unknowns
        for row in 1:number_of_unknowns
            system[row, column] = zero(element_type)
        end
    end
    @inbounds for column in 1:NC, row in 1:NC
        equation = row + NC * (column - 1)
        vector[equation] = rhs[row, column]
        for contracted in 1:NC
            unknown = contracted + NC * (column - 1)
            system[equation, unknown] += hermitian[row, contracted]
            unknown = row + NC * (contracted - 1)
            system[equation, unknown] += hermitian[contracted, column]
        end
    end

    # Solve the small dense system in place with partial pivoting.  This is
    # used only by the generic-color path; NC=3 retains the established LU
    # kernel above.
    @inbounds for pivot_column in 1:number_of_unknowns
        pivot_row = pivot_column
        pivot_magnitude = abs(system[pivot_column, pivot_column])
        for row in (pivot_column + 1):number_of_unknowns
            candidate = abs(system[row, pivot_column])
            if candidate > pivot_magnitude
                pivot_row = row
                pivot_magnitude = candidate
            end
        end
        if pivot_row != pivot_column
            for column in 1:number_of_unknowns
                system[pivot_column, column], system[pivot_row, column] =
                    system[pivot_row, column], system[pivot_column, column]
            end
            vector[pivot_column], vector[pivot_row] =
                vector[pivot_row], vector[pivot_column]
        end
        pivot = system[pivot_column, pivot_column]
        for column in pivot_column:number_of_unknowns
            system[pivot_column, column] /= pivot
        end
        vector[pivot_column] /= pivot
        for row in 1:number_of_unknowns
            row == pivot_column && continue
            multiplier = system[row, pivot_column]
            for column in pivot_column:number_of_unknowns
                system[row, column] -=
                    multiplier * system[pivot_column, column]
            end
            vector[row] -= multiplier * vector[pivot_column]
        end
    end
    @inbounds for column in 1:NC, row in 1:NC
        solution[row, column] = vector[row + NC * (column - 1)]
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

@inline function _kernel_hisq_project_un_pullback_core!(
    site_index, dinput, doutput, input,
    ::Val{NC}, ::Val{nw}, indexer,
) where {NC,nw}
    site = delinearize(indexer, site_index, nw)
    element_type = eltype(input)
    V = MMatrix{NC,NC,element_type}(undef)
    projected = MMatrix{NC,NC,element_type}(undef)
    hermitian = MMatrix{NC,NC,element_type}(undef)
    polar_work = MMatrix{NC,NC,element_type}(undef)
    inverse = MMatrix{NC,NC,element_type}(undef)
    next = MMatrix{NC,NC,element_type}(undef)
    output_cotangent = MMatrix{NC,NC,element_type}(undef)
    skew_rhs = MMatrix{NC,NC,element_type}(undef)
    sylvester_solution = MMatrix{NC,NC,element_type}(undef)
    gradient = MMatrix{NC,NC,element_type}(undef)
    system = MMatrix{NC * NC,NC * NC,element_type}(undef)
    vector = MVector{NC * NC,element_type}(undef)

    @inbounds for column in 1:NC, row in 1:NC
        V[row, column] = input[row, column, site...]
        output_cotangent[row, column] = doutput[row, column, site...]
    end
    _hisq_un_project_matrix!(
        projected, hermitian, V, polar_work, inverse, next, Val(NC))

    @inbounds for column in 1:NC, row in 1:NC
        value = zero(element_type)
        adjoint_value = zero(element_type)
        for contracted in 1:NC
            value += conj(projected[contracted, row]) *
                output_cotangent[contracted, column]
            adjoint_value += conj(output_cotangent[contracted, row]) *
                projected[contracted, column]
        end
        skew_rhs[row, column] = value - adjoint_value
    end
    _hisq_pullback_solve_sylvester!(
        sylvester_solution, hermitian, skew_rhs,
        system, vector, Val(NC))
    gemm!(gradient, projected, sylvester_solution)
    @inbounds for column in 1:NC, row in 1:NC
        dinput[row, column, site...] += gradient[row, column]
    end
    return nothing
end

function _hisq_project_un_pullback_accumulate!(dfat, dprojected, fat_links)
    NC = fat_links[1].NC1
    for mu in 1:4
        if NC == 3
            JACC.parallel_for(
                prod(fat_links[mu].PN),
                _kernel_hisq_project_u3_pullback_core!,
                dfat[mu].A, dprojected[mu].A, fat_links[mu].A,
                Val(fat_links[mu].nw), fat_links[mu].indexer)
        else
            JACC.parallel_for(
                prod(fat_links[mu].PN),
                _kernel_hisq_project_un_pullback_core!,
                dfat[mu].A, dprojected[mu].A, fat_links[mu].A,
                Val(NC), Val(fat_links[mu].nw), fat_links[mu].indexer)
        end
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
    reference.nw >= 3 || throw(ArgumentError(
        "HISQ link pullback requires halo width nw >= 3"))
    result_cotangent.NC1 == reference.NC1 &&
        result_cotangent.NC2 == 1 &&
        psi.NC1 == reference.NC1 && psi.NC2 == 1 || throw(ArgumentError(
            "HISQ link pullback requires one-column fermion fields matching " *
            "the thin-link color size"))
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
level-2 Fat7 smearing, U(N) projection, and level-1 Fat7 smearing. It does not
depend on Enzyme. `cache` supplies reusable temporary storage, and is refreshed
from `thin_links` when necessary. The four destination fields are not cleared,
so repeated calls accumulate. HISQ force evaluation requires `nw >= 3`.
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
            level2_coefficients, cache.fat7_pullback_workspace)

        clear_matrix!.(dFat)
        _hisq_project_un_pullback_accumulate!(
            dFat, dReunit, cache.level1_links)

        level1_coefficients = (
            one(real_type) / 8,
            one(real_type) / 16,
            one(real_type) / 64,
            one(real_type) / 384,
            zero(real_type),
        )
        _hisq_fat7_pullback_accumulate!(
            dthin_links, dFat, U, level1_coefficients,
            cache.fat7_pullback_workspace)
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
