import LatticeMatrices: hisq_fat7_level1!, hisq_fat7_level2!,
    mark_halo_dirty!,
    _hisq_link_element, _hisq_shift_site, _hisq_row_times_oriented_link,
    _hisq_five_signs, _hisq_seven_signs, _hisq_parallel_for,
    _hisq_flat_site_index3, _hisq_flat_link_element3,
    _hisq_row_times_oriented3_flat,
    _hisq_add_row3_flat!

using StaticArrays: SVector

@inline function _hisq_smearing_vector_shadow(annotation)
    hasproperty(annotation, :dval) || return nothing
    shadow = getproperty(annotation, :dval)
    shadow isa Base.RefValue && (shadow = shadow[])
    return shadow isa Union{AbstractVector,Tuple} ? shadow : nothing
end

@inline function _hisq_fat7_path_link_offset(path, occurrence)
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

@inline function _hisq_basis_vector(index, ::Type{T}, ::Val{NC}) where {T,NC}
    return SVector{NC}(ntuple(
        component -> ifelse(component == index, one(T), zero(T)),
        Val(NC)))
end

@inline function _hisq_path_site_before(origin, path, path_index)
    site = origin
    @inbounds for index in 1:(path_index - 1)
        site = _hisq_shift_site(site, path[index])
    end
    return site
end

@inline function _hisq_oriented_link_times_column(
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

@inline function _hisq_path_left_column(
    U1, U2, U3, U4, origin, path, occurrence, column,
    ::Type{T}, ::Val{NC},
) where {T,NC}
    values = _hisq_basis_vector(column, T, Val(NC))
    @inbounds for path_index in (occurrence - 1):-1:1
        site = _hisq_path_site_before(origin, path, path_index)
        values = _hisq_oriented_link_times_column(
            U1, U2, U3, U4, site, path[path_index], values, Val(NC))
    end
    return values
end

@inline function _hisq_path_right_row(
    U1, U2, U3, U4, origin, path, occurrence, row,
    ::Type{T}, ::Val{NC},
) where {T,NC}
    values = _hisq_basis_vector(row, T, Val(NC))
    site = _hisq_path_site_before(origin, path, occurrence + 1)
    @inbounds for path_index in (occurrence + 1):length(path)
        values, site = _hisq_row_times_oriented_link(
            values, U1, U2, U3, U4, site,
            path[path_index], Val(NC))
    end
    return values
end

@inline function _hisq_fat7_path_pullback_element(
    dV1, dV2, dV3, dV4, U1, U2, U3, U4,
    target, path, coefficient, output_direction, axis, row, column,
    ::Type{T}, ::Val{NC},
) where {T,NC}
    gradient = zero(T)
    @inbounds for occurrence in eachindex(path)
        direction = path[occurrence]
        abs(direction) == axis || continue

        link_offset = _hisq_fat7_path_link_offset(path, occurrence)
        origin = ntuple(d -> target[d] - link_offset[d], 4)
        hrow = ifelse(direction > 0, row, column)
        hcolumn = ifelse(direction > 0, column, row)
        left_column = _hisq_path_left_column(
            U1, U2, U3, U4, origin, path, occurrence, hrow,
            T, Val(NC))
        right_row = _hisq_path_right_row(
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
    # Generic owner kernels write one matrix element, so follow Julia's
    # column-major row -> column -> site ordering.
    zero_based = combined_index - 1
    row = mod(zero_based, NC) + 1
    column_site_and_axis = div(zero_based, NC)
    column = mod(column_site_and_axis, NC) + 1
    site_and_axis = div(column_site_and_axis, NC)
    site_index = mod(site_and_axis, volume) + 1
    axis = div(site_and_axis, volume) + 1
    return site_index, row, column, axis
end

@inline function _hisq_add_pullback_element!(
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

@inline function _kernel_hisq_fat7_pullback_one_link!(
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
        gradient += _hisq_fat7_path_pullback_element(
            dV1, dV2, dV3, dV4, U1, U2, U3, U4,
            target, (output_direction,), coefficient,
            output_direction, axis, row, column, T, Val(NC))
    end
    _hisq_add_pullback_element!(
        dU1, dU2, dU3, dU4, axis, row, column, target, gradient)
    return nothing
end

@inline function _kernel_hisq_fat7_pullback_staple3!(
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
        gradient += _hisq_fat7_path_pullback_element(
            dV1, dV2, dV3, dV4, U1, U2, U3, U4,
            target, (nu, output_direction, -nu), coefficient,
            output_direction, axis, row, column, T, Val(NC))
        gradient += _hisq_fat7_path_pullback_element(
            dV1, dV2, dV3, dV4, U1, U2, U3, U4,
            target, (-nu, output_direction, nu), coefficient,
            output_direction, axis, row, column, T, Val(NC))
    end
    _hisq_add_pullback_element!(
        dU1, dU2, dU3, dU4, axis, row, column, target, gradient)
    return nothing
end

@inline function _kernel_hisq_fat7_pullback_staple5!(
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
            gradient += _hisq_fat7_path_pullback_element(
                dV1, dV2, dV3, dV4, U1, U2, U3, U4,
                target, path, coefficient, output_direction,
                axis, row, column, T, Val(NC))
        end
    end
    _hisq_add_pullback_element!(
        dU1, dU2, dU3, dU4, axis, row, column, target, gradient)
    return nothing
end

@inline function _kernel_hisq_fat7_pullback_staple7!(
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
            gradient += _hisq_fat7_path_pullback_element(
                dV1, dV2, dV3, dV4, U1, U2, U3, U4,
                target, path, coefficient, output_direction,
                axis, row, column, T, Val(NC))
        end
    end
    _hisq_add_pullback_element!(
        dU1, dU2, dU3, dU4, axis, row, column, target, gradient)
    return nothing
end

@inline function _kernel_hisq_fat7_pullback_lepage!(
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
        gradient += _hisq_fat7_path_pullback_element(
            dV1, dV2, dV3, dV4, U1, U2, U3, U4,
            target, (nu, nu, output_direction, -nu, -nu), coefficient,
            output_direction, axis, row, column, T, Val(NC))
        gradient += _hisq_fat7_path_pullback_element(
            dV1, dV2, dV3, dV4, U1, U2, U3, U4,
            target, (-nu, -nu, output_direction, nu, nu), coefficient,
            output_direction, axis, row, column, T, Val(NC))
    end
    _hisq_add_pullback_element!(
        dU1, dU2, dU3, dU4, axis, row, column, target, gradient)
    return nothing
end

# The generic pullback above assigns one JACC work item to one matrix element.
# That avoids atomics on every backend, but for the physical NC=3 case it also
# repeats the same path geometry and one side of the path product for all three
# columns of a row.  The flat helpers below retain the owner-thread property
# while assigning one complete row to a thread.  For a forward-oriented target
# link the left product and L' * dV are shared by all three columns; for a
# backward-oriented target link the right product and dV * R' are shared.

@inline function _hisq_basis_tuple3(index, ::Type{T}) where T
    z = zero(T)
    o = one(T)
    return (
        ifelse(index == 1, o, z),
        ifelse(index == 2, o, z),
        ifelse(index == 3, o, z),
    )
end

@inline function _hisq_oriented_link_times_column3_flat(
    U1, U2, U3, U4, site, direction, column_values, padded_size,
)
    matrix_site = ifelse(
        direction > 0, site, _hisq_shift_site(site, direction))
    site_index = _hisq_flat_site_index3(matrix_site, padded_size)
    axis = abs(direction)
    c1, c2, c3 = column_values

    if direction > 0
        return (
            muladd(_hisq_flat_link_element3(
                    U1, U2, U3, U4, axis, 1, 1, site_index), c1,
                muladd(_hisq_flat_link_element3(
                        U1, U2, U3, U4, axis, 1, 2, site_index), c2,
                    _hisq_flat_link_element3(
                        U1, U2, U3, U4, axis, 1, 3, site_index) * c3)),
            muladd(_hisq_flat_link_element3(
                    U1, U2, U3, U4, axis, 2, 1, site_index), c1,
                muladd(_hisq_flat_link_element3(
                        U1, U2, U3, U4, axis, 2, 2, site_index), c2,
                    _hisq_flat_link_element3(
                        U1, U2, U3, U4, axis, 2, 3, site_index) * c3)),
            muladd(_hisq_flat_link_element3(
                    U1, U2, U3, U4, axis, 3, 1, site_index), c1,
                muladd(_hisq_flat_link_element3(
                        U1, U2, U3, U4, axis, 3, 2, site_index), c2,
                    _hisq_flat_link_element3(
                        U1, U2, U3, U4, axis, 3, 3, site_index) * c3)),
        )
    end

    return (
        muladd(conj(_hisq_flat_link_element3(
                U1, U2, U3, U4, axis, 1, 1, site_index)), c1,
            muladd(conj(_hisq_flat_link_element3(
                    U1, U2, U3, U4, axis, 2, 1, site_index)), c2,
                conj(_hisq_flat_link_element3(
                    U1, U2, U3, U4, axis, 3, 1, site_index)) * c3)),
        muladd(conj(_hisq_flat_link_element3(
                U1, U2, U3, U4, axis, 1, 2, site_index)), c1,
            muladd(conj(_hisq_flat_link_element3(
                    U1, U2, U3, U4, axis, 2, 2, site_index)), c2,
                conj(_hisq_flat_link_element3(
                    U1, U2, U3, U4, axis, 3, 2, site_index)) * c3)),
        muladd(conj(_hisq_flat_link_element3(
                U1, U2, U3, U4, axis, 1, 3, site_index)), c1,
            muladd(conj(_hisq_flat_link_element3(
                    U1, U2, U3, U4, axis, 2, 3, site_index)), c2,
                conj(_hisq_flat_link_element3(
                    U1, U2, U3, U4, axis, 3, 3, site_index)) * c3)),
    )
end

@inline function _hisq_path_left_column3_flat(
    U1, U2, U3, U4, origin, path::NTuple{L,Int}, occurrence, column,
    padded_size,
) where L
    values = _hisq_basis_tuple3(column, eltype(U1))
    @inbounds for path_index in (occurrence - 1):-1:1
        site = _hisq_path_site_before(origin, path, path_index)
        values = _hisq_oriented_link_times_column3_flat(
            U1, U2, U3, U4, site, path[path_index], values, padded_size)
    end
    return values
end

@inline function _hisq_path_right_row3_flat(
    U1, U2, U3, U4, origin, path::NTuple{L,Int}, occurrence, row,
    padded_size,
) where L
    values = _hisq_basis_tuple3(row, eltype(U1))
    site = _hisq_path_site_before(origin, path, occurrence + 1)
    @inbounds for path_index in (occurrence + 1):L
        values, site = _hisq_row_times_oriented3_flat(
            values, U1, U2, U3, U4, site,
            path[path_index], padded_size)
    end
    return values
end

@inline function _hisq_left_adjoint_times_dv3_flat(
    dV1, dV2, dV3, dV4, output_direction, origin, left, padded_size,
)
    site_index = _hisq_flat_site_index3(origin, padded_size)
    l1, l2, l3 = left
    return (
        muladd(conj(l1), _hisq_flat_link_element3(
                dV1, dV2, dV3, dV4, output_direction, 1, 1, site_index),
            muladd(conj(l2), _hisq_flat_link_element3(
                    dV1, dV2, dV3, dV4, output_direction, 2, 1, site_index),
                conj(l3) * _hisq_flat_link_element3(
                    dV1, dV2, dV3, dV4,
                    output_direction, 3, 1, site_index))),
        muladd(conj(l1), _hisq_flat_link_element3(
                dV1, dV2, dV3, dV4, output_direction, 1, 2, site_index),
            muladd(conj(l2), _hisq_flat_link_element3(
                    dV1, dV2, dV3, dV4, output_direction, 2, 2, site_index),
                conj(l3) * _hisq_flat_link_element3(
                    dV1, dV2, dV3, dV4,
                    output_direction, 3, 2, site_index))),
        muladd(conj(l1), _hisq_flat_link_element3(
                dV1, dV2, dV3, dV4, output_direction, 1, 3, site_index),
            muladd(conj(l2), _hisq_flat_link_element3(
                    dV1, dV2, dV3, dV4, output_direction, 2, 3, site_index),
                conj(l3) * _hisq_flat_link_element3(
                    dV1, dV2, dV3, dV4,
                    output_direction, 3, 3, site_index))),
    )
end

@inline function _hisq_dv_times_right_adjoint3_flat(
    dV1, dV2, dV3, dV4, output_direction, origin, right, padded_size,
)
    site_index = _hisq_flat_site_index3(origin, padded_size)
    r1, r2, r3 = right
    return (
        muladd(_hisq_flat_link_element3(
                dV1, dV2, dV3, dV4, output_direction, 1, 1, site_index),
                conj(r1),
            muladd(_hisq_flat_link_element3(
                    dV1, dV2, dV3, dV4, output_direction, 1, 2, site_index),
                    conj(r2),
                _hisq_flat_link_element3(
                    dV1, dV2, dV3, dV4,
                    output_direction, 1, 3, site_index) * conj(r3))),
        muladd(_hisq_flat_link_element3(
                dV1, dV2, dV3, dV4, output_direction, 2, 1, site_index),
                conj(r1),
            muladd(_hisq_flat_link_element3(
                    dV1, dV2, dV3, dV4, output_direction, 2, 2, site_index),
                    conj(r2),
                _hisq_flat_link_element3(
                    dV1, dV2, dV3, dV4,
                    output_direction, 2, 3, site_index) * conj(r3))),
        muladd(_hisq_flat_link_element3(
                dV1, dV2, dV3, dV4, output_direction, 3, 1, site_index),
                conj(r1),
            muladd(_hisq_flat_link_element3(
                    dV1, dV2, dV3, dV4, output_direction, 3, 2, site_index),
                    conj(r2),
                _hisq_flat_link_element3(
                    dV1, dV2, dV3, dV4,
                    output_direction, 3, 3, site_index) * conj(r3))),
    )
end

@inline function _hisq_dot_right_adjoint3(left_times_output, right)
    v1, v2, v3 = left_times_output
    r1, r2, r3 = right
    return muladd(v1, conj(r1), muladd(v2, conj(r2), v3 * conj(r3)))
end

@inline function _hisq_left_adjoint_dot3(left, output_times_right)
    l1, l2, l3 = left
    v1, v2, v3 = output_times_right
    return muladd(conj(l1), v1, muladd(conj(l2), v2, conj(l3) * v3))
end

@inline function _hisq_fat7_path_pullback_row3_occurrence_flat(
    dV1, dV2, dV3, dV4, U1, U2, U3, U4,
    target, path::NTuple{L,Int}, output_direction, axis, row, padded_size,
    occurrence,
) where L
    z = zero(eltype(U1))
    direction = path[occurrence]
    abs(direction) == axis || return (z, z, z)

    link_offset = _hisq_fat7_path_link_offset(path, occurrence)
    origin = ntuple(d -> target[d] - link_offset[d], 4)
    if direction > 0
        left = _hisq_path_left_column3_flat(
            U1, U2, U3, U4, origin, path, occurrence, row, padded_size)
        left_times_output = _hisq_left_adjoint_times_dv3_flat(
            dV1, dV2, dV3, dV4,
            output_direction, origin, left, padded_size)
        right1 = _hisq_path_right_row3_flat(
            U1, U2, U3, U4, origin, path, occurrence, 1, padded_size)
        right2 = _hisq_path_right_row3_flat(
            U1, U2, U3, U4, origin, path, occurrence, 2, padded_size)
        right3 = _hisq_path_right_row3_flat(
            U1, U2, U3, U4, origin, path, occurrence, 3, padded_size)
        return (
            _hisq_dot_right_adjoint3(left_times_output, right1),
            _hisq_dot_right_adjoint3(left_times_output, right2),
            _hisq_dot_right_adjoint3(left_times_output, right3),
        )
    end

    right = _hisq_path_right_row3_flat(
        U1, U2, U3, U4, origin, path, occurrence, row, padded_size)
    output_times_right = _hisq_dv_times_right_adjoint3_flat(
        dV1, dV2, dV3, dV4,
        output_direction, origin, right, padded_size)
    left1 = _hisq_path_left_column3_flat(
        U1, U2, U3, U4, origin, path, occurrence, 1, padded_size)
    left2 = _hisq_path_left_column3_flat(
        U1, U2, U3, U4, origin, path, occurrence, 2, padded_size)
    left3 = _hisq_path_left_column3_flat(
        U1, U2, U3, U4, origin, path, occurrence, 3, padded_size)
    return (
        conj(_hisq_left_adjoint_dot3(left1, output_times_right)),
        conj(_hisq_left_adjoint_dot3(left2, output_times_right)),
        conj(_hisq_left_adjoint_dot3(left3, output_times_right)),
    )
end

@inline function _hisq_fat7_path_pullback_row3_flat(
    dV1, dV2, dV3, dV4, U1, U2, U3, U4,
    target, path::NTuple{L,Int}, output_direction, axis, row, padded_size,
) where L
    z = zero(eltype(U1))
    gradient = (z, z, z)
    @inbounds for occurrence in 1:L
        value = _hisq_fat7_path_pullback_row3_occurrence_flat(
            dV1, dV2, dV3, dV4, U1, U2, U3, U4,
            target, path, output_direction, axis, row, padded_size,
            occurrence)
        gradient = (
            gradient[1] + value[1],
            gradient[2] + value[2],
            gradient[3] + value[3],
        )
    end
    return gradient
end

@inline function _hisq_add_pullback_row3(accumulator, value)
    return (
        accumulator[1] + value[1],
        accumulator[2] + value[2],
        accumulator[3] + value[3],
    )
end

# The NC=3 pullback performs substantially more work per site than the forward
# row kernels.  Keeping adjacent threads on adjacent sites gives its link reads
# better locality; the forward path deliberately uses the opposite, row-fast
# ordering so its stores follow the column-major matrix layout.
@inline function _hisq_pullback_combined_row3(combined_index, volume)
    zero_based = combined_index - 1
    site_index = mod(zero_based, volume) + 1
    row_and_axis = div(zero_based, volume)
    row = mod(row_and_axis, 3) + 1
    axis = div(row_and_axis, 3) + 1
    return site_index, row, axis
end

@inline function _kernel_hisq_fat7_pullback_one_link_nc3_jacc!(
    combined_index, dU1, dU2, dU3, dU4,
    dV1, dV2, dV3, dV4, U1, U2, U3, U4,
    coefficient, volume, padded_size, ::Val{nw}, indexer,
) where nw
    site_index, row, axis = _hisq_pullback_combined_row3(
        combined_index, volume)
    target = delinearize(indexer, site_index, nw)
    gradient = _hisq_fat7_path_pullback_row3_flat(
        dV1, dV2, dV3, dV4, U1, U2, U3, U4,
        target, (axis,), axis, axis, row, padded_size)
    _hisq_add_row3_flat!(
        dU1, dU2, dU3, dU4, axis, target, row,
        gradient, coefficient, padded_size)
    return nothing
end

@inline function _kernel_hisq_fat7_pullback_staple3_nc3_jacc!(
    combined_index, dU1, dU2, dU3, dU4,
    dV1, dV2, dV3, dV4, U1, U2, U3, U4,
    coefficient, volume, padded_size, ::Val{nw}, indexer,
) where nw
    site_index, row, axis = _hisq_pullback_combined_row3(
        combined_index, volume)
    target = delinearize(indexer, site_index, nw)
    z = zero(eltype(U1))
    gradient = (z, z, z)
    @inbounds for output_direction in 1:4, nu in 1:4
        if nu != output_direction
            gradient = _hisq_add_pullback_row3(gradient,
                _hisq_fat7_path_pullback_row3_flat(
                    dV1, dV2, dV3, dV4, U1, U2, U3, U4,
                    target, (nu, output_direction, -nu),
                    output_direction, axis, row, padded_size))
            gradient = _hisq_add_pullback_row3(gradient,
                _hisq_fat7_path_pullback_row3_flat(
                    dV1, dV2, dV3, dV4, U1, U2, U3, U4,
                    target, (-nu, output_direction, nu),
                    output_direction, axis, row, padded_size))
        end
    end
    _hisq_add_row3_flat!(
        dU1, dU2, dU3, dU4, axis, target, row,
        gradient, coefficient, padded_size)
    return nothing
end

@inline function _kernel_hisq_fat7_pullback_staple5_nc3_jacc!(
    combined_index, dU1, dU2, dU3, dU4,
    dV1, dV2, dV3, dV4, U1, U2, U3, U4,
    coefficient, volume, part::Val{P}, padded_size, ::Val{nw}, indexer,
) where {P,nw}
    site_index, row, axis = _hisq_pullback_combined_row3(
        combined_index, volume)
    target = delinearize(indexer, site_index, nw)
    z = zero(eltype(U1))
    gradient = (z, z, z)
    sign_nu, sign_rho = _hisq_five_signs(part)
    @inbounds for output_direction in 1:4, nu in 1:4
        if nu != output_direction
            for rho in 1:4
                if rho != output_direction && rho != nu
                    signed_nu = sign_nu * nu
                    signed_rho = sign_rho * rho
                    path = (signed_nu, signed_rho, output_direction,
                            -signed_rho, -signed_nu)
                    gradient = _hisq_add_pullback_row3(gradient,
                        _hisq_fat7_path_pullback_row3_flat(
                            dV1, dV2, dV3, dV4, U1, U2, U3, U4,
                            target, path, output_direction,
                            axis, row, padded_size))
                end
            end
        end
    end
    _hisq_add_row3_flat!(
        dU1, dU2, dU3, dU4, axis, target, row,
        gradient, coefficient, padded_size)
    return nothing
end

@inline function _kernel_hisq_fat7_pullback_staple7_nc3_jacc!(
    combined_index, dU1, dU2, dU3, dU4,
    dV1, dV2, dV3, dV4, U1, U2, U3, U4,
    coefficient, volume, part::Val{P}, padded_size, ::Val{nw}, indexer,
) where {P,nw}
    site_index, row, axis = _hisq_pullback_combined_row3(
        combined_index, volume)
    target = delinearize(indexer, site_index, nw)
    z = zero(eltype(U1))
    gradient = (z, z, z)
    sign_nu, sign_rho, sign_sigma = _hisq_seven_signs(part)
    @inbounds for output_direction in 1:4, nu in 1:4
        if nu != output_direction
            for rho in 1:4
                if rho != output_direction && rho != nu
                    sigma = 10 - output_direction - nu - rho
                    signed_nu = sign_nu * nu
                    signed_rho = sign_rho * rho
                    signed_sigma = sign_sigma * sigma
                    path = (signed_nu, signed_rho, signed_sigma,
                            output_direction, -signed_sigma,
                            -signed_rho, -signed_nu)
                    gradient = _hisq_add_pullback_row3(gradient,
                        _hisq_fat7_path_pullback_row3_flat(
                            dV1, dV2, dV3, dV4, U1, U2, U3, U4,
                            target, path, output_direction,
                            axis, row, padded_size))
                end
            end
        end
    end
    _hisq_add_row3_flat!(
        dU1, dU2, dU3, dU4, axis, target, row,
        gradient, coefficient, padded_size)
    return nothing
end

@inline function _kernel_hisq_fat7_pullback_lepage_nc3_jacc!(
    combined_index, dU1, dU2, dU3, dU4,
    dV1, dV2, dV3, dV4, U1, U2, U3, U4,
    coefficient, volume, padded_size, ::Val{nw}, indexer,
) where nw
    site_index, row, axis = _hisq_pullback_combined_row3(
        combined_index, volume)
    target = delinearize(indexer, site_index, nw)
    z = zero(eltype(U1))
    gradient = (z, z, z)
    @inbounds for output_direction in 1:4, nu in 1:4
        if nu != output_direction
            gradient = _hisq_add_pullback_row3(gradient,
                _hisq_fat7_path_pullback_row3_flat(
                    dV1, dV2, dV3, dV4, U1, U2, U3, U4,
                    target, (nu, nu, output_direction, -nu, -nu),
                    output_direction, axis, row, padded_size))
            gradient = _hisq_add_pullback_row3(gradient,
                _hisq_fat7_path_pullback_row3_flat(
                    dV1, dV2, dV3, dV4, U1, U2, U3, U4,
                    target, (-nu, -nu, output_direction, nu, nu),
                    output_direction, axis, row, padded_size))
        end
    end
    _hisq_add_row3_flat!(
        dU1, dU2, dU3, dU4, axis, target, row,
        gradient, coefficient, padded_size)
    return nothing
end

function _hisq_fat7_pullback_nc3_jacc!(dU, dV, U, coefficients)
    coefficient_1, coefficient_3, coefficient_5, coefficient_7,
        coefficient_lepage = coefficients
    volume = prod(U[1].PN)
    combined_volume = 12 * volume
    padded_size = ntuple(d -> size(U[1].A, d + 2), 4)
    common_arguments = (
        reshape(dU[1].A, :), reshape(dU[2].A, :),
        reshape(dU[3].A, :), reshape(dU[4].A, :),
        reshape(dV[1].A, :), reshape(dV[2].A, :),
        reshape(dV[3].A, :), reshape(dV[4].A, :),
        reshape(U[1].A, :), reshape(U[2].A, :),
        reshape(U[3].A, :), reshape(U[4].A, :),
    )
    geometry_arguments = (
        volume, padded_size, Val(U[1].nw), U[1].indexer)
    _hisq_parallel_for(
        combined_volume, _kernel_hisq_fat7_pullback_one_link_nc3_jacc!,
        common_arguments..., coefficient_1, geometry_arguments...)
    _hisq_parallel_for(
        combined_volume, _kernel_hisq_fat7_pullback_staple3_nc3_jacc!,
        common_arguments..., coefficient_3, geometry_arguments...)
    for part in 1:4
        _hisq_parallel_for(
            combined_volume, _kernel_hisq_fat7_pullback_staple5_nc3_jacc!,
            common_arguments..., coefficient_5, volume, Val(part),
            padded_size, Val(U[1].nw), U[1].indexer)
    end
    for part in 1:8
        _hisq_parallel_for(
            combined_volume, _kernel_hisq_fat7_pullback_staple7_nc3_jacc!,
            common_arguments..., coefficient_7, volume, Val(part),
            padded_size, Val(U[1].nw), U[1].indexer)
    end
    if !iszero(coefficient_lepage)
        _hisq_parallel_for(
            combined_volume, _kernel_hisq_fat7_pullback_lepage_nc3_jacc!,
            common_arguments..., coefficient_lepage, geometry_arguments...)
    end
    mark_halo_dirty!.(dU)
    return nothing
end

function ER.augmented_primal(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(hisq_fat7_level1!)},
    ::Type{RT},
    fat_links::ER.Annotation{<:Union{AbstractVector,Tuple}},
    thin_links::ER.Annotation{<:Union{AbstractVector,Tuple}},
) where RT
    iszero(thin_links.val[1].nw) && throw(ArgumentError(
        "Enzyme differentiation of hisq_fat7_level1! requires nw >= 1"))
    primal_return = hisq_fat7_level1!(fat_links.val, thin_links.val)
    tape = nothing
    primal = ER.needs_primal(cfg) ? primal_return : nothing
    shadow = ER.needs_shadow(cfg) ?
        _hisq_smearing_vector_shadow(fat_links) : nothing
    RetT = ER.augmented_rule_return_type(cfg, RT, tape)
    return RetT(primal, shadow, tape)
end

function ER.reverse(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(hisq_fat7_level1!)},
    _dresult_out, _tape,
    fat_links::ER.Annotation{<:Union{AbstractVector,Tuple}},
    thin_links::ER.Annotation{<:Union{AbstractVector,Tuple}},
)
    real_type = typeof(real(zero(eltype(thin_links.val[1].A))))
    coefficients = (
        one(real_type) / 8,
        one(real_type) / 16,
        one(real_type) / 64,
        one(real_type) / 384,
        zero(real_type),
    )
    _hisq_fat7_reverse!(
        fat_links, thin_links, coefficients, "hisq_fat7_level1!")
    return (nothing, nothing)
end

function _hisq_fat7_reverse!(
    fat_links, thin_links, coefficients, operation_name,
)
    dV = _hisq_smearing_vector_shadow(fat_links)
    dV isa Union{AbstractVector,Tuple} || return nothing
    dU = _hisq_smearing_vector_shadow(thin_links)
    return _hisq_fat7_pullback!(
        dU, dV, thin_links.val, fat_links.val,
        coefficients, operation_name)
end

function _hisq_fat7_pullback!(
    dU, dV, U, V, coefficients, operation_name,
)
    length(dV) == 4 && all(link -> link isa LatticeMatrix, dV) ||
        throw(ArgumentError(
            "$operation_name output shadow must contain four lattice fields"))

    dV_views = ntuple(
        mu -> _staggered_shadow_lattice(dV[mu], V[mu]), 4)
    for mu in 1:4
        zero_halo_region!(dV[mu])
        set_halo!(dV_views[mu])
    end

    if dU isa Union{AbstractVector,Tuple}
        length(dU) == 4 && all(link -> link isa LatticeMatrix, dU) ||
            throw(ArgumentError(
                "$operation_name input shadow must contain four lattice fields"))
        coefficient_1, coefficient_3, coefficient_5, coefficient_7,
            coefficient_lepage = coefficients
        volume = prod(U[1].PN)
        NC = U[1].NC1
        if NC == 3
            _hisq_fat7_pullback_nc3_jacc!(dU, dV, U, coefficients)
            for link in dV
                _zero_shadow!(link)
                zero_halo_region!(link)
            end
            return nothing
        end
        combined_volume = 4 * NC * NC * volume
        common_arguments = (
            dU[1].A, dU[2].A, dU[3].A, dU[4].A,
            dV[1].A, dV[2].A, dV[3].A, dV[4].A,
            U[1].A, U[2].A, U[3].A, U[4].A)
        geometry_arguments = (
            volume, Val(NC), Val(U[1].nw), U[1].indexer)
        _hisq_parallel_for(
            combined_volume, _kernel_hisq_fat7_pullback_one_link!,
            common_arguments..., coefficient_1, geometry_arguments...)
        _hisq_parallel_for(
            combined_volume, _kernel_hisq_fat7_pullback_staple3!,
            common_arguments..., coefficient_3, geometry_arguments...)
        for part in 1:4
            _hisq_parallel_for(
                combined_volume, _kernel_hisq_fat7_pullback_staple5!,
                common_arguments..., coefficient_5, volume, Val(part),
                Val(NC), Val(U[1].nw), U[1].indexer)
        end
        for part in 1:8
            _hisq_parallel_for(
                combined_volume, _kernel_hisq_fat7_pullback_staple7!,
                common_arguments..., coefficient_7, volume, Val(part),
                Val(NC), Val(U[1].nw), U[1].indexer)
        end
        if !iszero(coefficient_lepage)
            _hisq_parallel_for(
                combined_volume, _kernel_hisq_fat7_pullback_lepage!,
                common_arguments..., coefficient_lepage,
                geometry_arguments...)
        end
        mark_halo_dirty!.(dU)
    end

    for link in dV
        _zero_shadow!(link)
        zero_halo_region!(link)
    end
    return nothing
end

function ER.augmented_primal(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(hisq_fat7_level2!)},
    ::Type{RT},
    fat_links::ER.Annotation{<:Union{AbstractVector,Tuple}},
    reunitarized_links::ER.Annotation{<:Union{AbstractVector,Tuple}},
    naik_epsilon::ER.Annotation,
) where RT
    reunitarized_links.val[1].nw < 2 && throw(ArgumentError(
        "Enzyme differentiation of hisq_fat7_level2! requires nw >= 2"))
    primal_return = hisq_fat7_level2!(
        fat_links.val, reunitarized_links.val, naik_epsilon.val)
    tape = nothing
    primal = ER.needs_primal(cfg) ? primal_return : nothing
    shadow = ER.needs_shadow(cfg) ?
        _hisq_smearing_vector_shadow(fat_links) : nothing
    RetT = ER.augmented_rule_return_type(cfg, RT, tape)
    return RetT(primal, shadow, tape)
end

function ER.reverse(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(hisq_fat7_level2!)},
    _dresult_out, _tape,
    fat_links::ER.Annotation{<:Union{AbstractVector,Tuple}},
    reunitarized_links::ER.Annotation{<:Union{AbstractVector,Tuple}},
    naik_epsilon::ER.Annotation,
)
    real_type = typeof(real(zero(eltype(reunitarized_links.val[1].A))))
    epsilon = convert(real_type, naik_epsilon.val)
    coefficients = (
        one(real_type) + epsilon / 8,
        one(real_type) / 16,
        one(real_type) / 64,
        one(real_type) / 384,
        -one(real_type) / 8,
    )
    epsilon_cotangent = nothing
    if naik_epsilon isa ER.Active
        dV = _hisq_smearing_vector_shadow(fat_links)
        epsilon_cotangent = if dV isa Union{AbstractVector,Tuple}
            real(
                dot(dV[1], reunitarized_links.val[1]) +
                dot(dV[2], reunitarized_links.val[2]) +
                dot(dV[3], reunitarized_links.val[3]) +
                dot(dV[4], reunitarized_links.val[4])) / 8
        else
            zero(epsilon)
        end
    end
    _hisq_fat7_reverse!(
        fat_links, reunitarized_links, coefficients,
        "hisq_fat7_level2!")
    return (nothing, nothing, epsilon_cotangent)
end
