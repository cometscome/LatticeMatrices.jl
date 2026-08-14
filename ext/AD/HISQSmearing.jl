import LatticeMatrices: hisq_fat7_level1!, hisq_fat7_level2!,
    mark_halo_dirty!,
    _hisq_link_element, _hisq_shift_site, _hisq_row_times_oriented_link,
    _hisq_five_signs, _hisq_seven_signs, _hisq_parallel_for

using StaticArrays: SVector

@inline function _hisq_smearing_vector_shadow(annotation)
    hasproperty(annotation, :dval) || return nothing
    shadow = getproperty(annotation, :dval)
    shadow isa Base.RefValue && (shadow = shadow[])
    return shadow isa AbstractVector ? shadow : nothing
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
    zero_based = combined_index - 1
    site_index = mod(zero_based, volume) + 1
    matrix_element_and_axis = div(zero_based, volume)
    row = mod(matrix_element_and_axis, NC) + 1
    column_and_axis = div(matrix_element_and_axis, NC)
    column = mod(column_and_axis, NC) + 1
    axis = div(column_and_axis, NC) + 1
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

function ER.augmented_primal(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(hisq_fat7_level1!)},
    ::Type{RT},
    fat_links::ER.Annotation{<:Vector},
    thin_links::ER.Annotation{<:Vector},
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
    fat_links::ER.Annotation{<:Vector},
    thin_links::ER.Annotation{<:Vector},
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
    dV isa AbstractVector || return nothing
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
    fat_links::ER.Annotation{<:Vector},
    reunitarized_links::ER.Annotation{<:Vector},
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
    fat_links::ER.Annotation{<:Vector},
    reunitarized_links::ER.Annotation{<:Vector},
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
        epsilon_cotangent = if dV isa AbstractVector
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
