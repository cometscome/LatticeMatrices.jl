import LatticeMatrices: hisq_fat7_level1!, mark_halo_dirty!,
    _hisq_link_element, _hisq_shift_site, _hisq_load_oriented_link!,
    _hisq_transverse_signs, MMatrix, eye!, gemm!

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

# Accumulate the contribution of one path into the gradient of the single
# thin link owned by this kernel invocation.  Gathering all occurrences into
# one owner avoids complex-valued atomics when paths overlap on CPU or GPU.
@inline function _hisq_fat7_path_pullback_for_link!(
    dU, dV1, dV2, dV3, dV4, U1, U2, U3, U4,
    target, path, coefficient, output_direction,
    left_storage, right_storage, work_storage, link_storage,
    temporary_storage,
    ::Val{axis}, ::Val{NC},
) where {axis,NC}
    @inbounds for occurrence in eachindex(path)
        direction = path[occurrence]
        abs(direction) == axis || continue

        link_offset = _hisq_fat7_path_link_offset(path, occurrence)
        origin = ntuple(d -> target[d] - link_offset[d], 4)

        left = left_storage
        right = right_storage
        work = work_storage
        link = link_storage
        temporary = temporary_storage
        eye!(left)
        eye!(right)

        site = origin
        for path_index in eachindex(path)
            site = _hisq_load_oriented_link!(
                link, U1, U2, U3, U4, site,
                path[path_index], Val(NC))
            if path_index < occurrence
                gemm!(work, left, link)
                left, work = work, left
            elseif path_index > occurrence
                gemm!(work, right, link)
                right, work = work, right
            end
        end

        # For P = left * A * right and output cotangent G, the cotangent of
        # the oriented factor is H = left' * G * right'.
        for column in 1:NC, row in 1:NC
            value = zero(eltype(dU))
            for contracted in 1:NC
                value += conj(left[contracted, row]) *
                    _hisq_link_element(
                        dV1, dV2, dV3, dV4, output_direction,
                        contracted, column, origin)
            end
            temporary[row, column] = value
        end

        if direction > 0
            for column in 1:NC, row in 1:NC
                value = zero(eltype(dU))
                for contracted in 1:NC
                    value += temporary[row, contracted] *
                        conj(right[column, contracted])
                end
                dU[row, column, target...] += coefficient * value
            end
        else
            # A = U' for a backward-oriented factor, hence dU = H'.
            for column in 1:NC, row in 1:NC
                value = zero(eltype(dU))
                for contracted in 1:NC
                    value += temporary[column, contracted] *
                        conj(right[row, contracted])
                end
                dU[row, column, target...] += coefficient * conj(value)
            end
        end
    end
    return nothing
end

@inline function _hisq_fat7_output_pullback_for_link!(
    dU, dV1, dV2, dV3, dV4, U1, U2, U3, U4,
    target, output_direction,
    coefficient_1, coefficient_3, coefficient_5, coefficient_7,
    left, right, work, link, temporary,
    ::Val{axis}, ::Val{NC},
) where {axis,NC}
    _hisq_fat7_path_pullback_for_link!(
        dU, dV1, dV2, dV3, dV4, U1, U2, U3, U4,
        target, (output_direction,), coefficient_1, output_direction,
        left, right, work, link, temporary,
        Val(axis), Val(NC))

    for nu in 1:4
        nu == output_direction && continue
        for sign_nu in _hisq_transverse_signs
            signed_nu = sign_nu * nu
            _hisq_fat7_path_pullback_for_link!(
                dU, dV1, dV2, dV3, dV4, U1, U2, U3, U4,
                target, (signed_nu, output_direction, -signed_nu),
                coefficient_3, output_direction,
                left, right, work, link, temporary,
                Val(axis), Val(NC))
        end

        for rho in 1:4
            (rho == output_direction || rho == nu) && continue
            for sign_nu in _hisq_transverse_signs
                signed_nu = sign_nu * nu
                for sign_rho in _hisq_transverse_signs
                    signed_rho = sign_rho * rho
                    _hisq_fat7_path_pullback_for_link!(
                        dU, dV1, dV2, dV3, dV4, U1, U2, U3, U4,
                        target,
                        (signed_nu, signed_rho, output_direction,
                         -signed_rho, -signed_nu),
                        coefficient_5, output_direction,
                        left, right, work, link, temporary,
                        Val(axis), Val(NC))
                end
            end

            for sigma in 1:4
                (sigma == output_direction || sigma == nu || sigma == rho) &&
                    continue
                for sign_nu in _hisq_transverse_signs
                    signed_nu = sign_nu * nu
                    for sign_rho in _hisq_transverse_signs
                        signed_rho = sign_rho * rho
                        for sign_sigma in _hisq_transverse_signs
                            signed_sigma = sign_sigma * sigma
                            _hisq_fat7_path_pullback_for_link!(
                                dU, dV1, dV2, dV3, dV4,
                                U1, U2, U3, U4, target,
                                (signed_nu, signed_rho, signed_sigma,
                                 output_direction, -signed_sigma,
                                 -signed_rho, -signed_nu),
                                coefficient_7, output_direction,
                                left, right, work, link, temporary,
                                Val(axis), Val(NC))
                        end
                    end
                end
            end
        end
    end
    return nothing
end

@inline function _kernel_hisq_fat7_pullback!(
    site_index, dU, dV1, dV2, dV3, dV4, U1, U2, U3, U4,
    coefficient_1, coefficient_3, coefficient_5, coefficient_7,
    ::Val{axis}, ::Val{NC}, ::Val{nw}, indexer,
) where {axis,NC,nw}
    target = delinearize(indexer, site_index, nw)
    left = MMatrix{NC,NC,eltype(dU)}(undef)
    right = MMatrix{NC,NC,eltype(dU)}(undef)
    work = MMatrix{NC,NC,eltype(dU)}(undef)
    link = MMatrix{NC,NC,eltype(dU)}(undef)
    temporary = MMatrix{NC,NC,eltype(dU)}(undef)
    for output_direction in 1:4
        _hisq_fat7_output_pullback_for_link!(
            dU, dV1, dV2, dV3, dV4, U1, U2, U3, U4,
            target, output_direction,
            coefficient_1, coefficient_3, coefficient_5, coefficient_7,
            left, right, work, link, temporary,
            Val(axis), Val(NC))
    end
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
    dV = _hisq_smearing_vector_shadow(fat_links)
    dV isa AbstractVector || return (nothing, nothing)
    length(dV) == 4 && all(link -> link isa LatticeMatrix, dV) ||
        throw(ArgumentError(
            "hisq_fat7_level1! output shadow must contain four lattice fields"))

    dV_views = ntuple(
        mu -> _staggered_shadow_lattice(dV[mu], fat_links.val[mu]), 4)
    for mu in 1:4
        zero_halo_region!(dV[mu])
        set_halo!(dV_views[mu])
    end

    dU = _hisq_smearing_vector_shadow(thin_links)
    if dU isa AbstractVector
        length(dU) == 4 && all(link -> link isa LatticeMatrix, dU) ||
            throw(ArgumentError(
                "hisq_fat7_level1! input shadow must contain four lattice fields"))
        U = thin_links.val
        real_type = typeof(real(zero(eltype(U[1].A))))
        coefficients = (
            one(real_type) / 8,
            one(real_type) / 16,
            one(real_type) / 64,
            one(real_type) / 384,
        )
        for axis in 1:4
            JACC.parallel_for(
                prod(U[axis].PN), _kernel_hisq_fat7_pullback!,
                dU[axis].A,
                dV[1].A, dV[2].A, dV[3].A, dV[4].A,
                U[1].A, U[2].A, U[3].A, U[4].A,
                coefficients...,
                Val(axis), Val(U[axis].NC1), Val(U[axis].nw),
                U[axis].indexer,
            )
            mark_halo_dirty!(dU[axis])
        end
    end

    for link in dV
        _zero_shadow!(link)
        zero_halo_region!(link)
    end
    return (nothing, nothing)
end
