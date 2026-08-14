import LatticeMatrices: D5DW_MobiusDomainwallOperator5D,
    Adjoint_D5DW_MobiusDomainwallOperator5D,
    D5DW_GeneralizedDomainwallOperator5D,
    Adjoint_D5DW_GeneralizedDomainwallOperator5D, mark_halo_dirty!,
    kernel_D5DW_MobiusDomainwallOperator5D!,
    kernel_adjoint_D5DW_MobiusDomainwallOperator5D!,
    kernel_D5DW_GeneralizedDomainwallOperator5D!,
    kernel_adjoint_D5DW_GeneralizedDomainwallOperator5D!, kernel_add_4D!,
    mul_op, mul_op_1pg5, mul_op_1mg5
using StaticArrays: MMatrix

@inline function _domainwall_operator_shadow(operator)
    hasproperty(operator, :dval) || return nothing
    shadow = getproperty(operator, :dval)
    shadow isa Base.RefValue && (shadow = shadow[])
    shadow isa Adjoint_D5DW_MobiusDomainwallOperator5D &&
        (shadow = shadow.parent)
    shadow isa Adjoint_D5DW_GeneralizedDomainwallOperator5D &&
        (shadow = shadow.parent)
    return shadow isa Union{
        D5DW_MobiusDomainwallOperator5D,
        D5DW_GeneralizedDomainwallOperator5D,
    } ? shadow : nothing
end

@inline _domainwall_parent(operator::D5DW_MobiusDomainwallOperator5D) = operator
@inline _domainwall_parent(operator::Adjoint_D5DW_MobiusDomainwallOperator5D) =
    operator.parent
@inline _domainwall_parent(operator::D5DW_GeneralizedDomainwallOperator5D) =
    operator
@inline _domainwall_parent(
    operator::Adjoint_D5DW_GeneralizedDomainwallOperator5D) = operator.parent

@inline _domainwall_slice_coefficient(coefficient::Real, _) = coefficient
@inline _domainwall_slice_coefficient(coefficient::AbstractVector, s) =
    coefficient[s]

# Enzyme's structural shadow has zero cotangents in numeric metadata such as
# the boundary phases.  Halo exchange must therefore use the primal metadata
# while retaining the shadow storage.
@inline function _domainwall_shadow_lattice(
    shadow::T, primal::T,
) where {T<:LatticeMatrix}
    return T(
        primal.nw, primal.phases, primal.NC1, primal.NC2, primal.gsize,
        primal.cart, primal.coords, primal.dims, primal.nbr,
        shadow.A, shadow.buf, shadow.buf_host,
        primal.myrank, primal.PN, primal.comm, primal.indexer,
        shadow.temps, shadow.halo_epoch,
    )
end

function _validate_domainwall_ad_fields(operator, result, psi)
    result.nw > 0 || throw(ArgumentError(
        "Enzyme differentiation of D5DW_MobiusDomainwallOperator5D " *
        "requires nw >= 1"))
    length(result.PN) == 5 && length(psi.PN) == 5 || throw(ArgumentError(
        "Domain-wall pullbacks require five-dimensional fermion fields"))
    result.PN == psi.PN || throw(DimensionMismatch(
        "domain-wall result and input must have identical local extents"))
    result.nw == psi.nw || throw(DimensionMismatch(
        "domain-wall result and input must have the same halo width"))
    result.dims[5] == 1 && psi.dims[5] == 1 || throw(ArgumentError(
        "domain-wall pullbacks currently require process_grid[5] == 1"))

    L5 = typeof(operator).parameters[2]
    result.PN[5] == L5 || throw(DimensionMismatch(
        "fermion fifth extent $(result.PN[5]) does not match L5=$L5"))
    length(operator.U) == 4 || throw(ArgumentError(
        "domain-wall operator must contain four gauge links"))
    for (mu, link) in pairs(operator.U)
        link.nw == result.nw || throw(DimensionMismatch(
            "gauge link $mu and fermion fields must have the same halo width"))
        link.PN == result.PN[1:4] || throw(DimensionMismatch(
            "gauge link $mu and fermion fields have incompatible 4D extents"))
    end
    return nothing
end

function _validate_domainwall_shadow(parent, shadow)
    length(shadow.U) == 4 || throw(ArgumentError(
        "domain-wall operator shadow must contain four link fields"))
    for mu in 1:4
        primal_link = parent.U[mu]
        shadow_link = shadow.U[mu]
        shadow_link isa LatticeMatrix || throw(ArgumentError(
            "domain-wall link shadow $mu must be a LatticeMatrix"))
        shadow_link.PN == primal_link.PN || throw(DimensionMismatch(
            "domain-wall link shadow $mu has incompatible local extents"))
        shadow_link.nw == primal_link.nw || throw(DimensionMismatch(
            "domain-wall link shadow $mu has an incompatible halo width"))
        shadow_link.NC1 == primal_link.NC1 &&
            shadow_link.NC2 == primal_link.NC2 || throw(DimensionMismatch(
            "domain-wall link shadow $mu has incompatible matrix dimensions"))
    end
    if parent isa D5DW_GeneralizedDomainwallOperator5D
        shadow isa D5DW_GeneralizedDomainwallOperator5D || throw(ArgumentError(
            "a generalized domain-wall operator requires a generalized shadow"))
        L5 = typeof(parent).parameters[2]
        for name in (:a, :b, :c)
            length(getproperty(shadow, name)) == L5 || throw(DimensionMismatch(
                "domain-wall coefficient shadow $name must have length $L5"))
        end
    end
    return nothing
end

@inline function _domainwall_effective_mul_op(
    op, source, color, indices, coeff_diagonal, coeff_fifth,
    mass, ::Val{L5}, ::Val{nw},
) where {L5,nw}
    indices_5p = shiftindices(indices, LatticeMatrices.shift_5p5D)
    indices_5m = shiftindices(indices, LatticeMatrices.shift_5m5D)
    boundary_5p = ifelse(indices[5] == L5 + nw, -mass, one(mass))
    boundary_5m = ifelse(indices[5] == 1 + nw, -mass, one(mass))

    direct = mul_op(op, source, color, indices)
    projected_5p = mul_op_1pg5(op, source, color, indices_5p)
    projected_5m = mul_op_1mg5(op, source, color, indices_5m)
    return (
        coeff_diagonal * direct[1] + coeff_fifth *
            (boundary_5p * projected_5p[1] +
             boundary_5m * projected_5m[1]),
        coeff_diagonal * direct[2] + coeff_fifth *
            (boundary_5p * projected_5p[2] +
             boundary_5m * projected_5m[2]),
        coeff_diagonal * direct[3] + coeff_fifth *
            (boundary_5p * projected_5p[3] +
             boundary_5m * projected_5m[3]),
        coeff_diagonal * direct[4] + coeff_fifth *
            (boundary_5p * projected_5p[4] +
             boundary_5m * projected_5m[4]),
    )
end

@inline function _kernel_domainwall_link_pullback_direction_matrix!(
    dU, left, source, x, xplus_shift,
    coeff_scale, coeff_diagonal, coeff_fifth, mass,
    ::Val{NC}, ::Val{L5}, ::Val{nw}, op_plus, op_minus,
) where {NC,L5,nw}
    values = MMatrix{NC,NC,eltype(dU)}(undef)
    @inbounds for col in 1:NC, row in 1:NC
        values[row, col] = zero(eltype(dU))
    end
    @inbounds for s in 1:L5
        left_scale = _domainwall_slice_coefficient(coeff_scale, s)
        diagonal = _domainwall_slice_coefficient(coeff_diagonal, s)
        fifth = _domainwall_slice_coefficient(coeff_fifth, s)
        indices = (x[1], x[2], x[3], x[4], s + nw)
        indices_plus = shiftindices(indices, xplus_shift)
        plus_sources = MMatrix{4,NC,eltype(dU)}(undef)
        for col in 1:NC
            plus_source = _domainwall_effective_mul_op(
                op_plus, source, col, indices_plus,
                diagonal, fifth, mass, Val(L5), Val(nw))
            for spin in 1:4
                plus_sources[spin, col] = plus_source[spin]
            end
        end
        for row in 1:NC
            minus_source = _domainwall_effective_mul_op(
                op_minus, source, row, indices,
                diagonal, fifth, mass, Val(L5), Val(nw))
            for col in 1:NC
                value = values[row, col]
                for spin in 1:4
                    # Forward occurrence U_mu(x) in D_W at x, and the
                    # conjugated occurrence in the backward hop at x+mu.
                    value += left_scale * left[row, spin, indices...] *
                             conj(plus_sources[spin, col])
                    value += left_scale * minus_source[spin] *
                             conj(left[col, spin, indices_plus...])
                end
                values[row, col] = value
            end
        end
    end
    @inbounds for col in 1:NC, row in 1:NC
        dU[row, col, x...] += -(one(mass) / 2) * values[row, col]
    end
    return nothing
end

@inline function _kernel_domainwall_link_pullback_matrix!(
    item, dU1, dU2, dU3, dU4, left, source,
    coeff_scale, coeff_diagonal, coeff_fifth, mass,
    ::Val{NC}, ::Val{L5}, ::Val{nw}, gauge_indexer,
) where {NC,L5,nw}
    item0 = item - 1
    direction = item0 % 4 + 1
    site = item0 ÷ 4 + 1
    x = delinearize(gauge_indexer, site, nw)

    if direction == 1
        _kernel_domainwall_link_pullback_direction_matrix!(
            dU1, left, source, x, LatticeMatrices.shift_1p5D,
            coeff_scale, coeff_diagonal, coeff_fifth, mass,
            Val(NC), Val(L5), Val(nw),
            LatticeMatrices.oneminusγ1, LatticeMatrices.oneplusγ1)
    elseif direction == 2
        _kernel_domainwall_link_pullback_direction_matrix!(
            dU2, left, source, x, LatticeMatrices.shift_2p5D,
            coeff_scale, coeff_diagonal, coeff_fifth, mass,
            Val(NC), Val(L5), Val(nw),
            LatticeMatrices.oneminusγ2, LatticeMatrices.oneplusγ2)
    elseif direction == 3
        _kernel_domainwall_link_pullback_direction_matrix!(
            dU3, left, source, x, LatticeMatrices.shift_3p5D,
            coeff_scale, coeff_diagonal, coeff_fifth, mass,
            Val(NC), Val(L5), Val(nw),
            LatticeMatrices.oneminusγ3, LatticeMatrices.oneplusγ3)
    else
        _kernel_domainwall_link_pullback_direction_matrix!(
            dU4, left, source, x, LatticeMatrices.shift_4p5D,
            coeff_scale, coeff_diagonal, coeff_fifth, mass,
            Val(NC), Val(L5), Val(nw),
            LatticeMatrices.oneminusγ4, LatticeMatrices.oneplusγ4)
    end
    return nothing
end

# `temporary` contains either I-F+D_W or I-F+D_W F applied to `source`.
# Packing the base and Wilson inner products into one complex reduction keeps
# the coefficient pullback to two accelerator reductions per fifth slice.
@inline function _kernel_domainwall_coefficient_inner(
    item, left, source, temporary, s,
    mass, ::Val{NC}, ::Val{L5}, ::Val{nw}, gauge_indexer,
) where {NC,L5,nw}
    x = delinearize(gauge_indexer, item, nw)
    indices = (x[1], x[2], x[3], x[4], s + nw)
    indices_5p = shiftindices(indices, LatticeMatrices.shift_5p5D)
    indices_5m = shiftindices(indices, LatticeMatrices.shift_5m5D)
    boundary_5p = ifelse(s == L5, -mass, one(mass))
    boundary_5m = ifelse(s == 1, -mass, one(mass))
    base_inner = zero(real(eltype(source)))
    wilson_inner = zero(real(eltype(source)))

    @inbounds for color in 1:NC
        for spin in 1:4
            fifth_source = if spin <= 2
                boundary_5m * source[color, spin, indices_5m...]
            else
                boundary_5p * source[color, spin, indices_5p...]
            end
            base = source[color, spin, indices...] - fifth_source
            wilson = temporary[color, spin, indices...] - base
            left_value = left[color, spin, indices...]
            base_inner += real(conj(left_value) * base)
            wilson_inner += real(conj(left_value) * wilson)
        end
    end
    return complex(base_inner, wilson_inner)
end

@inline function _kernel_add_domainwall_coefficient_shadows!(
    s, da, db, dc, delta_a, delta_b, delta_c,
)
    @inbounds begin
        da[s] += delta_a[s]
        db[s] += delta_b[s]
        dc[s] += delta_c[s]
    end
    return nothing
end

function _accumulate_domainwall_coefficient_shadows!(
    shadow, gradient, ::Val{L5},
) where L5
    delta_a = ntuple(s -> gradient[s], Val(L5))
    delta_b = ntuple(s -> gradient[L5 + s], Val(L5))
    delta_c = ntuple(s -> gradient[2L5 + s], Val(L5))
    JACC.parallel_for(
        L5, _kernel_add_domainwall_coefficient_shadows!,
        shadow.a, shadow.b, shadow.c, delta_a, delta_b, delta_c)
    return nothing
end

function _domainwall_coefficient_pullback!(
    shadow, parent, left, source, prototype, ::Val{L5},
) where L5
    temporary, temporary_index = get_block(prototype.temps)
    U1, U2, U3, U4 = parent.U
    NC = prototype.NC1
    nw = prototype.nw
    n4 = prod(prototype.PN[1:4])
    R = typeof(parent.mass)
    local_gradient = zeros(R, 3L5)

    # K_b source = (I-F+D_W) source.
    JACC.parallel_for(
        prod(prototype.PN), kernel_D5DW_MobiusDomainwallOperator5D!,
        temporary, U1.A, U2.A, U3.A, U4.A,
        parent.mass, parent.wilson_params, source.A,
        Val(NC), Val(nw), prototype.indexer,
        Val(L5), one(R), zero(R))
    for s in 1:L5
        pair = JACC.parallel_reduce(
            n4, _kernel_domainwall_coefficient_inner,
            left.A, source.A, temporary, s,
            parent.mass, Val(NC), Val(L5), Val(nw), U1.indexer;
            init=zero(eltype(source.A)), op=+)
        local_gradient[s] = real(pair)
        local_gradient[L5 + s] = imag(pair)
    end

    # K_c source = (I-F+D_W F) source.
    JACC.parallel_for(
        prod(prototype.PN), kernel_D5DW_MobiusDomainwallOperator5D!,
        temporary, U1.A, U2.A, U3.A, U4.A,
        parent.mass, parent.wilson_params, source.A,
        Val(NC), Val(nw), prototype.indexer,
        Val(L5), zero(R), -one(R))
    for s in 1:L5
        pair = JACC.parallel_reduce(
            n4, _kernel_domainwall_coefficient_inner,
            left.A, source.A, temporary, s,
            parent.mass, Val(NC), Val(L5), Val(nw), U1.indexer;
            init=zero(eltype(source.A)), op=+)
        local_gradient[2L5 + s] = imag(pair)
    end
    unused!(prototype.temps, temporary_index)

    global_gradient = MPI.Allreduce(local_gradient, MPI.SUM, prototype.comm)
    a_host, b_host, c_host = Array(parent.a), Array(parent.b), Array(parent.c)
    @inbounds for s in 1:L5
        base = global_gradient[s]
        gb = global_gradient[L5 + s]
        gc = global_gradient[2L5 + s]
        global_gradient[s] = base + b_host[s] * gb + c_host[s] * gc
        global_gradient[L5 + s] = a_host[s] * gb
        global_gradient[2L5 + s] = a_host[s] * gc
    end
    _accumulate_domainwall_coefficient_shadows!(
        shadow, global_gradient, Val(L5))
    return nothing
end

function _domainwall_augmented_primal(cfg, ::Type{RT}, result, operator, psi) where RT
    _validate_domainwall_ad_fields(
        _domainwall_parent(operator.val), result.val, psi.val)
    primal_return = LinearAlgebra.mul!(result.val, operator.val, psi.val)
    tape = nothing
    primal = ER.needs_primal(cfg) ? primal_return : nothing
    shadow = ER.needs_shadow(cfg) ? _getshadow(result.dval) : nothing
    RetT = ER.augmented_rule_return_type(cfg, RT, tape)
    return RetT(primal, shadow, tape)
end

function ER.augmented_primal(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.mul!)},
    ::Type{RT},
    result::ER.Annotation{<:LatticeMatrix},
    operator::ER.Annotation{<:D5DW_MobiusDomainwallOperator5D},
    psi::ER.Annotation{<:LatticeMatrix},
) where RT
    return _domainwall_augmented_primal(cfg, RT, result, operator, psi)
end

function ER.augmented_primal(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.mul!)},
    ::Type{RT},
    result::ER.Annotation{<:LatticeMatrix},
    operator::ER.Annotation{<:D5DW_GeneralizedDomainwallOperator5D},
    psi::ER.Annotation{<:LatticeMatrix},
) where RT
    return _domainwall_augmented_primal(cfg, RT, result, operator, psi)
end

function ER.augmented_primal(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.mul!)},
    ::Type{RT},
    result::ER.Annotation{<:LatticeMatrix},
    operator::ER.Annotation{<:Adjoint_D5DW_GeneralizedDomainwallOperator5D},
    psi::ER.Annotation{<:LatticeMatrix},
) where RT
    return _domainwall_augmented_primal(cfg, RT, result, operator, psi)
end

@inline function _domainwall_pullback_coefficients(
    parent::D5DW_MobiusDomainwallOperator5D,
)
    return one(parent.mass), (parent.b + parent.c) / 2,
        (parent.b - parent.c) / 2
end

@inline function _domainwall_pullback_coefficients(
    parent::D5DW_GeneralizedDomainwallOperator5D,
)
    return parent.a, parent.b, parent.c
end

function ER.augmented_primal(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.mul!)},
    ::Type{RT},
    result::ER.Annotation{<:LatticeMatrix},
    operator::ER.Annotation{<:Adjoint_D5DW_MobiusDomainwallOperator5D},
    psi::ER.Annotation{<:LatticeMatrix},
) where RT
    return _domainwall_augmented_primal(cfg, RT, result, operator, psi)
end

function _domainwall_reverse!(
    adjoint_mode::Bool, dresult_out, result, operator, psi,
)
    dresult = _getshadow_out(dresult_out, result)
    dresult isa LatticeMatrix || (dresult = _getshadow(result.dval))
    dresult isa LatticeMatrix || return nothing

    parent = _domainwall_parent(operator.val)
    _validate_domainwall_ad_fields(parent, result.val, psi.val)
    L5 = typeof(parent).parameters[2]
    coeff_scale, coeff_diagonal, coeff_fifth =
        _domainwall_pullback_coefficients(parent)

    # Both the fermion input pullback and the link pullback read neighboring
    # output cotangents.  Rebuild the shadow lattice with primal metadata so
    # that complex boundary phases are applied correctly.
    zero_halo_region!(dresult)
    set_halo!(_domainwall_shadow_lattice(dresult, result.val))
    ensure_halo!(psi.val)

    operator_shadow = _domainwall_operator_shadow(operator)
    if operator_shadow !== nothing
        _validate_domainwall_shadow(parent, operator_shadow)
        dU = operator_shadow.U

        # Re(Dlambda, psi) = Re(lambda, D'psi), so the adjoint-operator
        # pullback is the same Wilson-link pullback with left/source swapped.
        left = adjoint_mode ? psi.val : dresult
        source = adjoint_mode ? dresult : psi.val
        NC = result.val.NC1
        JACC.parallel_for(
            prod(dU[1].PN) * 4,
            _kernel_domainwall_link_pullback_matrix!,
            dU[1].A, dU[2].A, dU[3].A, dU[4].A,
            left.A, source.A,
            coeff_scale, coeff_diagonal, coeff_fifth, parent.mass,
            Val(NC), Val(L5), Val(result.val.nw), dU[1].indexer,
        )
        mark_halo_dirty!.(dU)
        if parent isa D5DW_GeneralizedDomainwallOperator5D
            _domainwall_coefficient_pullback!(
                operator_shadow, parent, left, source, psi.val, Val(L5))
        end
    end

    dpsi = hasproperty(psi, :dval) ? _getshadow(psi.dval) : nothing
    if dpsi isa LatticeMatrix
        temporary, temporary_index = get_block(psi.val.temps)
        U1, U2, U3, U4 = parent.U
        if parent isa D5DW_GeneralizedDomainwallOperator5D
            kernel = adjoint_mode ?
                kernel_D5DW_GeneralizedDomainwallOperator5D! :
                kernel_adjoint_D5DW_GeneralizedDomainwallOperator5D!
            JACC.parallel_for(
                prod(psi.val.PN),
                kernel,
                temporary, U1.A, U2.A, U3.A, U4.A,
                parent.mass, parent.wilson_params, dresult.A,
                parent.a, parent.b, parent.c,
                Val(psi.val.NC1), Val(psi.val.nw), psi.val.indexer,
                Val(L5),
            )
        else
            coeff_minus = -coeff_fifth
            kernel = adjoint_mode ?
                kernel_D5DW_MobiusDomainwallOperator5D! :
                kernel_adjoint_D5DW_MobiusDomainwallOperator5D!
            JACC.parallel_for(
                prod(psi.val.PN),
                kernel,
                temporary, U1.A, U2.A, U3.A, U4.A,
                parent.mass, parent.wilson_params, dresult.A,
                Val(psi.val.NC1), Val(psi.val.nw), psi.val.indexer,
                Val(L5), coeff_diagonal, coeff_minus,
            )
        end
        JACC.parallel_for(
            prod(psi.val.PN),
            kernel_add_4D!,
            dpsi.A, temporary, dpsi.indexer,
            Val(dpsi.NC1), Val(dpsi.NC2), one(eltype(dpsi.A)), Val(dpsi.nw),
        )
        unused!(psi.val.temps, temporary_index)
        mark_halo_dirty!(dpsi)
    end

    _zero_shadow!(dresult)
    zero_halo_region!(dresult)
    return nothing
end

function ER.reverse(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.mul!)},
    dresult_out, _tape,
    result::ER.Annotation{<:LatticeMatrix},
    operator::ER.Annotation{<:D5DW_MobiusDomainwallOperator5D},
    psi::ER.Annotation{<:LatticeMatrix},
)
    _domainwall_reverse!(false, dresult_out, result, operator, psi)
    return (nothing, nothing, nothing)
end

function ER.reverse(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.mul!)},
    dresult_out, _tape,
    result::ER.Annotation{<:LatticeMatrix},
    operator::ER.Annotation{<:D5DW_GeneralizedDomainwallOperator5D},
    psi::ER.Annotation{<:LatticeMatrix},
)
    _domainwall_reverse!(false, dresult_out, result, operator, psi)
    return (nothing, nothing, nothing)
end

function ER.reverse(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.mul!)},
    dresult_out, _tape,
    result::ER.Annotation{<:LatticeMatrix},
    operator::ER.Annotation{<:Adjoint_D5DW_GeneralizedDomainwallOperator5D},
    psi::ER.Annotation{<:LatticeMatrix},
)
    _domainwall_reverse!(true, dresult_out, result, operator, psi)
    return (nothing, nothing, nothing)
end

function ER.reverse(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.mul!)},
    dresult_out, _tape,
    result::ER.Annotation{<:LatticeMatrix},
    operator::ER.Annotation{<:Adjoint_D5DW_MobiusDomainwallOperator5D},
    psi::ER.Annotation{<:LatticeMatrix},
)
    _domainwall_reverse!(true, dresult_out, result, operator, psi)
    return (nothing, nothing, nothing)
end
