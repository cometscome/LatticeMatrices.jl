using StaticArrays: SMatrix

import LatticeMatrices: WilsonDiracCloverOperator4D, HaloEpoch,
    clover_gamma_products, clover_plane_pairs,
    kernel_adjoint_WilsonDiracCloverOperator4D!

@inline function _clover_operator_shadow(operator)
    hasproperty(operator, :dval) || return nothing
    shadow = getproperty(operator, :dval)
    shadow isa Base.RefValue && (shadow = shadow[])
    return shadow isa WilsonDiracCloverOperator4D ? shadow : nothing
end

# Reuse a clover component's preallocated array as reverse scratch while
# retaining the primal communication metadata.  The buffers are safe to share:
# only one scratch field is exchanged at a time in the reverse pass.
@inline function _clover_scratch_lattice(array, primal::T) where {T<:LatticeMatrix}
    return T(
        primal.nw, primal.phases, primal.NC1, primal.NC2, primal.gsize,
        primal.cart, primal.coords, primal.dims, primal.nbr,
        array, primal.buf, primal.buf_host,
        primal.myrank, primal.PN, primal.comm, primal.indexer,
        primal.temps, HaloEpoch(),
    )
end

function _reserve_clover_pullback_scratch(operator::WilsonDiracCloverOperator4D)
    F = operator.clover.components
    a1, i1 = get_block(F[1].temps)
    a2, i2 = get_block(F[2].temps)
    a3, i3 = get_block(F[3].temps)
    a4, i4 = get_block(F[4].temps)
    a5, i5 = get_block(F[5].temps)
    a6, i6 = get_block(F[6].temps)
    scratch = (
        _clover_scratch_lattice(a1, F[1]),
        _clover_scratch_lattice(a2, F[2]),
        _clover_scratch_lattice(a3, F[3]),
        _clover_scratch_lattice(a4, F[4]),
        _clover_scratch_lattice(a5, F[5]),
        _clover_scratch_lattice(a6, F[6]),
    )
    return scratch, (i1, i2, i3, i4, i5, i6)
end

function _release_clover_pullback_scratch!(
    operator::WilsonDiracCloverOperator4D, indices)
    for plane in 1:6
        unused!(operator.clover[plane].temps, indices[plane])
    end
    return nothing
end

@inline function _kernel_clover_field_cotangent_plane!(
    dF, dresult, psi, coefficient, gamma, x, ::Val{NC}) where NC

    @inbounds for input_color in 1:NC
        for color in 1:NC
            value = zero(eltype(dF))
            for output_spin in 1:4
                transformed_psi = zero(eltype(dF))
                for input_spin in 1:4
                    transformed_psi += gamma[output_spin, input_spin] *
                                       psi[input_color, input_spin, x...]
                end
                value += dresult[color, output_spin, x...] *
                         conj(transformed_psi)
            end
            dF[color, input_color, x...] += conj(coefficient) * value
        end
    end
    return nothing
end

@inline function _kernel_clover_field_cotangent!(
    site, dF12, dF13, dF14, dF23, dF24, dF34,
    dresult, psi, coefficient, ::Val{NC}, ::Val{nw}, indexer,
) where {NC,nw}
    x = delinearize(indexer, site, nw)
    _kernel_clover_field_cotangent_plane!(
        dF12, dresult, psi, coefficient, clover_gamma_products[1], x, Val(NC))
    _kernel_clover_field_cotangent_plane!(
        dF13, dresult, psi, coefficient, clover_gamma_products[2], x, Val(NC))
    _kernel_clover_field_cotangent_plane!(
        dF14, dresult, psi, coefficient, clover_gamma_products[3], x, Val(NC))
    _kernel_clover_field_cotangent_plane!(
        dF23, dresult, psi, coefficient, clover_gamma_products[4], x, Val(NC))
    _kernel_clover_field_cotangent_plane!(
        dF24, dresult, psi, coefficient, clover_gamma_products[5], x, Val(NC))
    _kernel_clover_field_cotangent_plane!(
        dF34, dresult, psi, coefficient, clover_gamma_products[6], x, Val(NC))
    return nothing
end

# A path descriptor stores the positive link origin of each factor relative to
# the clover anchor, the direction, and whether that factor is adjointed.
@generated function _clover_path_descriptor(::Val{path}) where path
    position = zeros(Int, 4)
    origins = NTuple{4,Int}[]
    directions = Int[]
    daggers = Bool[]
    for signed_direction in path
        direction = abs(signed_direction)
        if signed_direction < 0
            position[direction] -= 1
        end
        push!(origins, Tuple(position))
        push!(directions, direction)
        push!(daggers, signed_direction < 0)
        if signed_direction > 0
            position[direction] += 1
        end
    end
    origin_expr = Expr(:tuple, (Expr(:tuple, origin...) for origin in origins)...)
    direction_expr = Expr(:tuple, directions...)
    dagger_expr = Expr(:tuple, daggers...)
    return :(($origin_expr, $direction_expr, $dagger_expr))
end

@inline function _clover_link_array(U1, U2, U3, U4, direction)
    direction == 1 && return U1
    direction == 2 && return U2
    direction == 3 && return U3
    return U4
end

@inline function _clover_factor_matrix(
    U1, U2, U3, U4, anchor, offset, direction, dagger, ::Val{NC},
) where NC
    U = _clover_link_array(U1, U2, U3, U4, direction)
    origin = ntuple(d -> anchor[d] + offset[d], Val(4))
    entries = ntuple(Val(NC * NC)) do linear_index
        row = mod(linear_index - 1, NC) + 1
        col = div(linear_index - 1, NC) + 1
        dagger ? conj(U[col, row, origin...]) : U[row, col, origin...]
    end
    return SMatrix{NC,NC}(entries)
end

@inline function _clover_projected_q_cotangent(dF, anchor, ::Val{NC}) where NC
    entries = ntuple(Val(NC * NC)) do linear_index
        row = mod(linear_index - 1, NC) + 1
        col = div(linear_index - 1, NC) + 1
        0.125 * (dF[row, col, anchor...] - conj(dF[col, row, anchor...]))
    end
    return SMatrix{NC,NC}(entries)
end

@inline function _clover_path_occurrence_pullback(
    U1, U2, U3, U4, dF, link_site,
    ::Val{path}, ::Val{K}, ::Val{NC},
) where {path,K,NC}
    origins, directions, daggers = _clover_path_descriptor(Val(path))
    anchor = ntuple(d -> link_site[d] - origins[K][d], Val(4))
    M1 = _clover_factor_matrix(
        U1, U2, U3, U4, anchor, origins[1], directions[1], daggers[1], Val(NC))
    M2 = _clover_factor_matrix(
        U1, U2, U3, U4, anchor, origins[2], directions[2], daggers[2], Val(NC))
    M3 = _clover_factor_matrix(
        U1, U2, U3, U4, anchor, origins[3], directions[3], daggers[3], Val(NC))
    M4 = _clover_factor_matrix(
        U1, U2, U3, U4, anchor, origins[4], directions[4], daggers[4], Val(NC))
    identity_matrix = one(M1)
    left = K == 1 ? identity_matrix :
           K == 2 ? M1 :
           K == 3 ? M1 * M2 : M1 * M2 * M3
    right = K == 1 ? M2 * M3 * M4 :
            K == 2 ? M3 * M4 :
            K == 3 ? M4 : identity_matrix
    dQ = _clover_projected_q_cotangent(dF, anchor, Val(NC))

    # Re tr(dQ' * left * delta(factor) * right)
    # gives the two cases below for U and U', respectively.
    return daggers[K] ? right * adjoint(dQ) * left :
                        adjoint(left) * dQ * adjoint(right)
end

@inline function _clover_path_pullback(
    U1, U2, U3, U4, dF, link_site,
    ::Val{path}, ::Val{mu}, ::Val{NC},
) where {path,mu,NC}
    zero_matrix = zero(SMatrix{NC,NC,eltype(U1)})
    dmu = zero_matrix
    dnu = zero_matrix

    h1 = _clover_path_occurrence_pullback(
        U1, U2, U3, U4, dF, link_site, Val(path), Val(1), Val(NC))
    h2 = _clover_path_occurrence_pullback(
        U1, U2, U3, U4, dF, link_site, Val(path), Val(2), Val(NC))
    h3 = _clover_path_occurrence_pullback(
        U1, U2, U3, U4, dF, link_site, Val(path), Val(3), Val(NC))
    h4 = _clover_path_occurrence_pullback(
        U1, U2, U3, U4, dF, link_site, Val(path), Val(4), Val(NC))

    dmu += abs(path[1]) == mu ? h1 : zero_matrix
    dnu += abs(path[1]) == mu ? zero_matrix : h1
    dmu += abs(path[2]) == mu ? h2 : zero_matrix
    dnu += abs(path[2]) == mu ? zero_matrix : h2
    dmu += abs(path[3]) == mu ? h3 : zero_matrix
    dnu += abs(path[3]) == mu ? zero_matrix : h3
    dmu += abs(path[4]) == mu ? h4 : zero_matrix
    dnu += abs(path[4]) == mu ? zero_matrix : h4
    return dmu, dnu
end

@inline function _kernel_clover_links_pullback!(
    site, dUmu, dUnu, U1, U2, U3, U4, dF,
    ::Val{mu}, ::Val{nu}, ::Val{NC}, ::Val{nw}, indexer,
) where {mu,nu,NC,nw}
    link_site = delinearize(indexer, site, nw)
    zero_matrix = zero(SMatrix{NC,NC,eltype(U1)})
    dmu = zero_matrix
    dnu = zero_matrix

    hmu, hnu = _clover_path_pullback(
        U1, U2, U3, U4, dF, link_site,
        Val((mu, nu, -mu, -nu)), Val(mu), Val(NC))
    dmu += hmu
    dnu += hnu
    hmu, hnu = _clover_path_pullback(
        U1, U2, U3, U4, dF, link_site,
        Val((nu, -mu, -nu, mu)), Val(mu), Val(NC))
    dmu += hmu
    dnu += hnu
    hmu, hnu = _clover_path_pullback(
        U1, U2, U3, U4, dF, link_site,
        Val((-mu, -nu, mu, nu)), Val(mu), Val(NC))
    dmu += hmu
    dnu += hnu
    hmu, hnu = _clover_path_pullback(
        U1, U2, U3, U4, dF, link_site,
        Val((-nu, mu, nu, -mu)), Val(mu), Val(NC))
    dmu += hmu
    dnu += hnu

    @inbounds for col in 1:NC
        for row in 1:NC
            dUmu[row, col, link_site...] += dmu[row, col]
            dUnu[row, col, link_site...] += dnu[row, col]
        end
    end
    return nothing
end

function _clover_links_pullback!(dU, U, dF, reference)
    U1, U2, U3, U4 = U
    for plane in 1:6
        mu, nu = clover_plane_pairs[plane]
        JACC.parallel_for(
            prod(reference.PN), _kernel_clover_links_pullback!,
            dU[mu].A, dU[nu].A, U1.A, U2.A, U3.A, U4.A, dF[plane].A,
            Val(mu), Val(nu), Val(reference.NC1), Val(reference.nw),
            reference.indexer,
        )
    end
    mark_halo_dirty!.(dU)
    return nothing
end

# The clover field strength is a cache.  Construct/update the operator before
# entering autodiff and pass the cached operator with a shadow operator (for
# example, `Duplicated(operator, shadow_operator)`).  The rule accumulates link
# cotangents directly in `shadow_operator.wilson.U`; its clover cache is unused.
function ER.augmented_primal(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.mul!)},
    ::Type{RT},
    result::ER.Annotation{<:LatticeMatrix},
    operator::ER.Annotation{<:WilsonDiracCloverOperator4D},
    psi::ER.Annotation{<:LatticeMatrix},
) where RT
    result.val.nw == 0 && throw(ArgumentError(
        "Enzyme differentiation of WilsonDiracCloverOperator4D requires nw >= 1"))
    primal_return = LinearAlgebra.mul!(result.val, operator.val, psi.val)
    scratch_tape = _clover_operator_shadow(operator) === nothing ? nothing :
                   _reserve_clover_pullback_scratch(operator.val)
    primal = ER.needs_primal(cfg) ? primal_return : nothing
    shadow = ER.needs_shadow(cfg) ? _getshadow(result.dval) : nothing
    RetT = ER.augmented_rule_return_type(cfg, RT, scratch_tape)
    return RetT(primal, shadow, scratch_tape)
end

function ER.reverse(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.mul!)},
    dresult_out, scratch_tape,
    result::ER.Annotation{<:LatticeMatrix},
    operator::ER.Annotation{<:WilsonDiracCloverOperator4D},
    psi::ER.Annotation{<:LatticeMatrix},
)
    dresult = _getshadow_out(dresult_out, result)
    dresult isa LatticeMatrix || (dresult = _getshadow(result.dval))
    dresult isa LatticeMatrix || return (nothing, nothing, nothing)

    zero_halo_region!(dresult)
    set_halo!(_wilson_shadow_lattice(dresult, result.val))

    primal = operator.val
    operator_shadow = _clover_operator_shadow(operator)
    if operator_shadow !== nothing
        dU = operator_shadow.wilson.U
        length(dU) == 4 || throw(ArgumentError(
            "WilsonDiracCloverOperator4D shadow must contain four link fields"))
        all(link -> link isa LatticeMatrix, dU) || throw(ArgumentError(
            "WilsonDiracCloverOperator4D link shadows must be LatticeMatrix objects"))

        U1, U2, U3, U4 = primal.wilson.U
        JACC.parallel_for(
            prod(result.val.PN), _kernel_wilson_link_pullback!,
            dU[1].A, dU[2].A, dU[3].A, dU[4].A,
            dresult.A, psi.val.A, -primal.wilson.κ,
            Val(result.val.NC1), Val(result.val.nw), result.val.indexer,
        )

        dF, scratch_indices = scratch_tape
        for field in dF
            _zero_shadow!(field)
            zero_halo_region!(field)
        end
        coefficient = -primal.wilson.κ * primal.cSW
        JACC.parallel_for(
            prod(result.val.PN), _kernel_clover_field_cotangent!,
            dF[1].A, dF[2].A, dF[3].A, dF[4].A, dF[5].A, dF[6].A,
            dresult.A, psi.val.A, coefficient,
            Val(result.val.NC1), Val(result.val.nw), result.val.indexer,
        )
        for field in dF
            mark_halo_dirty!(field)
            set_halo!(field)
        end
        _clover_links_pullback!(dU, primal.wilson.U, dF, result.val)
        _release_clover_pullback_scratch!(primal, scratch_indices)
    end

    dpsi = hasproperty(psi, :dval) ? _getshadow(psi.dval) : nothing
    if dpsi isa LatticeMatrix
        temporary, temporary_index = get_block(psi.val.temps)
        U1, U2, U3, U4 = primal.wilson.U
        F12, F13, F14, F23, F24, F34 = primal.clover.components
        coefficient = -primal.wilson.κ * primal.cSW
        JACC.parallel_for(
            prod(psi.val.PN), kernel_adjoint_WilsonDiracCloverOperator4D!,
            temporary, U1.A, U2.A, U3.A, U4.A, primal.wilson.κ, dresult.A,
            F12.A, F13.A, F14.A, F23.A, F24.A, F34.A, coefficient,
            Val(psi.val.NC1), Val(psi.val.nw), psi.val.indexer,
        )
        JACC.parallel_for(
            prod(psi.val.PN), kernel_add_4D!,
            dpsi.A, temporary, dpsi.indexer,
            Val(dpsi.NC1), Val(dpsi.NC2), one(eltype(dpsi.A)), Val(dpsi.nw),
        )
        unused!(psi.val.temps, temporary_index)
        mark_halo_dirty!(dpsi)
    end

    _zero_shadow!(dresult)
    zero_halo_region!(dresult)
    return (nothing, nothing, nothing)
end
