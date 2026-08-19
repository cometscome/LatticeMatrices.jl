using StaticArrays: SMatrix

# Reuse the clover cache's temporary pools for the six field-strength
# cotangents.  Only one scratch field communicates at a time, so the primal
# communication buffers can be shared safely.
@inline function _clover_pullback_scratch_lattice(
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

function _reserve_clover_pullback_scratch(
    operator::WilsonDiracCloverOperator4D,
)
    F = operator.clover.components
    blocks = ntuple(Val(6)) do plane
        get_block(F[plane].temps)
    end
    scratch = ntuple(Val(6)) do plane
        _clover_pullback_scratch_lattice(blocks[plane][1], F[plane])
    end
    indices = ntuple(plane -> blocks[plane][2], Val(6))
    return scratch, indices
end

function _release_clover_pullback_scratch!(
    operator::WilsonDiracCloverOperator4D, indices,
)
    for plane in 1:6
        unused!(operator.clover[plane].temps, indices[plane])
    end
    return nothing
end

@inline function _kernel_wilson_clover_link_pullback_direction!(
    dU, dresult, psi, indices, indices_plus, coefficient,
    ::Val{NC}, op_plus, op_minus,
) where NC
    @inbounds for row in 1:NC
        minus_psi = mul_op(op_minus, psi, row, indices)
        for col in 1:NC
            plus_psi = mul_op(op_plus, psi, col, indices_plus)
            value = zero(eltype(dU))
            for spin in 1:4
                value += dresult[row, spin, indices...] * conj(plus_psi[spin])
                value += minus_psi[spin] *
                         conj(dresult[col, spin, indices_plus...])
            end
            dU[row, col, indices...] += coefficient * value
        end
    end
    return nothing
end

@inline function _kernel_wilson_clover_link_pullback_direction!(
    dU, dresult, psi, indices, indices_plus, coefficient,
    ::Val{3}, ::Oneγ{-1,MU}, ::Oneγ{1,MU},
) where MU
    return _wilson_halfspin_link_pullback_direction3!(
        dU, dresult, psi, indices, indices_plus, coefficient, Val(MU))
end

@inline function _kernel_wilson_clover_link_pullback!(
    site, dU1, dU2, dU3, dU4, dresult, psi, coefficient,
    ::Val{NC}, ::Val{nw}, indexer,
) where {NC,nw}
    indices = delinearize(indexer, site, nw)
    indices_1p = shiftindices(indices, shift_1p)
    indices_2p = shiftindices(indices, shift_2p)
    indices_3p = shiftindices(indices, shift_3p)
    indices_4p = shiftindices(indices, shift_4p)

    _kernel_wilson_clover_link_pullback_direction!(
        dU1, dresult, psi, indices, indices_1p, coefficient,
        Val(NC), oneminusγ1, oneplusγ1,
    )
    _kernel_wilson_clover_link_pullback_direction!(
        dU2, dresult, psi, indices, indices_2p, coefficient,
        Val(NC), oneminusγ2, oneplusγ2,
    )
    _kernel_wilson_clover_link_pullback_direction!(
        dU3, dresult, psi, indices, indices_3p, coefficient,
        Val(NC), oneminusγ3, oneplusγ3,
    )
    _kernel_wilson_clover_link_pullback_direction!(
        dU4, dresult, psi, indices, indices_4p, coefficient,
        Val(NC), oneminusγ4, oneplusγ4,
    )
    return nothing
end

@inline function _kernel_clover_field_cotangent_plane!(
    dF, dresult, psi, coefficient, gamma, x, ::Val{NC},
) where NC
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

# A path descriptor stores the positive-link origin of each factor relative to
# the clover anchor, the direction, and whether that factor is adjointed.
@generated function _clover_path_descriptor(::Val{path}) where path
    position = zeros(Int, 4)
    origins = NTuple{4,Int}[]
    directions = Int[]
    daggers = Bool[]
    for signed_direction in path
        direction = abs(signed_direction)
        signed_direction < 0 && (position[direction] -= 1)
        push!(origins, Tuple(position))
        push!(directions, direction)
        push!(daggers, signed_direction < 0)
        signed_direction > 0 && (position[direction] += 1)
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

    @inbounds for col in 1:NC, row in 1:NC
        dUmu[row, col, link_site...] += dmu[row, col]
        dUnu[row, col, link_site...] += dnu[row, col]
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
    return nothing
end

function _validate_wilson_clover_link_pullback(
    dlinks, cache, links, result_cotangent, psi,
)
    length(dlinks) == 4 || throw(ArgumentError(
        "Wilson--clover link pullback requires four destination links"))
    length(links) == 4 || throw(ArgumentError(
        "Wilson--clover link pullback requires four primal links"))
    result_cotangent.nw >= 1 || throw(ArgumentError(
        "Wilson--clover link pullback requires nw >= 1"))
    all(link -> link isa LatticeMatrix{4}, dlinks) || throw(ArgumentError(
        "Wilson--clover destination links must be four-dimensional LatticeMatrix objects"))
    all(link -> link isa LatticeMatrix{4}, links) || throw(ArgumentError(
        "Wilson--clover primal links must be four-dimensional LatticeMatrix objects"))
    all(link -> link.NC1 == result_cotangent.NC1 &&
                link.NC2 == result_cotangent.NC1 &&
                link.nw == result_cotangent.nw, dlinks) || throw(ArgumentError(
        "Wilson--clover destination links must match the fermion lattice"))
    all(link -> link.NC1 == result_cotangent.NC1 &&
                link.NC2 == result_cotangent.NC1 &&
                link.nw == result_cotangent.nw, links) || throw(ArgumentError(
        "Wilson--clover primal links must match the fermion lattice"))
    result_cotangent.NC2 == 4 && psi.NC2 == 4 || throw(ArgumentError(
        "Wilson--clover pullback requires four-spinor fields"))
    for mu in 1:4, nu in 1:4
        (dlinks[mu] === links[nu] || dlinks[mu].A === links[nu].A) &&
            throw(ArgumentError(
                "Wilson--clover link cotangents must not alias primal links"))
    end
    length(cache.wilson.U) == 4 || throw(ArgumentError(
        "Wilson--clover cache must contain four links"))
    return nothing
end

"""
    wilson_clover_link_pullback!(
        dlinks, cache, links, result_cotangent, psi; coefficient=1)

Accumulate the analytic link pullback of
`coefficient * real(dot(result_cotangent,
mul_cached_clover!(..., cache, links..., psi)))` into `dlinks`.

Both the Wilson hopping term and the four-leaf clover term are included. The
four destination fields are not cleared, so repeated calls accumulate. The
implementation is independent of automatic-differentiation packages and uses
the same cached field strength as [`mul_cached_clover!`](@ref). It requires a
halo width of at least one.
"""
function wilson_clover_link_pullback!(
    dlinks::Union{AbstractVector,Tuple},
    cache::WilsonDiracCloverOperator4D,
    links::Union{AbstractVector,Tuple},
    result_cotangent::F,
    psi::F;
    coefficient=1,
) where {F<:LatticeMatrix{4}}
    coefficient isa Real || throw(ArgumentError(
        "Wilson--clover pullback coefficient must be real"))
    _validate_wilson_clover_link_pullback(
        dlinks, cache, links, result_cotangent, psi)
    U = (links[1], links[2], links[3], links[4])
    _ensure_clover_cache_current!(cache, U...)
    ensure_halo!.(U)
    ensure_halo!(result_cotangent)
    ensure_halo!(psi)

    scale = convert(typeof(cache.wilson.κ), coefficient)
    JACC.parallel_for(
        prod(result_cotangent.PN), _kernel_wilson_clover_link_pullback!,
        dlinks[1].A, dlinks[2].A, dlinks[3].A, dlinks[4].A,
        result_cotangent.A, psi.A, -cache.wilson.κ * scale,
        Val(result_cotangent.NC1), Val(result_cotangent.nw),
        result_cotangent.indexer,
    )

    dF, scratch_indices = _reserve_clover_pullback_scratch(cache)
    try
        clear_matrix!.(dF)
        clover_coefficient = -cache.wilson.κ * cache.cSW * scale
        JACC.parallel_for(
            prod(result_cotangent.PN), _kernel_clover_field_cotangent!,
            dF[1].A, dF[2].A, dF[3].A,
            dF[4].A, dF[5].A, dF[6].A,
            result_cotangent.A, psi.A, clover_coefficient,
            Val(result_cotangent.NC1), Val(result_cotangent.nw),
            result_cotangent.indexer,
        )
        mark_halo_dirty!.(dF)
        ensure_halo!.(dF)
        _clover_links_pullback!(dlinks, U, dF, result_cotangent)
    finally
        _release_clover_pullback_scratch!(cache, scratch_indices)
    end
    mark_halo_dirty!.(dlinks)
    return dlinks
end

function wilson_clover_link_pullback!(
    dU1::LatticeMatrix{4}, dU2::LatticeMatrix{4},
    dU3::LatticeMatrix{4}, dU4::LatticeMatrix{4},
    cache::WilsonDiracCloverOperator4D,
    U1::LatticeMatrix{4}, U2::LatticeMatrix{4},
    U3::LatticeMatrix{4}, U4::LatticeMatrix{4},
    result_cotangent::F, psi::F;
    coefficient=1,
) where {F<:LatticeMatrix{4}}
    return wilson_clover_link_pullback!(
        (dU1, dU2, dU3, dU4), cache, (U1, U2, U3, U4),
        result_cotangent, psi; coefficient)
end

export wilson_clover_link_pullback!
