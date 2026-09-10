"""
    StoutParameters(; rho=0.1)

Coefficient for one isotropic four-dimensional EXP/stout smearing step.  The
normalization follows Capitani--Durr--Hoelbling: at linear order in the gauge
field, `rho=0.1` corresponds to four-dimensional APE smearing with
`alpha=0.6`.
"""
struct StoutParameters{T<:AbstractFloat}
    rho::T

    function StoutParameters{T}(rho::T) where {T<:AbstractFloat}
        isfinite(rho) || throw(ArgumentError("the stout coefficient must be finite"))
        return new{T}(rho)
    end
end

StoutParameters(rho::Real) = StoutParameters{typeof(float(rho))}(float(rho))
StoutParameters(; rho=0.1) = StoutParameters(rho)

export StoutParameters

"""
    StoutSmearingCache4D(thin_links, parameters=StoutParameters())

Reusable forward intermediates and analytic-pullback workspace for one
isotropic four-dimensional EXP/stout step.  A cache belongs to one set of
thin-link objects and is not safe for concurrent use.
"""
struct StoutSmearingCache4D{T,P,S}
    staples::NTuple{4,T}
    omega::NTuple{4,T}
    exponentials::NTuple{4,T}
    staple_cotangent::NTuple{4,T}
    omega_cotangent::NTuple{4,T}
    exp_pullback::NTuple{4,T}
    scratch::NTuple{2,T}
    parameters::P
    state::S
end

function StoutSmearingCache4D(
    thin_links::Union{Vector{T},NTuple{4,T}},
    parameters::StoutParameters=StoutParameters(),
) where {T<:LatticeMatrix{4}}
    _validate_staggered_gauge_links(thin_links)
    thin_links[1].nw >= 1 || throw(ArgumentError(
        "stout smearing requires a halo width nw >= 1"))
    thin_links[1].NC1 in (2, 3) || throw(ArgumentError(
        "analytic stout smearing currently supports only SU(2) and SU(3) links"))
    real_type = typeof(real(zero(eltype(thin_links[1].A))))
    real_type <: AbstractFloat || throw(ArgumentError(
        "stout links must use floating-point matrix elements"))
    typed_parameters = StoutParameters(convert(real_type, parameters.rho))
    fields = ntuple(Val(26)) do _
        _nhyp_workspace_field(thin_links[1])
    end
    source_links = ntuple(mu -> thin_links[mu], Val(4))
    state = _NHYPCacheState(
        source_links, ntuple(_ -> UInt64(0), Val(4)), false)
    return StoutSmearingCache4D{T,typeof(typed_parameters),typeof(state)}(
        ntuple(i -> fields[i], Val(4)),
        ntuple(i -> fields[i + 4], Val(4)),
        ntuple(i -> fields[i + 8], Val(4)),
        ntuple(i -> fields[i + 12], Val(4)),
        ntuple(i -> fields[i + 16], Val(4)),
        ntuple(i -> fields[i + 20], Val(4)),
        ntuple(i -> fields[i + 24], Val(2)),
        typed_parameters,
        state,
    )
end

export StoutSmearingCache4D

function _validate_stout_forward(smeared_links, thin_links, cache)
    _validate_staggered_gauge_links(thin_links)
    _validate_nhyp_links(smeared_links, thin_links[1], "smeared_links")
    thin_links[1].nw >= 1 || throw(ArgumentError(
        "stout smearing requires a halo width nw >= 1"))
    thin_links[1].NC1 in (2, 3) || throw(ArgumentError(
        "analytic stout smearing currently supports only SU(2) and SU(3) links"))
    for output in smeared_links, input in thin_links
        _nhyp_aliases(output, input) && throw(ArgumentError(
            "stout output and thin links must not alias"))
    end
    _nhyp_same_layout(cache.staples[1], thin_links[1]) ||
        throw(ArgumentError("the stout cache has a different lattice layout"))
    return nothing
end

@inline function _stout_side_axes(mu)
    return ntuple(i -> i < mu ? i : i + 1, Val(3))
end

function _stout_build_staples!(staples, links)
    ensure_halo!.(links)
    for mu in 1:4
        side_axis1, side_axis2, side_axis3 = _stout_side_axes(mu)
        _nhyp_build_three_staples!(
            staples[mu], links[mu],
            links[side_axis1], links[mu],
            links[side_axis2], links[mu],
            links[side_axis3], links[mu],
            zero(real(eltype(links[mu].A))),
            one(real(eltype(links[mu].A))),
            side_axis1, side_axis2, side_axis3, mu)
    end
    JACC.synchronize()
    return staples
end

@inline function _record_stout_cache_state!(cache, thin_links)
    cache.state.source_links = ntuple(mu -> thin_links[mu], Val(4))
    cache.state.core_epochs = ntuple(
        mu -> thin_links[mu].halo_epoch.core, Val(4))
    cache.state.valid = true
    return cache
end

"""
    stout_smear!(smeared_links, thin_links, cache)

Apply one isotropic four-dimensional EXP/stout step,
`V_mu = exp(rho * TA(C_mu * U_mu')) * U_mu`, where `C_mu` is the sum of the
six forward and backward plaquette staples.
"""
function stout_smear!(
    smeared_links::Union{Vector{TO},NTuple{4,TO}},
    thin_links::Union{Vector{TI},NTuple{4,TI}},
    cache::StoutSmearingCache4D,
) where {TO<:LatticeMatrix{4},TI<:LatticeMatrix{4}}
    _validate_stout_forward(smeared_links, thin_links, cache)
    cache.state.valid = false
    _stout_build_staples!(cache.staples, thin_links)
    for mu in 1:4
        mul!(cache.omega[mu], cache.staples[mu], thin_links[mu]')
        expt_TA!(
            cache.exponentials[mu], cache.omega[mu], cache.parameters.rho)
        mul!(smeared_links[mu], cache.exponentials[mu], thin_links[mu])
    end
    JACC.synchronize()
    _record_stout_cache_state!(cache, thin_links)
    return smeared_links
end

function stout_smear(
    thin_links::Union{Vector{T},NTuple{4,T}},
    parameters::StoutParameters=StoutParameters(),
) where {T<:LatticeMatrix{4}}
    smeared_links = [similar(link) for link in thin_links]
    cache = StoutSmearingCache4D(thin_links, parameters)
    stout_smear!(smeared_links, thin_links, cache)
    return smeared_links, cache
end

export stout_smear!, stout_smear

function _validate_stout_cache_current(cache, thin_links)
    cache.state.valid || throw(ArgumentError(
        "stout_smear! must populate the cache before stout_pullback!"))
    for mu in 1:4
        cache.state.source_links[mu] === thin_links[mu] ||
            throw(ArgumentError("the stout cache belongs to different thin links"))
        cache.state.core_epochs[mu] == thin_links[mu].halo_epoch.core ||
            throw(ArgumentError(
                "thin link U[$mu] changed after the cached stout forward pass"))
    end
    return nothing
end

function _validate_stout_pullback(
    dthin_links, dsmeared_links, thin_links, cache,
)
    _validate_stout_forward(dsmeared_links, thin_links, cache)
    _validate_nhyp_links(dthin_links, thin_links[1], "dthin_links")
    for destination in dthin_links
        for input in thin_links
            _nhyp_aliases(destination, input) && throw(ArgumentError(
                "stout thin-link cotangents must not alias thin links"))
        end
        for source in dsmeared_links
            _nhyp_aliases(destination, source) && throw(ArgumentError(
                "stout input and output cotangents must not alias"))
        end
    end
    _validate_stout_cache_current(cache, thin_links)
    return nothing
end

"""
    stout_pullback!(dthin_links, dsmeared_links, thin_links, cache)

Apply the analytic reverse pass of a cached [`stout_smear!`](@ref) call.  The
cotangent convention is `real(sum(dot(dthin[mu], delta[mu])))`.
"""
function stout_pullback!(
    dthin_links::Union{Vector{TD},NTuple{4,TD}},
    dsmeared_links::Union{Vector{TC},NTuple{4,TC}},
    thin_links::Union{Vector{TI},NTuple{4,TI}},
    cache::StoutSmearingCache4D,
) where {
    TD<:LatticeMatrix{4},TC<:LatticeMatrix{4},TI<:LatticeMatrix{4},
}
    _validate_stout_pullback(
        dthin_links, dsmeared_links, thin_links, cache)
    clear_matrix!.(dthin_links)
    clear_matrix!.(cache.staple_cotangent)

    exp_cotangent, direct_work = cache.scratch
    for mu in 1:4
        # V = E * U: direct link term and the bilinear-trace cotangent of E.
        mul!(dthin_links[mu], cache.exponentials[mu]', dsmeared_links[mu])
        mul!(exp_cotangent, thin_links[mu], dsmeared_links[mu]')

        # exp_ta_pullback! uses tr(C*dE)=tr(R*dOmega).  Converting its result
        # back to the real Frobenius convention gives -TA(R).
        exp_ta_pullback!(
            cache.exp_pullback[mu], exp_cotangent,
            cache.omega[mu], cache.parameters.rho)
        traceless_antihermitian!(
            cache.omega_cotangent[mu], -1, cache.exp_pullback[mu])

        # Omega = C * U': dC = dOmega * U and
        # dU = dOmega' * C in the real Frobenius convention.
        mul!(
            cache.staple_cotangent[mu],
            cache.omega_cotangent[mu], thin_links[mu])
        mul!(
            direct_work, cache.omega_cotangent[mu]', cache.staples[mu])
        add_matrix!(dthin_links[mu], direct_work)
    end

    for mu in 1:4
        chain = cache.staple_cotangent[mu]
        for nu in 1:4
            nu == mu && continue
            _nhyp_staple_pullback!(
                dthin_links[nu], dthin_links[mu], chain,
                thin_links[nu], thin_links[mu], nu, mu,
                one(real(eltype(thin_links[mu].A))))
        end
    end
    return dthin_links
end

export stout_pullback!
