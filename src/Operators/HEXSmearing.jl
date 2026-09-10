"""
    HEXParameters(; alpha_outer=0.125, alpha_middle=0.15, alpha_inner=0.15)

Coefficients for one four-dimensional hypercubically nested EXP (HEX)
smearing step. The names describe the geometric nesting levels. The forward
map applies the standard `1/6`, `1/4`, and `1/2` geometric normalization at
the outer, middle, and inner levels, respectively.
"""
struct HEXParameters{T<:AbstractFloat}
    alpha_outer::T
    alpha_middle::T
    alpha_inner::T

    function HEXParameters{T}(
        alpha_outer::T, alpha_middle::T, alpha_inner::T,
    ) where {T<:AbstractFloat}
        all(isfinite, (alpha_outer, alpha_middle, alpha_inner)) ||
            throw(ArgumentError("HEX coefficients must be finite"))
        return new{T}(alpha_outer, alpha_middle, alpha_inner)
    end
end

function HEXParameters(
    alpha_outer::Real, alpha_middle::Real, alpha_inner::Real,
)
    promoted = promote(float(alpha_outer), float(alpha_middle), float(alpha_inner))
    return HEXParameters{typeof(promoted[1])}(promoted...)
end

HEXParameters(; alpha_outer=0.125, alpha_middle=0.15, alpha_inner=0.15) =
    HEXParameters(alpha_outer, alpha_middle, alpha_inner)

export HEXParameters

"""
    HEXSmearingCache4D(thin_links, parameters=HEXParameters())

Reusable forward intermediates and analytic-pullback workspace for one
four-dimensional HEX step.  The cache retains each restricted staple sum and
the inner and middle links.  Site-local exponential intermediates are
recomputed during the pullback to keep storage comparable to the nHYP cache.
"""
struct HEXSmearingCache4D{T,P,S}
    inner_staples::NTuple{12,T}
    inner_links::NTuple{12,T}
    middle_staples::NTuple{12,T}
    middle_links::NTuple{12,T}
    outer_staples::NTuple{4,T}
    inner_cotangent::NTuple{12,T}
    middle_cotangent::NTuple{12,T}
    scratch::NTuple{5,T}
    parameters::P
    state::S
end

function HEXSmearingCache4D(
    thin_links::Union{Vector{T},NTuple{4,T}},
    parameters::HEXParameters=HEXParameters(),
) where {T<:LatticeMatrix{4}}
    _validate_staggered_gauge_links(thin_links)
    thin_links[1].nw >= 1 || throw(ArgumentError(
        "HEX smearing requires a halo width nw >= 1"))
    thin_links[1].NC1 in (2, 3) || throw(ArgumentError(
        "HEX smearing currently supports only SU(2) and SU(3) links"))
    real_type = typeof(real(zero(eltype(thin_links[1].A))))
    real_type <: AbstractFloat || throw(ArgumentError(
        "HEX links must use floating-point matrix elements"))
    typed_parameters = HEXParameters(
        convert(real_type, parameters.alpha_outer),
        convert(real_type, parameters.alpha_middle),
        convert(real_type, parameters.alpha_inner),
    )
    fields = ntuple(Val(81)) do _
        _nhyp_workspace_field(thin_links[1])
    end
    source_links = ntuple(mu -> thin_links[mu], Val(4))
    state = _NHYPCacheState(
        source_links, ntuple(_ -> UInt64(0), Val(4)), false)
    return HEXSmearingCache4D{T,typeof(typed_parameters),typeof(state)}(
        ntuple(i -> fields[i], Val(12)),
        ntuple(i -> fields[i + 12], Val(12)),
        ntuple(i -> fields[i + 24], Val(12)),
        ntuple(i -> fields[i + 36], Val(12)),
        ntuple(i -> fields[i + 48], Val(4)),
        ntuple(i -> fields[i + 52], Val(12)),
        ntuple(i -> fields[i + 64], Val(12)),
        ntuple(i -> fields[i + 76], Val(5)),
        typed_parameters,
        state,
    )
end

export HEXSmearingCache4D

function _validate_hex_forward(smeared_links, thin_links, cache)
    _validate_staggered_gauge_links(thin_links)
    _validate_nhyp_links(smeared_links, thin_links[1], "smeared_links")
    thin_links[1].nw >= 1 || throw(ArgumentError(
        "HEX smearing requires a halo width nw >= 1"))
    thin_links[1].NC1 in (2, 3) || throw(ArgumentError(
        "HEX smearing currently supports only SU(2) and SU(3) links"))
    for output in smeared_links, input in thin_links
        _nhyp_aliases(output, input) && throw(ArgumentError(
            "HEX output and thin links must not alias"))
    end
    _nhyp_same_layout(cache.inner_links[1], thin_links[1]) ||
        throw(ArgumentError("the HEX cache has a different lattice layout"))
    return nothing
end

function _stout_retraction_forward!(
    output, central, staples, coefficient, omega, exponential,
)
    mul!(omega, staples, central')
    expt_TA!(exponential, omega, coefficient)
    mul!(output, exponential, central)
    return output
end

# Add the real-Frobenius pullback of
#     output = exp(coefficient * TA(staples * central')) * central
# to `dcentral` and return the staple cotangent in `omega` scratch storage.
function _stout_retraction_pullback_add!(
    dcentral, doutput, central, staples, coefficient, scratch,
)
    omega, exponential, temporary, exp_pullback, domega = scratch
    mul!(omega, staples, central')
    expt_TA!(exponential, omega, coefficient)

    mul!(exp_pullback, exponential', doutput)
    add_matrix!(dcentral, exp_pullback)

    mul!(temporary, central, doutput')
    exp_ta_pullback!(exp_pullback, temporary, omega, coefficient)
    traceless_antihermitian!(domega, -1, exp_pullback)

    mul!(omega, domega, central)
    mul!(temporary, domega', staples)
    add_matrix!(dcentral, temporary)
    return omega
end

@inline function _hex_build_inner_staple!(output, thin_links, mu, nu)
    _nhyp_build_one_staple!(
        output, thin_links[mu], thin_links[nu], thin_links[mu],
        zero(real(eltype(thin_links[mu].A))),
        one(real(eltype(thin_links[mu].A))),
        nu, mu)
    return output
end

@inline function _hex_build_middle_staple!(output, thin_links, inner, mu, nu)
    side_axis1, side_axis2 = _nhyp_other_two_axes(mu, nu)
    excluded_axis1 = 10 - mu - nu - side_axis1
    excluded_axis2 = 10 - mu - nu - side_axis2
    _nhyp_build_two_staples!(
        output, thin_links[mu],
        _nhyp_pair_field(inner, side_axis1, excluded_axis1),
        _nhyp_pair_field(inner, mu, excluded_axis1),
        _nhyp_pair_field(inner, side_axis2, excluded_axis2),
        _nhyp_pair_field(inner, mu, excluded_axis2),
        zero(real(eltype(thin_links[mu].A))),
        one(real(eltype(thin_links[mu].A))),
        side_axis1, side_axis2, mu)
    return output
end

@inline function _hex_build_outer_staple!(output, thin_links, middle, mu)
    side_axis1, side_axis2, side_axis3 = _stout_side_axes(mu)
    _nhyp_build_three_staples!(
        output, thin_links[mu],
        _nhyp_pair_field(middle, side_axis1, mu),
        _nhyp_pair_field(middle, mu, side_axis1),
        _nhyp_pair_field(middle, side_axis2, mu),
        _nhyp_pair_field(middle, mu, side_axis2),
        _nhyp_pair_field(middle, side_axis3, mu),
        _nhyp_pair_field(middle, mu, side_axis3),
        zero(real(eltype(thin_links[mu].A))),
        one(real(eltype(thin_links[mu].A))),
        side_axis1, side_axis2, side_axis3, mu)
    return output
end

@inline function _record_hex_cache_state!(cache, thin_links)
    cache.state.source_links = ntuple(mu -> thin_links[mu], Val(4))
    cache.state.core_epochs = ntuple(
        mu -> thin_links[mu].halo_epoch.core, Val(4))
    cache.state.valid = true
    return cache
end

"""Apply one four-dimensional hypercubically nested EXP (HEX) step."""
function hex_smear!(
    smeared_links::Union{Vector{TO},NTuple{4,TO}},
    thin_links::Union{Vector{TI},NTuple{4,TI}},
    cache::HEXSmearingCache4D,
) where {TO<:LatticeMatrix{4},TI<:LatticeMatrix{4}}
    _validate_hex_forward(smeared_links, thin_links, cache)
    cache.state.valid = false
    parameters = cache.parameters
    omega, exponential = cache.scratch[1], cache.scratch[2]

    ensure_halo!.(thin_links)
    for (mu, nu) in _nhyp_direction_pairs
        staples = _nhyp_pair_field(cache.inner_staples, mu, nu)
        output = _nhyp_pair_field(cache.inner_links, mu, nu)
        _hex_build_inner_staple!(staples, thin_links, mu, nu)
        _stout_retraction_forward!(
            output, thin_links[mu], staples,
            parameters.alpha_inner / 2, omega, exponential)
    end
    JACC.synchronize()
    ensure_halo!.(cache.inner_links)

    for (mu, nu) in _nhyp_direction_pairs
        staples = _nhyp_pair_field(cache.middle_staples, mu, nu)
        output = _nhyp_pair_field(cache.middle_links, mu, nu)
        _hex_build_middle_staple!(
            staples, thin_links, cache.inner_links, mu, nu)
        _stout_retraction_forward!(
            output, thin_links[mu], staples,
            parameters.alpha_middle / 4, omega, exponential)
    end
    JACC.synchronize()
    ensure_halo!.(cache.middle_links)

    for mu in 1:4
        staples = cache.outer_staples[mu]
        _hex_build_outer_staple!(
            staples, thin_links, cache.middle_links, mu)
        _stout_retraction_forward!(
            smeared_links[mu], thin_links[mu], staples,
            parameters.alpha_outer / 6, omega, exponential)
    end
    JACC.synchronize()
    _record_hex_cache_state!(cache, thin_links)
    return smeared_links
end

function hex_smear(
    thin_links::Union{Vector{T},NTuple{4,T}},
    parameters::HEXParameters=HEXParameters(),
) where {T<:LatticeMatrix{4}}
    smeared_links = [similar(link) for link in thin_links]
    cache = HEXSmearingCache4D(thin_links, parameters)
    hex_smear!(smeared_links, thin_links, cache)
    return smeared_links, cache
end

export hex_smear!, hex_smear

function _validate_hex_cache_current(cache, thin_links)
    cache.state.valid || throw(ArgumentError(
        "hex_smear! must populate the cache before hex_pullback!"))
    for mu in 1:4
        cache.state.source_links[mu] === thin_links[mu] ||
            throw(ArgumentError("the HEX cache belongs to different thin links"))
        cache.state.core_epochs[mu] == thin_links[mu].halo_epoch.core ||
            throw(ArgumentError(
                "thin link U[$mu] changed after the cached HEX forward pass"))
    end
    return nothing
end

function _validate_hex_pullback(
    dthin_links, dsmeared_links, thin_links, cache,
)
    _validate_hex_forward(dsmeared_links, thin_links, cache)
    _validate_nhyp_links(dthin_links, thin_links[1], "dthin_links")
    for destination in dthin_links
        for input in thin_links
            _nhyp_aliases(destination, input) && throw(ArgumentError(
                "HEX thin-link cotangents must not alias thin links"))
        end
        for source in dsmeared_links
            _nhyp_aliases(destination, source) && throw(ArgumentError(
                "HEX input and output cotangents must not alias"))
        end
    end
    _validate_hex_cache_current(cache, thin_links)
    return nothing
end

"""Apply the analytic reverse pass of a cached [`hex_smear!`](@ref) call."""
function hex_pullback!(
    dthin_links::Union{Vector{TD},NTuple{4,TD}},
    dsmeared_links::Union{Vector{TC},NTuple{4,TC}},
    thin_links::Union{Vector{TI},NTuple{4,TI}},
    cache::HEXSmearingCache4D,
) where {
    TD<:LatticeMatrix{4},TC<:LatticeMatrix{4},TI<:LatticeMatrix{4},
}
    _validate_hex_pullback(dthin_links, dsmeared_links, thin_links, cache)
    clear_matrix!.(dthin_links)
    clear_matrix!.(cache.inner_cotangent)
    clear_matrix!.(cache.middle_cotangent)
    parameters = cache.parameters

    for mu in 1:4
        dstaples = _stout_retraction_pullback_add!(
            dthin_links[mu], dsmeared_links[mu], thin_links[mu],
            cache.outer_staples[mu], parameters.alpha_outer / 6,
            cache.scratch)
        for nu in 1:4
            nu == mu && continue
            _nhyp_staple_pullback!(
                _nhyp_pair_field(cache.middle_cotangent, nu, mu),
                _nhyp_pair_field(cache.middle_cotangent, mu, nu),
                dstaples,
                _nhyp_pair_field(cache.middle_links, nu, mu),
                _nhyp_pair_field(cache.middle_links, mu, nu),
                nu, mu, one(real(eltype(thin_links[mu].A))))
        end
    end

    for (mu, nu) in _nhyp_direction_pairs
        dstaples = _stout_retraction_pullback_add!(
            dthin_links[mu],
            _nhyp_pair_field(cache.middle_cotangent, mu, nu),
            thin_links[mu],
            _nhyp_pair_field(cache.middle_staples, mu, nu),
            parameters.alpha_middle / 4, cache.scratch)
        for side_axis in 1:4
            (side_axis == mu || side_axis == nu) && continue
            excluded_axis = 10 - mu - nu - side_axis
            _nhyp_staple_pullback!(
                _nhyp_pair_field(
                    cache.inner_cotangent, side_axis, excluded_axis),
                _nhyp_pair_field(cache.inner_cotangent, mu, excluded_axis),
                dstaples,
                _nhyp_pair_field(cache.inner_links, side_axis, excluded_axis),
                _nhyp_pair_field(cache.inner_links, mu, excluded_axis),
                side_axis, mu, one(real(eltype(thin_links[mu].A))))
        end
    end

    for (mu, nu) in _nhyp_direction_pairs
        dstaples = _stout_retraction_pullback_add!(
            dthin_links[mu],
            _nhyp_pair_field(cache.inner_cotangent, mu, nu),
            thin_links[mu],
            _nhyp_pair_field(cache.inner_staples, mu, nu),
            parameters.alpha_inner / 2, cache.scratch)
        _nhyp_staple_pullback!(
            dthin_links[nu], dthin_links[mu], dstaples,
            thin_links[nu], thin_links[mu], nu, mu,
            one(real(eltype(thin_links[mu].A))))
    end
    return dthin_links
end

export hex_pullback!
