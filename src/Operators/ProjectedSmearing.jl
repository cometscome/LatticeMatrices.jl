@inline function _kernel_sun_phase_correct!(
    site_index, output, ::Val{2}, ::Val{nw}, indexer,
) where nw
    site = delinearize(indexer, site_index, nw)
    determinant =
        output[1, 1, site...] * output[2, 2, site...] -
        output[1, 2, site...] * output[2, 1, site...]
    RT = typeof(real(determinant))
    angle = atan(imag(determinant), real(determinant))
    sine, cosine = sincos(-angle / RT(2))
    phase = complex(cosine, sine)
    @inbounds for column in 1:2, row in 1:2
        output[row, column, site...] *= phase
    end
    return nothing
end

@inline function _kernel_sun_phase_correct!(
    site_index, output, ::Val{3}, ::Val{nw}, indexer,
) where nw
    site = delinearize(indexer, site_index, nw)
    a11 = output[1, 1, site...]
    a12 = output[1, 2, site...]
    a13 = output[1, 3, site...]
    a21 = output[2, 1, site...]
    a22 = output[2, 2, site...]
    a23 = output[2, 3, site...]
    a31 = output[3, 1, site...]
    a32 = output[3, 2, site...]
    a33 = output[3, 3, site...]
    determinant =
        a11 * (a22 * a33 - a23 * a32) -
        a12 * (a21 * a33 - a23 * a31) +
        a13 * (a21 * a32 - a22 * a31)
    RT = typeof(real(determinant))
    angle = atan(imag(determinant), real(determinant))
    sine, cosine = sincos(-angle / RT(3))
    phase = complex(cosine, sine)
    @inbounds for column in 1:3, row in 1:3
        output[row, column, site...] *= phase
    end
    return nothing
end

"""
    _sun_polar_project_field!(output, input)

Project each nonsingular input matrix first to U(N) by polar decomposition,
then apply the principal determinant phase to obtain SU(N).  The branch is
explicit and deterministic; unlike MaxReTr iteration, this map is suitable as
a portable forward reference but is intentionally not exposed with an HMC
pullback.
"""
function _sun_polar_project_field!(output, input)
    NC = input.NC1
    NC in (2, 3) || throw(ArgumentError(
        "principal-branch SU(N) polar projection supports only N=2 or N=3"))
    _nhyp_project_field!(output, input; sync=false)
    _parallel_for_mutating!(
        output, prod(output.PN), _kernel_sun_phase_correct!,
        output.A, Val(NC), Val(output.nw), output.indexer)
    return output
end

"""Parameters for one four-dimensional, principal-polar APE step."""
struct APEParameters{T<:AbstractFloat}
    alpha::T

    function APEParameters{T}(alpha::T) where {T<:AbstractFloat}
        isfinite(alpha) || throw(ArgumentError("the APE coefficient must be finite"))
        return new{T}(alpha)
    end
end


APEParameters(alpha::Real) = APEParameters{typeof(float(alpha))}(float(alpha))
APEParameters(; alpha=0.6) = APEParameters(alpha)

export APEParameters

struct APESmearingCache4D{T,P,S}
    unprojected::NTuple{4,T}
    parameters::P
    state::S
end

function APESmearingCache4D(
    thin_links::Union{Vector{T},NTuple{4,T}},
    parameters::APEParameters=APEParameters(),
) where {T<:LatticeMatrix{4}}
    _validate_staggered_gauge_links(thin_links)
    thin_links[1].nw >= 1 || throw(ArgumentError(
        "APE smearing requires a halo width nw >= 1"))
    thin_links[1].NC1 in (2, 3) || throw(ArgumentError(
        "APE smearing currently supports only SU(2) and SU(3) links"))
    real_type = typeof(real(zero(eltype(thin_links[1].A))))
    typed_parameters = APEParameters(convert(real_type, parameters.alpha))
    unprojected = ntuple(Val(4)) do _
        _nhyp_workspace_field(thin_links[1])
    end
    source_links = ntuple(mu -> thin_links[mu], Val(4))
    state = _NHYPCacheState(
        source_links, ntuple(_ -> UInt64(0), Val(4)), false)
    return APESmearingCache4D(unprojected, typed_parameters, state)
end

export APESmearingCache4D

function _validate_projected_forward(label, output, input, reference)
    _validate_staggered_gauge_links(input)
    _validate_nhyp_links(output, input[1], "smeared_links")
    input[1].nw >= 1 || throw(ArgumentError(
        "$label smearing requires a halo width nw >= 1"))
    input[1].NC1 in (2, 3) || throw(ArgumentError(
        "$label smearing currently supports only SU(2) and SU(3) links"))
    for destination in output, source in input
        _nhyp_aliases(destination, source) && throw(ArgumentError(
            "$label output and thin links must not alias"))
    end
    _nhyp_same_layout(reference, input[1]) || throw(ArgumentError(
        "the $label cache has a different lattice layout"))
    return nothing
end

"""
    ape_smear!(smeared_links, thin_links, cache)

Apply one four-dimensional APE step using the explicitly documented
principal-branch SU(N) polar projection.  This API is forward-only.
"""
function ape_smear!(
    smeared_links::Union{Vector{TO},NTuple{4,TO}},
    thin_links::Union{Vector{TI},NTuple{4,TI}},
    cache::APESmearingCache4D,
) where {TO<:LatticeMatrix{4},TI<:LatticeMatrix{4}}
    _validate_projected_forward(
        "APE", smeared_links, thin_links, cache.unprojected[1])
    cache.state.valid = false
    alpha = cache.parameters.alpha
    ensure_halo!.(thin_links)
    for mu in 1:4
        side_axis1, side_axis2, side_axis3 = _stout_side_axes(mu)
        _nhyp_build_three_staples!(
            cache.unprojected[mu], thin_links[mu],
            thin_links[side_axis1], thin_links[mu],
            thin_links[side_axis2], thin_links[mu],
            thin_links[side_axis3], thin_links[mu],
            1 - alpha, alpha / 6,
            side_axis1, side_axis2, side_axis3, mu)
        _sun_polar_project_field!(
            smeared_links[mu], cache.unprojected[mu])
    end
    JACC.synchronize()
    cache.state.source_links = ntuple(mu -> thin_links[mu], Val(4))
    cache.state.core_epochs = ntuple(
        mu -> thin_links[mu].halo_epoch.core, Val(4))
    cache.state.valid = true
    return smeared_links
end

function ape_smear(
    thin_links::Union{Vector{T},NTuple{4,T}},
    parameters::APEParameters=APEParameters(),
) where {T<:LatticeMatrix{4}}
    output = [similar(link) for link in thin_links]
    cache = APESmearingCache4D(thin_links, parameters)
    ape_smear!(output, thin_links, cache)
    return output, cache
end

export ape_smear!, ape_smear

"""Parameters for one four-dimensional, principal-polar HYP step."""
struct HYPParameters{T<:AbstractFloat}
    alpha_outer::T
    alpha_middle::T
    alpha_inner::T

    function HYPParameters{T}(
        alpha_outer::T, alpha_middle::T, alpha_inner::T,
    ) where {T<:AbstractFloat}
        all(isfinite, (alpha_outer, alpha_middle, alpha_inner)) ||
            throw(ArgumentError("HYP coefficients must be finite"))
        return new{T}(alpha_outer, alpha_middle, alpha_inner)
    end
end

function HYPParameters(
    alpha_outer::Real, alpha_middle::Real, alpha_inner::Real,
)
    promoted = promote(float(alpha_outer), float(alpha_middle), float(alpha_inner))
    return HYPParameters{typeof(promoted[1])}(promoted...)
end

HYPParameters(; alpha_outer=0.75, alpha_middle=0.6, alpha_inner=0.3) =
    HYPParameters(alpha_outer, alpha_middle, alpha_inner)

export HYPParameters

struct HYPSmearingCache4D{T,P,S}
    inner_unprojected::NTuple{12,T}
    inner_links::NTuple{12,T}
    middle_unprojected::NTuple{12,T}
    middle_links::NTuple{12,T}
    outer_unprojected::NTuple{4,T}
    parameters::P
    state::S
end

function HYPSmearingCache4D(
    thin_links::Union{Vector{T},NTuple{4,T}},
    parameters::HYPParameters=HYPParameters(),
) where {T<:LatticeMatrix{4}}
    _validate_staggered_gauge_links(thin_links)
    thin_links[1].nw >= 1 || throw(ArgumentError(
        "HYP smearing requires a halo width nw >= 1"))
    thin_links[1].NC1 in (2, 3) || throw(ArgumentError(
        "HYP smearing currently supports only SU(2) and SU(3) links"))
    real_type = typeof(real(zero(eltype(thin_links[1].A))))
    typed_parameters = HYPParameters(
        convert(real_type, parameters.alpha_outer),
        convert(real_type, parameters.alpha_middle),
        convert(real_type, parameters.alpha_inner),
    )
    fields = ntuple(Val(52)) do _
        _nhyp_workspace_field(thin_links[1])
    end
    source_links = ntuple(mu -> thin_links[mu], Val(4))
    state = _NHYPCacheState(
        source_links, ntuple(_ -> UInt64(0), Val(4)), false)
    return HYPSmearingCache4D(
        ntuple(i -> fields[i], Val(12)),
        ntuple(i -> fields[i + 12], Val(12)),
        ntuple(i -> fields[i + 24], Val(12)),
        ntuple(i -> fields[i + 36], Val(12)),
        ntuple(i -> fields[i + 48], Val(4)),
        typed_parameters,
        state,
    )
end

export HYPSmearingCache4D

"""
    hyp_smear!(smeared_links, thin_links, cache)

Apply one four-dimensional HYP step using the explicitly documented
principal-branch SU(N) polar projection.  This API is forward-only.
"""
function hyp_smear!(
    smeared_links::Union{Vector{TO},NTuple{4,TO}},
    thin_links::Union{Vector{TI},NTuple{4,TI}},
    cache::HYPSmearingCache4D,
) where {TO<:LatticeMatrix{4},TI<:LatticeMatrix{4}}
    _validate_projected_forward(
        "HYP", smeared_links, thin_links, cache.inner_links[1])
    cache.state.valid = false
    parameters = cache.parameters

    ensure_halo!.(thin_links)
    for (mu, nu) in _nhyp_direction_pairs
        candidate = _nhyp_pair_field(cache.inner_unprojected, mu, nu)
        projected = _nhyp_pair_field(cache.inner_links, mu, nu)
        _nhyp_build_one_staple!(
            candidate, thin_links[mu], thin_links[nu], thin_links[mu],
            1 - parameters.alpha_inner, parameters.alpha_inner / 2,
            nu, mu)
        _sun_polar_project_field!(projected, candidate)
    end
    JACC.synchronize()
    ensure_halo!.(cache.inner_links)

    for (mu, nu) in _nhyp_direction_pairs
        candidate = _nhyp_pair_field(cache.middle_unprojected, mu, nu)
        projected = _nhyp_pair_field(cache.middle_links, mu, nu)
        side_axis1, side_axis2 = _nhyp_other_two_axes(mu, nu)
        excluded_axis1 = 10 - mu - nu - side_axis1
        excluded_axis2 = 10 - mu - nu - side_axis2
        _nhyp_build_two_staples!(
            candidate, thin_links[mu],
            _nhyp_pair_field(cache.inner_links, side_axis1, excluded_axis1),
            _nhyp_pair_field(cache.inner_links, mu, excluded_axis1),
            _nhyp_pair_field(cache.inner_links, side_axis2, excluded_axis2),
            _nhyp_pair_field(cache.inner_links, mu, excluded_axis2),
            1 - parameters.alpha_middle, parameters.alpha_middle / 4,
            side_axis1, side_axis2, mu)
        _sun_polar_project_field!(projected, candidate)
    end
    JACC.synchronize()
    ensure_halo!.(cache.middle_links)

    for mu in 1:4
        candidate = cache.outer_unprojected[mu]
        side_axis1, side_axis2, side_axis3 = _stout_side_axes(mu)
        _nhyp_build_three_staples!(
            candidate, thin_links[mu],
            _nhyp_pair_field(cache.middle_links, side_axis1, mu),
            _nhyp_pair_field(cache.middle_links, mu, side_axis1),
            _nhyp_pair_field(cache.middle_links, side_axis2, mu),
            _nhyp_pair_field(cache.middle_links, mu, side_axis2),
            _nhyp_pair_field(cache.middle_links, side_axis3, mu),
            _nhyp_pair_field(cache.middle_links, mu, side_axis3),
            1 - parameters.alpha_outer, parameters.alpha_outer / 6,
            side_axis1, side_axis2, side_axis3, mu)
        _sun_polar_project_field!(smeared_links[mu], candidate)
    end
    JACC.synchronize()
    cache.state.source_links = ntuple(mu -> thin_links[mu], Val(4))
    cache.state.core_epochs = ntuple(
        mu -> thin_links[mu].halo_epoch.core, Val(4))
    cache.state.valid = true
    return smeared_links
end

function hyp_smear(
    thin_links::Union{Vector{T},NTuple{4,T}},
    parameters::HYPParameters=HYPParameters(),
) where {T<:LatticeMatrix{4}}
    output = [similar(link) for link in thin_links]
    cache = HYPSmearingCache4D(thin_links, parameters)
    hyp_smear!(output, thin_links, cache)
    return output, cache
end

export hyp_smear!, hyp_smear
