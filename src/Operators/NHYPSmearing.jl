const _nhyp_direction_pairs = (
    (1, 2), (1, 3), (1, 4),
    (2, 1), (2, 3), (2, 4),
    (3, 1), (3, 2), (3, 4),
    (4, 1), (4, 2), (4, 3),
)

@inline function _nhyp_pair_index(mu::Integer, nu::Integer)
    return 3 * (mu - 1) + nu - ifelse(nu > mu, 1, 0)
end

@inline _nhyp_pair_field(fields, mu, nu) =
    fields[_nhyp_pair_index(mu, nu)]

"""
    NHYPParameters(; alpha_outer=0.5, alpha_middle=0.5, alpha_inner=0.4)

Coefficients for normalized HYP (nHYP) smearing.  The names describe the
geometric nesting level and deliberately avoid the ambiguous `alpha1` /
`alpha2` / `alpha3` convention.  QEX's `(alpha1, alpha2, alpha3)` corresponds
to `(alpha_inner, alpha_middle, alpha_outer)` here.
"""
struct NHYPParameters{T<:AbstractFloat}
    alpha_outer::T
    alpha_middle::T
    alpha_inner::T

    function NHYPParameters{T}(
        alpha_outer::T, alpha_middle::T, alpha_inner::T,
    ) where {T<:AbstractFloat}
        all(isfinite, (alpha_outer, alpha_middle, alpha_inner)) ||
            throw(ArgumentError("nHYP coefficients must be finite"))
        return new{T}(alpha_outer, alpha_middle, alpha_inner)
    end
end

function NHYPParameters(
    alpha_outer::Real, alpha_middle::Real, alpha_inner::Real,
)
    promoted = promote(float(alpha_outer), float(alpha_middle), float(alpha_inner))
    return NHYPParameters{typeof(promoted[1])}(promoted...)
end

NHYPParameters(; alpha_outer=0.5, alpha_middle=0.5, alpha_inner=0.4) =
    NHYPParameters(alpha_outer, alpha_middle, alpha_inner)

export NHYPParameters

mutable struct _NHYPCacheState{T}
    source_links::NTuple{4,T}
    core_epochs::NTuple{4,UInt64}
    valid::Bool
end

"""
    NHYPSmearingCache4D(thin_links, parameters=NHYPParameters())

Reusable forward intermediates and pullback scratch for four-dimensional
nHYP smearing.  The cache retains the unprojected and U(N)-projected inner
and middle links, and the unprojected outer links required by the HMC force.
It is not safe to use one cache concurrently from multiple tasks.
"""
struct NHYPSmearingCache4D{T,P}
    inner_unprojected::NTuple{12,T}
    inner_links::NTuple{12,T}
    middle_unprojected::NTuple{12,T}
    middle_links::NTuple{12,T}
    outer_unprojected::NTuple{4,T}
    inner_cotangent::NTuple{12,T}
    middle_cotangent::NTuple{12,T}
    outer_cotangent::NTuple{4,T}
    parameters::P
    state::_NHYPCacheState{T}
end

@inline function _nhyp_workspace_field(reference::T) where {T<:LatticeMatrix{4}}
    return _lattice_alias_with_array(
        reference, similar(reference.A); halo_epoch=HaloEpoch())
end

function NHYPSmearingCache4D(
    thin_links::Union{Vector{T},NTuple{4,T}},
    parameters::NHYPParameters=NHYPParameters(),
) where {T<:LatticeMatrix{4}}
    _validate_staggered_gauge_links(thin_links)
    thin_links[1].nw >= 1 || throw(ArgumentError(
        "nHYP smearing requires a halo width nw >= 1"))

    real_type = typeof(real(zero(eltype(thin_links[1].A))))
    real_type <: AbstractFloat || throw(ArgumentError(
        "nHYP links must use floating-point matrix elements"))
    typed_parameters = NHYPParameters(
        convert(real_type, parameters.alpha_outer),
        convert(real_type, parameters.alpha_middle),
        convert(real_type, parameters.alpha_inner),
    )
    fields = ntuple(Val(80)) do _
        _nhyp_workspace_field(thin_links[1])
    end
    source_links = ntuple(mu -> thin_links[mu], Val(4))
    state = _NHYPCacheState(
        source_links, ntuple(_ -> UInt64(0), Val(4)), false)
    return NHYPSmearingCache4D{T,typeof(typed_parameters)}(
        ntuple(i -> fields[i], Val(12)),
        ntuple(i -> fields[i + 12], Val(12)),
        ntuple(i -> fields[i + 24], Val(12)),
        ntuple(i -> fields[i + 36], Val(12)),
        ntuple(i -> fields[i + 48], Val(4)),
        ntuple(i -> fields[i + 52], Val(12)),
        ntuple(i -> fields[i + 64], Val(12)),
        ntuple(i -> fields[i + 76], Val(4)),
        typed_parameters,
        state,
    )
end

export NHYPSmearingCache4D

@inline function _nhyp_same_layout(first, second)
    return first.NC1 == second.NC1 && first.NC2 == second.NC2 &&
        first.gsize == second.gsize && first.PN == second.PN &&
        first.dims == second.dims && first.nw == second.nw &&
        first.phases == second.phases && eltype(first.A) == eltype(second.A)
end

@inline _nhyp_aliases(first, second) =
    first === second || first.A === second.A

function _validate_nhyp_links(collection, reference, label)
    length(collection) == 4 || throw(ArgumentError(
        "$label must contain four link fields"))
    for (mu, link) in enumerate(collection)
        link isa LatticeMatrix{4} || throw(ArgumentError(
            "$label[$mu] must be a four-dimensional LatticeMatrix"))
        _nhyp_same_layout(link, reference) || throw(ArgumentError(
            "$label[$mu] does not match the nHYP lattice layout"))
    end
    return nothing
end

function _validate_nhyp_forward(smeared_links, thin_links, cache)
    _validate_staggered_gauge_links(thin_links)
    _validate_nhyp_links(smeared_links, thin_links[1], "smeared_links")
    thin_links[1].nw >= 1 || throw(ArgumentError(
        "nHYP smearing requires a halo width nw >= 1"))
    for output in smeared_links, input in thin_links
        _nhyp_aliases(output, input) && throw(ArgumentError(
            "nHYP output and thin links must not alias"))
    end
    for first_mu in 1:4, second_mu in (first_mu + 1):4
        _nhyp_aliases(
            smeared_links[first_mu], smeared_links[second_mu]) &&
            throw(ArgumentError("nHYP output links must not alias each other"))
    end
    primal_cache_fields = (
        cache.inner_unprojected..., cache.inner_links...,
        cache.middle_unprojected..., cache.middle_links...,
        cache.outer_unprojected...,
    )
    for output in smeared_links, cached in primal_cache_fields
        _nhyp_aliases(output, cached) && throw(ArgumentError(
            "nHYP output links must not alias cache intermediates"))
    end
    _nhyp_same_layout(cache.inner_links[1], thin_links[1]) ||
        throw(ArgumentError("the nHYP cache has a different lattice layout"))
    return nothing
end

@inline function _kernel_nhyp_scale_copy!(
    site_index, output, input, coefficient, ::Val{NC}, ::Val{nw}, indexer,
) where {NC,nw}
    site = delinearize(indexer, site_index, nw)
    @inbounds for column in 1:NC, row in 1:NC
        output[row, column, site...] = coefficient * input[row, column, site...]
    end
    return nothing
end

@inline function _nhyp_load_matrix!(
    matrix, field, site, ::Val{NC}, ::Val{adjoint},
) where {NC,adjoint}
    @inbounds for column in 1:NC, row in 1:NC
        matrix[row, column] = ifelse(
            adjoint,
            conj(field[column, row, site...]),
            field[row, column, site...],
        )
    end
    return matrix
end

@inline function _nhyp_zero_matrix!(matrix, ::Val{NC}) where NC
    @inbounds for column in 1:NC, row in 1:NC
        matrix[row, column] = zero(eltype(matrix))
    end
    return matrix
end

@inline function _nhyp_accumulate_matrix!(destination, source, ::Val{NC}) where NC
    @inbounds for column in 1:NC, row in 1:NC
        destination[row, column] += source[row, column]
    end
    return destination
end

@inline function _nhyp_add_to_field!(
    field, matrix, site, coefficient, ::Val{NC},
) where NC
    @inbounds for column in 1:NC, row in 1:NC
        field[row, column, site...] += coefficient * matrix[row, column]
    end
    return nothing
end

@inline function _kernel_nhyp_staple_add!(
    site_index, output, side, middle, side_axis, middle_axis, coefficient,
    ::Val{NC}, ::Val{nw}, indexer,
) where {NC,nw}
    origin = delinearize(indexer, site_index, nw)
    origin_plus_side = _hisq_shift_site(origin, side_axis)
    origin_plus_middle = _hisq_shift_site(origin, middle_axis)
    origin_minus_side = _hisq_shift_site(origin, -side_axis)
    minus_side_plus_middle = _hisq_shift_site(
        origin_minus_side, middle_axis)

    element_type = eltype(output)
    first = MMatrix{NC,NC,element_type}(undef)
    second = MMatrix{NC,NC,element_type}(undef)
    third = MMatrix{NC,NC,element_type}(undef)
    temporary = MMatrix{NC,NC,element_type}(undef)
    staple = MMatrix{NC,NC,element_type}(undef)
    term = MMatrix{NC,NC,element_type}(undef)

    _nhyp_load_matrix!(first, side, origin, Val(NC), Val(false))
    _nhyp_load_matrix!(second, middle, origin_plus_side, Val(NC), Val(false))
    _nhyp_load_matrix!(third, side, origin_plus_middle, Val(NC), Val(true))
    gemm!(temporary, first, second)
    gemm!(staple, temporary, third)

    _nhyp_load_matrix!(first, side, origin_minus_side, Val(NC), Val(true))
    _nhyp_load_matrix!(second, middle, origin_minus_side, Val(NC), Val(false))
    _nhyp_load_matrix!(third, side, minus_side_plus_middle, Val(NC), Val(false))
    gemm!(temporary, first, second)
    gemm!(term, temporary, third)
    _nhyp_accumulate_matrix!(staple, term, Val(NC))
    _nhyp_add_to_field!(output, staple, origin, coefficient, Val(NC))
    return nothing
end

function _nhyp_scale_copy!(output, input, coefficient)
    _parallel_for_mutating!(
        output, prod(output.PN), _kernel_nhyp_scale_copy!,
        output.A, input.A, coefficient, Val(output.NC1), Val(output.nw),
        output.indexer)
    return output
end

function _nhyp_staple_add!(
    output, side, middle, side_axis, middle_axis, coefficient,
)
    ensure_halo!(side)
    ensure_halo!(middle)
    _parallel_for_mutating!(
        output, prod(output.PN), _kernel_nhyp_staple_add!,
        output.A, side.A, middle.A, side_axis, middle_axis, coefficient,
        Val(output.NC1), Val(output.nw), output.indexer)
    return output
end

function _nhyp_project_field!(output, input)
    NC = input.NC1
    if NC == 3
        _parallel_for_mutating!(
            output, prod(output.PN), kernel_hisq_project_u3!,
            output.A, input.A, Val(input.nw), input.indexer)
    else
        _parallel_for_mutating!(
            output, prod(output.PN), kernel_hisq_project_un!,
            output.A, input.A, Val(NC), Val(input.nw), input.indexer)
    end
    return output
end

@inline function _record_nhyp_cache_state!(cache, thin_links)
    cache.state.source_links = ntuple(mu -> thin_links[mu], Val(4))
    cache.state.core_epochs = ntuple(
        mu -> thin_links[mu].halo_epoch.core, Val(4))
    cache.state.valid = true
    return cache
end

"""
    nhyp_smear!(smeared_links, thin_links, cache)

Apply three-level QEX-compatible nHYP smearing and retain the intermediates
needed by [`nhyp_pullback!`](@ref).  Each level uses a U(N) polar projection.
The cache coefficients are ordered explicitly as inner, middle, and outer
nesting levels rather than by the QEX `alpha1` / `alpha2` / `alpha3` names.
"""
function nhyp_smear!(
    smeared_links::Union{Vector{TO},NTuple{4,TO}},
    thin_links::Union{Vector{TI},NTuple{4,TI}},
    cache::NHYPSmearingCache4D,
) where {TO<:LatticeMatrix{4},TI<:LatticeMatrix{4}}
    _validate_nhyp_forward(smeared_links, thin_links, cache)
    cache.state.valid = false
    parameters = cache.parameters
    inner_staple_coefficient = parameters.alpha_inner / 2
    middle_staple_coefficient = parameters.alpha_middle / 4
    outer_staple_coefficient = parameters.alpha_outer / 6

    ensure_halo!.(thin_links)
    for (mu, nu) in _nhyp_direction_pairs
        candidate = _nhyp_pair_field(cache.inner_unprojected, mu, nu)
        projected = _nhyp_pair_field(cache.inner_links, mu, nu)
        _nhyp_scale_copy!(candidate, thin_links[mu], 1 - parameters.alpha_inner)
        _nhyp_staple_add!(
            candidate, thin_links[nu], thin_links[mu], nu, mu,
            inner_staple_coefficient)
        _nhyp_project_field!(projected, candidate)
    end
    ensure_halo!.(cache.inner_links)

    for (mu, nu) in _nhyp_direction_pairs
        candidate = _nhyp_pair_field(cache.middle_unprojected, mu, nu)
        projected = _nhyp_pair_field(cache.middle_links, mu, nu)
        _nhyp_scale_copy!(candidate, thin_links[mu], 1 - parameters.alpha_middle)
        for side_axis in 1:4
            (side_axis == mu || side_axis == nu) && continue
            excluded_axis = 10 - mu - nu - side_axis
            side = _nhyp_pair_field(
                cache.inner_links, side_axis, excluded_axis)
            middle = _nhyp_pair_field(
                cache.inner_links, mu, excluded_axis)
            _nhyp_staple_add!(
                candidate, side, middle, side_axis, mu,
                middle_staple_coefficient)
        end
        _nhyp_project_field!(projected, candidate)
    end
    ensure_halo!.(cache.middle_links)

    for mu in 1:4
        candidate = cache.outer_unprojected[mu]
        _nhyp_scale_copy!(candidate, thin_links[mu], 1 - parameters.alpha_outer)
        for nu in 1:4
            nu == mu && continue
            side = _nhyp_pair_field(cache.middle_links, nu, mu)
            middle = _nhyp_pair_field(cache.middle_links, mu, nu)
            _nhyp_staple_add!(
                candidate, side, middle, nu, mu,
                outer_staple_coefficient)
        end
        _nhyp_project_field!(smeared_links[mu], candidate)
    end
    _record_nhyp_cache_state!(cache, thin_links)
    return smeared_links
end

"""
    nhyp_smear(thin_links, parameters=NHYPParameters())

Allocating nHYP interface.  Returns `(smeared_links, cache)` so the same
forward intermediates can be passed directly to [`nhyp_pullback!`](@ref).
"""
function nhyp_smear(
    thin_links::Union{Vector{T},NTuple{4,T}},
    parameters::NHYPParameters=NHYPParameters(),
) where {T<:LatticeMatrix{4}}
    smeared_links = [similar(link) for link in thin_links]
    cache = NHYPSmearingCache4D(thin_links, parameters)
    nhyp_smear!(smeared_links, thin_links, cache)
    return smeared_links, cache
end

export nhyp_smear!, nhyp_smear

function _nhyp_project_pullback_accumulate!(dinput, doutput, input)
    NC = input.NC1
    if NC == 3
        _parallel_for_mutating!(
            dinput, prod(input.PN),
            _kernel_hisq_project_u3_pullback_core!,
            dinput.A, doutput.A, input.A, Val(input.nw), input.indexer)
    else
        _parallel_for_mutating!(
            dinput, prod(input.PN),
            _kernel_hisq_project_un_pullback_core!,
            dinput.A, doutput.A, input.A,
            Val(NC), Val(input.nw), input.indexer)
    end
    return dinput
end

@inline function _kernel_nhyp_staple_side_pullback!(
    site_index, dside, chain, side, middle,
    side_axis, middle_axis, coefficient,
    ::Val{NC}, ::Val{nw}, indexer,
) where {NC,nw}
    target = delinearize(indexer, site_index, nw)
    target_plus_middle = _hisq_shift_site(target, middle_axis)
    target_plus_side = _hisq_shift_site(target, side_axis)
    target_minus_middle = _hisq_shift_site(target, -middle_axis)
    minus_middle_plus_side = _hisq_shift_site(
        target_minus_middle, side_axis)
    plus_side_minus_middle = _hisq_shift_site(
        target_plus_side, -middle_axis)

    element_type = eltype(dside)
    first = MMatrix{NC,NC,element_type}(undef)
    second = MMatrix{NC,NC,element_type}(undef)
    third = MMatrix{NC,NC,element_type}(undef)
    temporary = MMatrix{NC,NC,element_type}(undef)
    term = MMatrix{NC,NC,element_type}(undef)
    gradient = MMatrix{NC,NC,element_type}(undef)
    _nhyp_zero_matrix!(gradient, Val(NC))

    # A(x) in A(x) B(x+nu) A(x+mu)'.
    _nhyp_load_matrix!(first, chain, target, Val(NC), Val(false))
    _nhyp_load_matrix!(second, side, target_plus_middle, Val(NC), Val(false))
    _nhyp_load_matrix!(third, middle, target_plus_side, Val(NC), Val(true))
    gemm!(temporary, first, second)
    gemm!(term, temporary, third)
    _nhyp_accumulate_matrix!(gradient, term, Val(NC))

    # A(x+mu)' in the positive staple, expressed at target x+mu.
    _nhyp_load_matrix!(first, chain, target_minus_middle, Val(NC), Val(true))
    _nhyp_load_matrix!(second, side, target_minus_middle, Val(NC), Val(false))
    _nhyp_load_matrix!(
        third, middle, minus_middle_plus_side, Val(NC), Val(false))
    gemm!(temporary, first, second)
    gemm!(term, temporary, third)
    _nhyp_accumulate_matrix!(gradient, term, Val(NC))

    # A(x-nu)' in the negative staple, expressed at target x-nu.
    _nhyp_load_matrix!(first, middle, target, Val(NC), Val(false))
    _nhyp_load_matrix!(second, side, target_plus_middle, Val(NC), Val(false))
    _nhyp_load_matrix!(third, chain, target_plus_side, Val(NC), Val(true))
    gemm!(temporary, first, second)
    gemm!(term, temporary, third)
    _nhyp_accumulate_matrix!(gradient, term, Val(NC))

    # A(x-nu+mu) in the negative staple, at target x-nu+mu.
    _nhyp_load_matrix!(first, middle, target_minus_middle, Val(NC), Val(true))
    _nhyp_load_matrix!(second, side, target_minus_middle, Val(NC), Val(false))
    _nhyp_load_matrix!(
        third, chain, plus_side_minus_middle, Val(NC), Val(false))
    gemm!(temporary, first, second)
    gemm!(term, temporary, third)
    _nhyp_accumulate_matrix!(gradient, term, Val(NC))

    _nhyp_add_to_field!(dside, gradient, target, coefficient, Val(NC))
    return nothing
end

@inline function _kernel_nhyp_staple_middle_pullback!(
    site_index, dmiddle, chain, side,
    side_axis, middle_axis, coefficient,
    ::Val{NC}, ::Val{nw}, indexer,
) where {NC,nw}
    target = delinearize(indexer, site_index, nw)
    target_minus_side = _hisq_shift_site(target, -side_axis)
    minus_side_plus_middle = _hisq_shift_site(
        target_minus_side, middle_axis)
    target_plus_side = _hisq_shift_site(target, side_axis)
    target_plus_middle = _hisq_shift_site(target, middle_axis)

    element_type = eltype(dmiddle)
    first = MMatrix{NC,NC,element_type}(undef)
    second = MMatrix{NC,NC,element_type}(undef)
    third = MMatrix{NC,NC,element_type}(undef)
    temporary = MMatrix{NC,NC,element_type}(undef)
    term = MMatrix{NC,NC,element_type}(undef)
    gradient = MMatrix{NC,NC,element_type}(undef)

    # B(x+nu) in the positive staple, expressed at target x+nu.
    _nhyp_load_matrix!(first, side, target_minus_side, Val(NC), Val(true))
    _nhyp_load_matrix!(second, chain, target_minus_side, Val(NC), Val(false))
    _nhyp_load_matrix!(
        third, side, minus_side_plus_middle, Val(NC), Val(false))
    gemm!(temporary, first, second)
    gemm!(gradient, temporary, third)

    # B(x-nu) in the negative staple, expressed at target x-nu.
    _nhyp_load_matrix!(first, side, target, Val(NC), Val(false))
    _nhyp_load_matrix!(second, chain, target_plus_side, Val(NC), Val(false))
    _nhyp_load_matrix!(third, side, target_plus_middle, Val(NC), Val(true))
    gemm!(temporary, first, second)
    gemm!(term, temporary, third)
    _nhyp_accumulate_matrix!(gradient, term, Val(NC))

    _nhyp_add_to_field!(dmiddle, gradient, target, coefficient, Val(NC))
    return nothing
end

function _nhyp_staple_pullback!(
    dside, dmiddle, chain, side, middle,
    side_axis, middle_axis, coefficient,
)
    ensure_halo!(chain)
    ensure_halo!(side)
    ensure_halo!(middle)
    _parallel_for_mutating!(
        dside, prod(dside.PN), _kernel_nhyp_staple_side_pullback!,
        dside.A, chain.A, side.A, middle.A,
        side_axis, middle_axis, coefficient,
        Val(dside.NC1), Val(dside.nw), dside.indexer)
    _parallel_for_mutating!(
        dmiddle, prod(dmiddle.PN), _kernel_nhyp_staple_middle_pullback!,
        dmiddle.A, chain.A, side.A,
        side_axis, middle_axis, coefficient,
        Val(dmiddle.NC1), Val(dmiddle.nw), dmiddle.indexer)
    return nothing
end

function _validate_nhyp_cache_current(cache, thin_links)
    cache.state.valid || throw(ArgumentError(
        "nhyp_smear! must populate the cache before nhyp_pullback!"))
    for mu in 1:4
        cache.state.source_links[mu] === thin_links[mu] ||
            throw(ArgumentError(
                "the nHYP cache belongs to different thin links"))
        cache.state.core_epochs[mu] == thin_links[mu].halo_epoch.core ||
            throw(ArgumentError(
                "thin link U[$mu] changed after the cached nHYP forward pass"))
    end
    return nothing
end

function _validate_nhyp_pullback(
    dthin_links, dsmeared_links, thin_links, cache,
)
    _validate_nhyp_forward(dsmeared_links, thin_links, cache)
    _validate_nhyp_links(dthin_links, thin_links[1], "dthin_links")
    for destination in dthin_links
        for input in thin_links
            _nhyp_aliases(destination, input) && throw(ArgumentError(
                "nHYP thin-link cotangents must not alias thin links"))
        end
        for source in dsmeared_links
            _nhyp_aliases(destination, source) && throw(ArgumentError(
                "nHYP input and output cotangents must not alias"))
        end
    end
    for first_mu in 1:4, second_mu in (first_mu + 1):4
        _nhyp_aliases(dthin_links[first_mu], dthin_links[second_mu]) &&
            throw(ArgumentError(
                "nHYP thin-link cotangents must not alias each other"))
    end
    _validate_nhyp_cache_current(cache, thin_links)
    return nothing
end

"""
    nhyp_pullback!(dthin_links, dsmeared_links, thin_links, cache)

Apply the analytic reverse pass for a cached [`nhyp_smear!`](@ref) call.
`dsmeared_links` is the cotangent of the four projected output links and
`dthin_links` is overwritten with the corresponding thin-link cotangent.
The inner product convention is `real(sum(dot(dthin[mu], delta[mu])))`.
"""
function nhyp_pullback!(
    dthin_links::Union{Vector{TD},NTuple{4,TD}},
    dsmeared_links::Union{Vector{TC},NTuple{4,TC}},
    thin_links::Union{Vector{TI},NTuple{4,TI}},
    cache::NHYPSmearingCache4D,
) where {
    TD<:LatticeMatrix{4},TC<:LatticeMatrix{4},TI<:LatticeMatrix{4},
}
    _validate_nhyp_pullback(
        dthin_links, dsmeared_links, thin_links, cache)
    clear_matrix!.(dthin_links)
    clear_matrix!.(cache.inner_cotangent)
    clear_matrix!.(cache.middle_cotangent)
    clear_matrix!.(cache.outer_cotangent)

    parameters = cache.parameters
    inner_staple_coefficient = parameters.alpha_inner / 2
    middle_staple_coefficient = parameters.alpha_middle / 4
    outer_staple_coefficient = parameters.alpha_outer / 6

    for mu in 1:4
        _nhyp_project_pullback_accumulate!(
            cache.outer_cotangent[mu], dsmeared_links[mu],
            cache.outer_unprojected[mu])
    end
    for mu in 1:4
        chain = cache.outer_cotangent[mu]
        add_matrix!(dthin_links[mu], chain, 1 - parameters.alpha_outer)
        for nu in 1:4
            nu == mu && continue
            dside = _nhyp_pair_field(cache.middle_cotangent, nu, mu)
            dmiddle = _nhyp_pair_field(cache.middle_cotangent, mu, nu)
            side = _nhyp_pair_field(cache.middle_links, nu, mu)
            middle = _nhyp_pair_field(cache.middle_links, mu, nu)
            _nhyp_staple_pullback!(
                dside, dmiddle, chain, side, middle, nu, mu,
                outer_staple_coefficient)
        end
    end

    projection_temporary = cache.outer_cotangent[1]
    for (mu, nu) in _nhyp_direction_pairs
        clear_matrix!(projection_temporary)
        _nhyp_project_pullback_accumulate!(
            projection_temporary,
            _nhyp_pair_field(cache.middle_cotangent, mu, nu),
            _nhyp_pair_field(cache.middle_unprojected, mu, nu))
        add_matrix!(
            dthin_links[mu], projection_temporary,
            1 - parameters.alpha_middle)
        for side_axis in 1:4
            (side_axis == mu || side_axis == nu) && continue
            excluded_axis = 10 - mu - nu - side_axis
            dside = _nhyp_pair_field(
                cache.inner_cotangent, side_axis, excluded_axis)
            dmiddle = _nhyp_pair_field(
                cache.inner_cotangent, mu, excluded_axis)
            side = _nhyp_pair_field(
                cache.inner_links, side_axis, excluded_axis)
            middle = _nhyp_pair_field(
                cache.inner_links, mu, excluded_axis)
            _nhyp_staple_pullback!(
                dside, dmiddle, projection_temporary, side, middle,
                side_axis, mu, middle_staple_coefficient)
        end
    end

    for (mu, nu) in _nhyp_direction_pairs
        clear_matrix!(projection_temporary)
        _nhyp_project_pullback_accumulate!(
            projection_temporary,
            _nhyp_pair_field(cache.inner_cotangent, mu, nu),
            _nhyp_pair_field(cache.inner_unprojected, mu, nu))
        add_matrix!(
            dthin_links[mu], projection_temporary,
            1 - parameters.alpha_inner)
        _nhyp_staple_pullback!(
            dthin_links[nu], dthin_links[mu], projection_temporary,
            thin_links[nu], thin_links[mu], nu, mu,
            inner_staple_coefficient)
    end
    return dthin_links
end

export nhyp_pullback!
