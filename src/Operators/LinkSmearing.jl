"""
    smearing_cache(links, parameters)

Construct a reusable cache for a supported link-smearing specification.
"""
smearing_cache(links, parameters::NHYPParameters) =
    NHYPSmearingCache4D(links, parameters)
smearing_cache(links, parameters::StoutParameters) =
    StoutSmearingCache4D(links, parameters)
smearing_cache(links, parameters::HEXParameters) =
    HEXSmearingCache4D(links, parameters)
smearing_cache(links, parameters::APEParameters) =
    APESmearingCache4D(links, parameters)
smearing_cache(links, parameters::HYPParameters) =
    HYPSmearingCache4D(links, parameters)

export smearing_cache

"""Apply one cached link-smearing step into caller-owned output links."""
smear_links!(output, input, cache::NHYPSmearingCache4D) =
    nhyp_smear!(output, input, cache)
smear_links!(output, input, cache::StoutSmearingCache4D) =
    stout_smear!(output, input, cache)
smear_links!(output, input, cache::HEXSmearingCache4D) =
    hex_smear!(output, input, cache)
smear_links!(output, input, cache::APESmearingCache4D) =
    ape_smear!(output, input, cache)
smear_links!(output, input, cache::HYPSmearingCache4D) =
    hyp_smear!(output, input, cache)

"""Apply the analytic pullback of one cached link-smearing step."""
smear_links_pullback!(dinput, doutput, input, cache::NHYPSmearingCache4D) =
    nhyp_pullback!(dinput, doutput, input, cache)
smear_links_pullback!(dinput, doutput, input, cache::StoutSmearingCache4D) =
    stout_pullback!(dinput, doutput, input, cache)
smear_links_pullback!(dinput, doutput, input, cache::HEXSmearingCache4D) =
    hex_pullback!(dinput, doutput, input, cache)
smear_links_pullback!(dinput, doutput, input, cache::APESmearingCache4D) =
    throw(ArgumentError(
        "principal-polar APE is forward-only; use stout for an analytic pullback"))
smear_links_pullback!(dinput, doutput, input, cache::HYPSmearingCache4D) =
    throw(ArgumentError(
        "principal-polar HYP is forward-only; use nHYP or HEX for an analytic pullback"))

function smear_links(input, parameters)
    output = [similar(link) for link in input]
    cache = smearing_cache(input, parameters)
    smear_links!(output, input, cache)
    return output, cache
end

export smear_links!, smear_links, smear_links_pullback!

"""
    IteratedSmearing(parameters, iterations)

Repeat a complete smearing transformation.  This is distinct from the three
restricted geometric levels inside HYP, nHYP, or HEX.
"""
struct IteratedSmearing{P}
    parameters::P
    iterations::Int

    function IteratedSmearing(parameters::P, iterations::Integer) where P
        iterations >= 1 || throw(ArgumentError(
            "smearing iterations must be positive; got $iterations"))
        return new{P}(parameters, Int(iterations))
    end
end

export IteratedSmearing

struct IteratedSmearingCache4D{T,C,P}
    intermediates::Vector{Vector{T}}
    stage_caches::Vector{C}
    cotangent_a::Vector{T}
    cotangent_b::Vector{T}
    specification::P
end

export IteratedSmearingCache4D

function smearing_cache(
    links::Union{Vector{T},NTuple{4,T}},
    specification::IteratedSmearing,
) where {T<:LatticeMatrix{4}}
    intermediates = Vector{Vector{T}}(undef, specification.iterations - 1)
    for stage in eachindex(intermediates)
        intermediates[stage] = [similar(link) for link in links]
    end
    first_cache = smearing_cache(links, specification.parameters)
    caches = Vector{typeof(first_cache)}(undef, specification.iterations)
    caches[1] = first_cache
    for stage in 2:specification.iterations
        caches[stage] = smearing_cache(
            intermediates[stage - 1], specification.parameters)
    end
    return IteratedSmearingCache4D(
        intermediates,
        caches,
        [similar(link) for link in links],
        [similar(link) for link in links],
        specification,
    )
end

function smear_links!(output, input, cache::IteratedSmearingCache4D)
    stages = cache.specification.iterations
    for stage in 1:stages
        stage_input = stage == 1 ? input : cache.intermediates[stage - 1]
        stage_output = stage == stages ? output : cache.intermediates[stage]
        smear_links!(stage_output, stage_input, cache.stage_caches[stage])
    end
    return output
end

function smear_links_pullback!(
    dinput, doutput, input, cache::IteratedSmearingCache4D,
)
    stages = cache.specification.iterations
    current = doutput
    use_a = true
    for stage in stages:-1:1
        stage_input = stage == 1 ? input : cache.intermediates[stage - 1]
        destination = if stage == 1
            dinput
        elseif use_a
            cache.cotangent_a
        else
            cache.cotangent_b
        end
        smear_links_pullback!(
            destination, current, stage_input, cache.stage_caches[stage])
        current = destination
        use_a = !use_a
    end
    return dinput
end
