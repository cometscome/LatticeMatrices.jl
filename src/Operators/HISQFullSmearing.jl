@inline function _hisq_u3_inverse_sqrt!(inverse_sqrt, sqrt_matrix, Q, Q2)
    real_type = typeof(real(zero(eltype(Q))))
    c0 = zero(real_type)
    trace_Q2 = zero(real_type)
    trace_Q3 = zero(real_type)
    @inbounds for i in 1:3
        c0 += real(Q[i, i])
        trace_Q2 += real(Q2[i, i])
        for j in 1:3
            trace_Q3 += real(Q2[i, j] * Q[j, i])
        end
    end
    c1 = trace_Q2 / 2
    c2 = trace_Q3 / 3

    S = c1 / 3 - c0 * c0 / 18
    tolerance = convert(real_type, 1e-7)
    g0 = zero(real_type)
    g1 = zero(real_type)
    g2 = zero(real_type)
    if abs(S) < tolerance
        g0 = c0 / 3
        g1 = g0
        g2 = g0
    else
        S = sqrt(max(S, zero(S)))
        R = c2 / 2 - c0 * c1 / 3 + c0 * c0 * c0 / 27
        ratio = clamp(R / (S * S * S), -one(real_type), one(real_type))
        theta = acos(ratio)
        two_pi_over_three = convert(real_type, 2pi / 3)
        center = c0 / 3
        amplitude = 2S
        g0 = center + amplitude * cos(theta / 3 - two_pi_over_three)
        g1 = center + amplitude * cos(theta / 3)
        g2 = center + amplitude * cos(theta / 3 + two_pi_over_three)
    end

    # SIMULATeQCD applies no force cutoff to the forward projection.  Clamp
    # only at machine precision to protect the square root from a tiny
    # negative roundoff error.  As for the reference implementation, the
    # differentiable Cayley--Hamilton branch assumes a nonsingular link.
    eigenvalue_floor = eps(real_type)
    g0 = max(g0, eigenvalue_floor)
    g1 = max(g1, eigenvalue_floor)
    g2 = max(g2, eigenvalue_floor)

    u = sqrt(g0) + sqrt(g1) + sqrt(g2)
    v = sqrt(g0 * g1) + sqrt(g0 * g2) + sqrt(g1 * g2)
    w = sqrt(g0 * g1 * g2)
    denominator = w * (u * v - w)
    f0 = (-w * (u * u + v) + u * v * v) / denominator
    f1 = (-w - u * u * u + 2u * v) / denominator
    f2 = u / denominator

    @inbounds for column in 1:3, row in 1:3
        identity_element = ifelse(row == column, one(eltype(Q)), zero(eltype(Q)))
        inverse_sqrt[row, column] =
            f0 * identity_element + f1 * Q[row, column] +
            f2 * Q2[row, column]
    end
    gemm!(sqrt_matrix, Q, inverse_sqrt)
    return nothing
end

@inline function _hisq_u3_project_matrix!(projected, V, Q, Q2, inverse_sqrt, sqrt_matrix)
    @inbounds for column in 1:3, row in 1:3
        value = zero(eltype(V))
        for contracted in 1:3
            value += conj(V[contracted, row]) * V[contracted, column]
        end
        Q[row, column] = value
    end
    gemm!(Q2, Q, Q)
    _hisq_u3_inverse_sqrt!(inverse_sqrt, sqrt_matrix, Q, Q2)
    gemm!(projected, V, inverse_sqrt)
    return nothing
end

@inline function kernel_hisq_project_u3!(
    site_index, output, input, ::Val{nw}, indexer,
) where nw
    site = delinearize(indexer, site_index, nw)
    element_type = eltype(output)
    V = MMatrix{3,3,element_type}(undef)
    Q = MMatrix{3,3,element_type}(undef)
    Q2 = MMatrix{3,3,element_type}(undef)
    inverse_sqrt = MMatrix{3,3,element_type}(undef)
    sqrt_matrix = MMatrix{3,3,element_type}(undef)
    projected = MMatrix{3,3,element_type}(undef)
    @inbounds for column in 1:3, row in 1:3
        V[row, column] = input[row, column, site...]
    end
    _hisq_u3_project_matrix!(
        projected, V, Q, Q2, inverse_sqrt, sqrt_matrix)
    @inbounds for column in 1:3, row in 1:3
        output[row, column, site...] = projected[row, column]
    end
    return nothing
end

function _validate_hisq_projection_output(projected_links, fat_links)
    _validate_hisq_smearing_output(projected_links, fat_links)
    fat_links[1].NC1 == 3 || throw(ArgumentError(
        "HISQ U(3) projection requires three-color links"))
    return nothing
end

"""
    hisq_project_u3!(projected_links, fat_links)
    hisq_project_u3(fat_links)

Project each unprojected Fat7 link to U(3) using
`V * (V' * V)^(-1/2)`, following SIMULATeQCD's HISQ convention.
"""
function hisq_project_u3!(
    projected_links::Vector{TO}, fat_links::Vector{TI},
) where {TO<:LatticeMatrix{4},TI<:LatticeMatrix{4}}
    _validate_hisq_projection_output(projected_links, fat_links)
    for mu in 1:4
        _parallel_for_mutating!(
            projected_links[mu], prod(projected_links[mu].PN),
            kernel_hisq_project_u3!, projected_links[mu].A,
            fat_links[mu].A, Val(fat_links[mu].nw), fat_links[mu].indexer)
    end
    return projected_links
end

function hisq_project_u3(fat_links::Vector{T}) where {T<:LatticeMatrix{4}}
    projected_links = [similar(link) for link in fat_links]
    return hisq_project_u3!(projected_links, fat_links)
end

export hisq_project_u3!, hisq_project_u3

@inline function kernel_hisq_naik_links!(
    combined_index, L1, L2, L3, L4, W1, W2, W3, W4,
    volume, ::Val{NC}, ::Val{nw}, indexer,
) where {NC,nw}
    site_index, row, mu = _hisq_combined_row(
        combined_index, volume, Val(NC))
    origin = delinearize(indexer, site_index, nw)
    path_row = _hisq_path_row(
        W1, W2, W3, W4, origin, (mu, mu, mu), row, Val(NC))
    _hisq_store_row!(
        L1, L2, L3, L4, mu, origin, row,
        path_row, one(eltype(L1)), Val(NC))
    return nothing
end

function _hisq_naik_links_nowing!(long_links, reunitarized_links)
    for mu in 1:4
        shifted_once = _materialize_periodic_shift(
            reunitarized_links[mu], shifts_p[mu])
        shifted_twice = _materialize_periodic_shift(
            reunitarized_links[mu], ntuple(d -> 2shifts_p[mu][d], 4))
        temporary = similar(long_links[mu])
        mul!(temporary, reunitarized_links[mu], shifted_once)
        mul!(long_links[mu], temporary, shifted_twice)
    end
    return long_links
end

"""
    hisq_naik_links!(long_links, reunitarized_links)
    hisq_naik_links(reunitarized_links)

Construct forward-anchored Naik transporters from reunitarized HISQ links,
`L_mu(x) = W_mu(x) W_mu(x+mu) W_mu(x+2mu)`.  This is the orientation
expected by [`HISQDiracOperator4D`](@ref).  The halo kernel requires
`nw >= 2`; `nw=0` uses periodic materialized shifts.
"""
function hisq_naik_links!(
    long_links::Vector{TO}, reunitarized_links::Vector{TI},
) where {TO<:LatticeMatrix{4},TI<:LatticeMatrix{4}}
    _validate_hisq_smearing_output(long_links, reunitarized_links)
    nw = reunitarized_links[1].nw
    iszero(nw) && return _hisq_naik_links_nowing!(
        long_links, reunitarized_links)
    nw < 2 && throw(ArgumentError(
        "HISQ Naik link construction requires nw >= 2 or nw == 0"))
    ensure_halo!.(reunitarized_links)
    volume = prod(long_links[1].PN)
    NC = long_links[1].NC1
    _hisq_parallel_for(
        4 * NC * volume, kernel_hisq_naik_links!,
        long_links[1].A, long_links[2].A,
        long_links[3].A, long_links[4].A,
        reunitarized_links[1].A, reunitarized_links[2].A,
        reunitarized_links[3].A, reunitarized_links[4].A,
        volume, Val(NC), Val(long_links[1].nw), long_links[1].indexer)
    mark_halo_dirty!.(long_links)
    return long_links
end

function hisq_naik_links(
    reunitarized_links::Vector{T},
) where {T<:LatticeMatrix{4}}
    long_links = [similar(link) for link in reunitarized_links]
    return hisq_naik_links!(long_links, reunitarized_links)
end

export hisq_naik_links!, hisq_naik_links

function _validate_hisq_full_workspace(
    fat_links, long_links, level1_links, reunitarized_links, thin_links,
)
    _validate_staggered_gauge_links(thin_links)
    thin_links[1].NC1 == 3 || throw(ArgumentError(
        "full HISQ smearing requires three-color thin links"))
    nw = thin_links[1].nw
    iszero(nw) || nw >= 3 || throw(ArgumentError(
        "full HISQ smearing requires nw=0 or nw>=3"))

    collections = (
        fat_links, long_links, level1_links, reunitarized_links, thin_links)
    labels = ("fat", "long", "level1", "reunitarized", "thin")
    for collection in collections
        _validate_staggered_gauge_links(collection)
        for link in collection
            link.NC1 == thin_links[1].NC1 &&
                link.NC2 == thin_links[1].NC2 &&
                link.gsize == thin_links[1].gsize &&
                link.PN == thin_links[1].PN &&
                link.dims == thin_links[1].dims &&
                link.nw == thin_links[1].nw &&
                eltype(link.A) == eltype(thin_links[1].A) ||
                throw(ArgumentError(
                    "all HISQ link fields must share one geometry and element type"))
        end
    end
    for first_collection in 1:length(collections)
        for second_collection in (first_collection + 1):length(collections)
            for first_direction in 1:4, second_direction in 1:4
                first_link = collections[first_collection][first_direction]
                second_link = collections[second_collection][second_direction]
                (first_link === second_link ||
                 first_link.A === second_link.A) && throw(ArgumentError(
                    "HISQ $(labels[first_collection]) and " *
                    "$(labels[second_collection]) work fields must not alias"))
            end
        end
    end
    return nothing
end

"""
    hisq_links_from_thin!(
        fat_links, long_links, level1_links, reunitarized_links,
        thin_links, naik_epsilon=0)

Build all links used by [`HISQDiracOperator4D`](@ref) from thin gauge links.
The caller owns the two work vectors `level1_links` and
`reunitarized_links`, making repeated construction allocation-free after
setup.  The stages are level-1 Fat7, U(3) reunitarization, level-2
Fat7/Lepage, and forward-anchored Naik construction.
"""
function hisq_links_from_thin!(
    fat_links::Vector{T}, long_links::Vector{T},
    level1_links::Vector{T}, reunitarized_links::Vector{T},
    thin_links::Vector{T}, naik_epsilon,
) where {T<:LatticeMatrix{4}}
    _validate_hisq_full_workspace(
        fat_links, long_links, level1_links, reunitarized_links, thin_links)
    hisq_fat7_level1!(level1_links, thin_links)
    hisq_project_u3!(reunitarized_links, level1_links)
    hisq_fat7_level2!(fat_links, reunitarized_links, naik_epsilon)
    hisq_naik_links!(long_links, reunitarized_links)
    return HISQLinks4D(fat_links, long_links)
end

hisq_links_from_thin!(
    fat_links, long_links, level1_links, reunitarized_links, thin_links;
    naik_epsilon=0,
) = hisq_links_from_thin!(
    fat_links, long_links, level1_links, reunitarized_links, thin_links,
    naik_epsilon)

"""
    hisq_links_from_thin(thin_links, naik_epsilon=0)

Allocating convenience form of [`hisq_links_from_thin!`](@ref).  Returns a
[`HISQLinks4D`](@ref) containing corrected fat and Naik links.
"""
function hisq_links_from_thin(
    thin_links::Vector{T}, naik_epsilon,
) where {T<:LatticeMatrix{4}}
    level1_links = [similar(link) for link in thin_links]
    reunitarized_links = [similar(link) for link in thin_links]
    fat_links = [similar(link) for link in thin_links]
    long_links = [similar(link) for link in thin_links]
    return hisq_links_from_thin!(
        fat_links, long_links, level1_links, reunitarized_links,
        thin_links, naik_epsilon)
end

hisq_links_from_thin(thin_links; naik_epsilon=0) =
    hisq_links_from_thin(thin_links, naik_epsilon)

function HISQDiracOperator4D(
    thin_links::Vector{T}, mass::Real; naik_epsilon::Real=0,
) where {T<:LatticeMatrix{4}}
    links = hisq_links_from_thin(thin_links, naik_epsilon)
    return HISQDiracOperator4D(links, mass; naik_epsilon)
end

export hisq_links_from_thin!, hisq_links_from_thin

mutable struct HISQCacheState{T}
    source_links::NTuple{4,T}
    core_epochs::NTuple{4,UInt64}
end

@inline _hisq_source_links(U) = ntuple(mu -> U[mu], Val(4))

@inline _hisq_core_epochs(U) =
    ntuple(mu -> U[mu].halo_epoch.core, Val(4))

@inline function _record_hisq_cache_state!(cache, U)
    cache.cache_state.source_links = _hisq_source_links(U)
    cache.cache_state.core_epochs = _hisq_core_epochs(U)
    return cache
end

"""
    HISQDiracCache4D(thin_links, mass; naik_epsilon=0)

Reusable storage for a HISQ operator built from thin links.  The level-1,
reunitarized, corrected-fat, and Naik links are retained so that repeated
Dirac applications do not repeat the smearing construction.

Use [`mul_cached_hisq!`](@ref) and [`mul_cached_hisq_adjoint!`](@ref) to apply
the operator.  Those entry points transparently rebuild the derived links on
the first call after a thin link is replaced or its core-data epoch changes.
"""
struct HISQDiracCache4D{T,L,O,S}
    level1_links::Vector{T}
    reunitarized_links::Vector{T}
    fat_links::Vector{T}
    long_links::Vector{T}
    links::L
    operator::O
    cache_state::S
end

function HISQDiracCache4D(
    thin_links::Vector{T}, mass::Real; naik_epsilon::Real=0,
) where {T<:LatticeMatrix{4}}
    level1_links = [similar(link) for link in thin_links]
    reunitarized_links = [similar(link) for link in thin_links]
    fat_links = [similar(link) for link in thin_links]
    long_links = [similar(link) for link in thin_links]
    links = hisq_links_from_thin!(
        fat_links, long_links, level1_links, reunitarized_links,
        thin_links, naik_epsilon)
    operator = HISQDiracOperator4D(links, mass; naik_epsilon)
    cache_state = HISQCacheState(
        _hisq_source_links(thin_links), _hisq_core_epochs(thin_links))
    return HISQDiracCache4D{
        T,typeof(links),typeof(operator),typeof(cache_state)
    }(
        level1_links, reunitarized_links, fat_links, long_links,
        links, operator, cache_state)
end

export HISQDiracCache4D

"""
    update_hisq_cache!(cache, thin_links)

Rebuild every derived HISQ link in `cache` from `thin_links`.  Normal callers
do not need to call this explicitly because the cached multiplication entry
points detect changes automatically.
"""
function update_hisq_cache!(
    cache::HISQDiracCache4D{T}, thin_links::Vector{T},
) where {T<:LatticeMatrix{4}}
    hisq_links_from_thin!(
        cache.fat_links, cache.long_links,
        cache.level1_links, cache.reunitarized_links,
        thin_links, cache.operator.naik_epsilon)
    return _record_hisq_cache_state!(cache, thin_links)
end

export update_hisq_cache!

@inline function _hisq_cache_is_current(
    cache::HISQDiracCache4D, U1, U2, U3, U4,
)
    U = (U1, U2, U3, U4)
    source_links = cache.cache_state.source_links
    epochs = cache.cache_state.core_epochs
    for mu in 1:4
        source_links[mu] === U[mu] || return false
        epochs[mu] == U[mu].halo_epoch.core || return false
    end
    return true
end

@inline function _ensure_hisq_cache_current!(
    cache::HISQDiracCache4D{T},
    U1::T, U2::T, U3::T, U4::T,
) where {T<:LatticeMatrix{4}}
    _hisq_cache_is_current(cache, U1, U2, U3, U4) && return cache
    return update_hisq_cache!(cache, T[U1, U2, U3, U4])
end

"""
    mul_cached_hisq!(result, cache, U1, U2, U3, U4, psi)

Apply the HISQ operator stored in `cache`, treating `U1`, ..., `U4` as its
thin-link inputs.  Identity and core-data epochs are checked before every
application, so all smearing stages are rebuilt only on the first application
after the thin links change and are then reused throughout a Krylov solve.

Mutations through `link.A` must be followed by `mark_halo_dirty!(link)` so
that the change is observable.  The Enzyme reverse rule propagates the force
through the cached Dirac, Naik, level-2, U(3), and level-1 stages without
differentiating through cache maintenance.
"""
function mul_cached_hisq!(
    result::T,
    cache::HISQDiracCache4D,
    U1::G, U2::G, U3::G, U4::G,
    psi::T,
) where {ET,AT,NC,nw,DI,
    T<:LatticeMatrix{4,ET,AT,NC,1,nw,DI},G<:LatticeMatrix{4}}
    _ensure_hisq_cache_current!(cache, U1, U2, U3, U4)
    return LinearAlgebra.mul!(result, cache.operator, psi)
end

"""
    mul_cached_hisq_adjoint!(result, cache, U1, U2, U3, U4, psi)

Apply the adjoint of [`mul_cached_hisq!`](@ref), sharing the same transparent
derived-link cache.
"""
function mul_cached_hisq_adjoint!(
    result::T,
    cache::HISQDiracCache4D,
    U1::G, U2::G, U3::G, U4::G,
    psi::T,
) where {ET,AT,NC,nw,DI,
    T<:LatticeMatrix{4,ET,AT,NC,1,nw,DI},G<:LatticeMatrix{4}}
    _ensure_hisq_cache_current!(cache, U1, U2, U3, U4)
    return LinearAlgebra.mul!(result, adjoint(cache.operator), psi)
end

export mul_cached_hisq!, mul_cached_hisq_adjoint!
