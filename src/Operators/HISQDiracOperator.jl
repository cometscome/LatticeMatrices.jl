"""
    HISQLinks4D(fat_links, long_links)

Precomputed links for the four-dimensional HISQ Dirac stencil. `fat_links[mu]`
connects `x` to `x + mu`, while `long_links[mu]` connects `x` to
`x + 3mu`. Both sets are unphased periodic gauge fields; staggered phases and
fermion boundary conditions are applied by [`HISQDiracOperator4D`](@ref).

This type deliberately separates the Dirac stencil from the expensive HISQ
smearing. A later link-building stage can construct the same container from
thin links without changing the stencil API.
"""
struct HISQLinks4D{T}
    fat_links::Union{Vector{T},NTuple{4,T}}
    long_links::Union{Vector{T},NTuple{4,T}}
end

function _validate_hisq_link_geometry(fat_links, long_links)
    _validate_staggered_gauge_links(fat_links)
    _validate_staggered_gauge_links(long_links)

    reference = fat_links[1]
    for (mu, link) in enumerate(long_links)
        link.NC1 == reference.NC1 && link.NC2 == reference.NC2 ||
            throw(ArgumentError(
                "long link L[$mu] and fat links have different matrix sizes"))
        link.gsize == reference.gsize && link.PN == reference.PN &&
            link.dims == reference.dims && link.nw == reference.nw ||
            throw(ArgumentError(
                "long link L[$mu] and fat links use different lattice geometry"))
        eltype(link.A) == eltype(reference.A) || throw(ArgumentError(
            "long link L[$mu] and fat links have different element types"))
    end

    nw = reference.nw
    iszero(nw) || nw >= 3 || throw(ArgumentError(
        "HISQ links require nw=0 or nw>=3 for the three-hop Naik term"))
    if !iszero(nw) && any(local_extent -> local_extent < nw, reference.PN)
        throw(ArgumentError(
            "each local lattice extent must be at least the halo width nw=$nw"))
    end
    return nothing
end

function HISQLinks4D(
    fat_links::Union{Vector{T},NTuple{4,T}},
    long_links::Union{Vector{T},NTuple{4,T}},
) where {T<:LatticeMatrix{4}}
    _validate_hisq_link_geometry(fat_links, long_links)
    return HISQLinks4D{T}(fat_links, long_links)
end

export HISQLinks4D

"""
    HISQDiracOperator4D(links, mass; naik_epsilon=0)
    HISQDiracOperator4D(fat_links, long_links, mass; naik_epsilon=0)

Four-dimensional HISQ Dirac stencil acting on an `NC x 1` staggered field,

```
(D psi)(x) = mass*psi(x) + sum_mu eta_mu(x) * (
    (1/2) * [X_mu(x) psi(x+mu) - X_mu(x-mu)' psi(x-mu)]
  - (1+epsilon)/48 *
    [L_mu(x) psi(x+3mu) - L_mu(x-3mu)' psi(x-3mu)]).
```

`X` is the corrected fat link and `L` is the forward-anchored three-link
transporter. The links must not contain staggered or boundary phases. Fermion
boundary phases are read from `psi.phases`. The fast fused path requires
`nw >= 3`; a halo-free `nw=0` fallback is also provided.
"""
struct HISQDiracOperator4D{LT,M<:Real,E<:Real} <: OperatorOnKernel
    links::LT
    mass::M
    naik_epsilon::E
end

function HISQDiracOperator4D(
    links::HISQLinks4D, mass::Real; naik_epsilon::Real=0,
)
    real_type = typeof(real(zero(eltype(links.fat_links[1].A))))
    typed_mass = convert(real_type, mass)
    typed_epsilon = convert(real_type, naik_epsilon)
    isfinite(typed_mass) || throw(ArgumentError("mass must be finite"))
    isfinite(typed_epsilon) ||
        throw(ArgumentError("naik_epsilon must be finite"))
    return HISQDiracOperator4D{
        typeof(links),typeof(typed_mass),typeof(typed_epsilon)
    }(links, typed_mass, typed_epsilon)
end

function HISQDiracOperator4D(
    fat_links::Union{Vector{T},NTuple{4,T}},
    long_links::Union{Vector{T},NTuple{4,T}}, mass::Real;
    naik_epsilon::Real=0,
) where {T<:LatticeMatrix{4}}
    return HISQDiracOperator4D(
        HISQLinks4D(fat_links, long_links), mass; naik_epsilon)
end

export HISQDiracOperator4D

struct Adjoint_HISQDiracOperator4D{T} <: OperatorOnKernel
    parent::T
end

Base.adjoint(operator::HISQDiracOperator4D) =
    Adjoint_HISQDiracOperator4D(operator)
Base.adjoint(operator::Adjoint_HISQDiracOperator4D) = operator.parent

const hisq_long_shifts_p = (
    (3, 0, 0, 0),
    (0, 3, 0, 0),
    (0, 0, 3, 0),
    (0, 0, 0, 3),
)
const hisq_long_shifts_m = (
    (-3, 0, 0, 0),
    (0, -3, 0, 0),
    (0, 0, -3, 0),
    (0, 0, 0, -3),
)

function _validate_hisq_application(result, operator, psi)
    result === psi && throw(ArgumentError(
        "HISQ mul! requires distinct result and input fields"))
    result.gsize == psi.gsize && result.PN == psi.PN &&
        result.dims == psi.dims && result.nw == psi.nw ||
        throw(ArgumentError(
            "result and fermion fields must use the same lattice geometry"))
    result.phases == psi.phases || throw(ArgumentError(
        "result and fermion fields must use the same boundary phases"))

    tolerance = sqrt(eps(typeof(real(zero(eltype(psi.A))))))
    for (mu, phase) in enumerate(psi.phases)
        isapprox(abs(phase), one(abs(phase)); atol=tolerance, rtol=tolerance) ||
            throw(ArgumentError(
                "fermion boundary phase $mu must have unit magnitude"))
    end

    for (label, links) in
        (("fat link X", operator.links.fat_links),
         ("long link L", operator.links.long_links))
        for (mu, link) in enumerate(links)
            link.NC1 == psi.NC1 || throw(ArgumentError(
                "$label[$mu] and fermion field have different color sizes"))
            eltype(link.A) == eltype(psi.A) || throw(ArgumentError(
                "$label[$mu] and fermion field have different element types"))
            link.gsize == psi.gsize && link.PN == psi.PN &&
                link.dims == psi.dims && link.nw == psi.nw ||
                throw(ArgumentError(
                    "$label[$mu] and fermion field use different lattice geometry"))
        end
    end
    return nothing
end

@inline function kernel_HISQDiracOperator4D!(
    site, result,
    X1, X2, X3, X4, L1, L2, L3, L4,
    mass, fat_coefficient, long_coefficient, psi,
    ::Val{NC}, ::Val{nw}, indexer, mpi_coordinates, local_size,
) where {NC,nw}
    x = delinearize(indexer, site, nw)

    eta2 = staggered_eta_global_halo(
        x, 2, nw, mpi_coordinates, local_size)
    eta3 = staggered_eta_global_halo(
        x, 3, nw, mpi_coordinates, local_size)
    eta4 = staggered_eta_global_halo(
        x, 4, nw, mpi_coordinates, local_size)

    @inbounds for output_color in 1:NC
        fat_hopping = _staggered_direction_matvec(
            X1, psi, x,
            shiftindices(x, shift_1p), shiftindices(x, shift_1m),
            output_color, Val(NC))
        long_hopping = _staggered_direction_matvec(
            L1, psi, x,
            shiftindices(x, hisq_long_shifts_p[1]),
            shiftindices(x, hisq_long_shifts_m[1]),
            output_color, Val(NC))

        fat_hopping += eta2 * _staggered_direction_matvec(
            X2, psi, x,
            shiftindices(x, shift_2p), shiftindices(x, shift_2m),
            output_color, Val(NC))
        long_hopping += eta2 * _staggered_direction_matvec(
            L2, psi, x,
            shiftindices(x, hisq_long_shifts_p[2]),
            shiftindices(x, hisq_long_shifts_m[2]),
            output_color, Val(NC))

        fat_hopping += eta3 * _staggered_direction_matvec(
            X3, psi, x,
            shiftindices(x, shift_3p), shiftindices(x, shift_3m),
            output_color, Val(NC))
        long_hopping += eta3 * _staggered_direction_matvec(
            L3, psi, x,
            shiftindices(x, hisq_long_shifts_p[3]),
            shiftindices(x, hisq_long_shifts_m[3]),
            output_color, Val(NC))

        fat_hopping += eta4 * _staggered_direction_matvec(
            X4, psi, x,
            shiftindices(x, shift_4p), shiftindices(x, shift_4m),
            output_color, Val(NC))
        long_hopping += eta4 * _staggered_direction_matvec(
            L4, psi, x,
            shiftindices(x, hisq_long_shifts_p[4]),
            shiftindices(x, hisq_long_shifts_m[4]),
            output_color, Val(NC))

        result[output_color, 1, x...] =
            mass * psi[output_color, 1, x...] +
            fat_coefficient * fat_hopping +
            long_coefficient * long_hopping
    end
    return nothing
end

function _hisq_hopping_coefficients(operator, adjoint_operator::Bool)
    direction_sign = adjoint_operator ? -one(operator.mass) : one(operator.mass)
    fat_coefficient = direction_sign / 2
    long_coefficient =
        -direction_sign * (one(operator.naik_epsilon) +
                           operator.naik_epsilon) / 48
    return fat_coefficient, long_coefficient
end

function _apply_hisq_halo!(result, operator, psi, adjoint_operator::Bool)
    _validate_hisq_application(result, operator, psi)
    fat_links = operator.links.fat_links
    long_links = operator.links.long_links
    for link in fat_links
        ensure_halo!(link)
    end
    for link in long_links
        ensure_halo!(link)
    end
    ensure_halo!(psi)

    fat_coefficient, long_coefficient =
        _hisq_hopping_coefficients(operator, adjoint_operator)
    _parallel_for_mutating!(result,
        prod(result.PN), kernel_HISQDiracOperator4D!, result.A,
        fat_links[1].A, fat_links[2].A, fat_links[3].A, fat_links[4].A,
        long_links[1].A, long_links[2].A, long_links[3].A,
        long_links[4].A, operator.mass, fat_coefficient,
        long_coefficient, psi.A, Val(result.NC1), Val(result.nw),
        result.indexer, result.coords, result.PN)
    return result
end

function _apply_hisq_nowing!(result, operator, psi, adjoint_operator::Bool)
    _validate_hisq_application(result, operator, psi)
    _parallel_for_mutating!(result,
        prod(result.PN), kernel_initialize_StaggeredDiracOperator4D_nowing!,
        result.A, psi.A, operator.mass, Val(result.NC1), result.indexer)

    fat_coefficient, long_coefficient =
        _hisq_hopping_coefficients(operator, adjoint_operator)
    for mu in 1:4
        for (links, plus_shift, minus_shift, coefficient) in (
            (operator.links.fat_links, shifts_p[mu], shifts_m[mu],
             fat_coefficient),
            (operator.links.long_links, hisq_long_shifts_p[mu],
             hisq_long_shifts_m[mu], long_coefficient),
        )
            psi_plus = _materialize_periodic_shift(psi, plus_shift)
            psi_minus = _materialize_periodic_shift(psi, minus_shift)
            link_minus = _materialize_periodic_shift(links[mu], minus_shift)
            _parallel_for_mutating!(result,
                prod(result.PN),
                kernel_StaggeredDiracOperator4D_direction_nowing!,
                result.A, links[mu].A, link_minus.A,
                psi_plus.A, psi_minus.A, coefficient,
                Val(result.NC1), result.indexer, Val(mu),
                result.coords, result.PN)
        end
    end
    return result
end

function _apply_hisq!(result, operator, psi, adjoint_operator::Bool)
    if iszero(result.nw)
        return _apply_hisq_nowing!(
            result, operator, psi, adjoint_operator)
    elseif result.nw >= 3
        return _apply_hisq_halo!(
            result, operator, psi, adjoint_operator)
    end
    throw(ArgumentError(
        "HISQ applications require nw=0 or nw>=3, got nw=$(result.nw)"))
end

function LinearAlgebra.mul!(
    result::T, operator::HISQDiracOperator4D, psi::T,
) where {ET,AT,NC,nw,DI,T<:LatticeMatrix{4,ET,AT,NC,1,nw,DI}}
    return _apply_hisq!(result, operator, psi, false)
end

function LinearAlgebra.mul!(
    result::T, operator::Adjoint_HISQDiracOperator4D, psi::T,
) where {ET,AT,NC,nw,DI,T<:LatticeMatrix{4,ET,AT,NC,1,nw,DI}}
    return _apply_hisq!(result, operator.parent, psi, true)
end
