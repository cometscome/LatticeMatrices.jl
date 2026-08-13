"""
    StaggeredDiracOperator4D(U, mass)

Four-dimensional one-link staggered Dirac operator in the Bridge++ mass
normalization,

```
(D psi)(x) = mass*psi(x)
  + (1/2) sum_mu eta_mu(x) [
        U_mu(x) psi(x+mu) - U_mu(x-mu)' psi(x-mu)] .
```

The staggered phases use zero-based global coordinates.  Fermion boundary
conditions are carried by `psi.phases`; the conventional choice is
`(1, 1, 1, -1)`.  Gauge links must be periodic.  The fermion field has
per-site shape `NC x 1`.
"""
struct StaggeredDiracOperator4D{T,M<:Real} <: OperatorOnKernel
    U::Vector{T}
    mass::M
end

function _validate_staggered_gauge_links(U)
    length(U) == 4 || throw(ArgumentError("U must be a vector of length 4"))
    reference = U[1]
    reference.NC1 == reference.NC2 || throw(ArgumentError(
        "staggered gauge links must be square matrices"))

    for (mu, link) in enumerate(U)
        link.NC1 == reference.NC1 && link.NC2 == reference.NC2 ||
            throw(ArgumentError("all gauge links must have the same matrix size"))
        link.gsize == reference.gsize || throw(ArgumentError(
            "all gauge links must have the same global lattice size"))
        link.PN == reference.PN || throw(ArgumentError(
            "all gauge links must have the same local lattice size"))
        link.nw == reference.nw || throw(ArgumentError(
            "all gauge links must have the same halo width"))
        link.dims == reference.dims || throw(ArgumentError(
            "all gauge links must use the same process grid"))
        all(isone, link.phases) || throw(ArgumentError(
            "gauge link U[$mu] must use periodic boundary phases"))
    end
    return nothing
end

function StaggeredDiracOperator4D(U::Vector{T}, mass::Real) where {
    T<:LatticeMatrix{4}
}
    _validate_staggered_gauge_links(U)
    real_type = typeof(real(zero(eltype(U[1].A))))
    typed_mass = convert(real_type, mass)
    isfinite(typed_mass) || throw(ArgumentError("mass must be finite"))
    return StaggeredDiracOperator4D{T,typeof(typed_mass)}(U, typed_mass)
end

export StaggeredDiracOperator4D

struct Adjoint_StaggeredDiracOperator4D{T} <: OperatorOnKernel
    parent::T
end

Base.adjoint(operator::StaggeredDiracOperator4D) =
    Adjoint_StaggeredDiracOperator4D(operator)
Base.adjoint(operator::Adjoint_StaggeredDiracOperator4D) = operator.parent

function _validate_staggered_application(result, operator, psi)
    result === psi && throw(ArgumentError(
        "staggered mul! requires distinct result and input fields"))
    result.gsize == psi.gsize && result.PN == psi.PN &&
        result.dims == psi.dims && result.nw == psi.nw || throw(ArgumentError(
            "result and fermion fields must use the same lattice geometry"))
    result.phases == psi.phases || throw(ArgumentError(
        "result and fermion fields must use the same boundary phases"))

    tolerance = sqrt(eps(typeof(real(zero(eltype(psi.A))))))
    for (mu, phase) in enumerate(psi.phases)
        isapprox(abs(phase), one(abs(phase)); atol=tolerance, rtol=tolerance) ||
            throw(ArgumentError(
                "fermion boundary phase $mu must have unit magnitude"))
    end

    for (mu, link) in enumerate(operator.U)
        link.NC1 == psi.NC1 || throw(ArgumentError(
            "gauge link U[$mu] and fermion field have different color sizes"))
        eltype(link.A) == eltype(psi.A) || throw(ArgumentError(
            "gauge link U[$mu] and fermion field have different element types"))
        link.gsize == psi.gsize && link.PN == psi.PN &&
            link.dims == psi.dims && link.nw == psi.nw || throw(ArgumentError(
                "gauge link U[$mu] and fermion field use different lattice geometry"))
    end
    return nothing
end

@inline function _staggered_direction_matvec(
    U, psi, x, xplus, xminus, output_color, ::Val{NC}) where NC
    forward = zero(eltype(psi))
    backward = zero(eltype(psi))
    @inbounds for input_color in 1:NC
        forward = muladd(
            U[output_color, input_color, x...],
            psi[input_color, 1, xplus...],
            forward)
        backward = muladd(
            conj(U[input_color, output_color, xminus...]),
            psi[input_color, 1, xminus...],
            backward)
    end
    return forward - backward
end

# The NC=3 path exposes all three complex multiply-adds to the compiler.  This
# is the bandwidth-dominated production case and avoids a small dynamic loop in
# accelerator kernels.
@inline function _staggered_direction_matvec(
    U, psi, x, xplus, xminus, output_color, ::Val{3})
    @inbounds begin
        forward = muladdmulti(
            U[output_color, 1, x...], psi[1, 1, xplus...],
            U[output_color, 2, x...], psi[2, 1, xplus...],
            U[output_color, 3, x...], psi[3, 1, xplus...])
        backward = muladdmulti(
            conj(U[1, output_color, xminus...]), psi[1, 1, xminus...],
            conj(U[2, output_color, xminus...]), psi[2, 1, xminus...],
            conj(U[3, output_color, xminus...]), psi[3, 1, xminus...])
    end
    return forward - backward
end

@inline function kernel_StaggeredDiracOperator4D!(
    site, result, U1, U2, U3, U4, mass, hopping_coefficient, psi,
    ::Val{NC}, ::Val{nw}, indexer, mpi_coordinates, local_size) where {NC,nw}

    x = delinearize(indexer, site, nw)
    x1p = shiftindices(x, shift_1p)
    x1m = shiftindices(x, shift_1m)
    x2p = shiftindices(x, shift_2p)
    x2m = shiftindices(x, shift_2m)
    x3p = shiftindices(x, shift_3p)
    x3m = shiftindices(x, shift_3m)
    x4p = shiftindices(x, shift_4p)
    x4m = shiftindices(x, shift_4m)

    eta2 = staggered_eta_global_halo(
        x, 2, nw, mpi_coordinates, local_size)
    eta3 = staggered_eta_global_halo(
        x, 3, nw, mpi_coordinates, local_size)
    eta4 = staggered_eta_global_halo(
        x, 4, nw, mpi_coordinates, local_size)

    @inbounds for output_color in 1:NC
        hopping = _staggered_direction_matvec(
            U1, psi, x, x1p, x1m, output_color, Val(NC))
        hopping += eta2 * _staggered_direction_matvec(
            U2, psi, x, x2p, x2m, output_color, Val(NC))
        hopping += eta3 * _staggered_direction_matvec(
            U3, psi, x, x3p, x3m, output_color, Val(NC))
        hopping += eta4 * _staggered_direction_matvec(
            U4, psi, x, x4p, x4m, output_color, Val(NC))
        result[output_color, 1, x...] =
            mass * psi[output_color, 1, x...] + hopping_coefficient * hopping
    end
    return nothing
end

function _apply_staggered_halo!(result, operator, psi, adjoint_operator::Bool)
    _validate_staggered_application(result, operator, psi)
    U1, U2, U3, U4 = operator.U
    ensure_halo!(U1)
    ensure_halo!(U2)
    ensure_halo!(U3)
    ensure_halo!(U4)
    ensure_halo!(psi)

    half = one(operator.mass) / 2
    hopping_coefficient = adjoint_operator ? -half : half
    _parallel_for_mutating!(result,
        prod(result.PN), kernel_StaggeredDiracOperator4D!, result.A,
        U1.A, U2.A, U3.A, U4.A, operator.mass, hopping_coefficient, psi.A,
        Val(result.NC1), Val(result.nw), result.indexer,
        result.coords, result.PN)
    return result
end

function LinearAlgebra.mul!(
    result::T,
    operator::StaggeredDiracOperator4D,
    psi::T,
) where {ET,AT,NC,nw,DI,T<:LatticeMatrix{4,ET,AT,NC,1,nw,DI}}
    return _apply_staggered_halo!(result, operator, psi, false)
end

function LinearAlgebra.mul!(
    result::T,
    operator::Adjoint_StaggeredDiracOperator4D,
    psi::T,
) where {ET,AT,NC,nw,DI,T<:LatticeMatrix{4,ET,AT,NC,1,nw,DI}}
    return _apply_staggered_halo!(result, operator.parent, psi, true)
end

# Halo-free fallback.  Periodic/twisted neighbor fields are materialized using
# the established nw=0 path, then accumulated one direction at a time.
@inline function kernel_initialize_StaggeredDiracOperator4D_nowing!(
    site, result, psi, mass, ::Val{NC}, indexer) where NC
    x = delinearize(indexer, site, 0)
    @inbounds for color in 1:NC
        result[color, 1, x...] = mass * psi[color, 1, x...]
    end
    return nothing
end

@inline function kernel_StaggeredDiracOperator4D_direction_nowing!(
    site, result, U, Uminus, psi_plus, psi_minus, coefficient,
    ::Val{NC}, indexer, ::Val{mu}, mpi_coordinates, local_size) where {NC,mu}

    x = delinearize(indexer, site, 0)
    eta = staggered_eta_global_halo(
        x, mu, 0, mpi_coordinates, local_size)
    @inbounds for output_color in 1:NC
        forward = zero(eltype(psi_plus))
        backward = zero(eltype(psi_minus))
        for input_color in 1:NC
            forward = muladd(
                U[output_color, input_color, x...],
                psi_plus[input_color, 1, x...],
                forward)
            backward = muladd(
                conj(Uminus[input_color, output_color, x...]),
                psi_minus[input_color, 1, x...],
                backward)
        end
        result[output_color, 1, x...] +=
            coefficient * eta * (forward - backward)
    end
    return nothing
end

function _apply_staggered_nowing!(result, operator, psi, adjoint_operator::Bool)
    _validate_staggered_application(result, operator, psi)
    all(link -> iszero(link.nw), operator.U) || throw(ArgumentError(
        "nw=0 staggered operators require nw=0 gauge fields"))

    _parallel_for_mutating!(result,
        prod(result.PN), kernel_initialize_StaggeredDiracOperator4D_nowing!,
        result.A, psi.A, operator.mass, Val(result.NC1), result.indexer)

    half = one(operator.mass) / 2
    coefficient = adjoint_operator ? -half : half
    for mu in 1:4
        psi_plus = _materialize_periodic_shift(psi, shifts_p[mu])
        psi_minus = _materialize_periodic_shift(psi, shifts_m[mu])
        Uminus = _materialize_periodic_shift(operator.U[mu], shifts_m[mu])
        _parallel_for_mutating!(result,
            prod(result.PN),
            kernel_StaggeredDiracOperator4D_direction_nowing!,
            result.A, operator.U[mu].A, Uminus.A,
            psi_plus.A, psi_minus.A, coefficient,
            Val(result.NC1), result.indexer, Val(mu),
            result.coords, result.PN)
    end
    return result
end

function LinearAlgebra.mul!(
    result::T,
    operator::StaggeredDiracOperator4D,
    psi::T,
) where {ET,AT,NC,DI,T<:LatticeMatrix{4,ET,AT,NC,1,0,DI}}
    return _apply_staggered_nowing!(result, operator, psi, false)
end

function LinearAlgebra.mul!(
    result::T,
    operator::Adjoint_StaggeredDiracOperator4D,
    psi::T,
) where {ET,AT,NC,DI,T<:LatticeMatrix{4,ET,AT,NC,1,0,DI}}
    return _apply_staggered_nowing!(result, operator.parent, psi, true)
end
