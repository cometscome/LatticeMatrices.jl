const clover_plane_pairs = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))

# Bridge++ uses its chiral gamma-matrix representation in the same basis as
# `gamma1`, ..., `gamma4` in this package.  In that convention the clover
# contribution to D is
#
#   -kappa * cSW * sum_{mu<nu} gamma_mu * gamma_nu * F_mu_nu,
#
# with F_mu_nu = (Q_mu_nu - Q_mu_nu') / 8 and Q the four-leaf clover.
const clover_gamma_products = ntuple(Val(6)) do plane
    mu, nu = clover_plane_pairs[plane]
    SMatrix{4,4}(γs[mu] * γs[nu])
end

"""
    CloverFieldStrength4D(U)

The six anti-Hermitian clover field-strength components, ordered as
`(12, 13, 14, 23, 24, 34)`.  Each component is

```
F_mu_nu(x) = (Q_mu_nu(x) - Q_mu_nu(x)') / 8,
```

where `Q_mu_nu` is the sum of the four counter-clockwise plaquettes touching
`x` in the `mu`-`nu` plane.
"""
struct CloverFieldStrength4D{T}
    components::NTuple{6,T}
end

Base.length(::CloverFieldStrength4D) = 6
Base.getindex(field::CloverFieldStrength4D, i::Integer) = field.components[i]

export CloverFieldStrength4D

function _validate_clover_gauge(U)
    length(U) == 4 || throw(ArgumentError("U must be a vector of length 4"))
    reference = U[1]
    reference.NC1 == reference.NC2 || throw(ArgumentError(
        "clover gauge links must be square matrices"))

    for mu in 2:4
        link = U[mu]
        link.NC1 == reference.NC1 && link.NC2 == reference.NC2 ||
            throw(ArgumentError("all gauge links must have the same matrix size"))
        link.gsize == reference.gsize ||
            throw(ArgumentError("all gauge links must have the same global lattice size"))
        link.PN == reference.PN ||
            throw(ArgumentError("all gauge links must have the same local lattice size"))
        link.nw == reference.nw ||
            throw(ArgumentError("all gauge links must have the same halo width"))
        link.dims == reference.dims ||
            throw(ArgumentError("all gauge links must use the same process grid"))
    end
    return nothing
end

@inline function _clover_q_element_halo(
    Umu, Unu, row, col, x, xpmu, xpnu, xmmu, xmnu,
    xpnu_mmu, xmmu_mnu, xpmu_mnu, ::Val{NC}) where NC

    value = zero(eltype(Umu))
    @inbounds for a in 1:NC
        for b in 1:NC
            for c in 1:NC
                # +mu, +nu, -mu, -nu
                value += Umu[row, a, x...] * Unu[a, b, xpmu...] *
                         conj(Umu[c, b, xpnu...]) * conj(Unu[col, c, x...])

                # +nu, -mu, -nu, +mu
                value += Unu[row, a, x...] * conj(Umu[b, a, xpnu_mmu...]) *
                         conj(Unu[c, b, xmmu...]) * Umu[c, col, xmmu...]

                # -mu, -nu, +mu, +nu
                value += conj(Umu[a, row, xmmu...]) *
                         conj(Unu[b, a, xmmu_mnu...]) *
                         Umu[b, c, xmmu_mnu...] * Unu[c, col, xmnu...]

                # -nu, +mu, +nu, -mu
                value += conj(Unu[a, row, xmnu...]) * Umu[a, b, xmnu...] *
                         Unu[b, c, xpmu_mnu...] * conj(Umu[col, c, x...])
            end
        end
    end
    return value
end

@inline function kernel_clover_field_strength_halo!(
    site, F, Umu, Unu, ::Val{NC}, nw, indexer,
    shift_mu_p, shift_mu_m, shift_nu_p, shift_nu_m,
    shift_nu_minus_mu, shift_minus_mu_minus_nu, shift_mu_minus_nu) where NC

    x = delinearize(indexer, site, nw)
    xpmu = shiftindices(x, shift_mu_p)
    xpnu = shiftindices(x, shift_nu_p)
    xmmu = shiftindices(x, shift_mu_m)
    xmnu = shiftindices(x, shift_nu_m)
    xpnu_mmu = shiftindices(x, shift_nu_minus_mu)
    xmmu_mnu = shiftindices(x, shift_minus_mu_minus_nu)
    xpmu_mnu = shiftindices(x, shift_mu_minus_nu)

    @inbounds for col in 1:NC
        for row in 1:col
            q_row_col = _clover_q_element_halo(
                Umu, Unu, row, col, x, xpmu, xpnu, xmmu, xmnu,
                xpnu_mmu, xmmu_mnu, xpmu_mnu, Val(NC))
            q_col_row = row == col ? q_row_col : _clover_q_element_halo(
                Umu, Unu, col, row, x, xpmu, xpnu, xmmu, xmnu,
                xpnu_mmu, xmmu_mnu, xpmu_mnu, Val(NC))
            value = 0.125 * (q_row_col - conj(q_col_row))
            F[row, col, x...] = value
            if row != col
                F[col, row, x...] = -conj(value)
            end
        end
    end
    return nothing
end

@inline function _clover_load_matrix3(U, x)
    @inbounds return (
        U[1, 1, x...], U[1, 2, x...], U[1, 3, x...],
        U[2, 1, x...], U[2, 2, x...], U[2, 3, x...],
        U[3, 1, x...], U[3, 2, x...], U[3, 3, x...],
    )
end

@inline function _clover_load_adjoint_matrix3(U, x)
    @inbounds return (
        conj(U[1, 1, x...]), conj(U[2, 1, x...]), conj(U[3, 1, x...]),
        conj(U[1, 2, x...]), conj(U[2, 2, x...]), conj(U[3, 2, x...]),
        conj(U[1, 3, x...]), conj(U[2, 3, x...]), conj(U[3, 3, x...]),
    )
end

@inline function _clover_mul_matrix3(A, B)
    return (
        muladdmulti(A[1], B[1], A[2], B[4], A[3], B[7]),
        muladdmulti(A[1], B[2], A[2], B[5], A[3], B[8]),
        muladdmulti(A[1], B[3], A[2], B[6], A[3], B[9]),
        muladdmulti(A[4], B[1], A[5], B[4], A[6], B[7]),
        muladdmulti(A[4], B[2], A[5], B[5], A[6], B[8]),
        muladdmulti(A[4], B[3], A[5], B[6], A[6], B[9]),
        muladdmulti(A[7], B[1], A[8], B[4], A[9], B[7]),
        muladdmulti(A[7], B[2], A[8], B[5], A[9], B[8]),
        muladdmulti(A[7], B[3], A[8], B[6], A[9], B[9]),
    )
end

@inline function _clover_add_matrix3(A, B)
    return (
        A[1] + B[1], A[2] + B[2], A[3] + B[3],
        A[4] + B[4], A[5] + B[5], A[6] + B[6],
        A[7] + B[7], A[8] + B[8], A[9] + B[9],
    )
end

@inline function _clover_path_matrix3(A, B, C, D)
    return _clover_mul_matrix3(
        _clover_mul_matrix3(_clover_mul_matrix3(A, B), C), D)
end

@inline function _clover_store_antihermitian3!(F, x, q)
    factor = one(real(q[1])) / 8
    value11 = factor * (q[1] - conj(q[1]))
    value12 = factor * (q[2] - conj(q[4]))
    value13 = factor * (q[3] - conj(q[7]))
    value22 = factor * (q[5] - conj(q[5]))
    value23 = factor * (q[6] - conj(q[8]))
    value33 = factor * (q[9] - conj(q[9]))
    @inbounds begin
        F[1, 1, x...] = value11
        F[1, 2, x...] = value12
        F[1, 3, x...] = value13
        F[2, 1, x...] = -conj(value12)
        F[2, 2, x...] = value22
        F[2, 3, x...] = value23
        F[3, 1, x...] = -conj(value13)
        F[3, 2, x...] = -conj(value23)
        F[3, 3, x...] = value33
    end
    return nothing
end

@inline function kernel_clover_field_strength_halo!(
    site, F, Umu, Unu, ::Val{3}, nw, indexer,
    shift_mu_p, shift_mu_m, shift_nu_p, shift_nu_m,
    shift_nu_minus_mu, shift_minus_mu_minus_nu, shift_mu_minus_nu)

    x = delinearize(indexer, site, nw)
    xpmu = shiftindices(x, shift_mu_p)
    xpnu = shiftindices(x, shift_nu_p)
    xmmu = shiftindices(x, shift_mu_m)
    xmnu = shiftindices(x, shift_nu_m)
    xpnu_mmu = shiftindices(x, shift_nu_minus_mu)
    xmmu_mnu = shiftindices(x, shift_minus_mu_minus_nu)
    xpmu_mnu = shiftindices(x, shift_mu_minus_nu)

    q = _clover_path_matrix3(
        _clover_load_matrix3(Umu, x),
        _clover_load_matrix3(Unu, xpmu),
        _clover_load_adjoint_matrix3(Umu, xpnu),
        _clover_load_adjoint_matrix3(Unu, x),
    )
    q = _clover_add_matrix3(q, _clover_path_matrix3(
        _clover_load_matrix3(Unu, x),
        _clover_load_adjoint_matrix3(Umu, xpnu_mmu),
        _clover_load_adjoint_matrix3(Unu, xmmu),
        _clover_load_matrix3(Umu, xmmu),
    ))
    q = _clover_add_matrix3(q, _clover_path_matrix3(
        _clover_load_adjoint_matrix3(Umu, xmmu),
        _clover_load_adjoint_matrix3(Unu, xmmu_mnu),
        _clover_load_matrix3(Umu, xmmu_mnu),
        _clover_load_matrix3(Unu, xmnu),
    ))
    q = _clover_add_matrix3(q, _clover_path_matrix3(
        _clover_load_adjoint_matrix3(Unu, xmnu),
        _clover_load_matrix3(Umu, xmnu),
        _clover_load_matrix3(Unu, xpmu_mnu),
        _clover_load_adjoint_matrix3(Umu, x),
    ))

    return _clover_store_antihermitian3!(F, x, q)
end

@inline function _clover_q_element_nowing(
    Umu, Unu, Unu_pmu, Umu_pnu, Umu_mmu_pnu, Unu_mmu, Umu_mmu,
    Unu_mnu, Umu_mmu_mnu, Unu_mmu_mnu, Umu_mnu, Unu_pmu_mnu,
    row, col, x, ::Val{NC}) where NC

    value = zero(eltype(Umu))
    @inbounds for a in 1:NC
        for b in 1:NC
            for c in 1:NC
                value += Umu[row, a, x...] * Unu_pmu[a, b, x...] *
                         conj(Umu_pnu[c, b, x...]) * conj(Unu[col, c, x...])

                value += Unu[row, a, x...] * conj(Umu_mmu_pnu[b, a, x...]) *
                         conj(Unu_mmu[c, b, x...]) * Umu_mmu[c, col, x...]

                value += conj(Umu_mmu[a, row, x...]) *
                         conj(Unu_mmu_mnu[b, a, x...]) *
                         Umu_mmu_mnu[b, c, x...] * Unu_mnu[c, col, x...]

                value += conj(Unu_mnu[a, row, x...]) * Umu_mnu[a, b, x...] *
                         Unu_pmu_mnu[b, c, x...] * conj(Umu[col, c, x...])
            end
        end
    end
    return value
end

@inline function kernel_clover_field_strength_nowing!(
    site, F, Umu, Unu, Unu_pmu, Umu_pnu, Umu_mmu_pnu, Unu_mmu,
    Umu_mmu, Unu_mnu, Umu_mmu_mnu, Unu_mmu_mnu, Umu_mnu, Unu_pmu_mnu,
    ::Val{NC}, indexer) where NC

    x = delinearize(indexer, site, 0)
    @inbounds for col in 1:NC
        for row in 1:col
            q_row_col = _clover_q_element_nowing(
                Umu, Unu, Unu_pmu, Umu_pnu, Umu_mmu_pnu, Unu_mmu,
                Umu_mmu, Unu_mnu, Umu_mmu_mnu, Unu_mmu_mnu, Umu_mnu,
                Unu_pmu_mnu, row, col, x, Val(NC))
            q_col_row = row == col ? q_row_col : _clover_q_element_nowing(
                Umu, Unu, Unu_pmu, Umu_pnu, Umu_mmu_pnu, Unu_mmu,
                Umu_mmu, Unu_mnu, Umu_mmu_mnu, Unu_mmu_mnu, Umu_mnu,
                Unu_pmu_mnu, col, row, x, Val(NC))
            value = 0.125 * (q_row_col - conj(q_col_row))
            F[row, col, x...] = value
            if row != col
                F[col, row, x...] = -conj(value)
            end
        end
    end
    return nothing
end

@inline function kernel_clover_field_strength_nowing!(
    site, F, Umu, Unu, Unu_pmu, Umu_pnu, Umu_mmu_pnu, Unu_mmu,
    Umu_mmu, Unu_mnu, Umu_mmu_mnu, Unu_mmu_mnu, Umu_mnu, Unu_pmu_mnu,
    ::Val{3}, indexer)

    x = delinearize(indexer, site, 0)
    q = _clover_path_matrix3(
        _clover_load_matrix3(Umu, x),
        _clover_load_matrix3(Unu_pmu, x),
        _clover_load_adjoint_matrix3(Umu_pnu, x),
        _clover_load_adjoint_matrix3(Unu, x),
    )
    q = _clover_add_matrix3(q, _clover_path_matrix3(
        _clover_load_matrix3(Unu, x),
        _clover_load_adjoint_matrix3(Umu_mmu_pnu, x),
        _clover_load_adjoint_matrix3(Unu_mmu, x),
        _clover_load_matrix3(Umu_mmu, x),
    ))
    q = _clover_add_matrix3(q, _clover_path_matrix3(
        _clover_load_adjoint_matrix3(Umu_mmu, x),
        _clover_load_adjoint_matrix3(Unu_mmu_mnu, x),
        _clover_load_matrix3(Umu_mmu_mnu, x),
        _clover_load_matrix3(Unu_mnu, x),
    ))
    q = _clover_add_matrix3(q, _clover_path_matrix3(
        _clover_load_adjoint_matrix3(Unu_mnu, x),
        _clover_load_matrix3(Umu_mnu, x),
        _clover_load_matrix3(Unu_pmu_mnu, x),
        _clover_load_adjoint_matrix3(Umu, x),
    ))

    return _clover_store_antihermitian3!(F, x, q)
end

@inline _clover_shift(mu, amount) = ntuple(d -> d == mu ? amount : 0, 4)
@inline _clover_shift(mu, amount_mu, nu, amount_nu) =
    ntuple(d -> d == mu ? amount_mu : (d == nu ? amount_nu : 0), 4)

function _update_clover_component_halo!(F, Umu, Unu, mu, nu)
    ensure_halo!(Umu)
    ensure_halo!(Unu)

    shift_mu_p = _clover_shift(mu, 1)
    shift_mu_m = _clover_shift(mu, -1)
    shift_nu_p = _clover_shift(nu, 1)
    shift_nu_m = _clover_shift(nu, -1)
    shift_nu_minus_mu = _clover_shift(mu, -1, nu, 1)
    shift_minus_mu_minus_nu = _clover_shift(mu, -1, nu, -1)
    shift_mu_minus_nu = _clover_shift(mu, 1, nu, -1)

    _parallel_for_mutating!(F,
        prod(F.PN), kernel_clover_field_strength_halo!, F.A, Umu.A, Unu.A,
        Val(F.NC1), F.nw, F.indexer, shift_mu_p, shift_mu_m, shift_nu_p,
        shift_nu_m, shift_nu_minus_mu, shift_minus_mu_minus_nu,
        shift_mu_minus_nu)
    return F
end

function _update_clover_component_nowing!(F, Umu, Unu, mu, nu)
    shift_mu_p = _clover_shift(mu, 1)
    shift_mu_m = _clover_shift(mu, -1)
    shift_nu_p = _clover_shift(nu, 1)
    shift_nu_m = _clover_shift(nu, -1)
    shift_nu_minus_mu = _clover_shift(mu, -1, nu, 1)
    shift_minus_mu_minus_nu = _clover_shift(mu, -1, nu, -1)
    shift_mu_minus_nu = _clover_shift(mu, 1, nu, -1)

    Unu_pmu = _materialize_periodic_shift(Unu, shift_mu_p)
    Umu_pnu = _materialize_periodic_shift(Umu, shift_nu_p)
    Umu_mmu_pnu = _materialize_periodic_shift(Umu, shift_nu_minus_mu)
    Unu_mmu = _materialize_periodic_shift(Unu, shift_mu_m)
    Umu_mmu = _materialize_periodic_shift(Umu, shift_mu_m)
    Unu_mnu = _materialize_periodic_shift(Unu, shift_nu_m)
    Umu_mmu_mnu = _materialize_periodic_shift(Umu, shift_minus_mu_minus_nu)
    Unu_mmu_mnu = _materialize_periodic_shift(Unu, shift_minus_mu_minus_nu)
    Umu_mnu = _materialize_periodic_shift(Umu, shift_nu_m)
    Unu_pmu_mnu = _materialize_periodic_shift(Unu, shift_mu_minus_nu)

    _parallel_for_mutating!(F,
        prod(F.PN), kernel_clover_field_strength_nowing!, F.A, Umu.A, Unu.A,
        Unu_pmu.A, Umu_pnu.A, Umu_mmu_pnu.A, Unu_mmu.A, Umu_mmu.A,
        Unu_mnu.A, Umu_mmu_mnu.A, Unu_mmu_mnu.A, Umu_mnu.A,
        Unu_pmu_mnu.A, Val(F.NC1), F.indexer)
    return F
end

function _update_clover_component!(F, Umu, Unu, mu, nu)
    if iszero(F.nw)
        return _update_clover_component_nowing!(F, Umu, Unu, mu, nu)
    end
    return _update_clover_component_halo!(F, Umu, Unu, mu, nu)
end

function CloverFieldStrength4D(U::Vector{T}) where {T<:LatticeMatrix{4}}
    _validate_clover_gauge(U)
    components = ntuple(_ -> similar(U[1]), Val(6))
    field = CloverFieldStrength4D{T}(components)
    update_clover!(field, U)
    return field
end

"""
    update_clover!(field, U)
    update_clover!(operator)
    update_clover!(operator, U)

Recompute the six clover field-strength components.  Call this after modifying
the gauge links stored by a `WilsonDiracCloverOperator4D`.  The two-argument
operator method also replaces the four Wilson-link references before refreshing
the cache.
"""
function update_clover!(field::CloverFieldStrength4D{T}, U::Vector{T}) where T
    _validate_clover_gauge(U)
    for plane in 1:6
        mu, nu = clover_plane_pairs[plane]
        _update_clover_component!(field[plane], U[mu], U[nu], mu, nu)
    end
    return field
end

export update_clover!

"""
    WilsonDiracCloverOperator4D(U, kappa, cSW)

Wilson--clover operator in the Bridge++ chiral-basis convention:

```
D = D_W - kappa*cSW*sum(gamma_mu*gamma_nu*F_mu_nu, mu < nu),
F_mu_nu = (Q_mu_nu - Q_mu_nu')/8.
```

The field strength is cached.  Direct `mul!` callers must call
`update_clover!(D)` after mutating `U`; the explicit-link
`mul_cached_clover!` entry point detects link changes and refreshes it
automatically.
"""
mutable struct CloverCacheState{T}
    source_links::NTuple{4,T}
    core_epochs::NTuple{4,UInt64}
end

@inline _clover_source_links(U) = ntuple(mu -> U[mu], Val(4))

@inline _clover_core_epochs(U) =
    ntuple(mu -> U[mu].halo_epoch.core, Val(4))

@inline function _record_clover_cache_state!(operator, U)
    operator.cache_state.source_links = _clover_source_links(U)
    operator.cache_state.core_epochs = _clover_core_epochs(U)
    return operator
end

struct WilsonDiracCloverOperator4D{W,C,S} <: OperatorOnKernel
    wilson::W
    cSW::Float64
    clover::C
    cache_state::S

    function WilsonDiracCloverOperator4D{W,C,S}(
        wilson::W, cSW, clover::C, cache_state::S,
    ) where {W,C,S<:CloverCacheState}
        return new{W,C,S}(wilson, cSW, clover, cache_state)
    end
end

function WilsonDiracCloverOperator4D(U::Vector{T}, kappa, cSW) where {T<:LatticeMatrix{4}}
    wilson = WilsonDiracOperator4D(U, kappa)
    clover = CloverFieldStrength4D(U)
    source_links = _clover_source_links(U)
    cache_state = CloverCacheState(source_links, _clover_core_epochs(U))
    return WilsonDiracCloverOperator4D{
        typeof(wilson),typeof(clover),typeof(cache_state)
    }(
        wilson, cSW, clover, cache_state)
end

export WilsonDiracCloverOperator4D

struct Adjoint_WilsonDiracCloverOperator4D{T} <: OperatorOnKernel
    parent::T
end

Base.adjoint(operator::WilsonDiracCloverOperator4D) =
    Adjoint_WilsonDiracCloverOperator4D(operator)
Base.adjoint(operator::Adjoint_WilsonDiracCloverOperator4D) = operator.parent

function update_clover!(operator::WilsonDiracCloverOperator4D)
    update_clover!(operator.clover, operator.wilson.U)
    return _record_clover_cache_state!(operator, operator.wilson.U)
end

function update_clover!(operator::WilsonDiracCloverOperator4D, U::Vector)
    _validate_clover_gauge(U)
    length(operator.wilson.U) == 4 || throw(ArgumentError(
        "WilsonDiracCloverOperator4D must contain four gauge links"))
    operator.wilson.U .= U
    return update_clover!(operator)
end

@inline function kernel_add_clover_term!(site, result,
    F12, F13, F14, F23, F24, F34, psi, coefficient, ::Val{NC}, nw,
    indexer) where NC

    x = delinearize(indexer, site, nw)
    imaginary_unit = one(eltype(psi)) * im
    @inbounds for color in 1:NC
        value1 = zero(eltype(psi))
        value2 = zero(eltype(psi))
        value3 = zero(eltype(psi))
        value4 = zero(eltype(psi))
        for input_color in 1:NC
            f12 = F12[color, input_color, x...]
            f13 = F13[color, input_color, x...]
            f14 = F14[color, input_color, x...]
            f23 = F23[color, input_color, x...]
            f24 = F24[color, input_color, x...]
            f34 = F34[color, input_color, x...]

            diagonal_plus = imaginary_unit * (f12 + f34)
            diagonal_minus = imaginary_unit * (f12 - f34)
            offdiagonal_plus = imaginary_unit * (f14 + f23)
            offdiagonal_minus = imaginary_unit * (f23 - f14)

            psi1 = psi[input_color, 1, x...]
            psi2 = psi[input_color, 2, x...]
            psi3 = psi[input_color, 3, x...]
            psi4 = psi[input_color, 4, x...]

            value1 += diagonal_plus * psi1 +
                      (-f13 + offdiagonal_plus + f24) * psi2
            value2 += -diagonal_plus * psi2 +
                      (f13 + offdiagonal_plus - f24) * psi1
            value3 += diagonal_minus * psi3 +
                      (-f13 + offdiagonal_minus - f24) * psi4
            value4 += -diagonal_minus * psi4 +
                      (f13 + offdiagonal_minus + f24) * psi3
        end
        result[color, 1, x...] += coefficient * value1
        result[color, 2, x...] += coefficient * value2
        result[color, 3, x...] += coefficient * value3
        result[color, 4, x...] += coefficient * value4
    end
    return nothing
end

function _add_clover_term!(result, operator::WilsonDiracCloverOperator4D, psi)
    F12, F13, F14, F23, F24, F34 = operator.clover.components
    coefficient = -operator.wilson.κ * operator.cSW
    _parallel_for_mutating!(result,
        prod(result.PN), kernel_add_clover_term!, result.A,
        F12.A, F13.A, F14.A, F23.A, F24.A, F34.A, psi.A, coefficient,
        Val(result.NC1), result.nw, result.indexer)
    return result
end

@inline function kernel_WilsonDiracCloverOperator4D!(site, result,
    U1, U2, U3, U4, kappa, psi, F12, F13, F14, F23, F24, F34,
    coefficient, ::Val{NC}, ::Val{nw}, indexer) where {NC,nw}

    kernel_WilsonDiracOperator4D!(
        site, result, U1, U2, U3, U4, kappa, psi, Val(NC), Val(nw), indexer)
    kernel_add_clover_term!(site, result, F12, F13, F14, F23, F24, F34,
        psi, coefficient, Val(NC), nw, indexer)
    return nothing
end

@inline function kernel_adjoint_WilsonDiracCloverOperator4D!(site, result,
    U1, U2, U3, U4, kappa, psi, F12, F13, F14, F23, F24, F34,
    coefficient, ::Val{NC}, ::Val{nw}, indexer) where {NC,nw}

    kernel_adjoint_WilsonDiracOperator4D!(
        site, result, U1, U2, U3, U4, kappa, psi, Val(NC), Val(nw), indexer)
    # The on-site clover term is Hermitian because both gamma_mu*gamma_nu
    # and F_mu_nu are anti-Hermitian and act on independent index spaces.
    kernel_add_clover_term!(site, result, F12, F13, F14, F23, F24, F34,
        psi, coefficient, Val(NC), nw, indexer)
    return nothing
end

function LinearAlgebra.mul!(result::T, operator::WilsonDiracCloverOperator4D,
    psi::T) where {T1,AT1,NC,nw,DI,
    T<:LatticeMatrix{4,T1,AT1,NC,4,nw,DI}}

    U1, U2, U3, U4 = operator.wilson.U
    ensure_halo!(U1)
    ensure_halo!(U2)
    ensure_halo!(U3)
    ensure_halo!(U4)
    ensure_halo!(psi)
    F12, F13, F14, F23, F24, F34 = operator.clover.components
    coefficient = -operator.wilson.κ * operator.cSW
    _parallel_for_mutating!(result,
        prod(result.PN), kernel_WilsonDiracCloverOperator4D!, result.A,
        U1.A, U2.A, U3.A, U4.A, operator.wilson.κ, psi.A,
        F12.A, F13.A, F14.A, F23.A, F24.A, F34.A, coefficient,
        Val(NC), Val(nw), result.indexer)
    return result
end

function LinearAlgebra.mul!(result::T,
    operator::Adjoint_WilsonDiracCloverOperator4D,
    psi::T) where {T1,AT1,NC,nw,DI,
    T<:LatticeMatrix{4,T1,AT1,NC,4,nw,DI}}

    parent = operator.parent
    U1, U2, U3, U4 = parent.wilson.U
    ensure_halo!(U1)
    ensure_halo!(U2)
    ensure_halo!(U3)
    ensure_halo!(U4)
    ensure_halo!(psi)
    F12, F13, F14, F23, F24, F34 = parent.clover.components
    coefficient = -parent.wilson.κ * parent.cSW
    _parallel_for_mutating!(result,
        prod(result.PN), kernel_adjoint_WilsonDiracCloverOperator4D!, result.A,
        U1.A, U2.A, U3.A, U4.A, parent.wilson.κ, psi.A,
        F12.A, F13.A, F14.A, F23.A, F24.A, F34.A, coefficient,
        Val(NC), Val(nw), result.indexer)
    return result
end

@inline function _clover_cache_is_current(
    operator::WilsonDiracCloverOperator4D,
    U1, U2, U3, U4,
)
    U = (U1, U2, U3, U4)
    source_links = operator.cache_state.source_links
    epochs = operator.cache_state.core_epochs
    for mu in 1:4
        source_links[mu] === U[mu] || return false
        epochs[mu] == U[mu].halo_epoch.core || return false
    end
    return true
end

@inline function _ensure_clover_cache_current!(
    operator::WilsonDiracCloverOperator4D,
    U1::T, U2::T, U3::T, U4::T,
) where {T<:LatticeMatrix{4}}
    _clover_cache_is_current(operator, U1, U2, U3, U4) && return operator
    return update_clover!(operator, T[U1, U2, U3, U4])
end

"""
    mul_cached_clover!(result, cache, U1, U2, U3, U4, psi)

Apply a Wilson--clover operator using the field-strength cache stored in
`cache`, while taking the Wilson links explicitly from `U1`, ..., `U4`.

This entry point is intended for repeated applications such as a Krylov solve.
It compares the identity and core-data epoch of each explicit link with the
links used to construct the field-strength cache.  The cache is refreshed
automatically on the first application after a link changes and is reused by
subsequent applications.  Mutations through `link.A` must be followed by
`mark_halo_dirty!(link)` so that the change is observable.

Its Enzyme reverse rule treats the cached field strength as derived from the
four explicit links and accumulates both the Wilson and clover link cotangents
without differentiating through the cache construction.
"""
function mul_cached_clover!(
    result::T,
    cache::WilsonDiracCloverOperator4D,
    U1::G, U2::G, U3::G, U4::G,
    psi::T,
) where {ET,AT,NC,nw,DI,
    T<:LatticeMatrix{4,ET,AT,NC,4,nw,DI},G<:LatticeMatrix{4}}
    _ensure_clover_cache_current!(cache, U1, U2, U3, U4)
    return LinearAlgebra.mul!(result, cache, psi)
end

"""
    mul_cached_clover_adjoint!(result, cache, U1, U2, U3, U4, psi)

Apply the adjoint of `mul_cached_clover!`.  The same refreshed field-strength
cache is shared by the forward and adjoint applications.
"""
function mul_cached_clover_adjoint!(
    result::T,
    cache::WilsonDiracCloverOperator4D,
    U1::G, U2::G, U3::G, U4::G,
    psi::T,
) where {ET,AT,NC,nw,DI,
    T<:LatticeMatrix{4,ET,AT,NC,4,nw,DI},G<:LatticeMatrix{4}}
    _ensure_clover_cache_current!(cache, U1, U2, U3, U4)
    return LinearAlgebra.mul!(result, adjoint(cache), psi)
end

export mul_cached_clover!, mul_cached_clover_adjoint!

# Without halo cells the Wilson part has to materialize each periodic shift.
# Keep that established path and add the local clover term in a second kernel.
function LinearAlgebra.mul!(result::T, operator::WilsonDiracCloverOperator4D,
    psi::T) where {T1,AT1,NC,DI,
    T<:LatticeMatrix{4,T1,AT1,NC,4,0,DI}}

    mul!(result, operator.wilson, psi)
    return _add_clover_term!(result, operator, psi)
end


function LinearAlgebra.mul!(result::T,
    operator::Adjoint_WilsonDiracCloverOperator4D,
    psi::T) where {T1,AT1,NC,DI,
    T<:LatticeMatrix{4,T1,AT1,NC,4,0,DI}}

    mul!(result, adjoint(operator.parent.wilson), psi)
    return _add_clover_term!(result, operator.parent, psi)
end
