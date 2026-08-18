struct Wilson_parameters{K<:Real,M<:Real}
    κ_wilson::K
    M_wilson::M
end
using InteractiveUtils
#using CUDA #debug

struct D5DW_MobiusDomainwallOperator5D{T,L5,R<:Real,WP<:Wilson_parameters} <: OperatorOnKernel
    U::Vector{T}
    mass::R
    wilson_params::WP
    b::R
    c::R


    function D5DW_MobiusDomainwallOperator5D(
        U::Vector{T}, L5, mass::Real, M::Real, b::Real, c::Real,
    ) where {T<:LatticeMatrix}
        length(U) == 4 || throw(ArgumentError(
            "D5DW_MobiusDomainwallOperator5D requires four gauge links"))
        L5 isa Integer || throw(ArgumentError("L5 must be an integer"))
        L5 > 0 || throw(ArgumentError("L5 must be positive"))

        R = promote_type(
            typeof(float(mass)), typeof(float(M)),
            typeof(float(b)), typeof(float(c)))
        mass_R, M_R, b_R, c_R = R(mass), R(M), R(b), R(c)
        r = one(R)
        Dim = length(U)
        κ_wilson = one(R) / (2 * Dim * r + 2M_R)
        wilsonparam = Wilson_parameters(κ_wilson, M_R)

        if b_R == 1 && c_R == 1
            println("Shamir kernel (standard DW) is used")
        elseif b_R == 2 && c_R == 0
            println("Borici/Wilson kernel (truncated overlap) is used")
        elseif b_R == 2 && c_R == 1
            println("scaled Shamir kernel (Mobius DW) is used")
        end

        return new{T,L5,R,typeof(wilsonparam)}(
            U, mass_R, wilsonparam, b_R, c_R)
    end
end
export D5DW_MobiusDomainwallOperator5D


"""
    D5DW_GeneralizedDomainwallOperator5D(U, L5, mass, M, a, b, c)

Generalized five-dimensional domain-wall operator

```math
D_5 = A\\left[I-F_m+D_W(B+C F_m)\\right],
```

where `A=diag(a)`, `B=diag(b)`, and `C=diag(c)` act on the fifth
coordinate.  The three coefficient arguments are vectors of length `L5`.
They are copied to the active JACC backend so that the same operator can be
used by threaded and accelerator kernels.
"""
struct D5DW_GeneralizedDomainwallOperator5D{
    T,L5,R<:Real,WP<:Wilson_parameters,CA<:AbstractVector{R},
} <: OperatorOnKernel
    U::Vector{T}
    mass::R
    wilson_params::WP
    a::CA
    b::CA
    c::CA

    function D5DW_GeneralizedDomainwallOperator5D(
        U::Vector{T}, L5, mass::Real, M::Real,
        a::AbstractVector{<:Real}, b::AbstractVector{<:Real},
        c::AbstractVector{<:Real},
    ) where {T<:LatticeMatrix}
        length(U) == 4 || throw(ArgumentError(
            "D5DW_GeneralizedDomainwallOperator5D requires four gauge links"))
        L5 isa Integer || throw(ArgumentError("L5 must be an integer"))
        L5 > 0 || throw(ArgumentError("L5 must be positive"))
        for (name, coefficients) in (("a", a), ("b", b), ("c", c))
            length(coefficients) == L5 || throw(DimensionMismatch(
                "$name must have length L5=$L5, got $(length(coefficients))"))
        end

        R = promote_type(
            typeof(float(mass)), typeof(float(M)),
            typeof(float(zero(eltype(a)))),
            typeof(float(zero(eltype(b)))),
            typeof(float(zero(eltype(c)))))
        mass_R, M_R = R(mass), R(M)
        a_host = R.(collect(a))
        b_host = R.(collect(b))
        c_host = R.(collect(c))
        for (name, coefficients) in
            (("a", a_host), ("b", b_host), ("c", c_host))
            all(isfinite, coefficients) || throw(ArgumentError(
                "$name coefficients must all be finite"))
        end

        Dim = length(U)
        kappa_wilson = one(R) / (2 * Dim + 2M_R)
        wilsonparam = Wilson_parameters(kappa_wilson, M_R)
        a_backend = JACC.array(a_host)
        b_backend = JACC.array(b_host)
        c_backend = JACC.array(c_host)
        CA = typeof(a_backend)
        b_backend isa CA && c_backend isa CA || error(
            "generalized domain-wall coefficients use incompatible backends")
        return new{T,L5,R,typeof(wilsonparam),CA}(
            U, mass_R, wilsonparam, a_backend, b_backend, c_backend)
    end
end
export D5DW_GeneralizedDomainwallOperator5D


struct Adjoint_D5DW_GeneralizedDomainwallOperator5D{T} <: OperatorOnKernel
    parent::T
end

function Base.adjoint(A::T) where {T<:D5DW_GeneralizedDomainwallOperator5D}
    Adjoint_D5DW_GeneralizedDomainwallOperator5D{typeof(A)}(A)
end
Base.adjoint(A::Adjoint_D5DW_GeneralizedDomainwallOperator5D) = A.parent



struct Adjoint_D5DW_MobiusDomainwallOperator5D{T} <: OperatorOnKernel
    parent::T
end

function Base.adjoint(A::T) where {T<:D5DW_MobiusDomainwallOperator5D}
    Adjoint_D5DW_MobiusDomainwallOperator5D{typeof(A)}(A)
end
Base.adjoint(A::Adjoint_D5DW_MobiusDomainwallOperator5D) = A.parent

@inline @inbounds function get_mass(x::T) where {T<:D5DW_MobiusDomainwallOperator5D}
    return x.mass
end

@inline @inbounds function get_wilson_params(x::T) where {T<:D5DW_MobiusDomainwallOperator5D}
    return x.wilson_params
end

@inline @inbounds function get_bc(x::T) where {T<:D5DW_MobiusDomainwallOperator5D}
    return x.b, x.c
end

@inline @inbounds get_mass(x::D5DW_GeneralizedDomainwallOperator5D) = x.mass
@inline @inbounds get_wilson_params(x::D5DW_GeneralizedDomainwallOperator5D) =
    x.wilson_params
@inline @inbounds get_abc(x::D5DW_GeneralizedDomainwallOperator5D) =
    (x.a, x.b, x.c)

@inline function _require_5d_halo(::Val{nw}) where nw
    nw > 0 || throw(ArgumentError(
        "5D Dirac operators do not support nw=0 yet; construct the 5D fields with nw >= 1"))
    return nothing
end

@inline function _ensure_5d_operator_halo!(U, ψ)
    ensure_halo!(U[1])
    ensure_halo!(U[2])
    ensure_halo!(U[3])
    ensure_halo!(U[4])
    ensure_halo!(ψ)
    return nothing
end

#LatticeMatrix_standard{D,T,AT,NC1,NC2,nw,DI}
function LinearAlgebra.mul!(C::TC,
    Dirac::TD, ψ::Tp) where {T1,AT1,NC1,nw,DI,L5,TU,
    TC<:LatticeMatrix{5,T1,AT1,NC1,4,nw,DI},TD<:D5DW_MobiusDomainwallOperator5D{TU,L5},
    Tp<:LatticeMatrix{5,T1,AT1,NC1,4,nw,DI}}

    _require_5d_halo(Val(nw))
    _ensure_5d_operator_halo!(Dirac.U, ψ)

    
    U1 = get_matrix(Dirac.U[1])
    U2 = get_matrix(Dirac.U[2])
    U3 = get_matrix(Dirac.U[3])
    U4 = get_matrix(Dirac.U[4])
    ψdata = get_matrix(ψ)
    Cdata = get_matrix(C)
    mass = get_mass(Dirac)
    wilson_params = get_wilson_params(Dirac)
    b, c = get_bc(Dirac)
    coeff_plus = (b + c) / 2
    coeff_minus = -(b - c) / 2
    
    #println("mass = ", mass)


    
    _parallel_for_mutating!(C,
        prod(C.PN), kernel_D5DW_MobiusDomainwallOperator5D!,
        Cdata, U1, U2, U3, U4, mass, wilson_params, ψdata,
        Val(NC1), Val(nw), C.indexer, Val(L5), coeff_plus, coeff_minus)
        

end

function LinearAlgebra.mul!(C::TC,
    Dirac::TD, psi::Tp) where {T1,AT1,NC1,nw,DI,L5,TU,
    TC<:LatticeMatrix{5,T1,AT1,NC1,4,nw,DI},
    TD<:D5DW_GeneralizedDomainwallOperator5D{TU,L5},
    Tp<:LatticeMatrix{5,T1,AT1,NC1,4,nw,DI}}

    _require_5d_halo(Val(nw))
    _ensure_5d_operator_halo!(Dirac.U, psi)

    U1 = get_matrix(Dirac.U[1])
    U2 = get_matrix(Dirac.U[2])
    U3 = get_matrix(Dirac.U[3])
    U4 = get_matrix(Dirac.U[4])
    a, b, c = get_abc(Dirac)
    _parallel_for_mutating!(C,
        prod(C.PN), kernel_D5DW_GeneralizedDomainwallOperator5D!,
        get_matrix(C), U1, U2, U3, U4, get_mass(Dirac),
        get_wilson_params(Dirac), get_matrix(psi), a, b, c,
        Val(NC1), Val(nw), C.indexer, Val(L5))
    return nothing
end

function kernel_D5DW_GeneralizedDomainwallOperator5D!(
    i, C, U1, U2, U3, U4, mass, wilson_params, psi,
    a, b, c, ::Val{NC1}, ::Val{nw}, dindexer, ::Val{L5},
) where {NC1,nw,L5}
    indices = delinearize(dindexer, i, nw)
    indices_1p = shiftindices(indices, shift_1p5D)
    indices_1m = shiftindices(indices, shift_1m5D)
    indices_2p = shiftindices(indices, shift_2p5D)
    indices_2m = shiftindices(indices, shift_2m5D)
    indices_3p = shiftindices(indices, shift_3p5D)
    indices_3m = shiftindices(indices, shift_3m5D)
    indices_4p = shiftindices(indices, shift_4p5D)
    indices_4m = shiftindices(indices, shift_4m5D)
    indices_5p = shiftindices(indices, shift_5p5D)
    indices_5m = shiftindices(indices, shift_5m5D)
    s = indices[5] - nw

    kernel_apply_1pD!(
        C, psi, U1, U2, U3, U4, wilson_params.κ_wilson, b[s],
        indices, Val(NC1), indices_1p, indices_1m, indices_2p, indices_2m,
        indices_3p, indices_3m, indices_4p, indices_4m)
    kernel_apply_1mD_F!(
        C, psi, U1, U2, U3, U4, wilson_params.κ_wilson, -c[s],
        indices, Val(NC1), indices_5p, indices_5m,
        mass, Val(L5), Val(nw))

    @inbounds for spin in 1:4, color in 1:NC1
        C[color, spin, indices...] *= a[s]
    end
    return nothing
end

const shift_1p5D = (1, 0, 0, 0, 0)
const shift_1m5D = (-1, 0, 0, 0, 0)
const shift_2p5D = (0, 1, 0, 0, 0)
const shift_2m5D = (0, -1, 0, 0, 0)
const shift_3p5D = (0, 0, 1, 0, 0)
const shift_3m5D = (0, 0, -1, 0, 0)
const shift_4p5D = (0, 0, 0, 1, 0)
const shift_4m5D = (0, 0, 0, -1, 0)
const shift_5p5D = (0, 0, 0, 0, 1)
const shift_5m5D = (0, 0, 0, 0, -1)

@inline function _domainwall_half_project_values3(
    psi1, psi2, psi3, psi4, ::Val{PM}, ::Val{MU},
) where {PM,MU}
    if MU == 1
        return psi1 - PM * im * psi4, psi2 - PM * im * psi3
    elseif MU == 2
        return psi1 - PM * psi4, psi2 + PM * psi3
    elseif MU == 3
        return psi1 - PM * im * psi3, psi2 + PM * im * psi4
    else
        return psi1 - PM * psi3, psi2 - PM * psi4
    end
end

# Project a virtual spinor whose lower and upper chiral components may come
# from different fifth-coordinate slices.  This keeps the established field
# layout while allowing B*psi + C*F*psi to be consumed directly by the
# Wilson hopping term, without materialising an intermediate 5D field.
@inline function _domainwall_combined_half_project3(
    psi, color, base_indices, low_indices, high_indices,
    base_coefficient, low_coefficient, high_coefficient,
    pm, mu,
)
    @inbounds begin
        psi1 = base_coefficient * psi[color, 1, base_indices...] +
               low_coefficient * psi[color, 1, low_indices...]
        psi2 = base_coefficient * psi[color, 2, base_indices...] +
               low_coefficient * psi[color, 2, low_indices...]
        psi3 = base_coefficient * psi[color, 3, base_indices...] +
               high_coefficient * psi[color, 3, high_indices...]
        psi4 = base_coefficient * psi[color, 4, base_indices...] +
               high_coefficient * psi[color, 4, high_indices...]
    end
    return _domainwall_half_project_values3(
        psi1, psi2, psi3, psi4, pm, mu)
end

# Form the two virtual spinors needed by the adjoint at the same time.  Their
# centre-slice contribution is identical, so reading it once avoids duplicate
# spinor traffic.  The returned half spinors correspond to the source slices
# selected by the upper and lower output chiralities, respectively.
@inline function _domainwall_paired_half_project3(
    psi, color, base_indices, low_indices, high_indices,
    base_coefficient, low_coefficient, high_coefficient,
    pm, mu,
)
    @inbounds begin
        base1 = psi[color, 1, base_indices...]
        base2 = psi[color, 2, base_indices...]
        base3 = psi[color, 3, base_indices...]
        base4 = psi[color, 4, base_indices...]
        low1 = base_coefficient * base1 +
            low_coefficient * psi[color, 1, low_indices...]
        low2 = base_coefficient * base2 +
            low_coefficient * psi[color, 2, low_indices...]
        low3 = base_coefficient * base3 +
            low_coefficient * psi[color, 3, low_indices...]
        low4 = base_coefficient * base4 +
            low_coefficient * psi[color, 4, low_indices...]
        high1 = base_coefficient * base1 +
            high_coefficient * psi[color, 1, high_indices...]
        high2 = base_coefficient * base2 +
            high_coefficient * psi[color, 2, high_indices...]
        high3 = base_coefficient * base3 +
            high_coefficient * psi[color, 3, high_indices...]
        high4 = base_coefficient * base4 +
            high_coefficient * psi[color, 4, high_indices...]
    end
    low_half1, low_half2 = _domainwall_half_project_values3(
        low1, low2, low3, low4, pm, mu)
    high_half1, high_half2 = _domainwall_half_project_values3(
        high1, high2, high3, high4, pm, mu)
    return low_half1, low_half2, high_half1, high_half2
end

@inline function _domainwall_paired_half_matvec_forward3(
    U, psi, gauge_indices, base_indices, low_indices, high_indices,
    base_coefficient, low_coefficient, high_coefficient, pm, mu,
)
    l11, l12, h11, h12 = _domainwall_paired_half_project3(
        psi, 1, base_indices, low_indices, high_indices,
        base_coefficient, low_coefficient, high_coefficient, pm, mu)
    l21, l22, h21, h22 = _domainwall_paired_half_project3(
        psi, 2, base_indices, low_indices, high_indices,
        base_coefficient, low_coefficient, high_coefficient, pm, mu)
    l31, l32, h31, h32 = _domainwall_paired_half_project3(
        psi, 3, base_indices, low_indices, high_indices,
        base_coefficient, low_coefficient, high_coefficient, pm, mu)
    @inbounds begin
        U11 = U[1, 1, gauge_indices...]
        U12 = U[1, 2, gauge_indices...]
        U13 = U[1, 3, gauge_indices...]
        U21 = U[2, 1, gauge_indices...]
        U22 = U[2, 2, gauge_indices...]
        U23 = U[2, 3, gauge_indices...]
        U31 = U[3, 1, gauge_indices...]
        U32 = U[3, 2, gauge_indices...]
        U33 = U[3, 3, gauge_indices...]
    end
    return (
        muladdmulti(U11, l11, U12, l21, U13, l31),
        muladdmulti(U11, l12, U12, l22, U13, l32),
        muladdmulti(U21, l11, U22, l21, U23, l31),
        muladdmulti(U21, l12, U22, l22, U23, l32),
        muladdmulti(U31, l11, U32, l21, U33, l31),
        muladdmulti(U31, l12, U32, l22, U33, l32),
        muladdmulti(U11, h11, U12, h21, U13, h31),
        muladdmulti(U11, h12, U12, h22, U13, h32),
        muladdmulti(U21, h11, U22, h21, U23, h31),
        muladdmulti(U21, h12, U22, h22, U23, h32),
        muladdmulti(U31, h11, U32, h21, U33, h31),
        muladdmulti(U31, h12, U32, h22, U33, h32),
    )
end

@inline function _domainwall_paired_half_matvec_backward3(
    U, psi, gauge_indices, base_indices, low_indices, high_indices,
    base_coefficient, low_coefficient, high_coefficient, pm, mu,
)
    l11, l12, h11, h12 = _domainwall_paired_half_project3(
        psi, 1, base_indices, low_indices, high_indices,
        base_coefficient, low_coefficient, high_coefficient, pm, mu)
    l21, l22, h21, h22 = _domainwall_paired_half_project3(
        psi, 2, base_indices, low_indices, high_indices,
        base_coefficient, low_coefficient, high_coefficient, pm, mu)
    l31, l32, h31, h32 = _domainwall_paired_half_project3(
        psi, 3, base_indices, low_indices, high_indices,
        base_coefficient, low_coefficient, high_coefficient, pm, mu)
    @inbounds begin
        U11 = conj(U[1, 1, gauge_indices...])
        U12 = conj(U[2, 1, gauge_indices...])
        U13 = conj(U[3, 1, gauge_indices...])
        U21 = conj(U[1, 2, gauge_indices...])
        U22 = conj(U[2, 2, gauge_indices...])
        U23 = conj(U[3, 2, gauge_indices...])
        U31 = conj(U[1, 3, gauge_indices...])
        U32 = conj(U[2, 3, gauge_indices...])
        U33 = conj(U[3, 3, gauge_indices...])
    end
    return (
        muladdmulti(U11, l11, U12, l21, U13, l31),
        muladdmulti(U11, l12, U12, l22, U13, l32),
        muladdmulti(U21, l11, U22, l21, U23, l31),
        muladdmulti(U21, l12, U22, l22, U23, l32),
        muladdmulti(U31, l11, U32, l21, U33, l31),
        muladdmulti(U31, l12, U32, l22, U33, l32),
        muladdmulti(U11, h11, U12, h21, U13, h31),
        muladdmulti(U11, h12, U12, h22, U13, h32),
        muladdmulti(U21, h11, U22, h21, U23, h31),
        muladdmulti(U21, h12, U22, h22, U23, h32),
        muladdmulti(U31, h11, U32, h21, U33, h31),
        muladdmulti(U31, h12, U32, h22, U33, h32),
    )
end

# Reconstruct only the upper components of the first half spinor and the
# lower components of the second.  Those are exactly the components selected
# by F' in the adjoint; constructing the other twelve components only raises
# register pressure and they would be discarded by the final write.
@inline function _domainwall_paired_reconstruct_add3(
    accumulator, halfspinors, ::Val{PM}, ::Val{MU},
) where {PM,MU}
    c11, c12, c13, c14,
    c21, c22, c23, c24,
    c31, c32, c33, c34 = accumulator
    l11, l12, l21, l22, l31, l32,
    h11, h12, h21, h22, h31, h32 = halfspinors
    if MU == 1
        return (
            c11 + l11, c12 + l12, c13 + PM * im * h12, c14 + PM * im * h11,
            c21 + l21, c22 + l22, c23 + PM * im * h22, c24 + PM * im * h21,
            c31 + l31, c32 + l32, c33 + PM * im * h32, c34 + PM * im * h31,
        )
    elseif MU == 2
        return (
            c11 + l11, c12 + l12, c13 + PM * h12, c14 - PM * h11,
            c21 + l21, c22 + l22, c23 + PM * h22, c24 - PM * h21,
            c31 + l31, c32 + l32, c33 + PM * h32, c34 - PM * h31,
        )
    elseif MU == 3
        return (
            c11 + l11, c12 + l12, c13 + PM * im * h11, c14 - PM * im * h12,
            c21 + l21, c22 + l22, c23 + PM * im * h21, c24 - PM * im * h22,
            c31 + l31, c32 + l32, c33 + PM * im * h31, c34 - PM * im * h32,
        )
    else
        return (
            c11 + l11, c12 + l12, c13 - PM * h11, c14 - PM * h12,
            c21 + l21, c22 + l22, c23 - PM * h21, c24 - PM * h22,
            c31 + l31, c32 + l32, c33 - PM * h31, c34 - PM * h32,
        )
    end
end

@inline function _domainwall_half_matvec_forward3(
    U, psi, gauge_indices, base_indices, low_indices, high_indices,
    base_coefficient, low_coefficient, high_coefficient, pm, mu,
)
    h11, h12 = _domainwall_combined_half_project3(
        psi, 1, base_indices, low_indices, high_indices,
        base_coefficient, low_coefficient, high_coefficient, pm, mu)
    h21, h22 = _domainwall_combined_half_project3(
        psi, 2, base_indices, low_indices, high_indices,
        base_coefficient, low_coefficient, high_coefficient, pm, mu)
    h31, h32 = _domainwall_combined_half_project3(
        psi, 3, base_indices, low_indices, high_indices,
        base_coefficient, low_coefficient, high_coefficient, pm, mu)
    @inbounds begin
        U11 = U[1, 1, gauge_indices...]
        U12 = U[1, 2, gauge_indices...]
        U13 = U[1, 3, gauge_indices...]
        U21 = U[2, 1, gauge_indices...]
        U22 = U[2, 2, gauge_indices...]
        U23 = U[2, 3, gauge_indices...]
        U31 = U[3, 1, gauge_indices...]
        U32 = U[3, 2, gauge_indices...]
        U33 = U[3, 3, gauge_indices...]
    end
    return (
        muladdmulti(U11, h11, U12, h21, U13, h31),
        muladdmulti(U11, h12, U12, h22, U13, h32),
        muladdmulti(U21, h11, U22, h21, U23, h31),
        muladdmulti(U21, h12, U22, h22, U23, h32),
        muladdmulti(U31, h11, U32, h21, U33, h31),
        muladdmulti(U31, h12, U32, h22, U33, h32),
    )
end

@inline function _domainwall_half_matvec_backward3(
    U, psi, gauge_indices, base_indices, low_indices, high_indices,
    base_coefficient, low_coefficient, high_coefficient, pm, mu,
)
    h11, h12 = _domainwall_combined_half_project3(
        psi, 1, base_indices, low_indices, high_indices,
        base_coefficient, low_coefficient, high_coefficient, pm, mu)
    h21, h22 = _domainwall_combined_half_project3(
        psi, 2, base_indices, low_indices, high_indices,
        base_coefficient, low_coefficient, high_coefficient, pm, mu)
    h31, h32 = _domainwall_combined_half_project3(
        psi, 3, base_indices, low_indices, high_indices,
        base_coefficient, low_coefficient, high_coefficient, pm, mu)
    @inbounds begin
        U11 = conj(U[1, 1, gauge_indices...])
        U12 = conj(U[2, 1, gauge_indices...])
        U13 = conj(U[3, 1, gauge_indices...])
        U21 = conj(U[1, 2, gauge_indices...])
        U22 = conj(U[2, 2, gauge_indices...])
        U23 = conj(U[3, 2, gauge_indices...])
        U31 = conj(U[1, 3, gauge_indices...])
        U32 = conj(U[2, 3, gauge_indices...])
        U33 = conj(U[3, 3, gauge_indices...])
    end
    return (
        muladdmulti(U11, h11, U12, h21, U13, h31),
        muladdmulti(U11, h12, U12, h22, U13, h32),
        muladdmulti(U21, h11, U22, h21, U23, h31),
        muladdmulti(U21, h12, U22, h22, U23, h32),
        muladdmulti(U31, h11, U32, h21, U33, h31),
        muladdmulti(U31, h12, U32, h22, U33, h32),
    )
end

@inline _domainwall_gauge_indices(indices) =
    (indices[1], indices[2], indices[3], indices[4])

@inline function _domainwall_paired_hopping_direction3(
    accumulator, U, psi, indices, low_indices, high_indices,
    shift_plus, shift_minus, base_coefficient, low_coefficient,
    high_coefficient, ::Val{FORWARD_PM}, ::Val{MU},
) where {FORWARD_PM,MU}
    indices_plus = shiftindices(indices, shift_plus)
    indices_minus = shiftindices(indices, shift_minus)
    low_plus = shiftindices(low_indices, shift_plus)
    low_minus = shiftindices(low_indices, shift_minus)
    high_plus = shiftindices(high_indices, shift_plus)
    high_minus = shiftindices(high_indices, shift_minus)
    forward_pm = Val(FORWARD_PM)
    backward_pm = Val(-FORWARD_PM)
    accumulator = _domainwall_paired_reconstruct_add3(
        accumulator,
        _domainwall_paired_half_matvec_forward3(
            U, psi, _domainwall_gauge_indices(indices),
            indices_plus, low_plus, high_plus,
            base_coefficient, low_coefficient, high_coefficient,
            forward_pm, Val(MU)),
        forward_pm, Val(MU))
    return _domainwall_paired_reconstruct_add3(
        accumulator,
        _domainwall_paired_half_matvec_backward3(
            U, psi, _domainwall_gauge_indices(indices_minus),
            indices_minus, low_minus, high_minus,
            base_coefficient, low_coefficient, high_coefficient,
            backward_pm, Val(MU)),
        backward_pm, Val(MU))
end

@inline function _domainwall_paired_hopping_accumulator3(
    U1, U2, U3, U4, psi, indices, low_indices, high_indices,
    base_coefficient, low_coefficient, high_coefficient,
    ::Val{FORWARD_PM},
) where FORWARD_PM
    zero_value = zero(@inbounds psi[1, 1, indices...])
    accumulator = (
        zero_value, zero_value, zero_value, zero_value,
        zero_value, zero_value, zero_value, zero_value,
        zero_value, zero_value, zero_value, zero_value,
    )
    accumulator = _domainwall_paired_hopping_direction3(
        accumulator, U1, psi, indices, low_indices, high_indices,
        shift_1p5D, shift_1m5D, base_coefficient, low_coefficient,
        high_coefficient, Val(FORWARD_PM), Val(1))
    accumulator = _domainwall_paired_hopping_direction3(
        accumulator, U2, psi, indices, low_indices, high_indices,
        shift_2p5D, shift_2m5D, base_coefficient, low_coefficient,
        high_coefficient, Val(FORWARD_PM), Val(2))
    accumulator = _domainwall_paired_hopping_direction3(
        accumulator, U3, psi, indices, low_indices, high_indices,
        shift_3p5D, shift_3m5D, base_coefficient, low_coefficient,
        high_coefficient, Val(FORWARD_PM), Val(3))
    return _domainwall_paired_hopping_direction3(
        accumulator, U4, psi, indices, low_indices, high_indices,
        shift_4p5D, shift_4m5D, base_coefficient, low_coefficient,
        high_coefficient, Val(FORWARD_PM), Val(4))
end

@inline function _domainwall_hopping_direction3(
    accumulator, U, psi, indices, low_indices, high_indices,
    shift_plus, shift_minus, base_coefficient, low_coefficient,
    high_coefficient, ::Val{FORWARD_PM}, ::Val{MU},
) where {FORWARD_PM,MU}
    indices_plus = shiftindices(indices, shift_plus)
    indices_minus = shiftindices(indices, shift_minus)
    low_plus = shiftindices(low_indices, shift_plus)
    low_minus = shiftindices(low_indices, shift_minus)
    high_plus = shiftindices(high_indices, shift_plus)
    high_minus = shiftindices(high_indices, shift_minus)
    forward_pm = Val(FORWARD_PM)
    backward_pm = Val(-FORWARD_PM)
    accumulator = _wilson_reconstruct_add3(
        accumulator,
        _domainwall_half_matvec_forward3(
            U, psi, _domainwall_gauge_indices(indices),
            indices_plus, low_plus, high_plus,
            base_coefficient, low_coefficient, high_coefficient,
            forward_pm, Val(MU)),
        forward_pm, Val(MU))
    return _wilson_reconstruct_add3(
        accumulator,
        _domainwall_half_matvec_backward3(
            U, psi, _domainwall_gauge_indices(indices_minus),
            indices_minus, low_minus, high_minus,
            base_coefficient, low_coefficient, high_coefficient,
            backward_pm, Val(MU)),
        backward_pm, Val(MU))
end

@inline function _domainwall_hopping_accumulator3(
    U1, U2, U3, U4, psi, indices, low_indices, high_indices,
    base_coefficient, low_coefficient, high_coefficient,
    ::Val{FORWARD_PM},
) where FORWARD_PM
    zero_value = zero(@inbounds psi[1, 1, indices...])
    accumulator = (
        zero_value, zero_value, zero_value, zero_value,
        zero_value, zero_value, zero_value, zero_value,
        zero_value, zero_value, zero_value, zero_value,
    )
    accumulator = _domainwall_hopping_direction3(
        accumulator, U1, psi, indices, low_indices, high_indices,
        shift_1p5D, shift_1m5D, base_coefficient, low_coefficient,
        high_coefficient, Val(FORWARD_PM), Val(1))
    accumulator = _domainwall_hopping_direction3(
        accumulator, U2, psi, indices, low_indices, high_indices,
        shift_2p5D, shift_2m5D, base_coefficient, low_coefficient,
        high_coefficient, Val(FORWARD_PM), Val(2))
    accumulator = _domainwall_hopping_direction3(
        accumulator, U3, psi, indices, low_indices, high_indices,
        shift_3p5D, shift_3m5D, base_coefficient, low_coefficient,
        high_coefficient, Val(FORWARD_PM), Val(3))
    return _domainwall_hopping_direction3(
        accumulator, U4, psi, indices, low_indices, high_indices,
        shift_4p5D, shift_4m5D, base_coefficient, low_coefficient,
        high_coefficient, Val(FORWARD_PM), Val(4))
end

# Apply only the four-dimensional Wilson hopping term to one fifth slice.
# Unlike `_domainwall_hopping_accumulator3`, this does not construct a virtual
# spinor from neighbouring fifth slices.  The two-stage adjoint stores this
# result once and lets a second, cheap kernel perform the fifth-direction
# mixing, matching Grid's DW^dag -> MeooeDag5D decomposition.
@inline function _domainwall_wilson_half_matvec_backward3(
    U, psi, gauge_indices, source_indices, pm, mu,
)
    h11, h12 = _wilson_half_project3(psi, 1, source_indices, pm, mu)
    h21, h22 = _wilson_half_project3(psi, 2, source_indices, pm, mu)
    h31, h32 = _wilson_half_project3(psi, 3, source_indices, pm, mu)
    @inbounds begin
        U11 = conj(U[1, 1, gauge_indices...])
        U12 = conj(U[2, 1, gauge_indices...])
        U13 = conj(U[3, 1, gauge_indices...])
        U21 = conj(U[1, 2, gauge_indices...])
        U22 = conj(U[2, 2, gauge_indices...])
        U23 = conj(U[3, 2, gauge_indices...])
        U31 = conj(U[1, 3, gauge_indices...])
        U32 = conj(U[2, 3, gauge_indices...])
        U33 = conj(U[3, 3, gauge_indices...])
    end
    return (
        muladdmulti(U11, h11, U12, h21, U13, h31),
        muladdmulti(U11, h12, U12, h22, U13, h32),
        muladdmulti(U21, h11, U22, h21, U23, h31),
        muladdmulti(U21, h12, U22, h22, U23, h32),
        muladdmulti(U31, h11, U32, h21, U33, h31),
        muladdmulti(U31, h12, U32, h22, U33, h32),
    )
end

@inline function _domainwall_wilson_hopping_direction3(
    accumulator, U, psi, indices, shift_plus, shift_minus,
    ::Val{FORWARD_PM}, ::Val{MU},
) where {FORWARD_PM,MU}
    indices_plus = shiftindices(indices, shift_plus)
    indices_minus = shiftindices(indices, shift_minus)
    forward_pm = Val(FORWARD_PM)
    backward_pm = Val(-FORWARD_PM)
    accumulator = _wilson_reconstruct_add3(
        accumulator,
        _wilson_half_matvec_forward3(
            U, psi, _domainwall_gauge_indices(indices), indices_plus,
            forward_pm, Val(MU)),
        forward_pm, Val(MU))
    return _wilson_reconstruct_add3(
        accumulator,
        _domainwall_wilson_half_matvec_backward3(
            U, psi, _domainwall_gauge_indices(indices_minus), indices_minus,
            backward_pm, Val(MU)),
        backward_pm, Val(MU))
end

@inline function _domainwall_wilson_hopping_accumulator3(
    U1, U2, U3, U4, psi, indices, ::Val{FORWARD_PM},
) where FORWARD_PM
    zero_value = zero(@inbounds psi[1, 1, indices...])
    accumulator = (
        zero_value, zero_value, zero_value, zero_value,
        zero_value, zero_value, zero_value, zero_value,
        zero_value, zero_value, zero_value, zero_value,
    )
    accumulator = _domainwall_wilson_hopping_direction3(
        accumulator, U1, psi, indices, shift_1p5D, shift_1m5D,
        Val(FORWARD_PM), Val(1))
    accumulator = _domainwall_wilson_hopping_direction3(
        accumulator, U2, psi, indices, shift_2p5D, shift_2m5D,
        Val(FORWARD_PM), Val(2))
    accumulator = _domainwall_wilson_hopping_direction3(
        accumulator, U3, psi, indices, shift_3p5D, shift_3m5D,
        Val(FORWARD_PM), Val(3))
    return _domainwall_wilson_hopping_direction3(
        accumulator, U4, psi, indices, shift_4p5D, shift_4m5D,
        Val(FORWARD_PM), Val(4))
end

@inline function _write_domainwall_forward_result3!(
    C, psi, indices, indices_5p, indices_5m, hopping,
    diagonal_coefficient, fifth_coefficient, scale, mass,
    kappa, ::Val{L5}, ::Val{nw},
) where {L5,nw}
    boundary_5p = ifelse(indices[5] == L5 + nw, -mass, one(mass))
    boundary_5m = ifelse(indices[5] == 1 + nw, -mass, one(mass))
    diagonal = scale * (one(kappa) + diagonal_coefficient / (2 * kappa))
    fifth = scale * (fifth_coefficient / (2 * kappa) - one(kappa))
    hopping_coefficient = -scale * (one(kappa) / 2)
    @inbounds begin
        C[1, 1, indices...] = diagonal * psi[1, 1, indices...] +
            fifth * boundary_5m * psi[1, 1, indices_5m...] +
            hopping_coefficient * hopping[1]
        C[1, 2, indices...] = diagonal * psi[1, 2, indices...] +
            fifth * boundary_5m * psi[1, 2, indices_5m...] +
            hopping_coefficient * hopping[2]
        C[1, 3, indices...] = diagonal * psi[1, 3, indices...] +
            fifth * boundary_5p * psi[1, 3, indices_5p...] +
            hopping_coefficient * hopping[3]
        C[1, 4, indices...] = diagonal * psi[1, 4, indices...] +
            fifth * boundary_5p * psi[1, 4, indices_5p...] +
            hopping_coefficient * hopping[4]
        C[2, 1, indices...] = diagonal * psi[2, 1, indices...] +
            fifth * boundary_5m * psi[2, 1, indices_5m...] +
            hopping_coefficient * hopping[5]
        C[2, 2, indices...] = diagonal * psi[2, 2, indices...] +
            fifth * boundary_5m * psi[2, 2, indices_5m...] +
            hopping_coefficient * hopping[6]
        C[2, 3, indices...] = diagonal * psi[2, 3, indices...] +
            fifth * boundary_5p * psi[2, 3, indices_5p...] +
            hopping_coefficient * hopping[7]
        C[2, 4, indices...] = diagonal * psi[2, 4, indices...] +
            fifth * boundary_5p * psi[2, 4, indices_5p...] +
            hopping_coefficient * hopping[8]
        C[3, 1, indices...] = diagonal * psi[3, 1, indices...] +
            fifth * boundary_5m * psi[3, 1, indices_5m...] +
            hopping_coefficient * hopping[9]
        C[3, 2, indices...] = diagonal * psi[3, 2, indices...] +
            fifth * boundary_5m * psi[3, 2, indices_5m...] +
            hopping_coefficient * hopping[10]
        C[3, 3, indices...] = diagonal * psi[3, 3, indices...] +
            fifth * boundary_5p * psi[3, 3, indices_5p...] +
            hopping_coefficient * hopping[11]
        C[3, 4, indices...] = diagonal * psi[3, 4, indices...] +
            fifth * boundary_5p * psi[3, 4, indices_5p...] +
            hopping_coefficient * hopping[12]
    end
    return nothing
end

@inline function _kernel_domainwall_forward3!(
    C, U1, U2, U3, U4, psi, indices,
    diagonal_coefficient, fifth_coefficient, scale,
    mass, kappa, ::Val{L5}, ::Val{nw},
) where {L5,nw}
    indices_5p = shiftindices(indices, shift_5p5D)
    indices_5m = shiftindices(indices, shift_5m5D)
    boundary_5p = ifelse(indices[5] == L5 + nw, -mass, one(mass))
    boundary_5m = ifelse(indices[5] == 1 + nw, -mass, one(mass))
    hopping = _domainwall_hopping_accumulator3(
        U1, U2, U3, U4, psi, indices, indices_5m, indices_5p,
        diagonal_coefficient,
        fifth_coefficient * boundary_5m,
        fifth_coefficient * boundary_5p,
        Val(-1))
    _write_domainwall_forward_result3!(
        C, psi, indices, indices_5p, indices_5m, hopping,
        diagonal_coefficient, fifth_coefficient, scale,
        mass, kappa, Val(L5), Val(nw))
    return nothing
end

function kernel_D5DW_GeneralizedDomainwallOperator5D!(
    i, C, U1, U2, U3, U4, mass, wilson_params, psi,
    a, b, c, ::Val{3}, ::Val{nw}, dindexer, ::Val{L5},
) where {nw,L5}
    indices = delinearize(dindexer, i, nw)
    s = indices[5] - nw
    _kernel_domainwall_forward3!(
        C, U1, U2, U3, U4, psi, indices,
        b[s], c[s], a[s], mass, wilson_params.κ_wilson,
        Val(L5), Val(nw))
    return nothing
end


 function kernel_D5DW_MobiusDomainwallOperator5D!(i, C, U1, U2, U3, U4,
    mass, wilson_params, ψdata,
    ::Val{NC1}, ::Val{nw}, dindexer, ::Val{L5},
    coeff_plus, coeff_minus) where {NC1,nw,L5}
    indices = delinearize(dindexer, i, nw) #5D indices
    
    indices_1p = shiftindices(indices, shift_1p5D)
    indices_1m = shiftindices(indices, shift_1m5D)
    indices_2p = shiftindices(indices, shift_2p5D)
    indices_2m = shiftindices(indices, shift_2m5D)
    indices_3p = shiftindices(indices, shift_3p5D)
    indices_3m = shiftindices(indices, shift_3m5D)
    indices_4p = shiftindices(indices, shift_4p5D)
    indices_4m = shiftindices(indices, shift_4m5D)
    indices_5p = shiftindices(indices, shift_5p5D)
    indices_5m = shiftindices(indices, shift_5m5D)
    

    kernel_apply_1pD!(C, ψdata, U1, U2, U3, U4, wilson_params.κ_wilson, coeff_plus, indices, Val(NC1),
        indices_1p, indices_1m, indices_2p, indices_2m,
        indices_3p, indices_3m, indices_4p, indices_4m)

    kernel_apply_1mD_F!(C, ψdata, U1, U2, U3, U4, wilson_params.κ_wilson, coeff_minus, indices, Val(NC1),
        indices_5p, indices_5m, mass, Val(L5), Val(nw))

end

function kernel_D5DW_MobiusDomainwallOperator5D!(
    i, C, U1, U2, U3, U4, mass, wilson_params, psi,
    ::Val{3}, ::Val{nw}, dindexer, ::Val{L5},
    coeff_plus, coeff_minus,
) where {nw,L5}
    indices = delinearize(dindexer, i, nw)
    _kernel_domainwall_forward3!(
        C, U1, U2, U3, U4, psi, indices,
        coeff_plus, -coeff_minus, one(mass), mass,
        wilson_params.κ_wilson, Val(L5), Val(nw))
    return nothing
end

function D4x_5D!(C::TC,U,ψ::Tp,coeff) where {T1,AT1,NC1,nw,DI,
    TC<:LatticeMatrix{5,T1,AT1,NC1,4,nw,DI},
    Tp<:LatticeMatrix{5,T1,AT1,NC1,4,nw,DI}}

    _require_5d_halo(Val(nw))
    _ensure_5d_operator_halo!(U, ψ)

    U1 = get_matrix(U[1])
    U2 = get_matrix(U[2])
    U3 = get_matrix(U[3])
    U4 = get_matrix(U[4])
    ψdata = get_matrix(ψ)
    Cdata = get_matrix(C)

    _parallel_for_mutating!(C,
        prod(C.PN), kernel_D4x_5D_single!,
        Cdata, U1, U2, U3, U4, ψdata,coeff,
        Val(NC1), Val(nw), C.indexer)
end

function kernel_D4x_5D_single!(i, C, U1, U2, U3, U4,ψdata,coeff,
    ::Val{NC1}, ::Val{nw}, dindexer) where {NC1,nw}
    indices = delinearize(dindexer, i, nw) #5D indices

    indices_1p = shiftindices(indices, shift_1p5D)
    indices_1m = shiftindices(indices, shift_1m5D)
    indices_2p = shiftindices(indices, shift_2p5D)
    indices_2m = shiftindices(indices, shift_2m5D)
    indices_3p = shiftindices(indices, shift_3p5D)
    indices_3m = shiftindices(indices, shift_3m5D)
    indices_4p = shiftindices(indices, shift_4p5D)
    indices_4m = shiftindices(indices, shift_4m5D)

    kernel_D4x_5D!(C, ψdata, U1, U2, U3, U4, indices,coeff, Val(NC1),
        indices_1p, indices_1m, indices_2p, indices_2m,
        indices_3p, indices_3m, indices_4p, indices_4m)
end

 function kernel_apply_1pD!(C, ψdata, U1, U2, U3, U4, κ, factor,
    indices, ::Val{NC1},
    indices_1p, indices_1m, indices_2p, indices_2m,
    indices_3p, indices_3m, indices_4p, indices_4m) where NC1

    massfactor = -(factor / (2 * κ) + 1)
    #println(massfactor)

    @inbounds for ic = 1:NC1
        for ia = 1:4
            #C[ic, ia, indices...] = -massfactor * ψdata[ic, ia, indices...]
            C[ic, ia, indices...] = -massfactor * ψdata[ic, ia, indices...]

            #C[ic, ia, indices...] = ψdata[ic, ia, indices...]

        end
    end
    #return

    kernel_D4x_5D!(C, ψdata, U1, U2, U3, U4, indices, -factor, Val(NC1),
        indices_1p, indices_1m, indices_2p, indices_2m,
        indices_3p, indices_3m, indices_4p, indices_4m)

end

@inline function mul_op_1pg5_addkappaU!(C,oneminusγ1, ψdata, jc, indices_1p,
        factor,U1,ic,indices)

        v1,v2,v3,v4 = mul_op_1pg5(oneminusγ1, ψdata, jc, indices_1p)
            #for ia = 1:4
        C[ic, 1, indices...] += factor * U1 * v1
        C[ic, 2, indices...] += factor * U1 * v2
        C[ic, 3, indices...] += factor * U1 * v3
        C[ic, 4, indices...] += factor * U1 * v4
end


function kernel_apply_1mD_F!(C, ψdata, U1, U2, U3, U4, κ, factor,
    indices, ::Val{NC1},
    indices_5p, indices_5m, mass, ::Val{L5}, ::Val{nw}) where {NC1,L5,nw}

    #massfactor = 1
    massfactor = -(factor / (2 * κ) + 1)
    coeff_1mg5 = ifelse(indices[5] == 1 + nw, -mass, 1)
    coeff_1pg5 = ifelse(indices[5] == L5 + nw, -mass, 1)
    #coeff_1pg5 = ifelse(indices[5] == 1 + nw, -mass, 0)
    #@info indices[5]
    #coeff_1mg5 = ifelse(indices[5] == L5 + nw, -mass, 0)

    @inbounds for ic = 1:NC1
        #(1+gamma_5) 3,4 only #LTK definition
        #if coeff_1mg5 != 0
        #@info ψdata[ic, 3, indices_5m...]
        #end
        C[ic, 3, indices...] += coeff_1pg5 * massfactor * ψdata[ic, 3, indices_5p...]
        C[ic, 4, indices...] += coeff_1pg5 * massfactor * ψdata[ic, 4, indices_5p...]

        #(1-gamma_5) 1,2 only #LTK definition
        C[ic, 1, indices...] += coeff_1mg5 * massfactor * ψdata[ic, 1, indices_5m...]
        C[ic, 2, indices...] += coeff_1mg5 * massfactor * ψdata[ic, 2, indices_5m...]


    end

    #return

    coeff = factor

    if factor == 0
        return
    end

    κ = -0.5 * coeff * coeff_1pg5
    #(1+gamma_5) 3,4 only #LTK definition
    indices_1p = shiftindices(indices_5p, shift_1p5D)
    indices_1m = shiftindices(indices_5p, shift_1m5D)
    indices_2p = shiftindices(indices_5p, shift_2p5D)
    indices_2m = shiftindices(indices_5p, shift_2m5D)
    indices_3p = shiftindices(indices_5p, shift_3p5D)
    indices_3m = shiftindices(indices_5p, shift_3m5D)
    indices_4p = shiftindices(indices_5p, shift_4p5D)
    indices_4m = shiftindices(indices_5p, shift_4m5D)

    indices_4 = (indices[1],indices[2],indices[3],indices[4])
    indices_1m_4 = (indices_1m[1],indices_1m[2],indices_1m[3],indices_1m[4])
    indices_2m_4 = (indices_2m[1],indices_2m[2],indices_2m[3],indices_2m[4])
    indices_3m_4 = (indices_3m[1],indices_3m[2],indices_3m[3],indices_3m[4])
    indices_4m_4 = (indices_4m[1],indices_4m[2],indices_4m[3],indices_4m[4])

    @inbounds for ic = 1:NC1
        for jc = 1:NC1
            #U_n[ν](1 - γν)*ψ_{n+ν} 

            Ui = U1[ic, jc, indices_4...] 
            mul_op_1pg5_addkappaU!(C,oneminusγ1, ψdata, jc, indices_1p,
                 -κ,Ui,ic,indices)
            #v1,v2,v3,v4 = mul_op_1pg5(oneminusγ1, ψdata, jc, indices_1p)
            #for ia = 1:4
            #    C[ic, 1, indices...] += -κ * U1[ic, jc, indices_4...] * v1
            #    C[ic, 2, indices...] += -κ * U1[ic, jc, indices_4...] * v2
            #    C[ic, 3, indices...] += -κ * U1[ic, jc, indices_4...] * v3
            #    C[ic, 4, indices...] += -κ * U1[ic, jc, indices_4...] * v4
            #end
            Ui = U2[ic, jc, indices_4...] 
            mul_op_1pg5_addkappaU!(C,oneminusγ2, ψdata, jc, indices_2p,
                 -κ,Ui,ic,indices)
            #v1,v2,v3,v4 = mul_op_1pg5(oneminusγ2, ψdata, jc, indices_2p)
            #for ia = 1:4
            #    C[ic, 1, indices...] += -κ * U2[ic, jc, indices_4...] * v1
            #    C[ic, 2, indices...] += -κ * U2[ic, jc, indices_4...] * v2
            #    C[ic, 3, indices...] += -κ * U2[ic, jc, indices_4...] * v3
            #    C[ic, 4, indices...] += -κ * U2[ic, jc, indices_4...] * v4
            #end

            Ui = U3[ic, jc, indices_4...] 
            mul_op_1pg5_addkappaU!(C,oneminusγ3, ψdata, jc, indices_3p,
                 -κ,Ui,ic,indices)
            #v1,v2,v3,v4 = mul_op_1pg5(oneminusγ3, ψdata, jc, indices_3p)
            #for ia = 1:4
            #    C[ic, 1, indices...] += -κ * U3[ic, jc, indices_4...] * v1
            #    C[ic, 2, indices...] += -κ * U3[ic, jc, indices_4...] * v2
            #    C[ic, 3, indices...] += -κ * U3[ic, jc, indices_4...] * v3
            #    C[ic, 4, indices...] += -κ * U3[ic, jc, indices_4...] * v4
            #end
            Ui = U4[ic, jc, indices_4...] 
            mul_op_1pg5_addkappaU!(C,oneminusγ4, ψdata, jc, indices_4p,
                 -κ,Ui,ic,indices)
            #v1,v2,v3,v4 = mul_op_1pg5(oneminusγ4, ψdata, jc, indices_4p)
            #for ia = 1:4
            #    C[ic, 1, indices...] += -κ * U4[ic, jc, indices_4...] * v1
            #    C[ic, 2, indices...] += -κ * U4[ic, jc, indices_4...] * v2
            #    C[ic, 3, indices...] += -κ * U4[ic, jc, indices_4...] * v3
            #    C[ic, 4, indices...] += -κ * U4[ic, jc, indices_4...] * v4
            #end


            # U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
            Ui = conj(U1[jc, ic, indices_1m_4...] )
            mul_op_1pg5_addkappaU!(C,oneplusγ1, ψdata, jc, indices_1m,
                 -κ,Ui,ic,indices)

            #v1,v2,v3,v4 = mul_op_1pg5(oneplusγ1, ψdata, jc, indices_1m)
            #for ia = 1:4
            #    C[ic, 1, indices...] += -κ * U1[jc, ic, indices_1m_4...]' * v1
            #    C[ic, 2, indices...] += -κ * U1[jc, ic, indices_1m_4...]' * v2
            #    C[ic, 3, indices...] += -κ * U1[jc, ic, indices_1m_4...]' * v3
            #    C[ic, 4, indices...] += -κ * U1[jc, ic, indices_1m_4...]' * v4
            #end

            Ui = conj(U2[jc, ic, indices_2m_4...] )
            mul_op_1pg5_addkappaU!(C,oneplusγ2, ψdata, jc, indices_2m,
                 -κ,Ui,ic,indices)

            # v1,v2,v3,v4  = mul_op_1pg5(oneplusγ2, ψdata, jc, indices_2m)
            #for ia = 1:4
            #    C[ic, 1, indices...] += -κ * U2[jc, ic, indices_2m_4...]' * v1
            #    C[ic, 2, indices...] += -κ * U2[jc, ic, indices_2m_4...]' * v2
            #    C[ic, 3, indices...] += -κ * U2[jc, ic, indices_2m_4...]' * v3
            #    C[ic, 4, indices...] += -κ * U2[jc, ic, indices_2m_4...]' * v4
            #end

            Ui = conj(U3[jc, ic, indices_3m_4...] )
            mul_op_1pg5_addkappaU!(C,oneplusγ3, ψdata, jc, indices_3m,
                 -κ,Ui,ic,indices)

            #v1,v2,v3,v4  = mul_op_1pg5(oneplusγ3, ψdata, jc, indices_3m)
            #for ia = 1:4
            #    C[ic, 1, indices...] += -κ * U3[jc, ic, indices_3m_4...]' * v1
            #    C[ic, 2, indices...] += -κ * U3[jc, ic, indices_3m_4...]' * v2
            #    C[ic, 3, indices...] += -κ * U3[jc, ic, indices_3m_4...]' * v3
            #    C[ic, 4, indices...] += -κ * U3[jc, ic, indices_3m_4...]' * v4
            #end

            Ui = conj(U4[jc, ic, indices_4m_4...] )
            mul_op_1pg5_addkappaU!(C,oneplusγ4, ψdata, jc, indices_4m,
                 -κ,Ui,ic,indices)
            #v1,v2,v3,v4 = mul_op_1pg5(oneplusγ4, ψdata, jc, indices_4m)
            #for ia = 1:4
            #    C[ic, 1, indices...] += -κ * U4[jc, ic, indices_4m_4...]' * v1
            #    C[ic, 2, indices...] += -κ * U4[jc, ic, indices_4m_4...]' * v2
            #    C[ic, 3, indices...] += -κ * U4[jc, ic, indices_4m_4...]' * v3
            #    C[ic, 4, indices...] += -κ * U4[jc, ic, indices_4m_4...]' * v4
            #end



        end
    end

    #(1-gamma_5) 1,2 only #LTK definition
    indices_1p = shiftindices(indices_5m, shift_1p5D)
    indices_1m = shiftindices(indices_5m, shift_1m5D)
    indices_2p = shiftindices(indices_5m, shift_2p5D)
    indices_2m = shiftindices(indices_5m, shift_2m5D)
    indices_3p = shiftindices(indices_5m, shift_3p5D)
    indices_3m = shiftindices(indices_5m, shift_3m5D)
    indices_4p = shiftindices(indices_5m, shift_4p5D)
    indices_4m = shiftindices(indices_5m, shift_4m5D)

    indices_4 = (indices[1],indices[2],indices[3],indices[4])
    indices_1m_4 = (indices_1m[1],indices_1m[2],indices_1m[3],indices_1m[4])
    indices_2m_4 = (indices_2m[1],indices_2m[2],indices_2m[3],indices_2m[4])
    indices_3m_4 = (indices_3m[1],indices_3m[2],indices_3m[3],indices_3m[4])
    indices_4m_4 = (indices_4m[1],indices_4m[2],indices_4m[3],indices_4m[4])


    κ = -0.5 * coeff * coeff_1mg5

    @inbounds for ic = 1:NC1
        for jc = 1:NC1
            v1,v2,v3,v4 = mul_op_1mg5(oneminusγ1, ψdata, jc, indices_1p)
            #for ia = 1:4
                C[ic, 1, indices...] += -κ * U1[ic, jc, indices_4...] * v1
                C[ic, 2, indices...] += -κ * U1[ic, jc, indices_4...] * v2
                C[ic, 3, indices...] += -κ * U1[ic, jc, indices_4...] * v3
                C[ic, 4, indices...] += -κ * U1[ic, jc, indices_4...] * v4
            #end
            v1,v2,v3,v4 = mul_op_1mg5(oneminusγ2, ψdata, jc, indices_2p)
            #for ia = 1:4
                C[ic, 1, indices...] += -κ * U2[ic, jc, indices_4...] * v1
                C[ic, 2, indices...] += -κ * U2[ic, jc, indices_4...] * v2
                C[ic, 3, indices...] += -κ * U2[ic, jc, indices_4...] * v3
                C[ic, 4, indices...] += -κ * U2[ic, jc, indices_4...] * v4
            #end
            v1,v2,v3,v4 = mul_op_1mg5(oneminusγ3, ψdata, jc, indices_3p)
            #for ia = 1:4
                C[ic, 1, indices...] += -κ * U3[ic, jc, indices_4...] * v1
                C[ic, 2, indices...] += -κ * U3[ic, jc, indices_4...] * v2
                C[ic, 3, indices...] += -κ * U3[ic, jc, indices_4...] * v3
                C[ic, 4, indices...] += -κ * U3[ic, jc, indices_4...] * v4
            #end
            v1,v2,v3,v4  = mul_op_1mg5(oneminusγ4, ψdata, jc, indices_4p)
            #for ia = 1:4
                C[ic, 1, indices...] += -κ * U4[ic, jc, indices_4...] * v1
                C[ic, 2, indices...] += -κ * U4[ic, jc, indices_4...] * v2
                C[ic, 3, indices...] += -κ * U4[ic, jc, indices_4...] * v3
                C[ic, 4, indices...] += -κ * U4[ic, jc, indices_4...] * v4
            #end


            # U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
            v1,v2,v3,v4  = mul_op_1mg5(oneplusγ1, ψdata, jc, indices_1m)
            #for ia = 1:4
                C[ic, 1, indices...] += -κ * U1[jc, ic, indices_1m_4...]' * v1
                C[ic, 2, indices...] += -κ * U1[jc, ic, indices_1m_4...]' * v2
                C[ic, 3, indices...] += -κ * U1[jc, ic, indices_1m_4...]' * v3
                C[ic, 4, indices...] += -κ * U1[jc, ic, indices_1m_4...]' * v4
            #end

            v1,v2,v3,v4 = mul_op_1mg5(oneplusγ2, ψdata, jc, indices_2m)
            #for ia = 1:4
                C[ic, 1, indices...] += -κ * U2[jc, ic, indices_2m_4...]' * v1
                C[ic, 2, indices...] += -κ * U2[jc, ic, indices_2m_4...]' * v2
                C[ic, 3, indices...] += -κ * U2[jc, ic, indices_2m_4...]' * v3
                C[ic, 4, indices...] += -κ * U2[jc, ic, indices_2m_4...]' * v4
            #end

            v1,v2,v3,v4 = mul_op_1mg5(oneplusγ3, ψdata, jc, indices_3m)
            #for ia = 1:4
                C[ic, 1, indices...] += -κ * U3[jc, ic, indices_3m_4...]' * v1
                C[ic, 2, indices...] += -κ * U3[jc, ic, indices_3m_4...]' * v2
                C[ic, 3, indices...] += -κ * U3[jc, ic, indices_3m_4...]' * v3
                C[ic, 4, indices...] += -κ * U3[jc, ic, indices_3m_4...]' * v4
            #end


            v1,v2,v3,v4  = mul_op_1mg5(oneplusγ4, ψdata, jc, indices_4m)
            #for ia = 1:4
                C[ic, 1, indices...] += -κ * U4[jc, ic, indices_4m_4...]' * v1
                C[ic, 2, indices...] += -κ * U4[jc, ic, indices_4m_4...]' * v2
                C[ic, 3, indices...] += -κ * U4[jc, ic, indices_4m_4...]' * v3
                C[ic, 4, indices...] += -κ * U4[jc, ic, indices_4m_4...]' * v4
            #end


        end
    end

end

function apply_F_5D!(C::TC,mass,L5,ψ::Tp) where {T1,AT1,NC1,nw,DI,
    TC<:LatticeMatrix{5,T1,AT1,NC1,4,nw,DI},
    Tp<:LatticeMatrix{5,T1,AT1,NC1,4,nw,DI}}

    _require_5d_halo(Val(nw))

    ψdata = get_matrix(ψ)
    Cdata = get_matrix(C)


    
    _parallel_for_mutating!(C,
        prod(C.PN), kernel_apply_F!,
        Cdata, ψdata,
        Val(NC1), mass,Val(L5), Val(nw), C.indexer)
        

end


function kernel_apply_F!(i,C, ψdata, ::Val{NC1},mass,::Val{L5},::Val{nw}, dindexer) where {NC1,L5,nw}
    indices = delinearize(dindexer, i, nw) #5D indices
    indices_5p = shiftindices(indices, shift_5p5D)
    indices_5m = shiftindices(indices, shift_5m5D)

    massfactor = 1
    coeff_1mg5 = ifelse(indices[5] == 1 + nw, -mass, 1)
    coeff_1pg5 = ifelse(indices[5] == L5 + nw, -mass, 1)
    #coeff_1pg5 = ifelse(indices[5] == 1 + nw, -mass, 0)
    #@info indices[5]
    #coeff_1mg5 = ifelse(indices[5] == L5 + nw, -mass, 0)

    @inbounds for ic = 1:NC1
        #(1+gamma_5) 3,4 only #LTK definition
        #if coeff_1mg5 != 0
        #@info ψdata[ic, 3, indices_5m...]
        #end
        C[ic, 3, indices...] += coeff_1pg5 * massfactor * ψdata[ic, 3, indices_5p...]
        C[ic, 4, indices...] += coeff_1pg5 * massfactor * ψdata[ic, 4, indices_5p...]

        #(1-gamma_5) 1,2 only #LTK definition
        C[ic, 1, indices...] += coeff_1mg5 * massfactor * ψdata[ic, 1, indices_5m...]
        C[ic, 2, indices...] += coeff_1mg5 * massfactor * ψdata[ic, 2, indices_5m...]


    end

    return


end

function apply_δF_5D!(C::TC,mass,L5,ψ::Tp) where {T1,AT1,NC1,nw,DI,
    TC<:LatticeMatrix{5,T1,AT1,NC1,4,nw,DI},
    Tp<:LatticeMatrix{5,T1,AT1,NC1,4,nw,DI}}

    _require_5d_halo(Val(nw))

    ψdata = get_matrix(ψ)
    Cdata = get_matrix(C)


    
    _parallel_for_mutating!(C,
        prod(C.PN), kernel_apply_δF!,
        Cdata, ψdata,
        Val(NC1), mass,Val(L5), Val(nw), C.indexer)
        

end

function kernel_apply_δF!(i,C, ψdata, ::Val{NC1},mass,::Val{L5},::Val{nw}, dindexer) where {NC1,L5,nw}
    indices = delinearize(dindexer, i, nw) #5D indices
    indices_5p = shiftindices(indices, shift_5p5D)
    indices_5m = shiftindices(indices, shift_5m5D)

    massfactor = 1
    coeff_1mg5 = ifelse(indices[5] == 1 + nw, -mass, 0)
    coeff_1pg5 = ifelse(indices[5] == L5 + nw, -mass, 0)
    #coeff_1pg5 = ifelse(indices[5] == 1 + nw, -mass, 0)
    #@info indices[5]
    #coeff_1mg5 = ifelse(indices[5] == L5 + nw, -mass, 0)

    @inbounds for ic = 1:NC1
        #(1+gamma_5) 3,4 only #LTK definition
        #if coeff_1mg5 != 0
        #@info ψdata[ic, 3, indices_5m...]
        #end
        if coeff_1pg5 != zero(coeff_1pg5)
            C[ic, 3, indices...] += coeff_1pg5 * massfactor * ψdata[ic, 3, indices_5p...]
            C[ic, 4, indices...] += coeff_1pg5 * massfactor * ψdata[ic, 4, indices_5p...]
        end

        #(1-gamma_5) 1,2 only #LTK definition
        if coeff_1mg5 != zero(coeff_1mg5)
            C[ic, 1, indices...] += coeff_1mg5 * massfactor * ψdata[ic, 1, indices_5m...]
            C[ic, 2, indices...] += coeff_1mg5 * massfactor * ψdata[ic, 2, indices_5m...]
        end

    end

    return


end


function kernel_D4x_5D!(C, ψdata, U1, U2, U3, U4, indices, coeff, ::Val{NC1},
    indices_1p, indices_1m, indices_2p, indices_2m,
    indices_3p, indices_3m, indices_4p, indices_4m) where {NC1}

    κ = -0.5 * coeff
    indices_4 = (indices[1],indices[2],indices[3],indices[4])
    indices_1m_4 = (indices_1m[1],indices_1m[2],indices_1m[3],indices_1m[4])
    indices_2m_4 = (indices_2m[1],indices_2m[2],indices_2m[3],indices_2m[4])
    indices_3m_4 = (indices_3m[1],indices_3m[2],indices_3m[3],indices_3m[4])
    indices_4m_4 = (indices_4m[1],indices_4m[2],indices_4m[3],indices_4m[4])


    @inbounds for ic = 1:NC1
        for jc = 1:NC1
            #U_n[ν](1 - γν)*ψ_{n+ν} 

            v1,v2,v3,v4 = mul_op(oneminusγ1, ψdata, jc, indices_1p)
            #for ia = 1:4
                C[ic, 1, indices...] += -κ * U1[ic, jc, indices_4...] * v1
                C[ic, 2, indices...] += -κ * U1[ic, jc, indices_4...] * v2
                C[ic, 3, indices...] += -κ * U1[ic, jc, indices_4...] * v3
                C[ic, 4, indices...] += -κ * U1[ic, jc, indices_4...] * v4
            #end
            v1,v2,v3,v4 = mul_op(oneminusγ2, ψdata, jc, indices_2p)
            #for ia = 1:4
                C[ic, 1, indices...] += -κ * U2[ic, jc, indices_4...] * v1
                C[ic, 2, indices...] += -κ * U2[ic, jc, indices_4...] * v2
                C[ic, 3, indices...] += -κ * U2[ic, jc, indices_4...] * v3
                C[ic, 4, indices...] += -κ * U2[ic, jc, indices_4...] * v4
            #end
            v1,v2,v3,v4  = mul_op(oneminusγ3, ψdata, jc, indices_3p)
            #for ia = 1:4
                C[ic, 1, indices...] += -κ * U3[ic, jc, indices_4...] * v1
                C[ic, 2, indices...] += -κ * U3[ic, jc, indices_4...] * v2
                C[ic, 3, indices...] += -κ * U3[ic, jc, indices_4...] * v3
                C[ic, 4, indices...] += -κ * U3[ic, jc, indices_4...] * v4
            #end
            v1,v2,v3,v4 = mul_op(oneminusγ4, ψdata, jc, indices_4p)
            #for ia = 1:4
                C[ic, 1, indices...] += -κ * U4[ic, jc, indices_4...] * v1
                C[ic, 2, indices...] += -κ * U4[ic, jc, indices_4...] * v2
                C[ic, 3, indices...] += -κ * U4[ic, jc, indices_4...] * v3
                C[ic, 4, indices...] += -κ * U4[ic, jc, indices_4...] * v4
            #end


            # U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
            v1,v2,v3,v4 = mul_op(oneplusγ1, ψdata, jc, indices_1m)
            #for ia = 1:4
                C[ic, 1, indices...] += -κ * U1[jc, ic, indices_1m_4...]' * v1
                C[ic, 2, indices...] += -κ * U1[jc, ic, indices_1m_4...]' * v2
                C[ic, 3, indices...] += -κ * U1[jc, ic, indices_1m_4...]' * v3
                C[ic, 4, indices...] += -κ * U1[jc, ic, indices_1m_4...]' * v4
            #end

            v1,v2,v3,v4 = mul_op(oneplusγ2, ψdata, jc, indices_2m)
            #for ia = 1:4
                C[ic, 1, indices...] += -κ * U2[jc, ic, indices_2m_4...]' * v1
                C[ic, 2, indices...] += -κ * U2[jc, ic, indices_2m_4...]' * v2
                C[ic, 3, indices...] += -κ * U2[jc, ic, indices_2m_4...]' * v3
                C[ic, 4, indices...] += -κ * U2[jc, ic, indices_2m_4...]' * v4
            #end

            v1,v2,v3,v4 = mul_op(oneplusγ3, ψdata, jc, indices_3m)
            #for ia = 1:4
                C[ic, 1, indices...] += -κ * U3[jc, ic, indices_3m_4...]' * v1
                C[ic, 2, indices...] += -κ * U3[jc, ic, indices_3m_4...]' * v2
                C[ic, 3, indices...] += -κ * U3[jc, ic, indices_3m_4...]' * v3
                C[ic, 4, indices...] += -κ * U3[jc, ic, indices_3m_4...]' * v4
            #end


            v1,v2,v3,v4 = mul_op(oneplusγ4, ψdata, jc, indices_4m)
            #for ia = 1:4
                C[ic, 1, indices...] += -κ * U4[jc, ic, indices_4m_4...]' * v1
                C[ic, 2, indices...] += -κ * U4[jc, ic, indices_4m_4...]' * v2
                C[ic, 3, indices...] += -κ * U4[jc, ic, indices_4m_4...]' * v3
                C[ic, 4, indices...] += -κ * U4[jc, ic, indices_4m_4...]' * v4
            #end


        end
    end

end

@inline function _write_domainwall_wilson_adjoint_stage3!(
    destination, psi, indices, hopping, kappa,
)
    diagonal = one(kappa) / (2 * kappa)
    hopping_coefficient = -(one(kappa) / 2)
    @inbounds begin
        destination[1, 1, indices...] = diagonal * psi[1, 1, indices...] +
            hopping_coefficient * hopping[1]
        destination[1, 2, indices...] = diagonal * psi[1, 2, indices...] +
            hopping_coefficient * hopping[2]
        destination[1, 3, indices...] = diagonal * psi[1, 3, indices...] +
            hopping_coefficient * hopping[3]
        destination[1, 4, indices...] = diagonal * psi[1, 4, indices...] +
            hopping_coefficient * hopping[4]
        destination[2, 1, indices...] = diagonal * psi[2, 1, indices...] +
            hopping_coefficient * hopping[5]
        destination[2, 2, indices...] = diagonal * psi[2, 2, indices...] +
            hopping_coefficient * hopping[6]
        destination[2, 3, indices...] = diagonal * psi[2, 3, indices...] +
            hopping_coefficient * hopping[7]
        destination[2, 4, indices...] = diagonal * psi[2, 4, indices...] +
            hopping_coefficient * hopping[8]
        destination[3, 1, indices...] = diagonal * psi[3, 1, indices...] +
            hopping_coefficient * hopping[9]
        destination[3, 2, indices...] = diagonal * psi[3, 2, indices...] +
            hopping_coefficient * hopping[10]
        destination[3, 3, indices...] = diagonal * psi[3, 3, indices...] +
            hopping_coefficient * hopping[11]
        destination[3, 4, indices...] = diagonal * psi[3, 4, indices...] +
            hopping_coefficient * hopping[12]
    end
    return nothing
end

function kernel_domainwall_wilson_adjoint_stage3!(
    site, destination, U1, U2, U3, U4, kappa, psi,
    ::Val{nw}, indexer,
) where nw
    indices = delinearize(indexer, site, nw)
    hopping = _domainwall_wilson_hopping_accumulator3(
        U1, U2, U3, U4, psi, indices, Val(1))
    _write_domainwall_wilson_adjoint_stage3!(
        destination, psi, indices, hopping, kappa)
    return nothing
end

@inline function _domainwall_wrapped_fifth_indices(
    indices, ::Val{L5}, ::Val{nw},
) where {L5,nw}
    fifth = indices[5]
    fifth_plus = ifelse(fifth == L5 + nw, 1 + nw, fifth + 1)
    fifth_minus = ifelse(fifth == 1 + nw, L5 + nw, fifth - 1)
    indices_plus = (
        indices[1], indices[2], indices[3], indices[4], fifth_plus)
    indices_minus = (
        indices[1], indices[2], indices[3], indices[4], fifth_minus)
    return indices_plus, indices_minus
end

@inline function _kernel_domainwall_adjoint_combine3!(
    destination, wilson_adjoint, psi, color, spin, indices,
    diagonal_coefficient, fifth_plus_coefficient, fifth_minus_coefficient,
    scale, scale_plus, scale_minus, mass,
    ::Val{L5}, ::Val{nw},
) where {L5,nw}
    indices_plus, indices_minus =
        _domainwall_wrapped_fifth_indices(indices, Val(L5), Val(nw))
    boundary_plus = ifelse(indices[5] == L5 + nw, -mass, one(mass))
    boundary_minus = ifelse(indices[5] == 1 + nw, -mass, one(mass))
    same = scale * diagonal_coefficient
    plus = boundary_plus * scale_plus
    minus = boundary_minus * scale_minus
    @inbounds if spin <= 2
        destination[color, spin, indices...] =
            scale * psi[color, spin, indices...] +
            same * wilson_adjoint[color, spin, indices...] +
            plus * (
                fifth_plus_coefficient *
                    wilson_adjoint[color, spin, indices_plus...] -
                psi[color, spin, indices_plus...])
    else
        destination[color, spin, indices...] =
            scale * psi[color, spin, indices...] +
            same * wilson_adjoint[color, spin, indices...] +
            minus * (
                fifth_minus_coefficient *
                    wilson_adjoint[color, spin, indices_minus...] -
                psi[color, spin, indices_minus...])
    end
    return nothing
end

@inline function _domainwall_component_site(linear_index)
    zero_based = linear_index - 1
    component = rem(zero_based, 12)
    site = fld(zero_based, 12) + 1
    color = rem(component, 3) + 1
    spin = fld(component, 3) + 1
    return color, spin, site
end

function kernel_domainwall_mobius_adjoint_combine3!(
    linear_index, destination, wilson_adjoint, psi, mass,
    coeff_plus, coeff_minus, ::Val{nw}, indexer, ::Val{L5},
) where {nw,L5}
    color, spin, site = _domainwall_component_site(linear_index)
    indices = delinearize(indexer, site, nw)
    scale = one(mass)
    fifth = -coeff_minus
    _kernel_domainwall_adjoint_combine3!(
        destination, wilson_adjoint, psi, color, spin, indices,
        coeff_plus, fifth, fifth, scale, scale, scale, mass,
        Val(L5), Val(nw))
    return nothing
end

function kernel_domainwall_generalized_adjoint_combine3!(
    linear_index, destination, wilson_adjoint, psi, mass, a, b, c,
    ::Val{nw}, indexer, ::Val{L5},
) where {nw,L5}
    color, spin, site = _domainwall_component_site(linear_index)
    indices = delinearize(indexer, site, nw)
    fifth = indices[5] - nw
    fifth_plus = ifelse(fifth == L5, 1, fifth + 1)
    fifth_minus = ifelse(fifth == 1, L5, fifth - 1)
    _kernel_domainwall_adjoint_combine3!(
        destination, wilson_adjoint, psi, color, spin, indices,
        b[fifth], c[fifth_plus], c[fifth_minus],
        a[fifth], a[fifth_plus], a[fifth_minus], mass,
        Val(L5), Val(nw))
    return nothing
end

@inline function _domainwall_scratch_lattice(array, primal::LatticeMatrix)
    return _lattice_alias_with_array(primal, array)
end


function LinearAlgebra.mul!(C::TC,
    Dirac::Adjoint_D5DW_MobiusDomainwallOperator5D{TD}, ψ::Tp) where {
    T1,AT1,NC1,nw,DI,T,L5,
    TC<:LatticeMatrix{5,T1,AT1,NC1,4,nw,DI},
    TD<:D5DW_MobiusDomainwallOperator5D{T,L5},
    Tp<:LatticeMatrix{5,T1,AT1,NC1,4,nw,DI}}

    _require_5d_halo(Val(nw))
    _ensure_5d_operator_halo!(Dirac.parent.U, ψ)

    U1 = get_matrix(Dirac.parent.U[1])
    U2 = get_matrix(Dirac.parent.U[2])
    U3 = get_matrix(Dirac.parent.U[3])
    U4 = get_matrix(Dirac.parent.U[4])
    ψdata = get_matrix(ψ)
    Cdata = get_matrix(C)
    mass = get_mass(Dirac.parent)
    wilson_params = get_wilson_params(Dirac.parent)
    b, c = get_bc(Dirac.parent)
    coeff_plus = (b + c) / 2
    coeff_minus = -(b - c) / 2
    if NC1 == 3
        scratch_array, scratch_index = get_block(ψ.temps)
        scratch = _domainwall_scratch_lattice(scratch_array, ψ)
        try
            scratch_data = get_matrix(scratch)
            _parallel_for_mutating!(scratch,
                prod(C.PN), kernel_domainwall_wilson_adjoint_stage3!,
                scratch_data, U1, U2, U3, U4,
                wilson_params.κ_wilson, ψdata, Val(nw), C.indexer)
            _parallel_for_mutating!(C,
                12 * prod(C.PN), kernel_domainwall_mobius_adjoint_combine3!,
                Cdata, scratch_data, ψdata, mass,
                coeff_plus, coeff_minus, Val(nw), C.indexer, Val(L5))
        finally
            unused!(ψ.temps, scratch_index)
        end
    else
        _parallel_for_mutating!(C,
            prod(C.PN), kernel_adjoint_D5DW_MobiusDomainwallOperator5D!,
            Cdata, U1, U2, U3, U4, mass, wilson_params, ψdata,
            Val(NC1), Val(nw), C.indexer, Val(L5),
            coeff_plus, coeff_minus)
    end
    return nothing
end

function kernel_adjoint_D5DW_MobiusDomainwallOperator5D!(i, C, U1, U2, U3, U4,
    mass, wilson_params, ψdata,
    ::Val{NC1}, ::Val{nw}, dindexer, ::Val{L5},
    coeff_plus, coeff_minus) where {NC1,nw,L5}
    indices = delinearize(dindexer, i, nw) #5D indices
    indices_1p = shiftindices(indices, shift_1p5D)
    indices_1m = shiftindices(indices, shift_1m5D)
    indices_2p = shiftindices(indices, shift_2p5D)
    indices_2m = shiftindices(indices, shift_2m5D)
    indices_3p = shiftindices(indices, shift_3p5D)
    indices_3m = shiftindices(indices, shift_3m5D)
    indices_4p = shiftindices(indices, shift_4p5D)
    indices_4m = shiftindices(indices, shift_4m5D)
    indices_5p = shiftindices(indices, shift_5p5D)
    indices_5m = shiftindices(indices, shift_5m5D)

    

    kernel_apply_1pDdag!(C, ψdata, U1, U2, U3, U4, wilson_params.κ_wilson, coeff_plus, indices, Val(NC1),
        indices_1p, indices_1m, indices_2p, indices_2m,
        indices_3p, indices_3m, indices_4p, indices_4m)

    kernel_apply_1mDdag_Fdag!(C, ψdata, U1, U2, U3, U4, wilson_params.κ_wilson, coeff_minus, indices, Val(NC1),
        indices_5p, indices_5m, mass, Val(L5), Val(nw))




end

function LinearAlgebra.mul!(C::TC,
    Dirac::Adjoint_D5DW_GeneralizedDomainwallOperator5D{TD},
    psi::Tp) where {T1,AT1,NC1,nw,DI,T,L5,
    TC<:LatticeMatrix{5,T1,AT1,NC1,4,nw,DI},
    TD<:D5DW_GeneralizedDomainwallOperator5D{T,L5},
    Tp<:LatticeMatrix{5,T1,AT1,NC1,4,nw,DI}}

    parent = Dirac.parent
    _require_5d_halo(Val(nw))
    _ensure_5d_operator_halo!(parent.U, psi)
    U1 = get_matrix(parent.U[1])
    U2 = get_matrix(parent.U[2])
    U3 = get_matrix(parent.U[3])
    U4 = get_matrix(parent.U[4])
    a, b, c = get_abc(parent)
    mass = get_mass(parent)
    wilson_params = get_wilson_params(parent)
    psi_data = get_matrix(psi)
    if NC1 == 3
        scratch_array, scratch_index = get_block(psi.temps)
        scratch = _domainwall_scratch_lattice(scratch_array, psi)
        try
            scratch_data = get_matrix(scratch)
            _parallel_for_mutating!(scratch,
                prod(C.PN), kernel_domainwall_wilson_adjoint_stage3!,
                scratch_data, U1, U2, U3, U4,
                wilson_params.κ_wilson, psi_data, Val(nw), C.indexer)
            _parallel_for_mutating!(C,
                12 * prod(C.PN), kernel_domainwall_generalized_adjoint_combine3!,
                get_matrix(C), scratch_data, psi_data, mass, a, b, c,
                Val(nw), C.indexer, Val(L5))
        finally
            unused!(psi.temps, scratch_index)
        end
    else
        _parallel_for_mutating!(C,
            prod(C.PN), kernel_adjoint_D5DW_GeneralizedDomainwallOperator5D!,
            get_matrix(C), U1, U2, U3, U4, mass,
            wilson_params, psi_data, a, b, c,
            Val(NC1), Val(nw), C.indexer, Val(L5))
    end
    return nothing
end

function kernel_adjoint_D5DW_GeneralizedDomainwallOperator5D!(
    i, C, U1, U2, U3, U4, mass, wilson_params, psi,
    a, b, c, ::Val{NC1}, ::Val{nw}, dindexer, ::Val{L5},
) where {NC1,nw,L5}
    indices = delinearize(dindexer, i, nw)
    indices_1p = shiftindices(indices, shift_1p5D)
    indices_1m = shiftindices(indices, shift_1m5D)
    indices_2p = shiftindices(indices, shift_2p5D)
    indices_2m = shiftindices(indices, shift_2m5D)
    indices_3p = shiftindices(indices, shift_3p5D)
    indices_3m = shiftindices(indices, shift_3m5D)
    indices_4p = shiftindices(indices, shift_4p5D)
    indices_4m = shiftindices(indices, shift_4m5D)
    indices_5p = shiftindices(indices, shift_5p5D)
    indices_5m = shiftindices(indices, shift_5m5D)
    s = indices[5] - nw

    kernel_apply_1pDdag!(
        C, psi, U1, U2, U3, U4, wilson_params.κ_wilson, b[s],
        indices, Val(NC1), indices_1p, indices_1m, indices_2p, indices_2m,
        indices_3p, indices_3m, indices_4p, indices_4m)
    @inbounds for spin in 1:4, color in 1:NC1
        C[color, spin, indices...] *= a[s]
    end

    source_5p = ifelse(s == L5, 1, s + 1)
    source_5m = ifelse(s == 1, L5, s - 1)
    _kernel_apply_1mDdag_Fdag_coefficients!(
        C, psi, U1, U2, U3, U4, wilson_params.κ_wilson,
        -c[source_5p], -c[source_5m], a[source_5p], a[source_5m],
        indices, Val(NC1), indices_5p, indices_5m,
        mass, Val(L5), Val(nw))
    return nothing
end

@inline function _write_domainwall_adjoint_result3!(
    C, psi, indices, indices_5p, indices_5m,
    hopping_low, hopping_high,
    diagonal_coefficient, fifth_5p, fifth_5m,
    scale, scale_5p, scale_5m, mass, kappa,
    ::Val{L5}, ::Val{nw},
) where {L5,nw}
    boundary_5p = ifelse(indices[5] == L5 + nw, -mass, one(mass))
    boundary_5m = ifelse(indices[5] == 1 + nw, -mass, one(mass))
    diagonal = scale * (one(kappa) + diagonal_coefficient / (2 * kappa))
    fifth_low = boundary_5p * scale_5p *
        (fifth_5p / (2 * kappa) - one(kappa))
    fifth_high = boundary_5m * scale_5m *
        (fifth_5m / (2 * kappa) - one(kappa))
    hopping_coefficient = -(one(kappa) / 2)
    @inbounds begin
        C[1, 1, indices...] = diagonal * psi[1, 1, indices...] +
            fifth_low * psi[1, 1, indices_5p...] +
            hopping_coefficient * hopping_low[1]
        C[1, 2, indices...] = diagonal * psi[1, 2, indices...] +
            fifth_low * psi[1, 2, indices_5p...] +
            hopping_coefficient * hopping_low[2]
        C[1, 3, indices...] = diagonal * psi[1, 3, indices...] +
            fifth_high * psi[1, 3, indices_5m...] +
            hopping_coefficient * hopping_high[3]
        C[1, 4, indices...] = diagonal * psi[1, 4, indices...] +
            fifth_high * psi[1, 4, indices_5m...] +
            hopping_coefficient * hopping_high[4]
        C[2, 1, indices...] = diagonal * psi[2, 1, indices...] +
            fifth_low * psi[2, 1, indices_5p...] +
            hopping_coefficient * hopping_low[5]
        C[2, 2, indices...] = diagonal * psi[2, 2, indices...] +
            fifth_low * psi[2, 2, indices_5p...] +
            hopping_coefficient * hopping_low[6]
        C[2, 3, indices...] = diagonal * psi[2, 3, indices...] +
            fifth_high * psi[2, 3, indices_5m...] +
            hopping_coefficient * hopping_high[7]
        C[2, 4, indices...] = diagonal * psi[2, 4, indices...] +
            fifth_high * psi[2, 4, indices_5m...] +
            hopping_coefficient * hopping_high[8]
        C[3, 1, indices...] = diagonal * psi[3, 1, indices...] +
            fifth_low * psi[3, 1, indices_5p...] +
            hopping_coefficient * hopping_low[9]
        C[3, 2, indices...] = diagonal * psi[3, 2, indices...] +
            fifth_low * psi[3, 2, indices_5p...] +
            hopping_coefficient * hopping_low[10]
        C[3, 3, indices...] = diagonal * psi[3, 3, indices...] +
            fifth_high * psi[3, 3, indices_5m...] +
            hopping_coefficient * hopping_high[11]
        C[3, 4, indices...] = diagonal * psi[3, 4, indices...] +
            fifth_high * psi[3, 4, indices_5m...] +
            hopping_coefficient * hopping_high[12]
    end
    return nothing
end

@inline function _kernel_domainwall_adjoint3!(
    C, U1, U2, U3, U4, psi, indices,
    diagonal_coefficient, fifth_5p, fifth_5m,
    scale, scale_5p, scale_5m, mass, kappa,
    ::Val{L5}, ::Val{nw},
) where {L5,nw}
    indices_5p = shiftindices(indices, shift_5p5D)
    indices_5m = shiftindices(indices, shift_5m5D)
    boundary_5p = ifelse(indices[5] == L5 + nw, -mass, one(mass))
    boundary_5m = ifelse(indices[5] == 1 + nw, -mass, one(mass))
    same_coefficient = scale * diagonal_coefficient
    low_coefficient = boundary_5p * scale_5p * fifth_5p
    high_coefficient = boundary_5m * scale_5m * fifth_5m

    # F' selects the upper output components from s+1 and the lower output
    # components from s-1.  Process both right-hand sides in one direction
    # traversal so the gauge matrix and the common centre slice are read once.
    hopping = _domainwall_paired_hopping_accumulator3(
        U1, U2, U3, U4, psi, indices, indices_5p, indices_5m,
        same_coefficient, low_coefficient, high_coefficient, Val(1))
    _write_domainwall_adjoint_result3!(
        C, psi, indices, indices_5p, indices_5m,
        hopping, hopping,
        diagonal_coefficient, fifth_5p, fifth_5m,
        scale, scale_5p, scale_5m, mass, kappa,
        Val(L5), Val(nw))
    return nothing
end

function kernel_adjoint_D5DW_GeneralizedDomainwallOperator5D!(
    i, C, U1, U2, U3, U4, mass, wilson_params, psi,
    a, b, c, ::Val{3}, ::Val{nw}, dindexer, ::Val{L5},
) where {nw,L5}
    indices = delinearize(dindexer, i, nw)
    s = indices[5] - nw
    source_5p = ifelse(s == L5, 1, s + 1)
    source_5m = ifelse(s == 1, L5, s - 1)
    _kernel_domainwall_adjoint3!(
        C, U1, U2, U3, U4, psi, indices,
        b[s], c[source_5p], c[source_5m],
        a[s], a[source_5p], a[source_5m],
        mass, wilson_params.κ_wilson, Val(L5), Val(nw))
    return nothing
end

function kernel_adjoint_D5DW_MobiusDomainwallOperator5D!(
    i, C, U1, U2, U3, U4, mass, wilson_params, psi,
    ::Val{3}, ::Val{nw}, dindexer, ::Val{L5},
    coeff_plus, coeff_minus,
) where {nw,L5}
    indices = delinearize(dindexer, i, nw)
    scale = one(mass)
    fifth = -coeff_minus
    _kernel_domainwall_adjoint3!(
        C, U1, U2, U3, U4, psi, indices,
        coeff_plus, fifth, fifth,
        scale, scale, scale, mass, wilson_params.κ_wilson,
        Val(L5), Val(nw))
    return nothing
end

function kernel_apply_1pDdag!(C, ψdata, U1, U2, U3, U4, κ, factor,
    indices, ::Val{NC1},
    indices_1p, indices_1m, indices_2p, indices_2m,
    indices_3p, indices_3m, indices_4p, indices_4m) where NC1

    massfactor = -(factor / (2 * κ) + 1)
    #println(massfactor)

     for ic = 1:NC1
        for ia = 1:4
            #C[ic, ia, indices...] = -massfactor * ψdata[ic, ia, indices...]
            C[ic, ia, indices...] = -massfactor * ψdata[ic, ia, indices...]

            #C[ic, ia, indices...] = ψdata[ic, ia, indices...]

        end
    end
    #return

    kernel_D4x_5Ddag!(C, ψdata, U1, U2, U3, U4, indices, -factor, Val(NC1),
        indices_1p, indices_1m, indices_2p, indices_2m,
        indices_3p, indices_3m, indices_4p, indices_4m)

end


function kernel_apply_1mDdag_Fdag!(C, ψdata, U1, U2, U3, U4, κ, factor,
    indices, ::Val{NC1},
    indices_5p, indices_5m, mass, ::Val{L5}, ::Val{nw}) where {NC1,L5,nw}

    return _kernel_apply_1mDdag_Fdag_coefficients!(
        C, ψdata, U1, U2, U3, U4, κ,
        factor, factor, one(factor), one(factor),
        indices, Val(NC1), indices_5p, indices_5m,
        mass, Val(L5), Val(nw))
end

function _kernel_apply_1mDdag_Fdag_coefficients!(
    C, ψdata, U1, U2, U3, U4, κ_wilson,
    factor_5p, factor_5m, scale_5p, scale_5m,
    indices, ::Val{NC1}, indices_5p, indices_5m,
    mass, ::Val{L5}, ::Val{nw},
) where {NC1,L5,nw}

    #massfactor = 1
    massfactor_5p = -(factor_5p / (2 * κ_wilson) + 1) * scale_5p
    massfactor_5m = -(factor_5m / (2 * κ_wilson) + 1) * scale_5m
    coeff_1mg5 = ifelse(indices[5] == 1 + nw, -mass, 1)
    coeff_1pg5 = ifelse(indices[5] == L5 + nw, -mass, 1)
    #coeff_1pg5 = ifelse(indices[5] == 1 + nw, -mass, 0)
    #@info indices[5]
    #coeff_1mg5 = ifelse(indices[5] == L5 + nw, -mass, 0)


    for ic = 1:NC1
        #(1+gamma_5) 3,4 only #LTK definition
        #if coeff_1mg5 != 0
        #@info ψdata[ic, 3, indices_5m...]
        #end
        C[ic, 1, indices...] += coeff_1pg5 * massfactor_5p * ψdata[ic, 1, indices_5p...]
        C[ic, 2, indices...] += coeff_1pg5 * massfactor_5p * ψdata[ic, 2, indices_5p...]

        #(1-gamma_5) 1,2 only #LTK definition
        C[ic, 3, indices...] += coeff_1mg5 * massfactor_5m * ψdata[ic, 3, indices_5m...]
        C[ic, 4, indices...] += coeff_1mg5 * massfactor_5m * ψdata[ic, 4, indices_5m...]

    end

    #return

    κ = -0.5 * factor_5p * coeff_1pg5 * scale_5p
    #(1+gamma_5) 3,4 only #LTK definition
    indices_1p = shiftindices(indices_5p, shift_1p5D)
    indices_1m = shiftindices(indices_5p, shift_1m5D)
    indices_2p = shiftindices(indices_5p, shift_2p5D)
    indices_2m = shiftindices(indices_5p, shift_2m5D)
    indices_3p = shiftindices(indices_5p, shift_3p5D)
    indices_3m = shiftindices(indices_5p, shift_3m5D)
    indices_4p = shiftindices(indices_5p, shift_4p5D)
    indices_4m = shiftindices(indices_5p, shift_4m5D)

    indices_4 = (indices[1],indices[2],indices[3],indices[4])
    indices_1m_4 = (indices_1m[1],indices_1m[2],indices_1m[3],indices_1m[4])
    indices_2m_4 = (indices_2m[1],indices_2m[2],indices_2m[3],indices_2m[4])
    indices_3m_4 = (indices_3m[1],indices_3m[2],indices_3m[3],indices_3m[4])
    indices_4m_4 = (indices_4m[1],indices_4m[2],indices_4m[3],indices_4m[4])


     for ic = 1:NC1
        for jc = 1:NC1
            #U_n[ν](1 - γν)*ψ_{n+ν} 
            v1,v2,v3,v4 = mul_op(oneplusγ1, ψdata, jc, indices_1p)
            #for ia = 1:2
                C[ic, 1, indices...] += -κ * U1[ic, jc, indices_4...] * v1
                C[ic, 2, indices...] += -κ * U1[ic, jc, indices_4...] * v2
            #end
            v1,v2,v3,v4 = mul_op(oneplusγ2, ψdata, jc, indices_2p)
            #for ia = 1:2
                C[ic, 1, indices...] += -κ * U2[ic, jc, indices_4...] * v1
                C[ic, 2, indices...] += -κ * U2[ic, jc, indices_4...] * v2
            #end
            v1,v2,v3,v4 = mul_op(oneplusγ3, ψdata, jc, indices_3p)
            #for ia = 1:2
                C[ic, 1, indices...] += -κ * U3[ic, jc, indices_4...] * v1
                C[ic, 2, indices...] += -κ * U3[ic, jc, indices_4...] * v2
            #end
            v1,v2,v3,v4 = mul_op(oneplusγ4, ψdata, jc, indices_4p)
            #for ia = 1:2
                C[ic, 1, indices...] += -κ * U4[ic, jc, indices_4...] * v1
                C[ic, 2, indices...] += -κ * U4[ic, jc, indices_4...] * v2
            #end


            # U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
            v1,v2,v3,v4 = mul_op(oneminusγ1, ψdata, jc, indices_1m)
            #for ia = 1:2
                C[ic, 1, indices...] += -κ * U1[jc, ic, indices_1m_4...]' * v1
                C[ic, 2, indices...] += -κ * U1[jc, ic, indices_1m_4...]' * v2
            #end

            v1,v2,v3,v4 = mul_op(oneminusγ2, ψdata, jc, indices_2m)
            #for ia = 1:2
                C[ic, 1, indices...] += -κ * U2[jc, ic, indices_2m_4...]' * v1
                C[ic, 2, indices...] += -κ * U2[jc, ic, indices_2m_4...]' * v2
            #end

            v1,v2,v3,v4 = mul_op(oneminusγ3, ψdata, jc, indices_3m)
            #for ia = 1:2
                C[ic, 1, indices...] += -κ * U3[jc, ic, indices_3m_4...]' * v1
                C[ic, 2, indices...] += -κ * U3[jc, ic, indices_3m_4...]' * v2
            #end


            v1,v2,v3,v4 = mul_op(oneminusγ4, ψdata, jc, indices_4m)
            #for ia = 1:2
                C[ic, 1, indices...] += -κ * U4[jc, ic, indices_4m_4...]' * v1
                C[ic, 2, indices...] += -κ * U4[jc, ic, indices_4m_4...]' * v2
            #end


        end
    end

    κ = -0.5 * factor_5m * coeff_1mg5 * scale_5m

    #(1-gamma_5) 1,2 only #LTK definition
    indices_1p = shiftindices(indices_5m, shift_1p5D)
    indices_1m = shiftindices(indices_5m, shift_1m5D)
    indices_2p = shiftindices(indices_5m, shift_2p5D)
    indices_2m = shiftindices(indices_5m, shift_2m5D)
    indices_3p = shiftindices(indices_5m, shift_3p5D)
    indices_3m = shiftindices(indices_5m, shift_3m5D)
    indices_4p = shiftindices(indices_5m, shift_4p5D)
    indices_4m = shiftindices(indices_5m, shift_4m5D)

    indices_4 = (indices[1],indices[2],indices[3],indices[4])
    indices_1m_4 = (indices_1m[1],indices_1m[2],indices_1m[3],indices_1m[4])
    indices_2m_4 = (indices_2m[1],indices_2m[2],indices_2m[3],indices_2m[4])
    indices_3m_4 = (indices_3m[1],indices_3m[2],indices_3m[3],indices_3m[4])
    indices_4m_4 = (indices_4m[1],indices_4m[2],indices_4m[3],indices_4m[4])

    for ic = 1:NC1
        for jc = 1:NC1
            #U_n[ν](1 - γν)*ψ_{n+ν} 

            v1,v2,v3,v4 = mul_op(oneplusγ1, ψdata, jc, indices_1p)
            #for ia = 3:4
                C[ic, 3, indices...] += -κ * U1[ic, jc, indices_4...] * v3
                C[ic, 4, indices...] += -κ * U1[ic, jc, indices_4...] * v4
            #end
            v1,v2,v3,v4 = mul_op(oneplusγ2, ψdata, jc, indices_2p)
            #for ia = 3:4
                C[ic, 3, indices...] += -κ * U2[ic, jc, indices_4...] * v3
                C[ic, 4, indices...] += -κ * U2[ic, jc, indices_4...] * v4
            #end
            v1,v2,v3,v4 = mul_op(oneplusγ3, ψdata, jc, indices_3p)
            #for ia = 3:4
                C[ic, 3, indices...] += -κ * U3[ic, jc, indices_4...] * v3
                C[ic, 4, indices...] += -κ * U3[ic, jc, indices_4...] * v4
            #end
            v1,v2,v3,v4 = mul_op(oneplusγ4, ψdata, jc, indices_4p)
            #for ia = 3:4
                C[ic, 3, indices...] += -κ * U4[ic, jc, indices_4...] * v3
                C[ic, 4, indices...] += -κ * U4[ic, jc, indices_4...] * v4
            #end


            # U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
            v1,v2,v3,v4  = mul_op(oneminusγ1, ψdata, jc, indices_1m)
            #for ia = 3:4
                C[ic, 3, indices...] += -κ * U1[jc, ic, indices_1m_4...]' * v3
                C[ic, 4, indices...] += -κ * U1[jc, ic, indices_1m_4...]' * v4
            #end

            v1,v2,v3,v4  = mul_op(oneminusγ2, ψdata, jc, indices_2m)
            #for ia = 3:4
                C[ic, 3, indices...] += -κ * U2[jc, ic, indices_2m_4...]' * v3
                C[ic, 4, indices...] += -κ * U2[jc, ic, indices_2m_4...]' * v4
            #end

            v1,v2,v3,v4  = mul_op(oneminusγ3, ψdata, jc, indices_3m)
            #for ia = 3:4
                C[ic, 3, indices...] += -κ * U3[jc, ic, indices_3m_4...]' * v3
                C[ic, 4, indices...] += -κ * U3[jc, ic, indices_3m_4...]' * v4
            #end


            v1,v2,v3,v4  = mul_op(oneminusγ4, ψdata, jc, indices_4m)
            #for ia = 3:4
                C[ic, 3, indices...] += -κ * U4[jc, ic, indices_4m_4...]' * v3
                C[ic, 4, indices...] += -κ * U4[jc, ic, indices_4m_4...]' * v4
            #end


        end
    end

end


function kernel_D4x_5Ddag!(C, ψdata, U1, U2, U3, U4, indices, coeff, ::Val{NC1},
    indices_1p, indices_1m, indices_2p, indices_2m,
    indices_3p, indices_3m, indices_4p, indices_4m) where {NC1}

    κ = -0.5 * coeff
    indices_4 = (indices[1],indices[2],indices[3],indices[4])
    indices_1m_4 = (indices_1m[1],indices_1m[2],indices_1m[3],indices_1m[4])
    indices_2m_4 = (indices_2m[1],indices_2m[2],indices_2m[3],indices_2m[4])
    indices_3m_4 = (indices_3m[1],indices_3m[2],indices_3m[3],indices_3m[4])
    indices_4m_4 = (indices_4m[1],indices_4m[2],indices_4m[3],indices_4m[4])


    for ic = 1:NC1
        for jc = 1:NC1
            #U_n[ν](1 - γν)*ψ_{n+ν} 

            v1,v2,v3,v4 = mul_op(oneplusγ1, ψdata, jc, indices_1p)
            #for ia = 1:4
                C[ic, 1, indices...] += -κ * U1[ic, jc, indices_4...] * v1
                C[ic, 2, indices...] += -κ * U1[ic, jc, indices_4...] * v2
                C[ic, 3, indices...] += -κ * U1[ic, jc, indices_4...] * v3
                C[ic, 4, indices...] += -κ * U1[ic, jc, indices_4...] * v4
            #end
            v1,v2,v3,v4  = mul_op(oneplusγ2, ψdata, jc, indices_2p)
            #for ia = 1:4
                C[ic, 1, indices...] += -κ * U2[ic, jc, indices_4...] * v1
                C[ic, 2, indices...] += -κ * U2[ic, jc, indices_4...] * v2
                C[ic, 3, indices...] += -κ * U2[ic, jc, indices_4...] * v3
                C[ic, 4, indices...] += -κ * U2[ic, jc, indices_4...] * v4
            #end
            v1,v2,v3,v4  = mul_op(oneplusγ3, ψdata, jc, indices_3p)
            #for ia = 1:4
                C[ic, 1, indices...] += -κ * U3[ic, jc, indices_4...] * v1
                C[ic, 2, indices...] += -κ * U3[ic, jc, indices_4...] * v2
                C[ic, 3, indices...] += -κ * U3[ic, jc, indices_4...] * v3
                C[ic, 4, indices...] += -κ * U3[ic, jc, indices_4...] * v4
            #end
            v1,v2,v3,v4  = mul_op(oneplusγ4, ψdata, jc, indices_4p)
            #for ia = 1:4
                C[ic, 1, indices...] += -κ * U4[ic, jc, indices_4...] * v1
                C[ic, 2, indices...] += -κ * U4[ic, jc, indices_4...] * v2
                C[ic, 3, indices...] += -κ * U4[ic, jc, indices_4...] * v3
                C[ic, 4, indices...] += -κ * U4[ic, jc, indices_4...] * v4
            #end


            # U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
            v1,v2,v3,v4 = mul_op(oneminusγ1, ψdata, jc, indices_1m)
            #for ia = 1:4
                C[ic, 1, indices...] += -κ * U1[jc, ic, indices_1m_4...]' * v1
                C[ic, 2, indices...] += -κ * U1[jc, ic, indices_1m_4...]' * v2
                C[ic, 3, indices...] += -κ * U1[jc, ic, indices_1m_4...]' * v3
                C[ic, 4, indices...] += -κ * U1[jc, ic, indices_1m_4...]' * v4
            #end

            v1,v2,v3,v4 = mul_op(oneminusγ2, ψdata, jc, indices_2m)
            #for ia = 1:4
                C[ic, 1, indices...] += -κ * U2[jc, ic, indices_2m_4...]' * v1
                C[ic, 2, indices...] += -κ * U2[jc, ic, indices_2m_4...]' * v2
                C[ic, 3, indices...] += -κ * U2[jc, ic, indices_2m_4...]' * v3
                C[ic, 4, indices...] += -κ * U2[jc, ic, indices_2m_4...]' * v4
            #end

            v1,v2,v3,v4 = mul_op(oneminusγ3, ψdata, jc, indices_3m)
            #for ia = 1:4
                C[ic, 1, indices...] += -κ * U3[jc, ic, indices_3m_4...]' * v1
                C[ic, 2, indices...] += -κ * U3[jc, ic, indices_3m_4...]' * v2
                C[ic, 3, indices...] += -κ * U3[jc, ic, indices_3m_4...]' * v3
                C[ic, 4, indices...] += -κ * U3[jc, ic, indices_3m_4...]' * v4
            #end


            v1,v2,v3,v4 = mul_op(oneminusγ4, ψdata, jc, indices_4m)
            #for ia = 1:4
                C[ic, 1, indices...] += -κ * U4[jc, ic, indices_4m_4...]' * v1
                C[ic, 2, indices...] += -κ * U4[jc, ic, indices_4m_4...]' * v2
                C[ic, 3, indices...] += -κ * U4[jc, ic, indices_4m_4...]' * v3
                C[ic, 4, indices...] += -κ * U4[jc, ic, indices_4m_4...]' * v4
            #end


        end
    end

end
