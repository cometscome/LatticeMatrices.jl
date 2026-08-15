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
    #println("mass = ", mass)


    _parallel_for_mutating!(C,
        prod(C.PN), kernel_adjoint_D5DW_MobiusDomainwallOperator5D!,
        Cdata, U1, U2, U3, U4, mass, wilson_params, ψdata,
        Val(NC1), Val(nw), C.indexer, Val(L5), coeff_plus, coeff_minus)

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
    _parallel_for_mutating!(C,
        prod(C.PN), kernel_adjoint_D5DW_GeneralizedDomainwallOperator5D!,
        get_matrix(C), U1, U2, U3, U4, get_mass(parent),
        get_wilson_params(parent), get_matrix(psi), a, b, c,
        Val(NC1), Val(nw), C.indexer, Val(L5))
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
