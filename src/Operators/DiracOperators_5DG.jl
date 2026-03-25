# struct Wilson_parameters
#     κ_wilson::Float64
#     M_wilson::Float64
# end
using InteractiveUtils
using StaticArrays
#using CUDA #debug

struct D5DW_GeneralizedDomainwallOperator5D{T,L5} <: OperatorOnKernel
    U::Vector{T}
    mass::Float64
    wilson_params::Wilson_parameters
    bs::Vector{Float64}
    cs::Vector{Float64}


    function D5DW_GeneralizedDomainwallOperator5D(U::Vector{T}, L5, mass, M, bs, cs) where {T<:LatticeMatrix}
        r = 1
        Dim = length(U)
        κ_wilson = 1 / (2 * Dim * r + 2M)
        wilsonparam = Wilson_parameters(κ_wilson, M)

        # if b == 1 && c == 1
        #     println("Shamir kernel (standard DW) is used")
        # elseif b == 2 && c == 0
        #     println("Borici/Wilson kernel (truncated overlap) is used")
        # elseif b == 2 && c == 1
        #     println("scaled Shamir kernel (Generalized DW) is used")
        # end

        return new{T,L5}(U, mass, wilsonparam, bs, cs)
    end
end
export D5DW_GeneralizedDomainwallOperator5D



struct Adjoint_D5DW_GeneralizedDomainwallOperator5D{T} <: OperatorOnKernel
    parent::T
end

function Base.adjoint(A::T) where {T<:D5DW_GeneralizedDomainwallOperator5D}
    Adjoint_D5DW_GeneralizedDomainwallOperator5D{typeof(A)}(A)
end

@inline @inbounds function get_mass(x::T) where {T<:D5DW_GeneralizedDomainwallOperator5D}
    return x.mass
end

@inline @inbounds function get_wilson_params(x::T) where {T<:D5DW_GeneralizedDomainwallOperator5D}
    return x.wilson_params
end

@inline @inbounds function get_bscs(x::T) where {T<:D5DW_GeneralizedDomainwallOperator5D}
    return x.bs, x.cs
end

#LatticeMatrix_standard{D,T,AT,NC1,NC2,nw,DI}
function LinearAlgebra.mul!(C::TC,
    Dirac::TD, ψ::Tp) where {T1,AT1,NC1,nw,DI,L5,TU,
    TC<:LatticeMatrix{5,T1,AT1,NC1,4,nw,DI},TD<:D5DW_GeneralizedDomainwallOperator5D{TU,L5},
    Tp<:LatticeMatrix{5,T1,AT1,NC1,4,nw,DI}}

    
    U1 = get_matrix(Dirac.U[1])
    U2 = get_matrix(Dirac.U[2])
    U3 = get_matrix(Dirac.U[3])
    U4 = get_matrix(Dirac.U[4])
    ψdata = get_matrix(ψ)
    Cdata = get_matrix(C)
    mass = get_mass(Dirac)
    wilson_params = get_wilson_params(Dirac)
    bs, cs = get_bscs(Dirac)
    coeffs_plus_svec = SVector{L5, Float64}(bs)
    coeffs_minus_svec = SVector{L5, Float64}(-cs)
    
    #println("mass = ", mass)
    # 確認用のコードを挿入
    if length(bs) != L5
        error("Dimension mismatch: bs has $(length(bs)) elements, but L5 is $L5")
    end

    JACC.parallel_for(
        prod(C.PN), kernel_D5DW_GeneralizedDomainwallOperator5D!,
        Cdata, U1, U2, U3, U4, mass, wilson_params, ψdata,
        Val(NC1), Val(nw), C.indexer, Val(L5), coeffs_plus_svec, coeffs_minus_svec)
        

end

# const shift_1p5D = (1, 0, 0, 0, 0)
# const shift_1m5D = (-1, 0, 0, 0, 0)
# const shift_2p5D = (0, 1, 0, 0, 0)
# const shift_2m5D = (0, -1, 0, 0, 0)
# const shift_3p5D = (0, 0, 1, 0, 0)
# const shift_3m5D = (0, 0, -1, 0, 0)
# const shift_4p5D = (0, 0, 0, 1, 0)
# const shift_4m5D = (0, 0, 0, -1, 0)
# const shift_5p5D = (0, 0, 0, 0, 1)
# const shift_5m5D = (0, 0, 0, 0, -1)


 function kernel_D5DW_GeneralizedDomainwallOperator5D!(i, C, U1, U2, U3, U4,
    mass, wilson_params, ψdata,
    ::Val{NC1}, ::Val{nw}, dindexer, ::Val{L5},
    coeffs_plus, coeffs_minus) where {NC1,nw,L5}
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

    coeff_plus = coeffs_plus[indices[5] - nw]
    coeff_minus = coeffs_minus[indices[5] - nw]
    

    kernel_apply_1pD!(C, ψdata, U1, U2, U3, U4, wilson_params.κ_wilson, coeff_plus, indices, Val(NC1),
        indices_1p, indices_1m, indices_2p, indices_2m,
        indices_3p, indices_3m, indices_4p, indices_4m)

    kernel_apply_1mD_F!(C, ψdata, U1, U2, U3, U4, wilson_params.κ_wilson, coeff_minus, indices, Val(NC1),
        indices_5p, indices_5m, mass, Val(L5), Val(nw))

end

function LinearAlgebra.mul!(C::TC,
    Dirac::TD, ψ::Tp) where {T1,AT1,NC1,nw,DI,T,L5,
    TC<:LatticeMatrix{5,T1,AT1,NC1,4,nw,DI},
    TD<:Adjoint_D5DW_GeneralizedDomainwallOperator5D{D5DW_GeneralizedDomainwallOperator5D{T,L5}},
    Tp<:LatticeMatrix{5,T1,AT1,NC1,4,nw,DI}}

    U1 = get_matrix(Dirac.parent.U[1])
    U2 = get_matrix(Dirac.parent.U[2])
    U3 = get_matrix(Dirac.parent.U[3])
    U4 = get_matrix(Dirac.parent.U[4])
    ψdata = get_matrix(ψ)
    Cdata = get_matrix(C)
    mass = get_mass(Dirac.parent)
    wilson_params = get_wilson_params(Dirac.parent)
    bs, cs = get_bscs(Dirac.parent)
    coeffs_plus_svec = SVector{L5, Float64}(bs)
    coeffs_minus_svec = SVector{L5, Float64}(-cs)
    #println("mass = ", mass)

    # 確認用のコードを挿入
    # if length(bs) != L5
    #     error("Dimension mismatch: bs has $(length(bs)) elements, but L5 is $L5")
    # end

    JACC.parallel_for(
        prod(C.PN), kernel_adjoint_D5DW_GeneralizedDomainwallOperator5D!,
        Cdata, U1, U2, U3, U4, mass, wilson_params, ψdata,
        Val(NC1), Val(nw), C.indexer, Val(L5), coeffs_plus_svec, coeffs_minus_svec)

end

function kernel_adjoint_D5DW_GeneralizedDomainwallOperator5D!(i, C, U1, U2, U3, U4,
    mass, wilson_params, ψdata,
    ::Val{NC1}, ::Val{nw}, dindexer, ::Val{L5},
    coeffs_plus, coeffs_minus) where {NC1,nw,L5}
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

    coeff_plus = coeffs_plus[indices[5] - nw]
    coeff_minus = coeffs_minus[indices[5] - nw]

    kernel_apply_1pDdag!(C, ψdata, U1, U2, U3, U4, wilson_params.κ_wilson, coeff_plus, indices, Val(NC1),
        indices_1p, indices_1m, indices_2p, indices_2m,
        indices_3p, indices_3m, indices_4p, indices_4m)

    kernel_apply_1mDdag_Fdag!(C, ψdata, U1, U2, U3, U4, wilson_params.κ_wilson, coeff_minus, indices, Val(NC1),
        indices_5p, indices_5m, mass, Val(L5), Val(nw))

end