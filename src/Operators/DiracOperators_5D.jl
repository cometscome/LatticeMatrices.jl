struct Wilson_parameters
    κ_wilson::Float64
    M_wilson::Float64
end
using InteractiveUtils
#using CUDA #debug

struct D5DW_MobiusDomainwallOperator5D{T,L5} <: OperatorOnKernel
    U::Vector{T}
    mass::Float64
    wilson_params::Wilson_parameters
    b::Float64
    c::Float64


    function D5DW_MobiusDomainwallOperator5D(U::Vector{T}, L5, mass, M, b, c) where {T<:LatticeMatrix}
        r = 1
        Dim = length(U)
        κ_wilson = 1 / (2 * Dim * r + 2M)
        wilsonparam = Wilson_parameters(κ_wilson, M)

        if b == 1 && c == 1
            println("Shamir kernel (standard DW) is used")
        elseif b == 2 && c == 0
            println("Borici/Wilson kernel (truncated overlap) is used")
        elseif b == 2 && c == 1
            println("scaled Shamir kernel (Mobius DW) is used")
        end

        return new{T,L5}(U, mass, wilsonparam, b, c)
    end
end
export D5DW_MobiusDomainwallOperator5D



struct Adjoint_D5DW_MobiusDomainwallOperator5D{T} <: OperatorOnKernel
    parent::T
end

function Base.adjoint(A::T) where {T<:D5DW_MobiusDomainwallOperator5D}
    Adjoint_D5DW_MobiusDomainwallOperator5D{typeof(A)}(A)
end

@inline @inbounds function get_mass(x::T) where {T<:D5DW_MobiusDomainwallOperator5D}
    return x.mass
end

@inline @inbounds function get_wilson_params(x::T) where {T<:D5DW_MobiusDomainwallOperator5D}
    return x.wilson_params
end

@inline @inbounds function get_bc(x::T) where {T<:D5DW_MobiusDomainwallOperator5D}
    return x.b, x.c
end

#LatticeMatrix_standard{D,T,AT,NC1,NC2,nw,DI}
function LinearAlgebra.mul!(C::TC,
    Dirac::TD, ψ::Tp) where {T1,AT1,NC1,nw,DI,L5,TU,
    TC<:LatticeMatrix{5,T1,AT1,NC1,4,nw,DI},TD<:D5DW_MobiusDomainwallOperator5D{TU,L5},
    Tp<:LatticeMatrix{5,T1,AT1,NC1,4,nw,DI}}

    
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


    
    JACC.parallel_for(
        prod(C.PN), kernel_D5DW_MobiusDomainwallOperator5D!,
        Cdata, U1, U2, U3, U4, mass, wilson_params, ψdata,
        Val(NC1), Val(nw), C.indexer, Val(L5), coeff_plus, coeff_minus)
        

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

    ψdata = get_matrix(ψ)
    Cdata = get_matrix(C)


    
    JACC.parallel_for(
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

    ψdata = get_matrix(ψ)
    Cdata = get_matrix(C)


    
    JACC.parallel_for(
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
    Dirac::TD, ψ::Tp) where {T1,AT1,NC1,nw,DI,T,L5,
    TC<:LatticeMatrix{5,T1,AT1,NC1,4,nw,DI},
    TD<:Adjoint_D5DW_MobiusDomainwallOperator5D{D5DW_MobiusDomainwallOperator5D{T,L5}},
    Tp<:LatticeMatrix{5,T1,AT1,NC1,4,nw,DI}}

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


    JACC.parallel_for(
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

    #massfactor = 1
    massfactor = -(factor / (2 * κ) + 1)
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
        C[ic, 1, indices...] += coeff_1pg5 * massfactor * ψdata[ic, 1, indices_5p...]
        C[ic, 2, indices...] += coeff_1pg5 * massfactor * ψdata[ic, 2, indices_5p...]

        #(1-gamma_5) 1,2 only #LTK definition
        C[ic, 3, indices...] += coeff_1mg5 * massfactor * ψdata[ic, 3, indices_5m...]
        C[ic, 4, indices...] += coeff_1mg5 * massfactor * ψdata[ic, 4, indices_5m...]

    end

    #return

    coeff = factor
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

    κ = -0.5 * coeff * coeff_1mg5

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