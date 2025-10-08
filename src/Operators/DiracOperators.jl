struct WilsonDiracOperator4D{T} <: OperatorOnKernel
    U::Vector{T}
    κ::Float64
end

function WilsonDiracOperator4D(U::Vector{T},κ) where {T<:LatticeMatrix}
    return WilsonDiracOperator4D{T}(U,κ)
end

const shift_1p = (1,0,0,0)
const shift_1m = (-1,0,0,0)
const shift_2p = (0,1,0,0)
const shift_2m = (0,-1,0,0)
const shift_3p = (0,0,1,0)
const shift_3m = (0,0,-1,0)
const shift_4p = (0,0,0,1)
const shift_4m = (0,0,0,-1)


"""
ψ_n - κ sum_ν U_n[ν](1 - γν)*ψ_{n+ν} + U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
"""
function LinearAlgebra.mul!(C::TC,
    Dirac::TD, ψ::TC) where {T1,AT1,NC1,nw,DI,
        TC <: LatticeMatrix{4,T1,AT1,NC1,4,nw,DI},TD<:WilsonDiracOperator4D}
    

    U1 = get_matrix(Dirac.U[1])
    U2 = get_matrix(Dirac.U[2])
    U3 = get_matrix(Dirac.U[3])
    U4 = get_matrix(Dirac.U[4])
    ψdata = get_matrix(ψ)
    Cdata = get_matrix(C)

    JACC.parallel_for(
        prod(C.PN), kernel_WilsonDiracOperator4D!,Cdata,U1,U2,U3,U4,Dirac.κ,ψdata,
        Val(NC1),Val(nw),C.indexer)    

end



function kernel_WilsonDiracOperator4D!(i,C,U1,U2,U3,U4,κ,ψdata,::Val{NC1},::Val{nw},dindexer) where {NC1,nw}
    indices = delinearize(dindexer, i, nw)
    indices_1p = shiftindices(indices, shift_1p)
    indices_1m = shiftindices(indices, shift_1m)
    indices_2p = shiftindices(indices, shift_2p)
    indices_2m = shiftindices(indices, shift_2m)
    indices_3p = shiftindices(indices, shift_3p)
    indices_3m = shiftindices(indices, shift_3m)
    indices_4p = shiftindices(indices, shift_4p)
    indices_4m = shiftindices(indices, shift_4m)


    @inbounds for ic = 1:NC1
        for ia = 1:4
            C[ic, ia, indices...] = ψdata[ic,ia, indices...]
        end
    end

    @inbounds for ic = 1:NC1
        for jc = 1:NC1
            #U_n[ν](1 - γν)*ψ_{n+ν} 
            v = mul_op(oneminusγ1, ψdata, jc, indices_1p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ*U1[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneminusγ2, ψdata, jc, indices_2p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ*U2[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneminusγ3, ψdata, jc, indices_3p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ*U3[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneminusγ4, ψdata, jc, indices_4p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ*U4[ic, jc, indices...] * v[ia]
            end

            # U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
            v = mul_op(oneplusγ1, ψdata, jc, indices_1m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ*U1[jc, ic, indices_1m...]' * v[ia]
            end
            v = mul_op(oneplusγ2, ψdata, jc, indices_2m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ*U2[jc, ic, indices_2m...]' * v[ia]
            end
            v = mul_op(oneplusγ3, ψdata, jc, indices_3m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ*U3[jc, ic, indices_3m...]' * v[ia]
            end
            v = mul_op(oneplusγ4, ψdata, jc, indices_4m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ*U4[jc, ic, indices_4m...]' * v[ia]
            end

        end
    end


end