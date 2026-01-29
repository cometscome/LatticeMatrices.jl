
struct WilsonDiracOperator4D{T} <: OperatorOnKernel
    U::Vector{T}
    κ::Float64

    function WilsonDiracOperator4D(U::Vector{T}, κ) where {T<:LatticeMatrix}
        @assert length(U) == 4 "U must be a vector of length 4."
        return new{T}(U, κ)
    end
end

export WilsonDiracOperator4D

struct Adjoint_WilsonDiracOperator4D{T} <: OperatorOnKernel
    parent::T
end

function Base.adjoint(A::T) where {T<:WilsonDiracOperator4D}
    Adjoint_WilsonDiracOperator4D{typeof(A)}(A)
end



"""
ψ_n - κ sum_ν U_n[ν](1 - γν)*ψ_{n+ν} + U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
"""
function LinearAlgebra.mul!(C::TC,
    Dirac::TD, ψ::TC) where {T1,AT1,NC1,nw,DI,
    TC<:LatticeMatrix{4,T1,AT1,NC1,4,nw,DI},TD<:WilsonDiracOperator4D}


    U1 = get_matrix(Dirac.U[1])
    U2 = get_matrix(Dirac.U[2])
    U3 = get_matrix(Dirac.U[3])
    U4 = get_matrix(Dirac.U[4])
    ψdata = get_matrix(ψ)
    Cdata = get_matrix(C)

    JACC.parallel_for(
        prod(C.PN), kernel_WilsonDiracOperator4D!, Cdata, U1, U2, U3, U4, Dirac.κ, ψdata,
        Val(NC1), Val(nw), C.indexer)

end



function kernel_WilsonDiracOperator4D!(i, C, U1, U2, U3, U4, κ, ψdata, ::Val{NC1}, ::Val{nw}, dindexer) where {NC1,nw}
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
            C[ic, ia, indices...] = ψdata[ic, ia, indices...]
        end
    end

    @inbounds for ic = 1:NC1
        for jc = 1:NC1
            #U_n[ν](1 - γν)*ψ_{n+ν} 

            v = mul_op(oneminusγ1, ψdata, jc, indices_1p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U1[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneminusγ2, ψdata, jc, indices_2p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U2[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneminusγ3, ψdata, jc, indices_3p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U3[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneminusγ4, ψdata, jc, indices_4p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U4[ic, jc, indices...] * v[ia]
            end


            # U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
            v = mul_op(oneplusγ1, ψdata, jc, indices_1m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U1[jc, ic, indices_1m...]' * v[ia]
            end

            v = mul_op(oneplusγ2, ψdata, jc, indices_2m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U2[jc, ic, indices_2m...]' * v[ia]
            end

            v = mul_op(oneplusγ3, ψdata, jc, indices_3m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U3[jc, ic, indices_3m...]' * v[ia]
            end


            v = mul_op(oneplusγ4, ψdata, jc, indices_4m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U4[jc, ic, indices_4m...]' * v[ia]
            end


        end
    end


end

@inline function muladdmulti(a1, b1, a2, b2, a3, b3)
    acc = zero(typeof(a1))
    acc = muladd(a1, b1, acc)
    acc = muladd(a2, b2, acc)
    acc = muladd(a3, b3, acc)
    return acc
end

@inline function kernel_Umgammax_p!(C, κ, U, ψdata, indices, indices_p, oneminusγ)
    v11, v12, v13, v14 = mul_op(oneminusγ, ψdata, 1, indices_p)
    v21, v22, v23, v24 = mul_op(oneminusγ, ψdata, 2, indices_p)
    v31, v32, v33, v34 = mul_op(oneminusγ, ψdata, 3, indices_p)

    U11 = U[1, 1, indices...]
    U12 = U[1, 2, indices...]
    U13 = U[1, 3, indices...]
    U21 = U[2, 1, indices...]
    U22 = U[2, 2, indices...]
    U23 = U[2, 3, indices...]
    U31 = U[3, 1, indices...]
    U32 = U[3, 2, indices...]
    U33 = U[3, 3, indices...]

    #C[1, 1, indices...] += -κ*(U11*v11 + U12*v21 + U13*v31)
    C[1, 1, indices...] += -κ * muladdmulti(U11, v11, U12, v21, U13, v31)
    C[2, 1, indices...] += -κ * muladdmulti(U21, v11, U22, v21, U23, v31)
    C[3, 1, indices...] += -κ * muladdmulti(U31, v11, U32, v21, U33, v31)

    C[1, 2, indices...] += -κ * muladdmulti(U11, v12, U12, v22, U13, v32)
    C[2, 2, indices...] += -κ * muladdmulti(U21, v12, U22, v22, U23, v32)
    C[3, 2, indices...] += -κ * muladdmulti(U31, v12, U32, v22, U33, v32)


    C[1, 3, indices...] += -κ * muladdmulti(U11, v13, U12, v23, U13, v33)
    C[2, 3, indices...] += -κ * muladdmulti(U21, v13, U22, v23, U23, v33)
    C[3, 3, indices...] += -κ * muladdmulti(U31, v13, U32, v23, U33, v33)

    C[1, 4, indices...] += -κ * muladdmulti(U11, v14, U12, v24, U13, v34)
    C[2, 4, indices...] += -κ * muladdmulti(U21, v14, U22, v24, U23, v34)
    C[3, 4, indices...] += -κ * muladdmulti(U31, v14, U32, v24, U33, v34)
end


@inline function kernel_Updaggammax_m!(C, κ, U, ψdata, indices, indices_m, oneplusγ)
    v11, v12, v13, v14 = mul_op(oneplusγ, ψdata, 1, indices_m)
    v21, v22, v23, v24 = mul_op(oneplusγ, ψdata, 2, indices_m)
    v31, v32, v33, v34 = mul_op(oneplusγ, ψdata, 3, indices_m)

    U11 = U[1, 1, indices_m...]'
    U12 = U[2, 1, indices_m...]'
    U13 = U[3, 1, indices_m...]'
    U21 = U[1, 2, indices_m...]'
    U22 = U[2, 2, indices_m...]'
    U23 = U[3, 2, indices_m...]'
    U31 = U[1, 3, indices_m...]'
    U32 = U[2, 3, indices_m...]'
    U33 = U[3, 3, indices_m...]'

    C[1, 1, indices...] += -κ * muladdmulti(U11, v11, U12, v21, U13, v31)
    C[2, 1, indices...] += -κ * muladdmulti(U21, v11, U22, v21, U23, v31)
    C[3, 1, indices...] += -κ * muladdmulti(U31, v11, U32, v21, U33, v31)

    C[1, 2, indices...] += -κ * muladdmulti(U11, v12, U12, v22, U13, v32)
    C[2, 2, indices...] += -κ * muladdmulti(U21, v12, U22, v22, U23, v32)
    C[3, 2, indices...] += -κ * muladdmulti(U31, v12, U32, v22, U33, v32)


    C[1, 3, indices...] += -κ * muladdmulti(U11, v13, U12, v23, U13, v33)
    C[2, 3, indices...] += -κ * muladdmulti(U21, v13, U22, v23, U23, v33)
    C[3, 3, indices...] += -κ * muladdmulti(U31, v13, U32, v23, U33, v33)

    C[1, 4, indices...] += -κ * muladdmulti(U11, v14, U12, v24, U13, v34)
    C[2, 4, indices...] += -κ * muladdmulti(U21, v14, U22, v24, U23, v34)
    C[3, 4, indices...] += -κ * muladdmulti(U31, v14, U32, v24, U33, v34)
end


function kernel_WilsonDiracOperator4D!(i, C, U1, U2, U3, U4, κ, ψdata, ::Val{3}, ::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)
    #U = (U1,U2,U3,U4)

    C[1, 1, indices...] = ψdata[1, 1, indices...]
    C[2, 1, indices...] = ψdata[2, 1, indices...]
    C[3, 1, indices...] = ψdata[3, 1, indices...]

    C[1, 2, indices...] = ψdata[1, 2, indices...]
    C[2, 2, indices...] = ψdata[2, 2, indices...]
    C[3, 2, indices...] = ψdata[3, 2, indices...]


    C[1, 3, indices...] = ψdata[1, 3, indices...]
    C[2, 3, indices...] = ψdata[2, 3, indices...]
    C[3, 3, indices...] = ψdata[3, 3, indices...]

    C[1, 4, indices...] = ψdata[1, 4, indices...]
    C[2, 4, indices...] = ψdata[2, 4, indices...]
    C[3, 4, indices...] = ψdata[3, 4, indices...]

    #@inbounds for ν=1:4
    @inbounds begin
        indices_p = shiftindices(indices, shift_1p)
        kernel_Umgammax_p!(C, κ, U1, ψdata, indices, indices_p, oneminusγ1)

        indices_m = shiftindices(indices, shift_1m)
        kernel_Updaggammax_m!(C, κ, U1, ψdata, indices, indices_m, oneplusγ1)

        indices_p = shiftindices(indices, shift_2p)
        kernel_Umgammax_p!(C, κ, U2, ψdata, indices, indices_p, oneminusγ2)

        indices_m = shiftindices(indices, shift_2m)
        kernel_Updaggammax_m!(C, κ, U2, ψdata, indices, indices_m, oneplusγ2)


        indices_p = shiftindices(indices, shift_3p)
        kernel_Umgammax_p!(C, κ, U3, ψdata, indices, indices_p, oneminusγ3)

        indices_m = shiftindices(indices, shift_3m)
        kernel_Updaggammax_m!(C, κ, U3, ψdata, indices, indices_m, oneplusγ3)


        indices_p = shiftindices(indices, shift_4p)
        kernel_Umgammax_p!(C, κ, U4, ψdata, indices, indices_p, oneminusγ4)

        indices_m = shiftindices(indices, shift_4m)
        kernel_Updaggammax_m!(C, κ, U4, ψdata, indices, indices_m, oneplusγ4)
    end

    #end


end


"""
ψ_n - κ sum_ν U_n[ν](1 + γν)*ψ_{n+ν} + U_{n-ν}[-ν]^+ (1 - γν)*ψ_{n-ν}
"""
function LinearAlgebra.mul!(C::TC,
    Dirac::TD, ψ::TC) where {T1,AT1,NC1,nw,DI,
    TC<:LatticeMatrix{4,T1,AT1,NC1,4,nw,DI},TD<:Adjoint_WilsonDiracOperator4D}


    U1 = get_matrix(Dirac.parent.U[1])
    U2 = get_matrix(Dirac.parent.U[2])
    U3 = get_matrix(Dirac.parent.U[3])
    U4 = get_matrix(Dirac.parent.U[4])
    ψdata = get_matrix(ψ)
    Cdata = get_matrix(C)

    JACC.parallel_for(
        prod(C.PN), kernel_adjoint_WilsonDiracOperator4D!, Cdata, U1, U2, U3, U4, Dirac.parent.κ, ψdata,
        Val(NC1), Val(nw), C.indexer)

end


function kernel_adjoint_WilsonDiracOperator4D!(i, C, U1, U2, U3, U4, κ, ψdata, ::Val{NC1}, ::Val{nw}, dindexer) where {NC1,nw}
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
            C[ic, ia, indices...] = ψdata[ic, ia, indices...]
        end
    end

    @inbounds for ic = 1:NC1
        for jc = 1:NC1
            #U_n[ν](1 - γν)*ψ_{n+ν} 

            v = mul_op(oneplusγ1, ψdata, jc, indices_1p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U1[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneplusγ2, ψdata, jc, indices_2p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U2[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneplusγ3, ψdata, jc, indices_3p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U3[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneplusγ4, ψdata, jc, indices_4p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U4[ic, jc, indices...] * v[ia]
            end


            # U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
            v = mul_op(oneminusγ1, ψdata, jc, indices_1m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U1[jc, ic, indices_1m...]' * v[ia]
            end

            v = mul_op(oneminusγ2, ψdata, jc, indices_2m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U2[jc, ic, indices_2m...]' * v[ia]
            end

            v = mul_op(oneminusγ3, ψdata, jc, indices_3m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U3[jc, ic, indices_3m...]' * v[ia]
            end


            v = mul_op(oneminusγ4, ψdata, jc, indices_4m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U4[jc, ic, indices_4m...]' * v[ia]
            end


        end
    end


end


function kernel_adjoint_WilsonDiracOperator4D!(i, C, U1, U2, U3, U4, κ, ψdata, ::Val{3}, ::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)
    #U = (U1,U2,U3,U4)

    C[1, 1, indices...] = ψdata[1, 1, indices...]
    C[2, 1, indices...] = ψdata[2, 1, indices...]
    C[3, 1, indices...] = ψdata[3, 1, indices...]

    C[1, 2, indices...] = ψdata[1, 2, indices...]
    C[2, 2, indices...] = ψdata[2, 2, indices...]
    C[3, 2, indices...] = ψdata[3, 2, indices...]


    C[1, 3, indices...] = ψdata[1, 3, indices...]
    C[2, 3, indices...] = ψdata[2, 3, indices...]
    C[3, 3, indices...] = ψdata[3, 3, indices...]

    C[1, 4, indices...] = ψdata[1, 4, indices...]
    C[2, 4, indices...] = ψdata[2, 4, indices...]
    C[3, 4, indices...] = ψdata[3, 4, indices...]

    #@inbounds for ν=1:4
    @inbounds begin
        indices_p = shiftindices(indices, shift_1p)
        kernel_Umgammax_p!(C, κ, U1, ψdata, indices, indices_p, oneplusγ1)

        indices_m = shiftindices(indices, shift_1m)
        kernel_Updaggammax_m!(C, κ, U1, ψdata, indices, indices_m, oneminusγ1)

        indices_p = shiftindices(indices, shift_2p)
        kernel_Umgammax_p!(C, κ, U2, ψdata, indices, indices_p, oneplusγ2)

        indices_m = shiftindices(indices, shift_2m)
        kernel_Updaggammax_m!(C, κ, U2, ψdata, indices, indices_m, oneminusγ2)


        indices_p = shiftindices(indices, shift_3p)
        kernel_Umgammax_p!(C, κ, U3, ψdata, indices, indices_p, oneplusγ3)

        indices_m = shiftindices(indices, shift_3m)
        kernel_Updaggammax_m!(C, κ, U3, ψdata, indices, indices_m, oneminusγ3)


        indices_p = shiftindices(indices, shift_4p)
        kernel_Umgammax_p!(C, κ, U4, ψdata, indices, indices_p, oneplusγ4)

        indices_m = shiftindices(indices, shift_4m)
        kernel_Updaggammax_m!(C, κ, U4, ψdata, indices, indices_m, oneminusγ4)
    end

    #end


end

struct WilsonDiracOperator4D_Donly{T} <: OperatorOnKernel
    U::Vector{T}

    function WilsonDiracOperator4D_Donly(U::Vector{T}) where {T<:LatticeMatrix}
        @assert length(U) == 4 "U must be a vector of length 4."
        return new{T}(U)
    end
end
export WilsonDiracOperator4D_Donly

"""
0.5 sum_ν U_n[ν](1 - γν)*ψ_{n+ν} + U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
"""
function LinearAlgebra.mul!(C::TC,
    Dirac::TD, ψ::TC) where {T1,AT1,NC1,nw,DI,
    TC<:LatticeMatrix{4,T1,AT1,NC1,4,nw,DI},TD<:WilsonDiracOperator4D_Donly}


    U1 = get_matrix(Dirac.U[1])
    U2 = get_matrix(Dirac.U[2])
    U3 = get_matrix(Dirac.U[3])
    U4 = get_matrix(Dirac.U[4])
    ψdata = get_matrix(ψ)
    Cdata = get_matrix(C)

    JACC.parallel_for(
        prod(C.PN), kernel_WilsonDiracOperator4D_Donly!, Cdata, U1, U2, U3, U4, ψdata,
        Val(NC1), Val(nw), C.indexer)

end


function kernel_WilsonDiracOperator4D_Donly!(i, C, U1, U2, U3, U4, ψdata, ::Val{NC1}, ::Val{nw}, dindexer) where {NC1,nw}
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
            C[ic, ia, indices...] = zero(ψdata[ic, ia, indices...])
        end
    end

    @inbounds for ic = 1:NC1
        for jc = 1:NC1
            #U_n[ν](1 - γν)*ψ_{n+ν} 

            v = mul_op(oneminusγ1, ψdata, jc, indices_1p)
            for ia = 1:4
                C[ic, ia, indices...] += 0.5 * U1[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneminusγ2, ψdata, jc, indices_2p)
            for ia = 1:4
                C[ic, ia, indices...] += 0.5 * U2[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneminusγ3, ψdata, jc, indices_3p)
            for ia = 1:4
                C[ic, ia, indices...] += 0.5 * U3[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneminusγ4, ψdata, jc, indices_4p)
            for ia = 1:4
                C[ic, ia, indices...] += 0.5 * U4[ic, jc, indices...] * v[ia]
            end


            # U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
            v = mul_op(oneplusγ1, ψdata, jc, indices_1m)
            for ia = 1:4
                C[ic, ia, indices...] += 0.5 * U1[jc, ic, indices_1m...]' * v[ia]
            end

            v = mul_op(oneplusγ2, ψdata, jc, indices_2m)
            for ia = 1:4
                C[ic, ia, indices...] += 0.5 * U2[jc, ic, indices_2m...]' * v[ia]
            end

            v = mul_op(oneplusγ3, ψdata, jc, indices_3m)
            for ia = 1:4
                C[ic, ia, indices...] += 0.5 * U3[jc, ic, indices_3m...]' * v[ia]
            end


            v = mul_op(oneplusγ4, ψdata, jc, indices_4m)
            for ia = 1:4
                C[ic, ia, indices...] += 0.5 * U4[jc, ic, indices_4m...]' * v[ia]
            end


        end
    end


end


function kernel_WilsonDiracOperator4D_Donly!(i, C, U1, U2, U3, U4, ψdata, ::Val{3}, ::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)
    #U = (U1,U2,U3,U4)
    v0 = zero(ψdata[1, 1, indices...])

    C[1, 1, indices...] = v0
    C[2, 1, indices...] = v0
    C[3, 1, indices...] = v0

    C[1, 2, indices...] = v0
    C[2, 2, indices...] = v0
    C[3, 2, indices...] = v0


    C[1, 3, indices...] = v0
    C[2, 3, indices...] = v0
    C[3, 3, indices...] = v0

    C[1, 4, indices...] = v0
    C[2, 4, indices...] = v0
    C[3, 4, indices...] = v0

    #@inbounds for ν=1:4
    κ = -0.5
    @inbounds begin
        indices_p = shiftindices(indices, shift_1p)
        kernel_Umgammax_p!(C, κ, U1, ψdata, indices, indices_p, oneminusγ1)

        indices_m = shiftindices(indices, shift_1m)
        kernel_Updaggammax_m!(C, κ, U1, ψdata, indices, indices_m, oneplusγ1)

        indices_p = shiftindices(indices, shift_2p)
        kernel_Umgammax_p!(C, κ, U2, ψdata, indices, indices_p, oneminusγ2)

        indices_m = shiftindices(indices, shift_2m)
        kernel_Updaggammax_m!(C, κ, U2, ψdata, indices, indices_m, oneplusγ2)


        indices_p = shiftindices(indices, shift_3p)
        kernel_Umgammax_p!(C, κ, U3, ψdata, indices, indices_p, oneminusγ3)

        indices_m = shiftindices(indices, shift_3m)
        kernel_Updaggammax_m!(C, κ, U3, ψdata, indices, indices_m, oneplusγ3)


        indices_p = shiftindices(indices, shift_4p)
        kernel_Umgammax_p!(C, κ, U4, ψdata, indices, indices_p, oneminusγ4)

        indices_m = shiftindices(indices, shift_4m)
        kernel_Updaggammax_m!(C, κ, U4, ψdata, indices, indices_m, oneplusγ4)
    end

    #end


end

struct Adjoint_WilsonDiracOperator4D_Donly{T} <: OperatorOnKernel
    parent::T
end

function Base.adjoint(A::T) where {T<:WilsonDiracOperator4D_Donly}
    Adjoint_WilsonDiracOperator4D_Donly{typeof(A)}(A)
end


"""
0.5 sum_ν U_n[ν](1 + γν)*ψ_{n+ν} + U_{n-ν}[-ν]^+ (1 - γν)*ψ_{n-ν}
"""
function LinearAlgebra.mul!(C::TC,
    Dirac::TD, ψ::TC) where {T1,AT1,NC1,nw,DI,
    TC<:LatticeMatrix{4,T1,AT1,NC1,4,nw,DI},TD<:Adjoint_WilsonDiracOperator4D_Donly}


    U1 = get_matrix(Dirac.parent.U[1])
    U2 = get_matrix(Dirac.parent.U[2])
    U3 = get_matrix(Dirac.parent.U[3])
    U4 = get_matrix(Dirac.parent.U[4])
    ψdata = get_matrix(ψ)
    Cdata = get_matrix(C)

    JACC.parallel_for(
        prod(C.PN), kernel_adjoint_WilsonDiracOperator4D_Donly!, Cdata, U1, U2, U3, U4, ψdata,
        Val(NC1), Val(nw), C.indexer)

end


function kernel_adjoint_WilsonDiracOperator4D_Donly!(i, C, U1, U2, U3, U4, ψdata, ::Val{NC1}, ::Val{nw}, dindexer) where {NC1,nw}
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
            C[ic, ia, indices...] = zero(ψdata[ic, ia, indices...])
        end
    end

    κ = -0.5
    @inbounds for ic = 1:NC1
        for jc = 1:NC1
            #U_n[ν](1 - γν)*ψ_{n+ν} 

            v = mul_op(oneplusγ1, ψdata, jc, indices_1p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U1[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneplusγ2, ψdata, jc, indices_2p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U2[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneplusγ3, ψdata, jc, indices_3p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U3[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneplusγ4, ψdata, jc, indices_4p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U4[ic, jc, indices...] * v[ia]
            end


            # U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
            v = mul_op(oneminusγ1, ψdata, jc, indices_1m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U1[jc, ic, indices_1m...]' * v[ia]
            end

            v = mul_op(oneminusγ2, ψdata, jc, indices_2m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U2[jc, ic, indices_2m...]' * v[ia]
            end

            v = mul_op(oneminusγ3, ψdata, jc, indices_3m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U3[jc, ic, indices_3m...]' * v[ia]
            end


            v = mul_op(oneminusγ4, ψdata, jc, indices_4m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U4[jc, ic, indices_4m...]' * v[ia]
            end


        end
    end


end


function kernel_adjoint_WilsonDiracOperator4D_Donly!(i, C, U1, U2, U3, U4, ψdata, ::Val{3}, ::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)
    #U = (U1,U2,U3,U4)

    κ = -0.5
    v0 = zero(ψdata[1, 1, indices...])

    C[1, 1, indices...] = v0
    C[2, 1, indices...] = v0
    C[3, 1, indices...] = v0

    C[1, 2, indices...] = v0
    C[2, 2, indices...] = v0
    C[3, 2, indices...] = v0


    C[1, 3, indices...] = v0
    C[2, 3, indices...] = v0
    C[3, 3, indices...] = v0

    C[1, 4, indices...] = v0
    C[2, 4, indices...] = v0
    C[3, 4, indices...] = v0

    #@inbounds for ν=1:4
    @inbounds begin
        indices_p = shiftindices(indices, shift_1p)
        kernel_Umgammax_p!(C, κ, U1, ψdata, indices, indices_p, oneplusγ1)

        indices_m = shiftindices(indices, shift_1m)
        kernel_Updaggammax_m!(C, κ, U1, ψdata, indices, indices_m, oneminusγ1)

        indices_p = shiftindices(indices, shift_2p)
        kernel_Umgammax_p!(C, κ, U2, ψdata, indices, indices_p, oneplusγ2)

        indices_m = shiftindices(indices, shift_2m)
        kernel_Updaggammax_m!(C, κ, U2, ψdata, indices, indices_m, oneminusγ2)


        indices_p = shiftindices(indices, shift_3p)
        kernel_Umgammax_p!(C, κ, U3, ψdata, indices, indices_p, oneplusγ3)

        indices_m = shiftindices(indices, shift_3m)
        kernel_Updaggammax_m!(C, κ, U3, ψdata, indices, indices_m, oneminusγ3)


        indices_p = shiftindices(indices, shift_4p)
        kernel_Umgammax_p!(C, κ, U4, ψdata, indices, indices_p, oneplusγ4)

        indices_m = shiftindices(indices, shift_4m)
        kernel_Updaggammax_m!(C, κ, U4, ψdata, indices, indices_m, oneminusγ4)
    end

    #end


end