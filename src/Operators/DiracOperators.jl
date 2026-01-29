


const shift_1p = (1, 0, 0, 0)
const shift_1m = (-1, 0, 0, 0)
const shift_2p = (0, 1, 0, 0)
const shift_2m = (0, -1, 0, 0)
const shift_3p = (0, 0, 1, 0)
const shift_3m = (0, 0, -1, 0)
const shift_4p = (0, 0, 0, 1)
const shift_4m = (0, 0, 0, -1)
const shifts_p = (shift_1p, shift_2p, shift_3p, shift_4p)
const shifts_m = (shift_1m, shift_2m, shift_3m, shift_4m)

include("WilsonDiracOperator.jl")


struct DiracOp{T,TF,Dmul,Ddagmul,P}
    U::Vector{T}
    apply::Dmul
    apply_dag::Ddagmul
    p::P
    temps::PreallocatedArray{T}
    phitemps::PreallocatedArray{TF}
end

function DiracOp(U, apply, apply_dag, p, phi; numtemp=4, numphitemp=4)
    T = eltype(U)
    Dmul = typeof(apply)
    Ddagmul = typeof(apply_dag)

    temps = PreallocatedArray(U[1]; num=numtemp, haslabel=false)
    phitemps = PreallocatedArray(phi; num=numphitemp, haslabel=false)
    TF = typeof(phi)


    return DiracOp{T,TF,Dmul,Ddagmul,typeof(p)}(U, apply, apply_dag, p, temps, phitemps)
end
export DiracOp

function LinearAlgebra.mul!(y, D::DiracOp, x)
    temp, ittemp = get_block(D.temps, 4)
    phitemp, itphitemp = get_block(D.phitemps, 4)
    D.apply(y, D.U[1], D.U[2], D.U[3], D.U[4], x, D.p, phitemp, temp)
    unused!(D.temps, ittemp)
    unused!(D.phitemps, itphitemp)
end



struct AdjointOp{Op}
    op::Op
end
Base.adjoint(D::DiracOp) = AdjointOp(D)

function LinearAlgebra.mul!(y, A::AdjointOp{<:DiracOp}, x)
    D = A.op
    temp, ittemp = get_block(D.temps, 4)
    phitemp, itphitemp = get_block(D.phitemps, 4)

    D.apply_dag(y, D.U[1], D.U[2], D.U[3], D.U[4], x, D.p, phitemp, temp)

    unused!(D.temps, ittemp)
    unused!(D.phitemps, itphitemp)
end



struct DdagDOp{T<:DiracOp}
    D::T
    function DdagDOp(D::T) where {T<:DiracOp}
        return new{T}(D)
    end
end
export DdagDOp

function LinearAlgebra.mul!(y, A::T, x) where {T<:DdagDOp}
    D = A.D
    phitemp1, itphitemp1 = get_block(D.phitemps)
    temp, ittemp = get_block(D.temps, 4)
    phitemp, itphitemp = get_block(D.phitemps, 4)

    D.apply(phitemp1, D.U[1], D.U[2], D.U[3], D.U[4], x, D.p, phitemp, temp)
    set_halo!(phitemp1)
    D.apply_dag(y, D.U[1], D.U[2], D.U[3], D.U[4], phitemp1, D.p, phitemp, temp)
    set_halo!(y)

    #DdagDmul!(y, D.U[1], D.U[2], D.U[3], D.U[4], x, D.p, phitemp1, temp, phitemp)
    unused!(D.phitemps, itphitemp1)
    unused!(D.temps, ittemp)
    unused!(D.phitemps, itphitemp)
end



function solve!(y, A::T, x; verboselevel=2) where {T<:DdagDOp}
    cg(y, A, x, A.D.phitemps; verboselevel)
end
export solve!

function pseudofermion_action(D::T, φ) where {T<:DiracOp}
    DdagD = DdagDOp(D)
    phitemp1, itphitemp1 = get_block(D.phitemps)
    η = phitemp1
    solve!(η, DdagD, φ)
    S = real(dot(φ, η))

    unused!(D.phitemps, itphitemp1)
    return S
end

export pseudofermion_action




function dSFdU end
export dSFdU