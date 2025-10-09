abstract type OperatorOnKernel end

abstract type OperatorSecond{NG1,NG2} <: OperatorOnKernel end

abstract type OperatorFirst{NC1,NC2} <: OperatorOnKernel end

struct ThreeOperators{T1<:AbstractLattice,T2<:OperatorSecond,T3<: AbstractLattice} <: OperatorOnKernel 
    op1::T1
    op2::T2
    op3::T3
end
export ThreeOperators


struct Oneγ{pm,μ} <: OperatorSecond{4,4}
    function Oneγ(μ, plus=true)
        pm = plus ? 1 : -1
        new{pm,μ}()
    end
end
export Oneγ


const oneplusγ1 = Oneγ(1, true)
export oneplusγ1
const oneplusγ2 = Oneγ(2, true)
export oneplusγ2
const oneplusγ3 = Oneγ(3, true)
export oneplusγ3
const oneplusγ4 = Oneγ(4, true)
export oneplusγ4
const oneplusγs = (oneplusγ1, oneplusγ2, oneplusγ3, oneplusγ4)

const oneminusγ1 = Oneγ(1, false)
export oneminusγ1
const oneminusγ2 = Oneγ(2, false)
export oneminusγ2
const oneminusγ3 = Oneγ(3, false)
export oneminusγ3
const oneminusγ4 = Oneγ(4, false)
export oneminusγ4
const oneminusγs = (oneminusγ1, oneminusγ2, oneminusγ3, oneminusγ4)

const γ1 = [0 0 0 -im;
    0 0 -im 0;
    0 im 0 0;
    im 0 0 0]
export γ1
const γ2 = [0 0 0 -1;
    0 0 1 0;
    0 1 0 0;
    -1 0 0 0]
export γ2

const γ3 = [0 0 -im 0;
    0 0 0 +im;
    im 0 0 0;
    0 -im 0 0]
export γ3

const γ4 = [0 0 -1 0;
    0 0 0 -1;
    -1 0 0 0;
    0 -1 0 0]
export γ4
const γs = (γ1, γ2, γ3, γ4)
export γs



@inline function mul_op(op::Oneγ{1,1}, x, ic, indices::NTuple{N,T}) where {N,T<:Integer}
    v1 = x[ic, 1, indices...] -   im * x[ic, 4, indices...]
    v2 = x[ic, 2, indices...] -   im * x[ic, 3, indices...]
    v3 = x[ic, 3, indices...] +  im * x[ic, 2, indices...]
    v4 = x[ic, 4, indices...] +  im * x[ic, 1, indices...]
    return v1, v2, v3, v4
end

@inline function mul_op(op::Oneγ{1,2}, x, ic, indices::NTuple{N,T}) where {N,T<:Integer}
    v1 = x[ic, 1, indices...] -  x[ic, 4, indices...]
    v2 = x[ic, 2, indices...] +  x[ic, 3, indices...]
    v3 = x[ic, 3, indices...] +  x[ic, 2, indices...]
    v4 = x[ic, 4, indices...] -  x[ic, 1, indices...]
    return v1, v2, v3, v4
end

@inline function mul_op(op::Oneγ{1,3}, x, ic, indices::NTuple{N,T}) where {N,T<:Integer}
    v1 = x[ic, 1, indices...] -  im * x[ic, 3, indices...]
    v2 = x[ic, 2, indices...] +  im * x[ic, 4, indices...]
    v3 = x[ic, 3, indices...] +  im * x[ic, 1, indices...]
    v4 = x[ic, 4, indices...] -  im * x[ic, 2, indices...]
    return v1, v2, v3, v4
end

@inline function mul_op(op::Oneγ{1,4}, x, ic, indices::NTuple{N,T}) where {N,T<:Integer}
    v1 = x[ic, 1, indices...] -  x[ic, 3, indices...]
    v2 = x[ic, 2, indices...] -  x[ic, 4, indices...]
    v3 = x[ic, 3, indices...] -  x[ic, 1, indices...]
    v4 = x[ic, 4, indices...] -  x[ic, 2, indices...]
    return v1, v2, v3, v4
end

@inline function mul_op(op::Oneγ{-1,1}, x, ic,indices::NTuple{N,T}) where {N,T<:Integer}
    v1 = x[ic, 1, indices...] + im * x[ic, 4, indices...]
    v2 = x[ic, 2, indices...] + im * x[ic, 3, indices...]
    v3 = x[ic, 3, indices...] - im * x[ic, 2, indices...]
    v4 = x[ic, 4, indices...] - im * x[ic, 1, indices...]
    return v1, v2, v3, v4
end

@inline @inline function mul_op(op::Oneγ{-1,2}, x, ic, indices::NTuple{N,T}) where {N,T<:Integer}
    v1 = x[ic, 1, indices...] + x[ic, 4, indices...]
    v2 = x[ic, 2, indices...] - x[ic, 3, indices...]
    v3 = x[ic, 3, indices...] - x[ic, 2, indices...]
    v4 = x[ic, 4, indices...] + x[ic, 1, indices...]
    return v1, v2, v3, v4
end

@inline function mul_op(op::Oneγ{-1,3}, x, ic, indices::NTuple{N,T}) where {N,T<:Integer}
    v1 = x[ic, 1, indices...] + im * x[ic, 3, indices...]
    v2 = x[ic, 2, indices...] - im * x[ic, 4, indices...]
    v3 = x[ic, 3, indices...] - im * x[ic, 1, indices...]
    v4 = x[ic, 4, indices...] + im * x[ic, 2, indices...]
    return v1, v2, v3, v4
end

@inline function mul_op(op::Oneγ{-1,4}, x, ic,indices::NTuple{N,T}) where {N,T<:Integer}
    v1 = x[ic, 1, indices...] + x[ic, 3, indices...]
    v2 = x[ic, 2, indices...] + x[ic, 4, indices...]
    v3 = x[ic, 3, indices...] + x[ic, 1, indices...]
    v4 = x[ic, 4, indices...] + x[ic, 2, indices...]
    return v1, v2, v3, v4
end

#C = B*A^T
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::TA, B::LatticeMatrix{D,T3,AT3,NC1,NC3,nw,DI}) where {D,T1,T3,AT1,AT3,NC1,NC2,NC3,nw,DI,TA<:OperatorSecond{NC2,NC3}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_OperatorSecond!, C.A, A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer
    )
    #set_halo!(C)
end

@inline function kernel_Dmatrix_mul_OperatorSecond!(i, C, A::TA, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer) where {
    NC1,NC2,NC3,nw,TA<:OperatorSecond{NC2,NC3}}
    indices = delinearize(dindexer, i, nw)
    @inbounds for ic = 1:NC1
        v = mul_op(A, B, ic, indices)
        #v = mul_op(B, ic, indices...)
        for jc = 1:NC2
            C[ic, jc, indices...] = v[jc]
        end
    end
end

#C = shiftedB*A^T
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::OperatorSecond{NC2,NC3}, B::Shifted_Lattice{L,shift}) where {D,T1,T3,AT1,AT3,NC1,NC2,NC3,nw,DI,L<:LatticeMatrix{D,T3,AT3,NC1,NC3,nw,DI},shift}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mulshifted_OperatorSecond!, C.A, A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift
    )
    #set_halo!(C)
end

@inline function kernel_Dmatrix_mulshifted_OperatorSecond!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)

    @inbounds for ic = 1:NC1
        v = mul_op(A, B, ic, indices_p)
        for jc = 1:NC2
            C[ic, jc, indices...] = v[jc]
        end
    end
end


#C = U*B*A^T
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    U::LatticeMatrix{D,T2,AT2,NC1,NC3,nw1,DI},
    A::OperatorSecond{NC2,NC4}, B::LatticeMatrix{D,T3,AT3,NC3,NC4,nw,DI}) where {
    D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,NC4,nw,nw1,DI}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_UOperatorSecondB!, C.A, U.A, A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(NC4), Val(nw), Val(nw1), C.indexer
    )
    #set_halo!(C)
end

function kernel_Dmatrix_mul_UOperatorSecondB!(i, C, U, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{NC4}, ::Val{nw}, ::Val{nw1}, dindexer) where {
    NC1,NC2,NC3,NC4,nw,nw1}
    indices = delinearize(dindexer, i, nw)

    #x[ic,a] = sum_{jc,b} U[ic,jc]*A[a,b]*psi[jc,b]
    #x[ic,a] = sum_{jc} U[ic,jc]*(sum_b A[a,b]*psi[jc,b])
    @inbounds for ic = 1:NC1
        for ia = 1:NC2
            C[ic, ia, indices...] = zero(eltype(C))
        end
    end
    @inbounds for ic = 1:NC1
        for jc = 1:NC3
            v = mul_op(A, B, jc, indices)
            for ia = 1:NC2
                C[ic, ia, indices...] += U[ic, jc, indices...] * v[ia]
            end
        end
    end
end

function kernel_Dmatrix_mul_UOperatorSecondB!(i, C, U, A, B, ::Val{3}, ::Val{4}, ::Val{3}, ::Val{4}, ::Val{nw}, ::Val{nw1}, dindexer) where {
    nw,nw1}
    indices = delinearize(dindexer, i, nw)

    #x[ic,a] = sum_{jc,b} U[ic,jc]*A[a,b]*psi[jc,b]
    #x[ic,a] = sum_{jc} U[ic,jc]*(sum_b A[a,b]*psi[jc,b])

    v11,v12,v13,v14 = mul_op(A, B, 1, indices)
    v21,v22,v23,v24 = mul_op(A, B, 2, indices)
    v31,v32,v33,v34 = mul_op(A, B, 3, indices)
    U11 = U[1, 1, indices...]
    U12 = U[1, 2, indices...]
    U13 = U[1, 3, indices...]
    U21 = U[2, 1, indices...]
    U22 = U[2, 2, indices...]
    U23 = U[2, 3, indices...]
    U31 = U[3, 1, indices...]
    U32 = U[3, 2, indices...]
    U33 = U[3, 3, indices...]

    C[1, 1, indices...] = U11*v11 + U12*v21 + U13*v31
    C[2, 1, indices...] = U21*v11 + U22*v21 + U23*v31
    C[3, 1, indices...] = U31*v11 + U32*v21 + U33*v31


    C[1, 2, indices...] = U11*v12 + U12*v22 + U13*v32
    C[2, 2, indices...] = U21*v12 + U22*v22 + U23*v32
    C[3, 2, indices...] = U31*v12 + U32*v22 + U33*v32


    C[1, 3, indices...] = U11*v13 + U12*v23 + U13*v33
    C[2, 3, indices...] = U21*v13 + U22*v23 + U23*v33
    C[3, 3, indices...] = U31*v13 + U32*v23 + U33*v33

    C[1, 4, indices...] = U11*v14 + U12*v24 + U13*v34
    C[2, 4, indices...] = U21*v14 + U22*v24 + U23*v34
    C[3, 4, indices...] = U31*v14 + U32*v24 + U33*v34

end

function kernel_Dmatrix_mul_UOperatorSecondB!(i, C, U, A, B, ::Val{2}, ::Val{4}, ::Val{2}, ::Val{4}, ::Val{nw}, ::Val{nw1}, dindexer) where {
    nw,nw1}
    indices = delinearize(dindexer, i, nw)

    #x[ic,a] = sum_{jc,b} U[ic,jc]*A[a,b]*psi[jc,b]
    #x[ic,a] = sum_{jc} U[ic,jc]*(sum_b A[a,b]*psi[jc,b])

    v11,v12,v13,v14 = mul_op(A, B, 1, indices)
    v21,v22,v23,v24 = mul_op(A, B, 2, indices)
    U11 = U[1, 1, indices...]
    U12 = U[1, 2, indices...]
    U21 = U[2, 1, indices...]
    U22 = U[2, 2, indices...]


    C[1, 1, indices...] = U11*v11 + U12*v21
    C[2, 1, indices...] = U21*v11 + U22*v21 
 
    C[1, 2, indices...] = U11*v12 + U12*v22
    C[2, 2, indices...] = U21*v12 + U22*v22 

    C[1, 3, indices...] = U11*v13 + U12*v23 
    C[2, 3, indices...] = U21*v13 + U22*v23 
 
    C[1, 4, indices...] = U11*v14 + U12*v24 
    C[2, 4, indices...] = U21*v14 + U22*v24 

end




#C = U*shiftedB*A^T
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    U::LatticeMatrix{D,T2,AT2,NC1,NC3,nw1,DI},
    A::OperatorSecond{NC2,NC4}, B::Shifted_Lattice{L,shift}) where {
    D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,NC4,nw,nw1,DI,L<:LatticeMatrix{D,T3,AT3,NC3,NC4,nw,DI},shift}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_UOperatorSecondshiftedB!, C.A, U.A, A, B.data.A,
        Val(NC1), Val(NC2), Val(NC3), Val(NC4), Val(nw), Val(nw1), C.indexer, shift
    )
    #set_halo!(C)
end

function kernel_Dmatrix_mul_UOperatorSecondshiftedB!(i, C, U, A, B,
    ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{NC4}, ::Val{nw}, ::Val{nw1}, dindexer, shift) where {
    NC1,NC2,NC3,NC4,nw,nw1}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)

    #x[ic,a] = sum_{jc,b} U[ic,jc]*A[a,b]*psi[jc,b]
    #x[ic,a] = sum_{jc} U[ic,jc]*(sum_b A[a,b]*psi[jc,b])
    @inbounds for ic = 1:NC1
        for ia = 1:NC2
            C[ic, ia, indices...] = zero(eltype(C))
        end
    end
    @inbounds for ic = 1:NC1
        for jc = 1:NC3
            v = mul_op(A, B, jc, indices_p)
            for ia = 1:NC2
                C[ic, ia, indices...] += U[ic, jc, indices...] * v[ia]
            end
        end
    end
end

function kernel_Dmatrix_mul_UOperatorSecondshiftedB!(i, C, U, A, B,
    ::Val{3}, ::Val{4}, ::Val{3}, ::Val{4}, ::Val{nw}, ::Val{nw1}, dindexer, shift) where {
    nw,nw1}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)

    v11,v12,v13,v14 = mul_op(A, B, 1, indices_p)
    v21,v22,v23,v24 = mul_op(A, B, 2, indices_p)
    v31,v32,v33,v34 = mul_op(A, B, 3, indices_p)
    U11 = U[1, 1, indices...]
    U12 = U[1, 2, indices...]
    U13 = U[1, 3, indices...]
    U21 = U[2, 1, indices...]
    U22 = U[2, 2, indices...]
    U23 = U[2, 3, indices...]
    U31 = U[3, 1, indices...]
    U32 = U[3, 2, indices...]
    U33 = U[3, 3, indices...]

    C[1, 1, indices...] = U11*v11 + U12*v21 + U13*v31
    C[2, 1, indices...] = U21*v11 + U22*v21 + U23*v31
    C[3, 1, indices...] = U31*v11 + U32*v21 + U33*v31


    C[1, 2, indices...] = U11*v12 + U12*v22 + U13*v32
    C[2, 2, indices...] = U21*v12 + U22*v22 + U23*v32
    C[3, 2, indices...] = U31*v12 + U32*v22 + U33*v32


    C[1, 3, indices...] = U11*v13 + U12*v23 + U13*v33
    C[2, 3, indices...] = U21*v13 + U22*v23 + U23*v33
    C[3, 3, indices...] = U31*v13 + U32*v23 + U33*v33

    C[1, 4, indices...] = U11*v14 + U12*v24 + U13*v34
    C[2, 4, indices...] = U21*v14 + U22*v24 + U23*v34
    C[3, 4, indices...] = U31*v14 + U32*v24 + U33*v34

end


#C = Udag*B*A^T
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    U::Adjoint_Lattice{L1},
    A::OperatorSecond{NC2,NC4}, B::LatticeMatrix{D,T3,AT3,NC3,NC4,nw,DI}) where {
    D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,NC4,nw,nw1,DI,L1<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw1,DI}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_UdagOperatorSecondB!, C.A, U.data.A, A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(NC4), Val(nw), Val(nw1), C.indexer
    )
    #set_halo!(C)
end

function kernel_Dmatrix_mul_UdagOperatorSecondB!(i, C, U, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{NC4}, ::Val{nw}, ::Val{nw1}, dindexer) where {
    NC1,NC2,NC3,NC4,nw,nw1}
    indices = delinearize(dindexer, i, nw)

    #x[ic,a] = sum_{jc,b} U[ic,jc]*A[a,b]*psi[jc,b]
    #x[ic,a] = sum_{jc} U[ic,jc]*(sum_b A[a,b]*psi[jc,b])
    @inbounds for ic = 1:NC1
        for ia = 1:NC2
            C[ic, ia, indices...] = zero(eltype(C))
        end
    end
    @inbounds for ic = 1:NC1
        for jc = 1:NC3
            v = mul_op(A, B, jc, indices)
            for ia = 1:NC2
                C[ic, ia, indices...] += conj(U[jc, ic, indices...]) * v[ia]
            end
        end
    end
end

function kernel_Dmatrix_mul_UdagOperatorSecondB!(i, C, U, A, B, ::Val{3}, ::Val{4}, ::Val{3}, ::Val{4}, ::Val{nw}, ::Val{nw1}, dindexer) where {
    nw,nw1}
    indices = delinearize(dindexer, i, nw)

    v11,v12,v13,v14 = mul_op(A, B, 1, indices)
    v21,v22,v23,v24 = mul_op(A, B, 2, indices)
    v31,v32,v33,v34 = mul_op(A, B, 3, indices)
    U11 = U[1, 1, indices...]'
    U12 = U[2, 1, indices...]'
    U13 = U[3, 1, indices...]'
    U21 = U[1, 2, indices...]'
    U22 = U[2, 2, indices...]'
    U23 = U[3, 2, indices...]'
    U31 = U[1, 3, indices...]'
    U32 = U[2, 3, indices...]'
    U33 = U[3, 3, indices...]'

    C[1, 1, indices...] = U11*v11 + U12*v21 + U13*v31
    C[2, 1, indices...] = U21*v11 + U22*v21 + U23*v31
    C[3, 1, indices...] = U31*v11 + U32*v21 + U33*v31


    C[1, 2, indices...] = U11*v12 + U12*v22 + U13*v32
    C[2, 2, indices...] = U21*v12 + U22*v22 + U23*v32
    C[3, 2, indices...] = U31*v12 + U32*v22 + U33*v32


    C[1, 3, indices...] = U11*v13 + U12*v23 + U13*v33
    C[2, 3, indices...] = U21*v13 + U22*v23 + U23*v33
    C[3, 3, indices...] = U31*v13 + U32*v23 + U33*v33

    C[1, 4, indices...] = U11*v14 + U12*v24 + U13*v34
    C[2, 4, indices...] = U21*v14 + U22*v24 + U23*v34
    C[3, 4, indices...] = U31*v14 + U32*v24 + U33*v34

end


function kernel_Dmatrix_mul_UdagOperatorSecondB!(i, C, U, A, B, ::Val{2}, ::Val{4}, ::Val{2}, ::Val{4}, ::Val{nw}, ::Val{nw1}, dindexer) where {
    nw,nw1}
    indices = delinearize(dindexer, i, nw)

    v11,v12,v13,v14 = mul_op(A, B, 1, indices)
    v21,v22,v23,v24 = mul_op(A, B, 2, indices)
    U11 = U[1, 1, indices...]'
    U12 = U[2, 1, indices...]'
    U21 = U[1, 2, indices...]'
    U22 = U[2, 2, indices...]'


    C[1, 1, indices...] = U11*v11 + U12*v21 #+ U13*v31
    C[2, 1, indices...] = U21*v11 + U22*v21 #+ U23*v31
    #C[3, 1, indices...] = U31*v11 + U32*v21 #+ U33*v31


    C[1, 2, indices...] = U11*v12 + U12*v22 #+ U13*v32
    C[2, 2, indices...] = U21*v12 + U22*v22 #+ U23*v32
    #C[3, 2, indices...] = U31*v12 + U32*v22 #+ U33*v32


    C[1, 3, indices...] = U11*v13 + U12*v23 #+ U13*v33
    C[2, 3, indices...] = U21*v13 + U22*v23 #+ U23*v33
    #C[3, 3, indices...] = U31*v13 + U32*v23 #+ U33*v33

    C[1, 4, indices...] = U11*v14 + U12*v24 #+ U13*v34
    C[2, 4, indices...] = U21*v14 + U22*v24 #+ U23*v34
    #C[3, 4, indices...] = U31*v14 + U32*v24 #+ U33*v34

end


#C = shiftedUdag*B*A^T
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    U::Adjoint_Lattice{Shifted_Lattice{L1,shift}},
    A::OperatorSecond{NC2,NC4}, B::LatticeMatrix{D,T3,AT3,NC3,NC4,nw,DI}) where {
    D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,NC4,nw,nw1,DI,shift,L1<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw1,DI}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftedUdagOperatorSecondB!, C.A, U.data.data.A, A, B.A, 
        Val(NC1), Val(NC2), Val(NC3), Val(NC4), Val(nw), Val(nw1), C.indexer,shift
    )
    #set_halo!(C)
end

function kernel_Dmatrix_mul_shiftedUdagOperatorSecondB!(i, C, U, A, B,
         ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{NC4}, ::Val{nw}, ::Val{nw1}, dindexer,shift) where {
    NC1,NC2,NC3,NC4,nw,nw1}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)

    #x[ic,a] = sum_{jc,b} U[ic,jc]*A[a,b]*psi[jc,b]
    #x[ic,a] = sum_{jc} U[ic,jc]*(sum_b A[a,b]*psi[jc,b])
    @inbounds for ic = 1:NC1
        for ia = 1:NC2
            C[ic, ia, indices...] = zero(eltype(C))
        end
    end
    @inbounds for ic = 1:NC1
        for jc = 1:NC3
            v = mul_op(A, B, jc, indices)
            for ia = 1:NC2
                C[ic, ia, indices...] += conj(U[jc, ic,  indices_p...]) * v[ia]
            end
        end
    end
end


function kernel_Dmatrix_mul_shiftedUdagOperatorSecondB!(i, C, U, A, B,
         ::Val{3}, ::Val{4}, ::Val{3}, ::Val{4}, ::Val{nw}, ::Val{nw1}, dindexer,shift) where {
    nw,nw1}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)


    v11,v12,v13,v14 = mul_op(A, B, 1, indices)
    v21,v22,v23,v24 = mul_op(A, B, 2, indices)
    v31,v32,v33,v34 = mul_op(A, B, 3, indices)
    U11 = U[1, 1, indices_p...]'
    U12 = U[2, 1, indices_p...]'
    U13 = U[3, 1, indices_p...]'
    U21 = U[1, 2, indices_p...]'
    U22 = U[2, 2, indices_p...]'
    U23 = U[3, 2, indices_p...]'
    U31 = U[1, 3, indices_p...]'
    U32 = U[2, 3, indices_p...]'
    U33 = U[3, 3, indices_p...]'

    C[1, 1, indices...] = U11*v11 + U12*v21 + U13*v31
    C[2, 1, indices...] = U21*v11 + U22*v21 + U23*v31
    C[3, 1, indices...] = U31*v11 + U32*v21 + U33*v31


    C[1, 2, indices...] = U11*v12 + U12*v22 + U13*v32
    C[2, 2, indices...] = U21*v12 + U22*v22 + U23*v32
    C[3, 2, indices...] = U31*v12 + U32*v22 + U33*v32


    C[1, 3, indices...] = U11*v13 + U12*v23 + U13*v33
    C[2, 3, indices...] = U21*v13 + U22*v23 + U23*v33
    C[3, 3, indices...] = U31*v13 + U32*v23 + U33*v33

    C[1, 4, indices...] = U11*v14 + U12*v24 + U13*v34
    C[2, 4, indices...] = U21*v14 + U22*v24 + U23*v34
    C[3, 4, indices...] = U31*v14 + U32*v24 + U33*v34

end

function kernel_Dmatrix_mul_shiftedUdagOperatorSecondB!(i, C, U, A, B,
         ::Val{2}, ::Val{4}, ::Val{2}, ::Val{4}, ::Val{nw}, ::Val{nw1}, dindexer,shift) where {
    nw,nw1}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)


    v11,v12,v13,v14 = mul_op(A, B, 1, indices)
    v21,v22,v23,v24 = mul_op(A, B, 2, indices)
    #v31,v32,v33,v34 = mul_op(A, B, 3, indices)
    U11 = U[1, 1, indices_p...]'
    U12 = U[2, 1, indices_p...]'
    #U13 = U[3, 1, indices_p...]'
    U21 = U[1, 2, indices_p...]'
    U22 = U[2, 2, indices_p...]'
    #U23 = U[3, 2, indices_p...]'
    #U31 = U[1, 3, indices_p...]'
    #U32 = U[2, 3, indices_p...]'
    #U33 = U[3, 3, indices_p...]'

    C[1, 1, indices...] = U11*v11 + U12*v21# + U13*v31
    C[2, 1, indices...] = U21*v11 + U22*v21# + U23*v31
    #C[3, 1, indices...] = U31*v11 + U32*v21 + U33*v31


    C[1, 2, indices...] = U11*v12 + U12*v22# + U13*v32
    C[2, 2, indices...] = U21*v12 + U22*v22# + U23*v32
    #C[3, 2, indices...] = U31*v12 + U32*v22 + U33*v32


    C[1, 3, indices...] = U11*v13 + U12*v23 #+ U13*v33
    C[2, 3, indices...] = U21*v13 + U22*v23 #+ U23*v33
    #C[3, 3, indices...] = U31*v13 + U32*v23 + U33*v33

    C[1, 4, indices...] = U11*v14 + U12*v24 #+ U13*v34
    C[2, 4, indices...] = U21*v14 + U22*v24 #+ U23*v34
    #C[3, 4, indices...] = U31*v14 + U32*v24 + U33*v34

end


#C = shiftedUdag*shiftedB*A^T
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    U::Adjoint_Lattice{Shifted_Lattice{L1,shift}},
    A::OperatorSecond{NC2,NC4}, B::Shifted_Lattice{L2,shift2}) where {
    D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,NC4,nw,nw1,DI,shift,L1<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw1,DI},
    shift2,L2<:LatticeMatrix{D,T3,AT3,NC3,NC4,nw,DI}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftedUdagOperatorSecondshiftedB!, C.A, U.data.data.A, A, B.data.A, 
        Val(NC1), Val(NC2), Val(NC3), Val(NC4), Val(nw), Val(nw1), C.indexer,shift,shift2
    )
    #set_halo!(C)
end

function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},T::ThreeOperators{Top1,Top2,Top3}) where {
    D,T1,Top1,Top2,Top3,AT1,NC1,NC2,nw,DI}
    mul!(C,T.op1,T.op2,T.op3)
end

function kernel_Dmatrix_mul_shiftedUdagOperatorSecondshiftedB!(i, C, U, A, B,
         ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{NC4}, ::Val{nw}, ::Val{nw1}, dindexer,shift,shift2) where {
    NC1,NC2,NC3,NC4,nw,nw1}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)
    indices_p2 = shiftindices(indices, shift2)

    #x[ic,a] = sum_{jc,b} U[ic,jc]*A[a,b]*psi[jc,b]
    #x[ic,a] = sum_{jc} U[ic,jc]*(sum_b A[a,b]*psi[jc,b])
    @inbounds for ic = 1:NC1
        for ia = 1:NC2
            C[ic, ia, indices...] = zero(eltype(C))
        end
    end
    @inbounds for ic = 1:NC1
        for jc = 1:NC3
            v = mul_op(A, B, jc, indices_p2)
            for ia = 1:NC2
                C[ic, ia, indices...] += conj(U[jc, ic,  indices_p...]) * v[ia]
            end
        end
    end
end

function kernel_Dmatrix_mul_shiftedUdagOperatorSecondshiftedB!(i, C, U, A, B,
          ::Val{3}, ::Val{4}, ::Val{3}, ::Val{4}, ::Val{nw}, ::Val{nw1}, dindexer,shift,shift2) where {
    nw,nw1}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)
    indices_p2 = shiftindices(indices, shift2)

    v11,v12,v13,v14 = mul_op(A, B, 1, indices_p2)
    v21,v22,v23,v24 = mul_op(A, B, 2, indices_p2)
    v31,v32,v33,v34 = mul_op(A, B, 3, indices_p2)
    U11 = U[1, 1, indices_p...]'
    U12 = U[2, 1, indices_p...]'
    U13 = U[3, 1, indices_p...]'
    U21 = U[1, 2, indices_p...]'
    U22 = U[2, 2, indices_p...]'
    U23 = U[3, 2, indices_p...]'
    U31 = U[1, 3, indices_p...]'
    U32 = U[2, 3, indices_p...]'
    U33 = U[3, 3, indices_p...]'

    C[1, 1, indices...] = U11*v11 + U12*v21 + U13*v31
    C[2, 1, indices...] = U21*v11 + U22*v21 + U23*v31
    C[3, 1, indices...] = U31*v11 + U32*v21 + U33*v31


    C[1, 2, indices...] = U11*v12 + U12*v22 + U13*v32
    C[2, 2, indices...] = U21*v12 + U22*v22 + U23*v32
    C[3, 2, indices...] = U31*v12 + U32*v22 + U33*v32


    C[1, 3, indices...] = U11*v13 + U12*v23 + U13*v33
    C[2, 3, indices...] = U21*v13 + U22*v23 + U23*v33
    C[3, 3, indices...] = U31*v13 + U32*v23 + U33*v33

    C[1, 4, indices...] = U11*v14 + U12*v24 + U13*v34
    C[2, 4, indices...] = U21*v14 + U22*v24 + U23*v34
    C[3, 4, indices...] = U31*v14 + U32*v24 + U33*v34

end

function kernel_Dmatrix_mul_shiftedUdagOperatorSecondshiftedB!(i, C, U, A, B,
          ::Val{2}, ::Val{4}, ::Val{2}, ::Val{4}, ::Val{nw}, ::Val{nw1}, dindexer,shift,shift2) where {
    nw,nw1}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)
    indices_p2 = shiftindices(indices, shift2)

    v11,v12,v13,v14 = mul_op(A, B, 1, indices_p2)
    v21,v22,v23,v24 = mul_op(A, B, 2, indices_p2)
    #v31,v32,v33,v34 = mul_op(A, B, 3, indices_p2)
    U11 = U[1, 1, indices_p...]'
    U12 = U[2, 1, indices_p...]'
    #U13 = U[3, 1, indices_p...]'
    U21 = U[1, 2, indices_p...]'
    U22 = U[2, 2, indices_p...]'
    #U23 = U[3, 2, indices_p...]'
    #U31 = U[1, 3, indices_p...]'
    #U32 = U[2, 3, indices_p...]'
    #U33 = U[3, 3, indices_p...]'

    C[1, 1, indices...] = U11*v11 + U12*v21 #+ U13*v31
    C[2, 1, indices...] = U21*v11 + U22*v21 #+ U23*v31
    #C[3, 1, indices...] = U31*v11 + U32*v21 + U33*v31


    C[1, 2, indices...] = U11*v12 + U12*v22 #+ U13*v32
    C[2, 2, indices...] = U21*v12 + U22*v22 #+ U23*v32
    #C[3, 2, indices...] = U31*v12 + U32*v22 + U33*v32


    C[1, 3, indices...] = U11*v13 + U12*v23 #+ U13*v33
    C[2, 3, indices...] = U21*v13 + U22*v23 #+ U23*v33
    #C[3, 3, indices...] = U31*v13 + U32*v23 + U33*v33

    C[1, 4, indices...] = U11*v14 + U12*v24 #+ U13*v34
    C[2, 4, indices...] = U21*v14 + U22*v24 #+ U23*v34
    #C[3, 4, indices...] = U31*v14 + U32*v24 + U33*v34

end

#C = U1*x1*A1^T + U2*x2*A2^T 
function mul_and_sum!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    U1::LatticeMatrix{D,T2,AT2,NC1,NC3,nw1,DI}, A1::OperatorSecond{NC2,NC4}, B1::LatticeMatrix{D,T3,AT3,NC3,NC4,nw,DI},
    U2::LatticeMatrix{D,T4,AT4,NC1,NC5,nw2,DI}, A2::OperatorSecond{NC2,NC6}, B2::LatticeMatrix{D,T5,AT5,NC5,NC6,nw,DI}) where {
    D,T1,T2,T3,T4,T5,AT1,AT2,AT3,AT4,AT5,NC1,NC2,NC3,NC4,NC5,NC6,nw,nw1,nw2,DI}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_UOperatorSecondB_sum!, C.A,
        U1.A, A1, B1.A,
        U2.A, A2, B2.A,
        Val(NC1), Val(NC2), Val(NC3), Val(NC4), Val(NC5), Val(NC6), Val(nw), Val(nw1), Val(nw2), C.indexer
    )
    #set_halo!(C)
end

function mul_and_sum!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    T_1::ThreeOperators{Top1_1,Top2_1,Top3_1},
    T_2::ThreeOperators{Top1_2,Top2_2,Top3_2}) where {
    D,T1,Top1_1,Top2_1,Top3_1,Top1_2,Top2_2,Top3_2,AT1,NC1,NC2,nw,DI}
    mul_and_sum!(C,T_1.op1,T_1.op2,T_1.op3,T_2.op1,T_2.op2,T_2.op3)
end

function kernel_Dmatrix_mul_UOperatorSecondB_sum!(i, C,
    U1, A1, B1,
    U2, A2, B2,
    ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{NC4}, ::Val{NC5}, ::Val{NC6}, ::Val{nw}, ::Val{nw1}, ::Val{nw2}, dindexer) where {
    NC1,NC2,NC3,NC4,NC5,NC6,nw,nw1,nw2}
    indices = delinearize(dindexer, i, nw)

        #x[ic,a] = sum_{jc,b} U[ic,jc]*A[a,b]*psi[jc,b]
    #x[ic,a] = sum_{jc} U[ic,jc]*(sum_b A[a,b]*psi[jc,b])

    #@inbounds for ic = 1:NC1
    #    for ia = 1:NC2
    #        C[ic, ia, indices...] = zero(eltype(C))
    #    end
    #end

    @inbounds for ic = 1:NC1
        for jc = 1:NC3
            v = mul_op(A1, B1, jc, indices)
            for ia = 1:NC2
                C[ic, ia, indices...] += U1[ic, jc, indices...] * v[ia]
            end
        end
    end

    @inbounds for ic = 1:NC1
        for jc = 1:NC5
            v = mul_op(A2, B2, jc, indices)
            for ia = 1:NC2
                C[ic, ia, indices...] += U2[ic, jc, indices...] * v[ia]
            end
        end
    end

end

function kernel_Dmatrix_mul_UOperatorSecondB_sum!(i, C,
    U1, A1, B1,
    U2, A2, B2,
    ::Val{3}, ::Val{4}, ::Val{3}, ::Val{4}, ::Val{3}, ::Val{4}, ::Val{nw}, ::Val{nw1}, ::Val{nw2}, dindexer) where {
    nw,nw1,nw2}
    indices = delinearize(dindexer, i, nw)


    
    v11_1,v12_1,v13_1,v14_1 = mul_op(A1, B1, 1, indices)
    v21_1,v22_1,v23_1,v24_1 = mul_op(A1, B1, 2, indices)
    v31_1,v32_1,v33_1,v34_1 = mul_op(A1, B1, 3, indices)

    v11_2,v12_2,v13_2,v14_2 = mul_op(A2, B2, 1, indices)
    v21_2,v22_2,v23_2,v24_2 = mul_op(A2, B2, 2, indices)
    v31_2,v32_2,v33_2,v34_2 = mul_op(A2, B2, 3, indices)


    U11_1 = U1[1, 1, indices...]
    U12_1 = U1[1, 2, indices...]
    U13_1 = U1[1, 3, indices...]
    U21_1 = U1[2, 1, indices...]
    U22_1 = U1[2, 2, indices...]
    U23_1 = U1[2, 3, indices...]
    U31_1 = U1[3, 1, indices...]
    U32_1 = U1[3, 2, indices...]
    U33_1 = U1[3, 3, indices...]

    U11_2 = U2[1, 1, indices...]
    U12_2 = U2[1, 2, indices...]
    U13_2 = U2[1, 3, indices...]
    U21_2 = U2[2, 1, indices...]
    U22_2 = U2[2, 2, indices...]
    U23_2 = U2[2, 3, indices...]
    U31_2 = U2[3, 1, indices...]
    U32_2 = U2[3, 2, indices...]
    U33_2 = U2[3, 3, indices...]

    C[1, 1, indices...] += U11_1*v11_1 + U12_1*v21_1 + U13_1*v31_1 + U11_2*v11_2 + U12_2*v21_2 + U13_2*v31_2
    C[2, 1, indices...] += U21_1*v11_1 + U22_1*v21_1 + U23_1*v31_1 + U21_2*v11_2 + U22_2*v21_2 + U23_2*v31_2
    C[3, 1, indices...] += U31_1*v11_1 + U32_1*v21_1 + U33_1*v31_1 + U31_2*v11_2 + U32_2*v21_2 + U33_2*v31_2

    C[1, 2, indices...] += U11_1*v12_1 + U12_1*v22_1 + U13_1*v32_1 + U11_2*v12_2 + U12_2*v22_2 + U13_2*v32_2
    C[2, 2, indices...] += U21_1*v12_1 + U22_1*v22_1 + U23_1*v32_1 + U21_2*v12_2 + U22_2*v22_2 + U23_2*v32_2
    C[3, 2, indices...] += U31_1*v12_1 + U32_1*v22_1 + U33_1*v32_1 + U31_2*v12_2 + U32_2*v22_2 + U33_2*v32_2

    C[1, 3, indices...] += U11_1*v13_1 + U12_1*v23_1 + U13_1*v33_1 + U11_2*v13_2 + U12_2*v23_2 + U13_2*v33_2
    C[2, 3, indices...] += U21_1*v13_1 + U22_1*v23_1 + U23_1*v33_1 + U21_2*v13_2 + U22_2*v23_2 + U23_2*v33_2
    C[3, 3, indices...] += U31_1*v13_1 + U32_1*v23_1 + U33_1*v33_1 + U31_2*v13_2 + U32_2*v23_2 + U33_2*v33_2

    C[1, 4, indices...] += U11_1*v14_1 + U12_1*v24_1 + U13_1*v34_1 + U11_2*v14_2 + U12_2*v24_2 + U13_2*v34_2
    C[2, 4, indices...] += U21_1*v14_1 + U22_1*v24_1 + U23_1*v34_1 + U21_2*v14_2 + U22_2*v24_2 + U23_2*v34_2
    C[3, 4, indices...] += U31_1*v14_1 + U32_1*v24_1 + U33_1*v34_1 + U31_2*v14_2 + U32_2*v24_2 + U33_2*v34_2
end

export mul_and_sum!

#C = U1*x1*A1^T + shiftedU2dag*shiftgedx2*A2^T 
function mul_and_sum!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    U1::LatticeMatrix{D,T2,AT2,NC1,NC3,nw1,DI}, A1::OperatorSecond{NC2,NC4}, B1::LatticeMatrix{D,T3,AT3,NC3,NC4,nw,DI},
    U2::Adjoint_Lattice{Shifted_Lattice{L2,shift1}}, A2::OperatorSecond{NC2,NC6}, B2::Shifted_Lattice{BL2,shift2}) where {
    D,T1,T2,T3,T4,T5,AT1,AT2,AT3,AT4,AT5,NC1,NC2,NC3,NC4,NC5,NC6,nw,nw1,nw2,DI,L2<:LatticeMatrix{D,T4,AT4,NC1,NC5,nw2,DI},
    BL2<:LatticeMatrix{D,T5,AT5,NC5,NC6,nw,DI},shift1,shift2}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftedUdagOperatorSecondshiftedB_sum!, C.A,
        U1.A, A1, B1.A,
        U2.data.data.A, A2, B2.data.A,
        Val(NC1), Val(NC2), Val(NC3), Val(NC4), Val(NC5), Val(NC6), Val(nw), Val(nw1), Val(nw2), C.indexer,
        shift1,shift2
    )
    #set_halo!(C)
end

function mul_and_sum!(C,U1,A1,B1,U2,A2,B2)
    error("No method for mul_and_sum! with types: $(typeof(C)), $(typeof(U1)), $(typeof(A1)), $(typeof(B1)), $(typeof(U2)), $(typeof(A2)), $(typeof(B2))")
end

function kernel_Dmatrix_mul_shiftedUdagOperatorSecondshiftedB_sum!(i, C,
    U1, A1, B1,
    U2, A2, B2,
    ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{NC4}, ::Val{NC5}, ::Val{NC6}, ::Val{nw}, ::Val{nw1}, ::Val{nw2}, dindexer,
    shift1,shift2) where {
    NC1,NC2,NC3,NC4,NC5,NC6,nw,nw1,nw2}
    indices = delinearize(dindexer, i, nw)
    indices_p1 = shiftindices(indices, shift1)
    indices_p2 = shiftindices(indices, shift2)

        #x[ic,a] = sum_{jc,b} U[ic,jc]*A[a,b]*psi[jc,b]
    #x[ic,a] = sum_{jc} U[ic,jc]*(sum_b A[a,b]*psi[jc,b])

    #@inbounds for ic = 1:NC1
    #    for ia = 1:NC2
    #        C[ic, ia, indices...] = zero(eltype(C))
    #    end
    #end

    @inbounds for ic = 1:NC1
        for jc = 1:NC3
            v = mul_op(A1, B1, jc, indices)
            for ia = 1:NC2
                C[ic, ia, indices...] += U1[ic, jc, indices...] * v[ia]
            end
        end
    end

    @inbounds for ic = 1:NC1
        for jc = 1:NC5
            v = mul_op(A2, B2, jc, indices_p2)
            for ia = 1:NC2
                C[ic, ia, indices...] += U2[jc, ic, indices_p1...]' * v[ia]
            end
        end
    end

end

function kernel_Dmatrix_mul_shiftedUdagOperatorSecondshiftedB_sum!(i, C,
    U1, A1, B1,
    U2, A2, B2,
    ::Val{3}, ::Val{4}, ::Val{3}, ::Val{4}, ::Val{3}, ::Val{4}, ::Val{nw}, ::Val{nw1}, ::Val{nw2}, dindexer,
    shift1,shift2) where {
    nw,nw1,nw2}
    indices = delinearize(dindexer, i, nw)
    indices_p1 = shiftindices(indices, shift1)
    indices_p2 = shiftindices(indices, shift2)

    v11_1,v12_1,v13_1,v14_1 = mul_op(A1, B1, 1, indices)
    v21_1,v22_1,v23_1,v24_1 = mul_op(A1, B1, 2, indices)
    v31_1,v32_1,v33_1,v34_1 = mul_op(A1, B1, 3, indices)

    v11_2,v12_2,v13_2,v14_2 = mul_op(A2, B2, 1, indices_p2)
    v21_2,v22_2,v23_2,v24_2 = mul_op(A2, B2, 2, indices_p2)
    v31_2,v32_2,v33_2,v34_2 = mul_op(A2, B2, 3, indices_p2)


    U11_1 = U1[1, 1, indices...]
    U12_1 = U1[1, 2, indices...]
    U13_1 = U1[1, 3, indices...]
    U21_1 = U1[2, 1, indices...]
    U22_1 = U1[2, 2, indices...]
    U23_1 = U1[2, 3, indices...]
    U31_1 = U1[3, 1, indices...]
    U32_1 = U1[3, 2, indices...]
    U33_1 = U1[3, 3, indices...]

    U11_2 = U2[1, 1, indices_p1...]'
    U12_2 = U2[2, 1, indices_p1...]'
    U13_2 = U2[3, 1, indices_p1...]'
    U21_2 = U2[1, 2, indices_p1...]'
    U22_2 = U2[2, 2, indices_p1...]'
    U23_2 = U2[3, 2, indices_p1...]'
    U31_2 = U2[1, 3, indices_p1...]'
    U32_2 = U2[2, 3, indices_p1...]'
    U33_2 = U2[3, 3, indices_p1...]'

    C[1, 1, indices...] += U11_1*v11_1 + U12_1*v21_1 + U13_1*v31_1 + U11_2*v11_2 + U12_2*v21_2 + U13_2*v31_2
    C[2, 1, indices...] += U21_1*v11_1 + U22_1*v21_1 + U23_1*v31_1 + U21_2*v11_2 + U22_2*v21_2 + U23_2*v31_2
    C[3, 1, indices...] += U31_1*v11_1 + U32_1*v21_1 + U33_1*v31_1 + U31_2*v11_2 + U32_2*v21_2 + U33_2*v31_2

    C[1, 2, indices...] += U11_1*v12_1 + U12_1*v22_1 + U13_1*v32_1 + U11_2*v12_2 + U12_2*v22_2 + U13_2*v32_2
    C[2, 2, indices...] += U21_1*v12_1 + U22_1*v22_1 + U23_1*v32_1 + U21_2*v12_2 + U22_2*v22_2 + U23_2*v32_2
    C[3, 2, indices...] += U31_1*v12_1 + U32_1*v22_1 + U33_1*v32_1 + U31_2*v12_2 + U32_2*v22_2 + U33_2*v32_2

    C[1, 3, indices...] += U11_1*v13_1 + U12_1*v23_1 + U13_1*v33_1 + U11_2*v13_2 + U12_2*v23_2 + U13_2*v33_2
    C[2, 3, indices...] += U21_1*v13_1 + U22_1*v23_1 + U23_1*v33_1 + U21_2*v13_2 + U22_2*v23_2 + U23_2*v33_2
    C[3, 3, indices...] += U31_1*v13_1 + U32_1*v23_1 + U33_1*v33_1 + U31_2*v13_2 + U32_2*v23_2 + U33_2*v33_2

    C[1, 4, indices...] += U11_1*v14_1 + U12_1*v24_1 + U13_1*v34_1 + U11_2*v14_2 + U12_2*v24_2 + U13_2*v34_2
    C[2, 4, indices...] += U21_1*v14_1 + U22_1*v24_1 + U23_1*v34_1 + U21_2*v14_2 + U22_2*v24_2 + U23_2*v34_2
    C[3, 4, indices...] += U31_1*v14_1 + U32_1*v24_1 + U33_1*v34_1 + U31_2*v14_2 + U32_2*v24_2 + U33_2*v34_2

end

#C = U1*shiftedx1*A1^T + shiftedU2dag*shiftgedx2*A2^T 
function mul_and_sum!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    U1::LatticeMatrix{D,T2,AT2,NC1,NC3,nw1,DI}, A1::OperatorSecond{NC2,NC4}, B1::Shifted_Lattice{BL1,shift1},
    U2::Adjoint_Lattice{Shifted_Lattice{L2,shiftU}}, A2::OperatorSecond{NC2,NC6}, B2::Shifted_Lattice{BL2,shift2}) where {
    D,T1,T2,T3,T4,T5,AT1,AT2,AT3,AT4,AT5,NC1,NC2,NC3,NC4,NC5,NC6,nw,nw1,nw2,DI,L2<:LatticeMatrix{D,T4,AT4,NC1,NC5,nw2,DI},
    BL1<:LatticeMatrix{D,T3,AT3,NC3,NC4,nw,DI},
    BL2<:LatticeMatrix{D,T5,AT5,NC5,NC6,nw,DI},shift1,shiftU,shift2}

    

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftedUdagOperatorSecondshiftedshiftedB_sum!, C.A,
        U1.A, A1, B1.data.A,
        U2.data.data.A, A2, B2.data.A,
        Val(NC1), Val(NC2), Val(NC3), Val(NC4), Val(NC5), Val(NC6), Val(nw), Val(nw1), Val(nw2), C.indexer,
        shift1,shiftU,shift2
    )
    #set_halo!(C)
end

function kernel_Dmatrix_mul_shiftedUdagOperatorSecondshiftedshiftedB_sum!(i, C,
    U1, A1, B1,
    U2, A2, B2,
    ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{NC4}, ::Val{NC5}, ::Val{NC6}, ::Val{nw}, ::Val{nw1}, ::Val{nw2}, dindexer,
    shift1,shiftU,shift2) where {
    NC1,NC2,NC3,NC4,NC5,NC6,nw,nw1,nw2}
    indices = delinearize(dindexer, i, nw)
    indices_p1 = shiftindices(indices, shift1)
    indices_pU = shiftindices(indices, shiftU)
    indices_p2 = shiftindices(indices, shift2)

        #x[ic,a] = sum_{jc,b} U[ic,jc]*A[a,b]*psi[jc,b]
    #x[ic,a] = sum_{jc} U[ic,jc]*(sum_b A[a,b]*psi[jc,b])

    #@inbounds for ic = 1:NC1
    #    for ia = 1:NC2
    #        C[ic, ia, indices...] = zero(eltype(C))
    #    end
    #end

    @inbounds for ic = 1:NC1
        for jc = 1:NC3
            v = mul_op(A1, B1, jc, indices_p1)
            for ia = 1:NC2
                C[ic, ia, indices...] += U1[ic, jc, indices...] * v[ia]
            end
        end
    end

    @inbounds for ic = 1:NC1
        for jc = 1:NC5
            v = mul_op(A2, B2, jc, indices_p2)
            for ia = 1:NC2
                C[ic, ia, indices...] += U2[jc, ic, indices_pU...]' * v[ia]
            end
        end
    end

end


function kernel_Dmatrix_mul_shiftedUdagOperatorSecondshiftedshiftedB_sum!(i, C,
    U1, A1, B1,
    U2, A2, B2,
    ::Val{3}, ::Val{4}, ::Val{3}, ::Val{4}, ::Val{3}, ::Val{4}, ::Val{nw}, ::Val{nw1}, ::Val{nw2}, dindexer,
    shift1,shiftU,shift2) where {
    nw,nw1,nw2}
    indices = delinearize(dindexer, i, nw)
    indices_p1 = shiftindices(indices, shift1)
    indices_pU = shiftindices(indices, shiftU)
    indices_p2 = shiftindices(indices, shift2)

    v11_1,v12_1,v13_1,v14_1 = mul_op(A1, B1, 1, indices_p1)
    v21_1,v22_1,v23_1,v24_1 = mul_op(A1, B1, 2, indices_p1)
    v31_1,v32_1,v33_1,v34_1 = mul_op(A1, B1, 3, indices_p1)

    v11_2,v12_2,v13_2,v14_2 = mul_op(A2, B2, 1, indices_p2)
    v21_2,v22_2,v23_2,v24_2 = mul_op(A2, B2, 2, indices_p2)
    v31_2,v32_2,v33_2,v34_2 = mul_op(A2, B2, 3, indices_p2)


    U11_1 = U1[1, 1, indices...]
    U12_1 = U1[1, 2, indices...]
    U13_1 = U1[1, 3, indices...]
    U21_1 = U1[2, 1, indices...]
    U22_1 = U1[2, 2, indices...]
    U23_1 = U1[2, 3, indices...]
    U31_1 = U1[3, 1, indices...]
    U32_1 = U1[3, 2, indices...]
    U33_1 = U1[3, 3, indices...]

    U11_2 = U2[1, 1, indices_pU...]'
    U12_2 = U2[2, 1, indices_pU...]'
    U13_2 = U2[3, 1, indices_pU...]'
    U21_2 = U2[1, 2, indices_pU...]'
    U22_2 = U2[2, 2, indices_pU...]'
    U23_2 = U2[3, 2, indices_pU...]'
    U31_2 = U2[1, 3, indices_pU...]'
    U32_2 = U2[2, 3, indices_pU...]'
    U33_2 = U2[3, 3, indices_pU...]'

    C[1, 1, indices...] += U11_1*v11_1 + U12_1*v21_1 + U13_1*v31_1 + U11_2*v11_2 + U12_2*v21_2 + U13_2*v31_2
    C[2, 1, indices...] += U21_1*v11_1 + U22_1*v21_1 + U23_1*v31_1 + U21_2*v11_2 + U22_2*v21_2 + U23_2*v31_2
    C[3, 1, indices...] += U31_1*v11_1 + U32_1*v21_1 + U33_1*v31_1 + U31_2*v11_2 + U32_2*v21_2 + U33_2*v31_2

    C[1, 2, indices...] += U11_1*v12_1 + U12_1*v22_1 + U13_1*v32_1 + U11_2*v12_2 + U12_2*v22_2 + U13_2*v32_2
    C[2, 2, indices...] += U21_1*v12_1 + U22_1*v22_1 + U23_1*v32_1 + U21_2*v12_2 + U22_2*v22_2 + U23_2*v32_2
    C[3, 2, indices...] += U31_1*v12_1 + U32_1*v22_1 + U33_1*v32_1 + U31_2*v12_2 + U32_2*v22_2 + U33_2*v32_2

    C[1, 3, indices...] += U11_1*v13_1 + U12_1*v23_1 + U13_1*v33_1 + U11_2*v13_2 + U12_2*v23_2 + U13_2*v33_2
    C[2, 3, indices...] += U21_1*v13_1 + U22_1*v23_1 + U23_1*v33_1 + U21_2*v13_2 + U22_2*v23_2 + U23_2*v33_2
    C[3, 3, indices...] += U31_1*v13_1 + U32_1*v23_1 + U33_1*v33_1 + U31_2*v13_2 + U32_2*v23_2 + U33_2*v33_2

    C[1, 4, indices...] += U11_1*v14_1 + U12_1*v24_1 + U13_1*v34_1 + U11_2*v14_2 + U12_2*v24_2 + U13_2*v34_2
    C[2, 4, indices...] += U21_1*v14_1 + U22_1*v24_1 + U23_1*v34_1 + U21_2*v14_2 + U22_2*v24_2 + U23_2*v34_2
    C[3, 4, indices...] += U31_1*v14_1 + U32_1*v24_1 + U33_1*v34_1 + U31_2*v14_2 + U32_2*v24_2 + U33_2*v34_2

end


#C = U1*shiftedx1*A1^T + shiftedU2dag*shiftgedx2*A2^T 
function mul_and_sum!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    U2::Adjoint_Lattice{Shifted_Lattice{L2,shiftU}}, A2::OperatorSecond{NC2,NC6}, B2::Shifted_Lattice{BL2,shift2},
    U1::LatticeMatrix{D,T2,AT2,NC1,NC3,nw1,DI}, A1::OperatorSecond{NC2,NC4}, B1::Shifted_Lattice{BL1,shift1}
    ) where {
    D,T1,T2,T3,T4,T5,AT1,AT2,AT3,AT4,AT5,NC1,NC2,NC3,NC4,NC5,NC6,nw,nw1,nw2,DI,L2<:LatticeMatrix{D,T4,AT4,NC1,NC5,nw2,DI},
    BL1<:LatticeMatrix{D,T3,AT3,NC3,NC4,nw,DI},
    BL2<:LatticeMatrix{D,T5,AT5,NC5,NC6,nw,DI},shift1,shiftU,shift2}

    

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftedUdagOperatorSecondshiftedshiftedB_sum!, C.A,
        U1.A, A1, B1.data.A,
        U2.data.data.A, A2, B2.data.A,
        Val(NC1), Val(NC2), Val(NC3), Val(NC4), Val(NC5), Val(NC6), Val(nw), Val(nw1), Val(nw2), C.indexer,
        shift1,shiftU,shift2
    )
    #set_halo!(C)
end

function mul_and_sum!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    Ts::NTuple{N,ThreeOperators}) where {N,
    D,T1,AT1,NC1,NC2,nw,DI}
    mul_and_sum!(C,Ts...)
end