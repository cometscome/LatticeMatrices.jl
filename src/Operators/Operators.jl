abstract type OperatorOnKernel end

abstract type OperatorSecond{NG1,NG2} <: OperatorOnKernel end

abstract type OperatorFirst{NG1,NG2} <: OperatorOnKernel end


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
const oneminusγ1 = Oneγ(1, false)
export oneminusγ1
const oneminusγ2 = Oneγ(2, false)
export oneminusγ2
const oneminusγ3 = Oneγ(3, false)
export oneminusγ3
const oneminusγ4 = Oneγ(4, false)
export oneminusγ4

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