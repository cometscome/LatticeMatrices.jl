using LinearAlgebra


struct Staggered_Lattice{D,μ} <: AbstractLattice
    data::D

    function Staggered_Lattice(data::D, μ) where {D}
        return new{D,μ}(data)
    end
end
export Staggered_Lattice

#coords: 1-index
@inline function staggered_eta(coords::NTuple{D,Int}, μ) where {D}
    μ == 1 && return 1
    s = 0
    @inbounds for ν in 1:μ-1
        s += coords[ν] - 1
    end
    return iseven(s) ? 1 : -1
end

#coords: 1-index
@inline function staggered_eta_halo(coords::NTuple{D,Int}, μ, nw) where {D}
    μ == 1 && return 1
    s = 0
    @inbounds for ν in 1:μ-1
        s += coords[ν] - 1 - nw
    end
    return iseven(s) ? 1 : -1
end

#coords: 1-index
@inline function staggered_eta_halo0(coords::NTuple{D,Int}, μ, nw) where {D}
    μ == 0 && return 1
    return staggered_eta_halo(coords, μ, nw)
end

function Base.adjoint(data::TS) where {D,μ,TS<:Staggered_Lattice{D,μ}}
    return Adjoint_Lattice{typeof(data)}(data)
end


#C = stA B
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Staggered_Lattice{TA,μA}, B::TB) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,μA,
    TA<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},TB<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}

    Adata = A.data
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_etaAB!, C.A, Adata.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), Val(μA), Val(0), C.indexer
    )
    #set_halo!(C)
end


#C = A stB
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::TA, B::Staggered_Lattice{TB,μB}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,μB,
    TA<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},TB<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}

    Bdata = B.data
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_etaAB!, C.A, A.A, Bdata.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), Val(0), Val(μB), C.indexer
    )
    #set_halo!(C)
end

#C = stA stB
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Staggered_Lattice{TA,μA}, B::Staggered_Lattice{TB,μB}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,μA,μB,
    TA<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},TB<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}

    Adata = A.data
    Bdata = B.data
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_etaAB!, C.A, Adata.A, Bdata.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), Val(μA), Val(μB), C.indexer
    )
    #set_halo!(C)
end

#C = α*stA*B + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Staggered_Lattice{TA,μA}, B::TB, α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,μA,
    TA<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},TB<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}

    Adata = A.data
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_etaAB!, C.A, Adata.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), Val(μA), Val(0), C.indexer, α, β
    )
    #set_halo!(C)
end

#C = α*A*stB + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::TA, B::Staggered_Lattice{TB,μB}, α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,μB,
    TA<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},TB<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}

    Bdata = B.data
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_etaAB!, C.A, A.A, Bdata.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), Val(0), Val(μB), C.indexer, α, β
    )
    #set_halo!(C)
end

#C = α*stA*stB + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Staggered_Lattice{TA,μA}, B::Staggered_Lattice{TB,μB}, α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,μA,μB,
    TA<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},TB<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}

    Adata = A.data
    Bdata = B.data
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_etaAB!, C.A, Adata.A, Bdata.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), Val(μA), Val(μB), C.indexer, α, β
    )
    #set_halo!(C)
end

#C = A'*stB
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{L}, B::Staggered_Lattice{TB,μB}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,μB,
    L<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},TB<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}

    Bdata = B.data
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_etaAdagB!, C.A, A.data.A, Bdata.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), Val(0), Val(μB), C.indexer
    )
    #set_halo!(C)
end

#C = stA'*B
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Staggered_Lattice{TA,μA}}, B::TB) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,μA,
    TA<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},TB<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}

    Adata = A.data.data
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_etaAdagB!, C.A, Adata.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), Val(μA), Val(0), C.indexer
    )
    #set_halo!(C)
end

#C = stA'*stB
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Staggered_Lattice{TA,μA}}, B::Staggered_Lattice{TB,μB}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,μA,μB,
    TA<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},TB<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}

    Adata = A.data.data
    Bdata = B.data
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_etaAdagB!, C.A, Adata.A, Bdata.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), Val(μA), Val(μB), C.indexer
    )
    #set_halo!(C)
end

#C = α*A'*stB + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{L}, B::Staggered_Lattice{TB,μB}, α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,μB,
    L<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},TB<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}

    Bdata = B.data
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_etaAdagB!, C.A, A.data.A, Bdata.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), Val(0), Val(μB), C.indexer, α, β
    )
    #set_halo!(C)
end

#C = α*stA'*B + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Staggered_Lattice{TA,μA}}, B::TB, α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,μA,
    TA<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},TB<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}

    Adata = A.data.data
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_etaAdagB!, C.A, Adata.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), Val(μA), Val(0), C.indexer, α, β
    )
    #set_halo!(C)
end

#C = α*stA'*stB + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Staggered_Lattice{TA,μA}}, B::Staggered_Lattice{TB,μB}, α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,μA,μB,
    TA<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},TB<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}

    Adata = A.data.data
    Bdata = B.data
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_etaAdagB!, C.A, Adata.A, Bdata.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), Val(μA), Val(μB), C.indexer, α, β
    )
    #set_halo!(C)
end

#C = stA*B'
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Staggered_Lattice{TA,μA}, B::Adjoint_Lattice{L}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,μA,
    TA<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    Adata = A.data
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_etaABdag!, C.A, Adata.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), Val(μA), Val(0), C.indexer
    )
    #set_halo!(C)
end

#C = A*stB'
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::TA, B::Adjoint_Lattice{Staggered_Lattice{TB,μB}}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,μB,
    TA<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},TB<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    Bdata = B.data.data
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_etaABdag!, C.A, A.A, Bdata.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), Val(0), Val(μB), C.indexer
    )
    #set_halo!(C)
end

#C = stA*stB'
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Staggered_Lattice{TA,μA}, B::Adjoint_Lattice{Staggered_Lattice{TB,μB}}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,μA,μB,
    TA<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},TB<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    Adata = A.data
    Bdata = B.data.data
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_etaABdag!, C.A, Adata.A, Bdata.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), Val(μA), Val(μB), C.indexer
    )
    #set_halo!(C)
end

#C = α*stA*B' + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Staggered_Lattice{TA,μA}, B::Adjoint_Lattice{L}, α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,μA,
    TA<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    Adata = A.data
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_etaABdag!, C.A, Adata.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), Val(μA), Val(0), C.indexer, α, β
    )
    #set_halo!(C)
end

#C = α*A*stB' + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::TA, B::Adjoint_Lattice{Staggered_Lattice{TB,μB}}, α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,μB,
    TA<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},TB<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    Bdata = B.data.data
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_etaABdag!, C.A, A.A, Bdata.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), Val(0), Val(μB), C.indexer, α, β
    )
    #set_halo!(C)
end

#C = α*stA*stB' + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Staggered_Lattice{TA,μA}, B::Adjoint_Lattice{Staggered_Lattice{TB,μB}}, α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,μA,μB,
    TA<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},TB<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    Adata = A.data
    Bdata = B.data.data
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_etaABdag!, C.A, Adata.A, Bdata.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), Val(μA), Val(μB), C.indexer, α, β
    )
    #set_halo!(C)
end

#C = stA'*B'
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Staggered_Lattice{TA,μA}}, B::Adjoint_Lattice{L}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,μA,
    TA<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    Adata = A.data.data
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_etaAdagBdag!, C.A, Adata.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), Val(μA), Val(0), C.indexer
    )
    #set_halo!(C)
end

#C = A'*stB'
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{L}, B::Adjoint_Lattice{Staggered_Lattice{TB,μB}}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,μB,
    L<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},TB<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    Bdata = B.data.data
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_etaAdagBdag!, C.A, A.data.A, Bdata.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), Val(0), Val(μB), C.indexer
    )
    #set_halo!(C)
end

#C = stA'*stB'
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Staggered_Lattice{TA,μA}}, B::Adjoint_Lattice{Staggered_Lattice{TB,μB}}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,μA,μB,
    TA<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},TB<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    Adata = A.data.data
    Bdata = B.data.data
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_etaAdagBdag!, C.A, Adata.A, Bdata.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), Val(μA), Val(μB), C.indexer
    )
    #set_halo!(C)
end

#C = α*stA'*B' + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Staggered_Lattice{TA,μA}}, B::Adjoint_Lattice{L}, α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,μA,
    TA<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    Adata = A.data.data
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_etaAdagBdag!, C.A, Adata.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), Val(μA), Val(0), C.indexer, α, β
    )
    #set_halo!(C)
end

#C = α*A'*stB' + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{L}, B::Adjoint_Lattice{Staggered_Lattice{TB,μB}}, α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,μB,
    L<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},TB<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    Bdata = B.data.data
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_etaAdagBdag!, C.A, A.data.A, Bdata.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), Val(0), Val(μB), C.indexer, α, β
    )
    #set_halo!(C)
end

#C = α*stA'*stB' + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Staggered_Lattice{TA,μA}}, B::Adjoint_Lattice{Staggered_Lattice{TB,μB}}, α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,μA,μB,
    TA<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},TB<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    Adata = A.data.data
    Bdata = B.data.data
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_etaAdagBdag!, C.A, Adata.A, Bdata.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), Val(μA), Val(μB), C.indexer, α, β
    )
    #set_halo!(C)
end

@inline function kernel_Dmatrix_mul_etaAB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, ::Val{μA}, ::Val{μB}, dindexer) where {NC1,NC2,NC3,nw,μA,μB}
    indices = delinearize(dindexer, i, nw)

    η = staggered_eta_halo0(indices, μA, nw) * staggered_eta_halo0(indices, μB, nw)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            s = zero(eltype(C))
            for kc = 1:NC3
                s += A[ic, kc, indices...] * B[kc, jc, indices...]
            end
            C[ic, jc, indices...] = η * s
        end
    end
end

@inline function kernel_Dmatrix_mul_etaAB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, ::Val{μA}, ::Val{μB}, dindexer, α, β) where {NC1,NC2,NC3,nw,μA,μB}
    indices = delinearize(dindexer, i, nw)

    η = staggered_eta_halo0(indices, μA, nw) * staggered_eta_halo0(indices, μB, nw)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            s = zero(eltype(C))
            for kc = 1:NC3
                s += A[ic, kc, indices...] * B[kc, jc, indices...]
            end
            C[ic, jc, indices...] = β * C[ic, jc, indices...] + α * η * s
        end
    end
end

@inline function kernel_Dmatrix_mul_etaAdagB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, ::Val{μA}, ::Val{μB}, dindexer) where {NC1,NC2,NC3,nw,μA,μB}
    indices = delinearize(dindexer, i, nw)

    η = staggered_eta_halo0(indices, μA, nw) * staggered_eta_halo0(indices, μB, nw)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            s = zero(eltype(C))
            for kc = 1:NC3
                s += A[kc, ic, indices...]' * B[kc, jc, indices...]
            end
            C[ic, jc, indices...] = η * s
        end
    end
end

@inline function kernel_Dmatrix_mul_etaAdagB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, ::Val{μA}, ::Val{μB}, dindexer, α, β) where {NC1,NC2,NC3,nw,μA,μB}
    indices = delinearize(dindexer, i, nw)

    η = staggered_eta_halo0(indices, μA, nw) * staggered_eta_halo0(indices, μB, nw)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            s = zero(eltype(C))
            for kc = 1:NC3
                s += A[kc, ic, indices...]' * B[kc, jc, indices...]
            end
            C[ic, jc, indices...] = β * C[ic, jc, indices...] + α * η * s
        end
    end
end

@inline function kernel_Dmatrix_mul_etaABdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, ::Val{μA}, ::Val{μB}, dindexer) where {NC1,NC2,NC3,nw,μA,μB}
    indices = delinearize(dindexer, i, nw)

    η = staggered_eta_halo0(indices, μA, nw) * staggered_eta_halo0(indices, μB, nw)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            s = zero(eltype(C))
            for kc = 1:NC3
                s += A[ic, kc, indices...] * B[jc, kc, indices...]'
            end
            C[ic, jc, indices...] = η * s
        end
    end
end

@inline function kernel_Dmatrix_mul_etaABdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, ::Val{μA}, ::Val{μB}, dindexer, α, β) where {NC1,NC2,NC3,nw,μA,μB}
    indices = delinearize(dindexer, i, nw)

    η = staggered_eta_halo0(indices, μA, nw) * staggered_eta_halo0(indices, μB, nw)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            s = zero(eltype(C))
            for kc = 1:NC3
                s += A[ic, kc, indices...] * B[jc, kc, indices...]'
            end
            C[ic, jc, indices...] = β * C[ic, jc, indices...] + α * η * s
        end
    end
end

@inline function kernel_Dmatrix_mul_etaAdagBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, ::Val{μA}, ::Val{μB}, dindexer) where {NC1,NC2,NC3,nw,μA,μB}
    indices = delinearize(dindexer, i, nw)

    η = staggered_eta_halo0(indices, μA, nw) * staggered_eta_halo0(indices, μB, nw)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            s = zero(eltype(C))
            for kc = 1:NC3
                s += A[kc, ic, indices...]' * B[jc, kc, indices...]'
            end
            C[ic, jc, indices...] = η * s
        end
    end
end

@inline function kernel_Dmatrix_mul_etaAdagBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, ::Val{μA}, ::Val{μB}, dindexer, α, β) where {NC1,NC2,NC3,nw,μA,μB}
    indices = delinearize(dindexer, i, nw)

    η = staggered_eta_halo0(indices, μA, nw) * staggered_eta_halo0(indices, μB, nw)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            s = zero(eltype(C))
            for kc = 1:NC3
                s += A[kc, ic, indices...]' * B[jc, kc, indices...]'
            end
            C[ic, jc, indices...] = β * C[ic, jc, indices...] + α * η * s
        end
    end
end

