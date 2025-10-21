module LatticeMatrices
using MPI
using LinearAlgebra
using JACC

include("utilities/randomgenerator.jl")

abstract type AbstractLattice end

abstract type Lattice{D,T,AT,NC1,NC2,NW} <: AbstractLattice end


#include("HaloComm.jl")
#include("1D/1Dlatticevector.jl")
#include("1D/1Dlatticematrix.jl")

struct Shifted_Lattice{D,shift} <: AbstractLattice
    data::D
end



export Shifted_Lattice

struct Adjoint_Lattice{D} <: AbstractLattice
    data::D
end



function Base.adjoint(data::Lattice{D,T,AT}) where {D,T,AT}
    return Adjoint_Lattice{typeof(data)}(data)
end

function Base.adjoint(data::Shifted_Lattice{D,shift}) where {D,shift}
    return Adjoint_Lattice{typeof(data)}(data)
end

include("Latticeindices.jl")
include("LatticeMatrices_core.jl")
include("LinearAlgebras/linearalgebra.jl")
include("TA/TA.jl")

function Shifted_Lattice(data::TD, shift) where {D,T,AT,TD<:Lattice{D,T,AT}}
    return Shifted_Lattice{typeof(data),Tuple(shift)}(data)
end


function Shifted_Lattice(data::TL, shift) where {D,T,AT,NC1,NC2,nw,DI,TL<:LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}}
    #set_halo!(data)
    #nw = data.nw
    isinside = true
    for i in 1:D
        if shift[i] < -nw || shift[i] > nw
            isinside = false
            break
        end
    end
    #println("Shifted_Lattice: shift = ", shift, " isinside = ", isinside)
    if isinside
        sl = Shifted_Lattice{typeof(data),Tuple(shift)}(data)
    else
        sl0 = similar(data)
        sl1 = similar(data)
        shift0 = zeros(Int64, D)
        substitute!(sl0, data)
        for i in 1:D
            if shift[i] > nw
                smallshift = shift[i] ÷ nw
                shift0 .= 0
                shift0[i] = nw
                for k = 1:smallshift
                    sls = Shifted_Lattice{typeof(data),Tuple(shift0)}(sl0)
                    substitute!(sl1, sls)
                    substitute!(sl0, sl1)
                end
                shift0 .= 0
                shift0[i] = shift[i] % nw
                sls = Shifted_Lattice{typeof(data),Tuple(shift0)}(sl0)
                substitute!(sl1, sls)
                substitute!(sl0, sl1)
            elseif shift[i] < -nw
                smallshift = abs(shift[i]) ÷ nw
                shift0 .= 0
                shift0[i] = -nw
                #println(shift0)
                for k = 1:smallshift
                    println(shift0)
                    sls = Shifted_Lattice{typeof(data),Tuple(shift0)}(sl0)
                    substitute!(sl1, sls)
                    substitute!(sl0, sl1)
                end
                shift0 .= 0
                shift0[i] = -(abs(shift[i]) % nw)
                #println(shift0)
                sls = Shifted_Lattice{typeof(data),Tuple(shift0)}(sl0)
                substitute!(sl1, sls)
                substitute!(sl0, sl1)
            else
                shift0 .= 0
                shift0[i] = shift[i]
                sls = Shifted_Lattice{typeof(data),Tuple(shift0)}(sl0)
                substitute!(sl1, sls)
                substitute!(sl0, sl1)
            end
        end
        zeroshift = ntuple(_ -> 0, D)
        sl = Shifted_Lattice{typeof(data),zeroshift}(sl0)
    end
    return sl
end

function get_matrix(a::T) where {T<:LatticeMatrix}
    return a.A
end

function get_matrix(a::T) where {T<:Shifted_Lattice}
    return a.data.A
end


function get_matrix(a::T) where {T<:Adjoint_Lattice}
    return a.data.A
end

function get_matrix(a::Adjoint_Lattice{T}) where {T<:Shifted_Lattice}
    return a.data.data.A
end

function JACC.parallel_for(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}, variables...) where {D,T1,AT1,NC1,NG,nw,DI}
    JACC.parallel_for(
        prod(C.PN), kernelfunction, C.A, variables..., Val(NC1), Val(NG), Val(nw), C.indexer
    )
end

function JACC.parallel_reduce(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}, variables...) where {D,T1,AT1,NC1,NG,nw,DI}
    s = JACC.parallel_reduce(
        prod(C.PN), +, kernelfunction, C.A, variables..., Val(NC1), Val(NG), Val(nw), C.indexer
        ; init=zero(eltype(C.A))
    )
    s = MPI.Allreduce(s, MPI.SUM, C.comm)
end

function JACC.parallel_for(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}) where {D,T1,AT1,NC1,NG,nw,DI}
    JACC.parallel_for(
        prod(C.PN), kernelfunction, C.A, Val(NC1), Val(NG), Val(nw), C.indexer
    )
end

function JACC.parallel_reduce(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}) where {D,T1,AT1,NC1,NG,nw,DI}
    s = JACC.parallel_reduce(
        prod(C.PN), +, kernelfunction, C.A, Val(NC1), Val(NG), Val(nw), C.indexer
        ; init=zero(eltype(C.A))
    )
    s = MPI.Allreduce(s, MPI.SUM, C.comm)
end

function JACC.parallel_for(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}, A::Lattice{D,T2,AT2,NC2,NG2,nw2}, variables...) where {D,T1,AT1,NC1,NG,nw,DI,
    T2,AT2,NC2,NG2,nw2}
    a = get_matrix(A)
    JACC.parallel_for(
        prod(C.PN), kernelfunction, C.A, a, variables..., Val(NC1), Val(NG), Val(nw), Val(NC2), Val(NG2), Val(nw2), C.indexer
    )

end

function JACC.parallel_reduce(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}, A::Lattice{D,T2,AT2,NC2,NG2,nw2}, variables...) where {D,T1,AT1,NC1,NG,nw,DI,
    T2,AT2,NC2,NG2,nw2}
    a = get_matrix(A)
    s = JACC.parallel_reduce(
        prod(C.PN), +, kernelfunction, C.A, a, variables..., Val(NC1), Val(NG), Val(nw), Val(NC2), Val(NG2), Val(nw2), C.indexer
        ; init=zero(eltype(C.A))
    )
    s = MPI.Allreduce(s, MPI.SUM, C.comm)
end

function JACC.parallel_for(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}, A::Lattice{D,T2,AT2,NC2,NG2,nw2}) where {D,T1,AT1,NC1,NG,nw,DI,
    T2,AT2,NC2,NG2,nw2}
    a = get_matrix(A)
    JACC.parallel_for(
        prod(C.PN), kernelfunction, C.A, a, Val(NC1), Val(NG), Val(nw), Val(NC2), Val(NG2), Val(nw2), C.indexer
    )

end

function JACC.parallel_reduce(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}, A::Lattice{D,T2,AT2,NC2,NG2,nw2}) where {D,T1,AT1,NC1,NG,nw,DI,
    T2,AT2,NC2,NG2,nw2}
    a = get_matrix(A)
    s = JACC.parallel_reduce(
        prod(C.PN), +, kernelfunction, C.A, a, Val(NC1), Val(NG), Val(nw), Val(NC2), Val(NG2), Val(nw2), C.indexer
        ; init=zero(eltype(C.A))
    )
    s = MPI.Allreduce(s, MPI.SUM, C.comm)
end

function JACC.parallel_for(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}, A::Lattice{D,T2,AT2,NC2,NG2,nw2},
    B::Lattice{D,T3,AT3,NC3,NG3,nw3},
    variables...) where {D,T1,AT1,NC1,NG,nw,DI,
    T2,AT2,NC2,NG2,nw2,
    T3,AT3,NC3,NG3,nw3}
    a = get_matrix(A)
    b = get_matrix(B)
    JACC.parallel_for(
        prod(C.PN), kernelfunction, C.A, a, b, variables..., Val(NC1), Val(NG), Val(nw), Val(NC2), Val(NG2), Val(nw2), Val(NC3), Val(NG3), Val(nw3), C.indexer
    )
end

function JACC.parallel_reduce(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}, A::Lattice{D,T2,AT2,NC2,NG2,nw2},
    B::Lattice{D,T3,AT3,NC3,NG3,nw3},
    variables...) where {D,T1,AT1,NC1,NG,nw,DI,
    T2,AT2,NC2,NG2,nw2,
    T3,AT3,NC3,NG3,nw3}
    a = get_matrix(A)
    b = get_matrix(B)
    s = JACC.parallel_reduce(
        prod(C.PN), +, kernelfunction, C.A, a, b, variables..., Val(NC1), Val(NG), Val(nw), Val(NC2), Val(NG2), Val(nw2), Val(NC3), Val(NG3), Val(nw3), C.indexer
        ; init=zero(eltype(C.A))
    )
    s = MPI.Allreduce(s, MPI.SUM, C.comm)
end

function JACC.parallel_for(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}, A::Lattice{D,T2,AT2,NC2,NG2,nw2},
    B::Lattice{D,T3,AT3,NC3,NG3,nw3},
) where {D,T1,AT1,NC1,NG,nw,DI,
    T2,AT2,NC2,NG2,nw2,
    T3,AT3,NC3,NG3,nw3}
    a = get_matrix(A)
    b = get_matrix(B)
    JACC.parallel_for(
        prod(C.PN), kernelfunction, C.A, a, b, Val(NC1), Val(NG), Val(nw), Val(NC2), Val(NG2), Val(nw2), Val(NC3), Val(NG3), Val(nw3), C.indexer
    )
end

function JACC.parallel_reduce(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}, A::Lattice{D,T2,AT2,NC2,NG2,nw2},
    B::Lattice{D,T3,AT3,NC3,NG3,nw3},
) where {D,T1,AT1,NC1,NG,nw,DI,
    T2,AT2,NC2,NG2,nw2,
    T3,AT3,NC3,NG3,nw3}
    a = get_matrix(A)
    b = get_matrix(B)
    s = JACC.parallel_reduce(
        prod(C.PN), +, kernelfunction, C.A, a, b, Val(NC1), Val(NG), Val(nw), Val(NC2), Val(NG2), Val(nw2), Val(NC3), Val(NG3), Val(nw3), C.indexer
        ; init=zero(eltype(C.A))
    )
    s = MPI.Allreduce(s, MPI.SUM, C.comm)
end


function get_PEs(ls::LatticeMatrix{D,T,AT,NC1,NC2}) where {D,T,AT,NC1,NC2}
    return ls.dims
end
export get_PEs

function get_shift(::Shifted_Lattice{<:LatticeMatrix{D,T,AT,NC1,NC2,nw},shift}) where {D,T,AT,NC1,NC2,nw,shift}
    return shift
end

include("Operators/Operators.jl")
include("Operators/DiracOperators.jl")


end
