module LatticeMatrices
using MPI
using LinearAlgebra
using JACC
using Enzyme

include("utilities/randomgenerator.jl")

abstract type AbstractLattice end

abstract type Lattice{D,T,AT,NC1,NC2,NW} <: AbstractLattice end


#include("HaloComm.jl")
#include("1D/1Dlatticevector.jl")
#include("1D/1Dlatticematrix.jl")

struct Shifted_Lattice{D,Dim} <: AbstractLattice
    data::D
    shift::NTuple{Dim,Int64}

    function Shifted_Lattice(data, shift, ::Val{Dim}) where {Dim}
        return new{typeof(data),Dim}(data, shift)
    end

end



export Shifted_Lattice

struct Adjoint_Lattice{D} <: AbstractLattice
    data::D
end



function Base.adjoint(data::TD) where {D,T,AT,TD<:Lattice{D,T,AT}}
    return Adjoint_Lattice{typeof(data)}(data)
end

function Base.adjoint(data::T) where {D,Dim,T<:Shifted_Lattice{D,Dim}}
    return Adjoint_Lattice{typeof(data)}(data)
end

function Base.adjoint(data::TD) where {TD<:Adjoint_Lattice}
    return data.data
end




include("Latticeindices.jl")
include("LatticeMatrices_core.jl")
include("LinearAlgebras/linearalgebra.jl")
include("TA/TA.jl")
include("AD/AD.jl")
include("AD/ND.jl")

function get_shift(x::Shifted_Lattice{Tx,D}) where {D,T,AT,NC1,NC2,nw,Tx<:LatticeMatrix{D,T,AT,NC1,NC2,nw}}
    return x.shift
end

function get_shift(x::Adjoint_Lattice{Shifted_Lattice{Tx,D}}) where {D,T,AT,NC1,NC2,nw,Tx<:LatticeMatrix{D,T,AT,NC1,NC2,nw}}
    return x.data.shift
end




#function Shifted_Lattice(data::TD, shift::TS) where {D,T,AT,TD<:Lattice{D,T,AT},TS}
#    return Shifted_Lattice{typeof(data),D}(data, shift)
#end



@inline function _as_shift_tuple(shift_in, ::Val{D}) where {D}
    if shift_in isa NTuple{D,Int}
        return shift_in
    elseif shift_in isa AbstractVector{<:Integer}
        @assert length(shift_in) == D "shift length must be $D"
        return ntuple(i -> Int(shift_in[i]), D)
    elseif shift_in isa Tuple
        @assert length(shift_in) == D "shift length must be $D"
        return ntuple(i -> Int(shift_in[i]), D)
    else
        error("Unsupported shift type: $(typeof(shift_in)). Provide NTuple{$D,Int} or Vector{Int}.")
    end
end

@inline make_step(i, r, ::Val{D}) where {D} =
    ntuple(j -> ifelse(j == i, r, 0), D)

function Shifted_Lattice(data::TL, shift_in::TS) where {
    D,T,AT,NC1,NC2,nw,DI,
    TL<:LatticeMatrix{D,T,AT,NC1,NC2,nw,DI},TS
}
    shift = _as_shift_tuple(shift_in, Val(D))

    @inbounds begin
        isinside = true
        for i in 1:D
            s = shift[i]
            if (s < -nw) | (s > nw)
                isinside = false
                break
            end
        end
        if isinside
            return Shifted_Lattice(data, shift, Val(D))
        end
    end

    sl0 = similar(data)
    sl1 = similar(data)
    substitute!(sl0, data)

    zeroT = ntuple(_ -> 0, D)

    @inbounds for i in 1:D
        s = shift[i]
        if s == 0
            continue
        end

        if s > nw
            smallshift = s ÷ nw
            step = ntuple(j -> (j == i ? nw : 0), D)
            for _ in 1:smallshift
                sls = Shifted_Lattice(sl0, step, Val(D))
                substitute!(sl1, sls)
                substitute!(sl0, sl1)
            end
            rems = s % nw
            step2 = make_step(i, rems, Val(D))
            #step2 = ntuple(j -> (j == i ? rems : 0), D)
            sls = Shifted_Lattice(sl0, step2, Val(D))
            substitute!(sl1, sls)
            substitute!(sl0, sl1)

        elseif s < -nw
            as = -s
            smallshift = as ÷ nw
            step = ntuple(j -> (j == i ? -nw : 0), D)
            for _ in 1:smallshift
                sls = Shifted_Lattice(sl0, step, Val(D))
                substitute!(sl1, sls)
                substitute!(sl0, sl1)
            end
            rems = -(as % nw)
            step2 = make_step(i, rems, Val(D))
            #step2 = ntuple(j -> (j == i ? rems : 0), D)
            sls = Shifted_Lattice(sl0, step2, Val(D))
            substitute!(sl1, sls)
            substitute!(sl0, sl1)

        else
            step = ntuple(j -> (j == i ? s : 0), D)
            sls = Shifted_Lattice(sl0, step, Val(D))
            substitute!(sl1, sls)
            substitute!(sl0, sl1)
        end
    end

    zeroshift = ntuple(_ -> 0, D)
    return Shifted_Lattice(sl0, zeroshift, Val(D))
end

#=
function Shifted_Lattice(data::TL, shift) where {D,T,AT,NC1,NC2,nw,DI,TL<:LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}}
    #set_halo!(data)
    #error("dd")
    #nw = data.nw
    #println("shift")
    isinside = true
    for i in 1:D
        if shift[i] < -nw || shift[i] > nw
            isinside = false
            break
        end
    end
    #println("Shifted_Lattice: shift = ", shift, " isinside = ", isinside)
    if isinside
        sl = Shifted_Lattice{typeof(data),typeof(shift)}(data, Tuple(shift))
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
                    sls = Shifted_Lattice{typeof(data),typeof(shift0)}(sl0, Tuple(shift0))
                    substitute!(sl1, sls)
                    substitute!(sl0, sl1)
                end
                shift0 .= 0
                shift0[i] = shift[i] % nw
                sls = Shifted_Lattice{typeof(data),typeof(shift0)}(sl0, Tuple(shift0))
                substitute!(sl1, sls)
                substitute!(sl0, sl1)
            elseif shift[i] < -nw
                smallshift = abs(shift[i]) ÷ nw
                shift0 .= 0
                shift0[i] = -nw
                #println(shift0)
                for k = 1:smallshift
                    println(shift0)
                    sls = Shifted_Lattice{typeof(data),typeof(shift0)}(sl0, Tuple(shift0))
                    substitute!(sl1, sls)
                    substitute!(sl0, sl1)
                end
                shift0 .= 0
                shift0[i] = -(abs(shift[i]) % nw)
                #println(shift0)
                sls = Shifted_Lattice{typeof(data),typeof(shift0)}(sl0, Tuple(shift0))
                substitute!(sl1, sls)
                substitute!(sl0, sl1)
            else
                shift0 .= 0
                shift0[i] = shift[i]
                sls = Shifted_Lattice{typeof(data),typeof(shift0)}(sl0, Tuple(shift0))
                substitute!(sl1, sls)
                substitute!(sl0, sl1)
            end
        end
        zeroshift = ntuple(_ -> 0, D)
        sl = Shifted_Lattice{typeof(data),typeof(zeroshift)}(sl0, zeroshift)
    end
    return sl
end
=#

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



include("Operators/Operators.jl")
include("Operators/DiracOperators.jl")
include("Operators/DiracOperators_5D.jl")


end
