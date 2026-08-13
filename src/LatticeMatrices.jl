module LatticeMatrices
using MPI
using LinearAlgebra
using JACC
#using Enzyme

include("utilities/randomgenerator.jl")

abstract type AbstractLattice end

abstract type Lattice{D,T,AT,NC1,NC2,NW} <: AbstractLattice end


#include("HaloComm.jl")
#include("1D/1Dlatticevector.jl")
#include("1D/1Dlatticematrix.jl")

struct Shifted_Lattice{D,Dim} <: AbstractLattice
    data::D
    shift::NTuple{Dim,Int64}

    @inline function Shifted_Lattice(data, shift, ::Val{Dim}) where {Dim}
        return new{typeof(data),Dim}(data, shift)
    end

end

# Internal lazy representation used only by halo-free kernels that implement
# periodic indexing themselves. Public shift constructors never return it.
struct _LazyShifted_Lattice{D,Dim} <: AbstractLattice
    data::D
    shift::NTuple{Dim,Int64}
end

struct Traceless_AntiHermitian{D} <: AbstractLattice
    data::D
end
export Traceless_AntiHermitian


export Shifted_Lattice
export shift_L
export add_matrix_shiftedA!

struct Adjoint_Lattice{D} <: AbstractLattice
    data::D
end



function Base.adjoint(data::TD) where {D,T,AT,TD<:Lattice{D,T,AT}}
    return Adjoint_Lattice{typeof(data)}(data)
end

@inline function Base.adjoint(data::T) where {D,Dim,T<:Shifted_Lattice{D,Dim}}
    return Adjoint_Lattice{typeof(data)}(data)
end

@inline function Base.adjoint(data::T) where {D,Dim,T<:_LazyShifted_Lattice{D,Dim}}
    return Adjoint_Lattice{typeof(data)}(data)
end

function Base.adjoint(data::TD) where {TD<:Adjoint_Lattice}
    return data.data
end




include("Latticeindices.jl")
include("LatticeMatrices_core.jl")
include("LinearAlgebras/linearalgebra.jl")
include("TA/TA.jl")
#include("AD/AD.jl")
include("ND.jl")
include("LinearAlgebras/staggered.jl")


@inline function get_shift(x::Shifted_Lattice{Tx,D}) where {D,T,AT,NC1,NC2,nw,Tx<:LatticeMatrix{D,T,AT,NC1,NC2,nw}}
    ensure_halo!(x)
    return x.shift
end

@inline function get_shift(x::Adjoint_Lattice{Shifted_Lattice{Tx,D}}) where {D,T,AT,NC1,NC2,nw,Tx<:LatticeMatrix{D,T,AT,NC1,NC2,nw}}
    ensure_halo!(x)
    return x.data.shift
end

@inline get_shift(x::_LazyShifted_Lattice) = x.shift
@inline get_shift(x::Adjoint_Lattice{<:_LazyShifted_Lattice}) = x.data.shift




#function Shifted_Lattice(data::TD, shift::TS) where {D,T,AT,TD<:Lattice{D,T,AT},TS}
#    return Shifted_Lattice{typeof(data),D}(data, shift)
#end

function zero_halo_region! end
export zero_halo_region!
function zero_halo_dim! end
export zero_halo_dim!

function fold_halo_dim_to_core_grad! end
export fold_halo_dim_to_core_grad!


@inline function _as_shift_tuple(shift_in, ::Val{D}) where {D}
    if shift_in isa NTuple{D,Int}
        return shift_in
    elseif shift_in isa AbstractVector{<:Integer}
        len = length(shift_in)
        len > D && throw(ArgumentError("shift length must be <= $D"))
        return ntuple(i -> i <= len ? Int(shift_in[i]) : 0, D)
    elseif shift_in isa Tuple
        len = length(shift_in)
        len > D && throw(ArgumentError("shift length must be <= $D"))
        return ntuple(i -> i <= len ? Int(shift_in[i]) : 0, D)
    else
        error("Unsupported shift type: $(typeof(shift_in)). Provide NTuple{$D,Int} or Vector{Int}.")
    end
end

@inline make_step(i, r, ::Val{D}) where {D} =
    ntuple(j -> ifelse(j == i, r, 0), D)

Base.@noinline function Shifted_Lattice(data::TL, shift_in::TS) where {
    D,T,AT,NC1,NC2,nw,DI,
    TL<:LatticeMatrix{D,T,AT,NC1,NC2,nw,DI},TS
}
    return Shifted_Lattice_construct(data, shift_in)
end

@inline function _periodic_shift_index(i::Integer, shift::Integer, n::Integer)
    raw = i + shift
    return mod(raw - 1, n) + 1, fld(raw - 1, n)
end

@inline function _global_core_indices(local_indices::NTuple{D,<:Integer}, coords, local_size) where D
    return ntuple(d -> coords[d] * local_size[d] + local_indices[d], D)
end

"""
    global_site_coordinates(local_indices, mpi_coordinates, local_size)
    global_site_coordinates(lattice, local_indices)

Convert one-based core-local lattice coordinates to one-based global
coordinates. MPI Cartesian coordinates are expected to be zero-based, as in a
[`LatticeMatrix`](@ref).
"""
@inline function global_site_coordinates(
    local_indices::NTuple{D,<:Integer},
    coords::NTuple{D,<:Integer},
    local_size::NTuple{D,<:Integer},
) where D
    return _global_core_indices(local_indices, coords, local_size)
end

@inline function global_site_coordinates(
    lattice::LatticeMatrix{D},
    local_indices::NTuple{D,<:Integer},
) where D
    return global_site_coordinates(local_indices, lattice.coords, lattice.PN)
end

"""
    global_site_id(global_coordinates, global_size)
    global_site_id(lattice, local_indices)

Return the zero-based global linear site id in Julia column-major order.  The
result depends only on the global coordinates and lattice size, not on MPI rank
or process-grid decomposition.
"""
@inline function global_site_id(
    global_indices::NTuple{D,<:Integer},
    global_size::NTuple{D,<:Integer},
) where D
    site = UInt64(0)
    stride = UInt64(1)
    @inbounds for d in 1:D
        site += UInt64(global_indices[d] - 1) * stride
        stride *= UInt64(global_size[d])
    end
    return site
end

@inline function global_site_id(
    lattice::LatticeMatrix{D},
    local_indices::NTuple{D,<:Integer},
) where D
    global_indices = global_site_coordinates(lattice, local_indices)
    return global_site_id(global_indices, lattice.gsize)
end

export global_site_coordinates, global_site_id

@inline function _global_site_is_even(local_indices::NTuple{D,<:Integer}, coords, local_size) where D
    coordinate_sum = 0
    @inbounds for d in 1:D
        coordinate_sum += coords[d] * local_size[d] + local_indices[d]
    end
    return iszero(coordinate_sum & 1)
end

@inline function _shifted_global_indices_and_phase(indices::NTuple{D,<:Integer}, shift,
    global_size, phases, ::Type{T}) where {D,T}
    shifted_indices = ntuple(d -> begin
        shifted, _ = _periodic_shift_index(indices[d], shift[d], global_size[d])
        shifted
    end, D)

    factor = one(T)
    @inbounds for d in 1:D
        _, wraps = _periodic_shift_index(indices[d], shift[d], global_size[d])
        factor *= phases[d]^wraps
    end
    return shifted_indices, factor
end

@inline function kernel_periodic_shift_nowing!(i, C, A, ::Val{NC1}, ::Val{NC2},
    dindexer, shift, coords, local_size, global_size, phases) where {NC1,NC2}
    local_indices = delinearize(dindexer, i, 0)
    global_indices = _global_core_indices(local_indices, coords, local_size)
    source_indices, factor = _shifted_global_indices_and_phase(
        global_indices, shift, global_size, phases, eltype(C))

    @inbounds for jc in 1:NC2
        for ic in 1:NC1
            C[ic, jc, local_indices...] = factor * A[ic, jc, source_indices...]
        end
    end
    return nothing
end

@inline _nowing_slab_indices(A, d, range) =
    ntuple(i -> i == d + 2 ? range : Colon(), ndims(A))

function _shift_one_dimension_host!(destination, source, data, d, direction)
    local_length = data.PN[d]
    if direction > 0
        destination_range = 1:(local_length-1)
        source_range = 2:local_length
        send_range = 1:1
        receive_range = local_length:local_length
        send_rank = data.nbr[d][1]
        receive_rank = data.nbr[d][2]
        crosses_global_boundary = data.coords[d] == data.dims[d] - 1
    else
        destination_range = 2:local_length
        source_range = 1:(local_length-1)
        send_range = local_length:local_length
        receive_range = 1:1
        send_rank = data.nbr[d][2]
        receive_rank = data.nbr[d][1]
        crosses_global_boundary = data.coords[d] == 0
    end

    destination_indices = _nowing_slab_indices(destination, d, destination_range)
    source_indices = _nowing_slab_indices(source, d, source_range)
    @views copyto!(destination[destination_indices...], source[source_indices...])

    send_indices = _nowing_slab_indices(source, d, send_range)
    send_buffer = Array(@view source[send_indices...])
    receive_buffer = similar(send_buffer)

    if send_rank == data.myrank && receive_rank == data.myrank
        copyto!(receive_buffer, send_buffer)
    else
        tag = 1200 + 2d + ifelse(direction > 0, 0, 1)
        requests = MPI.Request[]
        push!(requests, MPI.Irecv!(receive_buffer, receive_rank, tag, data.cart))
        push!(requests, MPI.Isend(send_buffer, send_rank, tag, data.cart))
        MPI.Waitall!(requests)
    end

    if crosses_global_boundary
        phase = direction > 0 ? data.phases[d] : inv(data.phases[d])
        _mul_phase!(receive_buffer, phase)
    end
    receive_indices = _nowing_slab_indices(destination, d, receive_range)
    @views copyto!(destination[receive_indices...], receive_buffer)
    return nothing
end

function _materialize_periodic_shift_mpi(data::TL, shift::NTuple{D,Int}) where {
    D,T,AT,NC1,NC2,DI,
    TL<:LatticeMatrix{D,T,AT,NC1,NC2,0,DI}
}
    current = Array(data.A)
    scratch = similar(current)

    for d in 1:D
        direction = sign(shift[d])
        for _ in 1:abs(shift[d])
            _shift_one_dimension_host!(scratch, current, data, d, direction)
            current, scratch = scratch, current
        end
    end

    shifted = similar(data)
    shifted.A .= JACC.array(current)
    mark_halo_dirty!(shifted)
    return shifted
end

function _materialize_periodic_shift(data::TL, shift::NTuple{D,Int}) where {
    D,T,AT,NC1,NC2,DI,
    TL<:LatticeMatrix{D,T,AT,NC1,NC2,0,DI}
}
    all(iszero, shift) && return data

    if MPI.Comm_size(data.cart) > 1
        return _materialize_periodic_shift_mpi(data, shift)
    end

    shifted = similar(data)
    _parallel_for_mutating!(shifted,
        prod(data.PN), kernel_periodic_shift_nowing!, shifted.A, data.A,
        Val(NC1), Val(NC2), data.indexer, shift, data.coords, data.PN,
        data.gsize, data.phases)
    return shifted
end

@inline function Shifted_Lattice_construct(data::TL, shift_in::TS) where {
    D,T,AT,NC1,NC2,DI,
    TL<:LatticeMatrix{D,T,AT,NC1,NC2,0,DI},TS
}
    shift = _as_shift_tuple(shift_in, Val(D))
    shifted = _materialize_periodic_shift(data, shift)
    zero_shift = ntuple(_ -> 0, D)
    return Shifted_Lattice(shifted, zero_shift, Val(D))
end

@inline function _lazy_shift_nowing(data::TL, shift_in) where {
    D,T,AT,NC1,NC2,DI,
    TL<:LatticeMatrix{D,T,AT,NC1,NC2,0,DI}
}
    MPI.Comm_size(data.cart) == 1 || throw(ArgumentError(
        "lazy nw=0 shifts are only available on a single MPI rank"))
    shift = _as_shift_tuple(shift_in, Val(D))
    return _LazyShifted_Lattice{typeof(data),D}(data, shift)
end

Base.@noinline function Shifted_Lattice_construct(data::TL, shift_in::TS) where {
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
            any(s -> !iszero(s), shift) && ensure_halo!(data)
            return Shifted_Lattice(data, shift, Val(D))
        end
    end

    sl0 = similar(data)
    sl1 = similar(data)
    substitute!(sl0, data)
    set_halo!(sl0)

    @inbounds for i in 1:D
        remaining = shift[i]
        while !iszero(remaining)
            amount = clamp(remaining, -nw, nw)
            step = make_step(i, amount, Val(D))
            sls = Shifted_Lattice(sl0, step, Val(D))
            substitute!(sl1, sls)
            set_halo!(sl1)
            sl0, sl1 = sl1, sl0
            remaining -= amount
        end
    end

    zeroshift = ntuple(_ -> 0, D)
    return Shifted_Lattice(sl0, zeroshift, Val(D))
end

#Base.@noinline function shift_L(A::LatticeMatrix, shift)
#    return Shifted_Lattice(A, shift)
#end

Base.@noinline function shift_L(B, sh::NTuple{Dim,Int}) where {Dim}
    #println("shift_L: Dim=$(Dim) length(sh)=$(length(sh)) sh=$(sh) typeof(B)=$(typeof(B))")
    return Shifted_Lattice(B, sh)
    #return Shifted_Lattice{typeof(B),Dim}(B, sh)
end

include("LinearAlgebras/mul_nowing.jl")

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
    ensure_halo!(a)
    return a.data.A
end


function get_matrix(a::T) where {T<:Adjoint_Lattice}
    return a.data.A
end

function get_matrix(a::Adjoint_Lattice{T}) where {T<:Shifted_Lattice}
    ensure_halo!(a)
    return a.data.data.A
end

@inline function ensure_halo!(a::Shifted_Lattice)
    shift = getfield(a, :shift)
    any(s -> !iszero(s), shift) && ensure_halo!(getfield(a, :data))
    return nothing
end

@inline function ensure_halo!(a::Adjoint_Lattice{<:Shifted_Lattice})
    ensure_halo!(getfield(a, :data))
    return nothing
end

@inline function Base.getproperty(a::Shifted_Lattice, name::Symbol)
    if name === :data
        ensure_halo!(a)
    end
    return getfield(a, name)
end

function JACC.parallel_for(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}, variables...) where {D,T1,AT1,NC1,NG,nw,DI}
    _parallel_for_mutating!(C,
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
    _parallel_for_mutating!(C,
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
    _parallel_for_mutating!(C,
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
    _parallel_for_mutating!(C,
        prod(C.PN), kernelfunction, C.A, a, Val(NC1), Val(NG), Val(nw), Val(NC2), Val(NG2), Val(nw2), C.indexer
    )

end

function JACC.parallel_reduce(kernelfunction::Function, C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}, A::Lattice{D,T2,AT2,NC2,NG2,nw2}) where {D,T1,AT1,NC1,NG,nw,DI,
    T2,AT2,NC2,NG2,nw2}
    a = get_matrix(A)
    s = JACC.parallel_reduce(
        prod(C.PN), kernelfunction, C.A, a, Val(NC1), Val(NG), Val(nw), Val(NC2), Val(NG2), Val(nw2), C.indexer
        ; init=zero(eltype(C.A)), op=+
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
    _parallel_for_mutating!(C,
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
        prod(C.PN), kernelfunction, C.A, a, b, variables..., Val(NC1), Val(NG), Val(nw), Val(NC2), Val(NG2), Val(nw2), Val(NC3), Val(NG3), Val(nw3), C.indexer
        ; init=zero(eltype(C.A)), op=+
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
    _parallel_for_mutating!(C,
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
        prod(C.PN), kernelfunction, C.A, a, b, Val(NC1), Val(NG), Val(nw), Val(NC2), Val(NG2), Val(nw2), Val(NC3), Val(NG3), Val(nw3), C.indexer
        ; init=zero(eltype(C.A)), op=+
    )
    s = MPI.Allreduce(s, MPI.SUM, C.comm)
end


function get_PEs(ls::LatticeMatrix{D,T,AT,NC1,NC2}) where {D,T,AT,NC1,NC2}
    return ls.dims
end
export get_PEs

function Wirtinger! end
export Wirtinger!

"""
    Wirtinger!(gradient)

Convert an Enzyme complex gradient in place to the Wirtinger derivative
`(∂f/∂x - im*∂f/∂y)/2`. Only the local core is transformed; the halo is
marked stale and will be synchronized on the next shifted read.
"""
function Wirtinger!(gradient::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}) where {D,T<:Complex,AT,NC1,NC2,nw,DI}
    half = convert(T, 0.5)
    _parallel_for_mutating!(
        gradient,
        prod(gradient.PN),
        _wirtinger_kernel!,
        gradient.A,
        half,
        Val(NC1),
        Val(NC2),
        Val(nw),
        gradient.indexer,
    )
    return gradient
end

@inline function _wirtinger_kernel!(i, gradient, half, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, indexer) where {NC1,NC2,nw}
    indices = delinearize(indexer, i, nw)
    @inbounds for jc in 1:NC2, ic in 1:NC1
        gradient[ic, jc, indices...] = half * conj(gradient[ic, jc, indices...])
    end
    return nothing
end

function realtrace end
export realtrace
function Wirtinger_derivative! end
export Wirtinger_derivative!

# Compatibility with the misspelled pre-v1 API.
Base.@deprecate Wiltinger! Wirtinger!
Base.@deprecate Wiltinger_derivative! Wirtinger_derivative!
export Wiltinger!, Wiltinger_derivative!
function Enzyme_derivative! end
export Enzyme_derivative!
function fold_halo_to_core_grad! end

struct DiffArg{T}
    x::T
end
struct NoDiffArg{T}
    x::T
end
# User-facing helpers
diff(x) = DiffArg(x)      # argument should be differentiated
nodiff(x) = NoDiffArg(x)    # argument is treated as constant
export diff, nodiff
function toann end
export toann

export mul_AshiftB!
export mul_shiftAshiftB!
export mul_A_shiftBdag!

include("Operators/Operators.jl")
include("Operators/DiracOperators.jl")
include("Operators/DiracOperators_5D.jl")


end
