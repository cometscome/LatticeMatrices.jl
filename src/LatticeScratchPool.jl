"""
    LatticeScratchPool(prototype; num=0, Nmax=1000, reusemode=false)

Lazily-owned scratch storage for a `LatticeMatrix`. A zero-capacity pool keeps
only the already-owned lattice array as an allocation prototype and creates its
first scratch array on demand. Positive `num` values preserve the historical
eager preallocation behavior.

This wrapper intentionally exposes the existing `PreallocatedArrays` API so
downstream code can continue to use `get_block`, `unused!`, and indexed access.
"""
mutable struct LatticeScratchPool{TG}
    prototype::TG
    pool::Union{
        Nothing,
        PreallocatedArray{TG,Union{Nothing,String},false},
    }
    initial_capacity::Int
    Nmax::Int
    reusemode::Bool

    function LatticeScratchPool(
        prototype::TG;
        num::Integer=0,
        Nmax::Integer=1000,
        reusemode::Bool=false,
    ) where {TG}
        num >= 0 || throw(ArgumentError(
            "the number of scratch blocks must be non-negative, got $num"))
        Nmax > 0 || throw(ArgumentError(
            "the maximum number of scratch blocks must be positive, got $Nmax"))
        num <= Nmax || throw(ArgumentError(
            "the initial number of scratch blocks $num exceeds Nmax=$Nmax"))
        inner = iszero(num) ? nothing : PreallocatedArray(
            prototype;
            num=Int(num),
            Nmax=Int(Nmax),
            reusemode,
            haslabel=false,
        )
        return new{TG}(prototype, inner, Int(num), Int(Nmax), reusemode)
    end
end

@inline function _ensure_scratch_pool!(
    scratch::LatticeScratchPool,
    minimum_capacity::Integer=1,
)
    minimum_capacity >= 0 || throw(ArgumentError(
        "minimum scratch capacity must be non-negative, got $minimum_capacity"))
    minimum_capacity <= scratch.Nmax || throw(ArgumentError(
        "requested scratch capacity $minimum_capacity exceeds Nmax=$(scratch.Nmax)"))
    if scratch.pool === nothing && !iszero(minimum_capacity)
        capacity = max(scratch.initial_capacity, Int(minimum_capacity))
        scratch.pool = PreallocatedArray(
            scratch.prototype;
            num=capacity,
            Nmax=scratch.Nmax,
            reusemode=scratch.reusemode,
            haslabel=false,
        )
    end
    return scratch.pool
end

function _grow_scratch_pool!(scratch::LatticeScratchPool, capacity::Integer)
    capacity >= 0 || throw(ArgumentError(
        "scratch capacity must be non-negative, got $capacity"))
    capacity <= scratch.Nmax || throw(ArgumentError(
        "requested scratch capacity $capacity exceeds Nmax=$(scratch.Nmax)"))
    pool = _ensure_scratch_pool!(scratch, capacity)
    pool === nothing && return nothing
    for _ in (length(pool) + 1):Int(capacity)
        push!(pool._data, similar(scratch.prototype))
        push!(pool._flagusing, false)
        push!(pool._indices, 0)
    end
    return pool
end

@inline _scratch_inner(scratch::LatticeScratchPool) = scratch.pool
@inline scratch_capacity(scratch::LatticeScratchPool) = length(scratch)
@inline scratch_inuse(scratch::LatticeScratchPool) =
    scratch.pool === nothing ? 0 : count(scratch.pool._flagusing)

Base.eltype(::Type{LatticeScratchPool{TG}}) where {TG} = TG
Base.eltype(::LatticeScratchPool{TG}) where {TG} = TG
Base.length(scratch::LatticeScratchPool) =
    scratch.pool === nothing ? 0 : length(scratch.pool)
Base.size(scratch::LatticeScratchPool) = (length(scratch),)
Base.firstindex(::LatticeScratchPool) = 1
Base.lastindex(scratch::LatticeScratchPool) = length(scratch)

function Base.getindex(scratch::LatticeScratchPool, index::Int)
    index > 0 || throw(BoundsError(scratch, index))
    pool = _grow_scratch_pool!(scratch, index)
    if iszero(pool._indices[index])
        storage_index = findfirst(!, pool._flagusing)
        storage_index === nothing && error(
            "scratch bookkeeping has no free storage for index $index")
        pool._flagusing[storage_index] = true
        pool._indices[index] = storage_index
    elseif !scratch.reusemode
        error("scratch index $index is already in use")
    end
    return pool._data[pool._indices[index]]
end

function Base.getindex(
    scratch::LatticeScratchPool{TG},
    indices::Vararg{Int,N},
) where {TG,N}
    blocks = TG[]
    sizehint!(blocks, N)
    for index in indices
        push!(blocks, scratch[index])
    end
    return blocks
end

function Base.getindex(
    scratch::LatticeScratchPool{TG},
    indices::AbstractVector{T},
) where {TG,T<:Integer}
    blocks = TG[]
    sizehint!(blocks, length(indices))
    for index in indices
        push!(blocks, scratch[Int(index)])
    end
    return blocks
end

function PreallocatedArrays.get_block(scratch::LatticeScratchPool)
    pool = _ensure_scratch_pool!(scratch)
    index = findfirst(iszero, pool._indices)
    if index === nothing
        index = length(pool) + 1
        _grow_scratch_pool!(scratch, index)
    end
    return scratch[index], index
end

function PreallocatedArrays.get_block(
    scratch::LatticeScratchPool{TG},
    number::Integer,
) where {TG}
    number >= 0 || throw(ArgumentError(
        "the number of requested scratch blocks must be non-negative, got $number"))
    iszero(number) && return TG[], Int64[]
    blocks = TG[]
    indices = Int64[]
    sizehint!(blocks, number)
    sizehint!(indices, number)
    for _ in 1:number
        block, index = PreallocatedArrays.get_block(scratch)
        push!(blocks, block)
        push!(indices, index)
    end
    return blocks, indices
end

function PreallocatedArrays.unused!(scratch::LatticeScratchPool, index)
    scratch.pool === nothing && return nothing
    PreallocatedArrays.unused!(scratch.pool, index)
    return nothing
end

function PreallocatedArrays.unused!(scratch::LatticeScratchPool)
    scratch.pool === nothing && return nothing
    PreallocatedArrays.unused!(scratch.pool)
    return nothing
end

function PreallocatedArrays.set_reusemode!(
    scratch::LatticeScratchPool,
    reusemode,
)
    scratch.reusemode = Bool(reusemode)
    scratch.pool === nothing ||
        PreallocatedArrays.set_reusemode!(scratch.pool, reusemode)
    return scratch.reusemode
end

# Preserve read-only compatibility for downstream diagnostics which currently
# inspect PreallocatedArray internals. New code should use `length`,
# `scratch_capacity`, and `scratch_inuse` instead.
function Base.getproperty(
    scratch::LatticeScratchPool{TG},
    name::Symbol,
) where {TG}
    if name === :_data
        pool = getfield(scratch, :pool)
        return pool === nothing ? TG[] : getfield(pool, :_data)
    elseif name === :_flagusing
        pool = getfield(scratch, :pool)
        return pool === nothing ? Bool[] : getfield(pool, :_flagusing)
    elseif name === :_indices
        pool = getfield(scratch, :pool)
        return pool === nothing ? Int64[] : getfield(pool, :_indices)
    elseif name === :_labels
        pool = getfield(scratch, :pool)
        return pool === nothing ? Union{Nothing,String}[] : getfield(pool, :_labels)
    elseif name === :_reusemode
        return getfield(scratch, :reusemode)
    elseif name === :_haslabel
        return false
    end
    return getfield(scratch, name)
end

function Base.propertynames(::LatticeScratchPool, ::Bool=false)
    return (
        :prototype,
        :pool,
        :initial_capacity,
        :Nmax,
        :reusemode,
        :_data,
        :_flagusing,
        :_indices,
        :_labels,
        :_reusemode,
        :_haslabel,
    )
end

function Base.display(scratch::LatticeScratchPool)
    if scratch.pool === nothing
        println("The total number of fields: 0")
        println("The total number of fields used: 0")
        return nothing
    end
    return display(scratch.pool)
end
