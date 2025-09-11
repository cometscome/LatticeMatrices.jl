using StaticArrays
struct DIndexer{D,dims,strides}
end
export DIndexer

# ---- helpers ----
@inline _i32(x::Integer) = Int32(x)

# compute strides for row-major order
function make_strides(dims::NTuple{D,Int32}) where {D}
    @assert D > 0 "dims must be non-empty (D ≥ 1)."
    s = Vector{Int32}(undef, D)
    s[1] = 1
    @inbounds for d in 2:D
        s[d] = s[d-1] * dims[d-1]
    end
    return Tuple(s)::NTuple{D,Int32}
end

# constructor for NTuple input
DIndexer(dims_in::NTuple{D,<:Integer}) where {D} = begin
    @assert D > 0 "dims must be non-empty (D ≥ 1)."
    dims = ntuple(i -> Int32(dims_in[i]), D)
    DIndexer{D,dims,make_strides(dims)}()
end

# constructor for AbstractVector input
DIndexer(dims_in::AbstractVector{<:Integer}) = begin
    D = length(dims_in)
    @assert D > 0 "dims must be non-empty (D ≥ 1)."
    dims = ntuple(i -> Int32(dims_in[i]), D)
    DIndexer{D,dims,make_strides(dims)}()
end

# ======= GPU-kernel friendly versions (no heap allocation) =======
# idx(1-based) -> L(1-based)
@inline function linearize(
    ::DIndexer{D,dims,strides}, idx::NTuple{D,Int32}
) where {D,dims,strides}
    acc = Int32(0)
    @inbounds for d in 1:D
        acc += (idx[d] - 1) * strides[d]
    end
    return acc + 1
end

# L(1-based) -> idx(1-based) 
@inline function delinearize(
    ::DIndexer{D,dims,strides}, L::Integer, offset=Int32(0)
) where {D,dims,strides}
    m = MVector{D,Int32}(undef)
    r = L - 1
    @inbounds for d in D:-1:2
        q = r ÷ strides[d]
        r = r % strides[d]
        m[d] = q + offset + 1
    end
    m[1] = r + offset + 1
    return NTuple{D,Int32}(m)
end

@inline function shiftindices(indices, shift)
    return ntuple(i -> indices[i] + shift[i], length(indices))
end
export shiftindices


export delinearize
export linearize