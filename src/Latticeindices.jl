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
    ::DIndexer{D,dims,strides}, idx::NTuple{D,T}
) where {D,dims,strides,T<:Integer}
    acc = Int32(0)
    @inbounds for d in 1:D
        acc += (Int32(idx[d]) - 1) * strides[d]
    end
    return acc + 1
end

# L(1-based) -> idx(1-based) 
#=
@inline function delinearize(
    ::DIndexer{D,dims,strides}, L::Integer, offset=0
) where {D,dims,strides}
    m = MVector{D,Int64}(undef)
    r = L - 1
    @inbounds for d in D:-1:2
        q = r ÷ strides[d]
        r = r % strides[d]
        m[d] = q + offset + 1
        #m[d] = q + 1
    end
    #m[1] = r  + 1
    m[1] = r + offset + 1
    #m .+= offset
    return NTuple{D,Int64}(m)
end
=#

@inline function delinearize(::DIndexer{1,dims,strides}, L::Integer, offset::Integer=0) where {dims,strides}
    r   = Int(L) - 1; off = Int(offset)
    i1 = r + off + 1
    return (i1,)
end

@inline function delinearize(::DIndexer{2,dims,strides}, L::Integer, offset::Integer=0) where {dims,strides}
    r   = Int(L) - 1; off = Int(offset)
    q2, r = Base.divrem(r, Int(strides[2])); i2 = q2 + off + 1
    i1 = r + off + 1
    return (i1, i2)
end

@inline function delinearize(::DIndexer{3,dims,strides}, L::Integer, offset::Integer=0) where {dims,strides}
    r   = Int(L) - 1; off = Int(offset)
    q3, r = Base.divrem(r, Int(strides[3])); i3 = q3 + off + 1
    q2, r = Base.divrem(r, Int(strides[2])); i2 = q2 + off + 1
    i1 = r + off + 1
    return (i1, i2, i3)
end

@inline function delinearize(::DIndexer{4,dims,strides}, L::Integer, offset::Integer=0) where {dims,strides}
    r   = Int(L) - 1; off = Int(offset)
    q4, r = Base.divrem(r, Int(strides[4])); i4 = q4 + off + 1
    q3, r = Base.divrem(r, Int(strides[3])); i3 = q3 + off + 1
    q2, r = Base.divrem(r, Int(strides[2])); i2 = q2 + off + 1
    i1 = r + off + 1
    return (i1, i2, i3, i4)
end

@inline function delinearize(::DIndexer{5,dims,strides}, L::Integer, offset::Integer=0) where {dims,strides}
    r   = Int(L) - 1; off = Int(offset)
    q5, r = Base.divrem(r, Int(strides[5])); i5 = q5 + off + 1
    q4, r = Base.divrem(r, Int(strides[4])); i4 = q4 + off + 1
    q3, r = Base.divrem(r, Int(strides[3])); i3 = q3 + off + 1
    q2, r = Base.divrem(r, Int(strides[2])); i2 = q2 + off + 1
    i1 = r + off + 1
    return (i1, i2, i3, i4, i5)
end

@inline function delinearize(::DIndexer{6,dims,strides}, L::Integer, offset::Integer=0) where {dims,strides}
    r   = Int(L) - 1; off = Int(offset)
    q6, r = Base.divrem(r, Int(strides[6])); i6 = q6 + off + 1
    q5, r = Base.divrem(r, Int(strides[5])); i5 = q5 + off + 1
    q4, r = Base.divrem(r, Int(strides[4])); i4 = q4 + off + 1
    q3, r = Base.divrem(r, Int(strides[3])); i3 = q3 + off + 1
    q2, r = Base.divrem(r, Int(strides[2])); i2 = q2 + off + 1
    i1 = r + off + 1
    return (i1, i2, i3, i4, i5, i6)
end


@inline _delinearize(::Val{1}, r::Int, off::Int, strides) = (r + off + 1,)

@inline function _delinearize(::Val{N}, r::Int, off::Int, strides) where {N}
    @inbounds q, r2 = Base.divrem(r, Int(strides[N]))
    head = _delinearize(Val(N-1), r2, off, strides)
    return (head..., q + off + 1)
end

@inline function delinearize(::DIndexer{D,dims,strides}, L::Integer, offset::Integer=0) where {D,dims,strides}
    _delinearize(Val(D), Int(L) - 1, Int(offset), strides)
end


#=
@generated function delinearize(
    ::DIndexer{D,dims,strides}, L::Integer, offset::Integer=0
) where {D,dims,strides}
    # Build the block to return
    body = Expr(:block,
        :(Base.@_inline_meta true),
        :(r   = Int(L) - 1),
        :(off = Int(offset)),
    )

    comps = Vector{Any}(undef, D)

    # Divide for d = D..2
    for d = D:-1:2
        sd = Int(strides[d])
        push!(body.args, :(q, r = Base.divrem(r, $(sd))))
        comps[d] = :(q + off + 1)
    end
    # Remainder for d = 1
    comps[1] = :(r + off + 1)

    push!(body.args, Expr(:tuple, comps...))
    return body
end
=#

# Wrapper (OK to add @inline here)
@inline function delinearize(
    idx::DIndexer{D,dims,strides}, L::Integer, ::Val{nw}
) where {D,dims,strides,nw}
    return delinearize(idx, L, Int(nw))
end



#=
# L(1-based) -> idx(1-based)
@inline @Base.propagate_inbounds function delinearize(
    ::DIndexer{D,dims,strides}, L::Integer, offset = Int32(0)
) where {D,dims,strides}
    offset32 = Int32(offset)
    L32 = Int32(L)
    one32 = Int32(1)

    m = MVector{D,Int32}(undef)
    r = L32 - one32

    @inbounds for d = D:-1:2
        sd = Int32(strides[d])      
        q  = r ÷ sd                 
        r  = r % sd                  
        m[d] = q + offset32 + one32   
    end
    m[1] = r + offset32 + one32

    return NTuple{D,Int32}(m)       
end
=#



@inline function shiftindices(indices, shift)
    return ntuple(i -> indices[i] + shift[i], length(indices))
end
export shiftindices


export delinearize
export linearize
