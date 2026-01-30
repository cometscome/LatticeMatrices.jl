#C = a*x
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI},
    a::TA, x::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}) where {T1,AT1,NC1,nw,NG,TA<:Number,D,DI}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mulsx!, C.A, a, x.A, Val(NC1), Val(NG), Val(nw), C.indexer
    )
    #set_halo!(C)
end

@inline function kernel_Dmatrix_mulsx!(i, C, a, x, ::Val{NC1}, ::Val{NG}, ::Val{nw}, dindexer) where {NC1,NG,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for ig = 1:NG
        for ic = 1:NC1
            C[ic, ig, indices...] = a * x[ic, ig, indices...]
        end
    end
    return
end

@inline function kernel_Dmatrix_mulsx!(i, C, a, x, ::Val{2}, ::Val{NG}, ::Val{nw}, dindexer) where {NG,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for ig = 1:NG
        C[1, ig, indices...] = a * x[1, ig, indices...]
        C[2, ig, indices...] = a * x[2, ig, indices...]
    end
    return
end

@inline function kernel_Dmatrix_mulsx!(i, C, a, x, ::Val{3}, ::Val{NG}, ::Val{nw}, dindexer) where {NG,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for ig = 1:NG
        C[1, ig, indices...] = a * x[1, ig, indices...]
        C[2, ig, indices...] = a * x[2, ig, indices...]
        C[3, ig, indices...] = a * x[3, ig, indices...]
    end
    return
end


#C = x*a
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI},
    x::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}, a::TA) where {T1,AT1,NC1,nw,NG,TA<:Number,D,DI}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mulsx!, C.A, a, x.A, Val(NC1), Val(NG), Val(nw), C.indexer
    )
    #set_halo!(C)
end

#C = C*A where a is a scalar
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI},
    a::TA) where {T1,AT1,NC1,nw,TA<:Number,DI,D,NG}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mulsC!, C.A, a, Val(NC1), Val(NG), Val(nw), C.indexer
    )
end

@inline function kernel_Dmatrix_mulsC!(i, C, a, ::Val{NC1}, ::Val{NG}, ::Val{nw}, dindexer) where {NC1,NG,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for ig = 1:NG
        for ic = 1:NC1
            C[ic, ig, indices...] = a * C[ic, ig, indices...]
        end
    end
    return
end

@inline function kernel_Dmatrix_mulsC!(i, C, a, ::Val{2}, ::Val{NG}, ::Val{nw}, dindexer) where {NG,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for ig = 1:NG
        C[1, ig, indices...] = a * C[1, ig, indices...]
        C[2, ig, indices...] = a * C[2, ig, indices...]
    end
    return
end

@inline function kernel_Dmatrix_mulsC!(i, C, a, ::Val{3}, ::Val{NG}, ::Val{nw}, dindexer) where {NG,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for ig = 1:NG
        C[1, ig, indices...] = a * C[1, ig, indices...]
        C[2, ig, indices...] = a * C[2, ig, indices...]
        C[3, ig, indices...] = a * C[3, ig, indices...]
    end
    return
end



#C = C*A where A is a regular matrix
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI},
    A::TA) where {T1,AT1,NC1,nw,TA<:AbstractMatrix,DI,D,NG}
    At = JACC.array(A[:, :])
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mulA!, C.A, At, Val(NC1), Val(NG), Val(nw), C.indexer
    )
end

# A :: NG×NG matrix (on device); eltype(A) == eltype(C)
function kernel_Dmatrix_mulA!(i, C, A, ::Val{NC1}, ::Val{NG}, ::Val{nw}, dindexer) where {NC1,NG,nw}
    indices = delinearize(dindexer, i, nw)

    @inbounds for ic = 1:NC1
        # 1) load e_j = C[ic, j, indices...] into a stack-allocated tuple (no heap alloc)
        e = ntuple(j -> C[ic, j, indices...], NG)

        # 2) r_j = Σ_k A[j,k] * e_k  (also as a tuple; unrolled by Val(NG))
        r = ntuple(k -> begin
                s = zero(eltype(C))
                @inbounds for j = 1:NG
                    s += e[j] * A[j, k]
                end
                s
            end, NG)

        # 3) write back
        @inbounds for j = 1:NG
            C[ic, j, indices...] = r[j]
        end
    end
    return
end

function kernel_Dmatrix_mulA!(i, C, A, ::Val{NC1}, ::Val{2}, ::Val{nw}, dindexer) where {NC1,nw}
    indices = delinearize(dindexer, i, nw)

    @inbounds for ic = 1:NC1
        e1 = C[ic, 1, indices...]
        e2 = C[ic, 2, indices...]

        C[ic, 1, indices...] = A[1, 1] * e1 + A[2, 1] * e2
        C[ic, 2, indices...] = A[1, 2] * e1 + A[2, 2] * e2
    end
    return
end

function kernel_Dmatrix_mulA!(i, C, A, ::Val{NC1}, ::Val{3}, ::Val{nw}, dindexer) where {NC1,nw}
    indices = delinearize(dindexer, i, nw)

    @inbounds for ic = 1:NC1
        e1 = C[ic, 1, indices...]
        e2 = C[ic, 2, indices...]
        e3 = C[ic, 3, indices...]

        C[ic, 1, indices...] = A[1, 1] * e1 + A[2, 1] * e2 + A[3, 1] * e3
        C[ic, 2, indices...] = A[1, 2] * e1 + A[2, 2] * e2 + A[3, 2] * e3
        C[ic, 3, indices...] = A[1, 3] * e1 + A[2, 3] * e2 + A[3, 3] * e3
    end
    return
end

function kernel_Dmatrix_mulA!(i, C, A, ::Val{NC1}, ::Val{4}, ::Val{nw}, dindexer) where {NC1,nw}
    indices = delinearize(dindexer, i, nw)

    @inbounds for ic = 1:NC1
        e1 = C[ic, 1, indices...]
        e2 = C[ic, 2, indices...]
        e3 = C[ic, 3, indices...]
        e4 = C[ic, 4, indices...]

        C[ic, 1, indices...] =
            A[1, 1] * e1 + A[2, 1] * e2 + A[3, 1] * e3 + A[4, 1] * e4
        C[ic, 2, indices...] =
            A[1, 2] * e1 + A[2, 2] * e2 + A[3, 2] * e3 + A[4, 2] * e4
        C[ic, 3, indices...] =
            A[1, 3] * e1 + A[2, 3] * e2 + A[3, 3] * e3 + A[4, 3] * e4
        C[ic, 4, indices...] =
            A[1, 4] * e1 + A[2, 4] * e2 + A[3, 4] * e3 + A[4, 4] * e4
    end
    return
end

function kernel_Dmatrix_mulA!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds begin
        a11 = A[1, 1]
        a12 = A[1, 2]
        a21 = A[2, 1]
        a22 = A[2, 2]

        b11 = B[1, 1, indices...]
        b21 = B[2, 1, indices...]
        b12 = B[1, 2, indices...]
        b22 = B[2, 2, indices...]

        C[1, 1, indices...] = a11 * b11 + a12 * b21
        C[2, 1, indices...] = a21 * b11 + a22 * b21
        C[1, 2, indices...] = a11 * b12 + a12 * b22
        C[2, 2, indices...] = a21 * b12 + a22 * b22
    end
end

function kernel_Dmatrix_mulA!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds begin
        a11 = A[1, 1]
        a12 = A[1, 2]
        a13 = A[1, 3]
        a21 = A[2, 1]
        a22 = A[2, 2]
        a23 = A[2, 3]
        a31 = A[3, 1]
        a32 = A[3, 2]
        a33 = A[3, 3]

        b11 = B[1, 1, indices...]
        b21 = B[2, 1, indices...]
        b31 = B[3, 1, indices...]
        b12 = B[1, 2, indices...]
        b22 = B[2, 2, indices...]
        b32 = B[3, 2, indices...]
        b13 = B[1, 3, indices...]
        b23 = B[2, 3, indices...]
        b33 = B[3, 3, indices...]

        C[1, 1, indices...] = a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = a31 * b13 + a32 * b23 + a33 * b33
    end
end




#C = A B
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul!, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer
    )
    #set_halo!(C)
end




@inline function kernel_Dmatrix_mul!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = zero(eltype(C))
        end

        for kc = 1:NC3
            b = B[kc, jc, indices...]
            for ic = 1:NC1
                C[ic, jc, indices...] += A[ic, kc, indices...] * b# B[kc, jc, indices...]
            end
        end
    end
end



#C = A B 
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC1,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC1,nw,DI}, B::LatticeMatrix{D,T3,AT3,NC1,NC1,nw,DI}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,nw,DI}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul!, C.A, A.A, B.A, Val(NC1), Val(nw), C.indexer
    )
    #set_halo!(C)
end

@inline function kernel_Dmatrix_mul!(i, C, A, B, ::Val{NC1}, ::Val{nw}, dindexer) where {NC1,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC1
        for ic = 1:NC1
            C[ic, jc, indices...] = zero(eltype(C))
        end

        for kc = 1:NC1
            b = B[kc, jc, indices...]
            for ic = 1:NC1
                C[ic, jc, indices...] += A[ic, kc, indices...] * b# B[kc, jc, indices...]
            end
        end
    end
end

@inline function kernel_Dmatrix_mul!(i, C, A, B, ::Val{3}, ::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        a11 = A[1, 1, indices...]
        a21 = A[2, 1, indices...]
        a31 = A[3, 1, indices...]
        a12 = A[1, 2, indices...]
        a22 = A[2, 2, indices...]
        a32 = A[3, 2, indices...]
        a13 = A[1, 3, indices...]
        a23 = A[2, 3, indices...]
        a33 = A[3, 3, indices...]

        b11 = B[1, 1, indices...]
        b21 = B[2, 1, indices...]
        b31 = B[3, 1, indices...]
        b12 = B[1, 2, indices...]
        b22 = B[2, 2, indices...]
        b32 = B[3, 2, indices...]
        b13 = B[1, 3, indices...]
        b23 = B[2, 3, indices...]
        b33 = B[3, 3, indices...]
        C[1, 1, indices...] = a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = a31 * b13 + a32 * b23 + a33 * b33
    end
end

@inline function kernel_Dmatrix_mul!(i, C, A, B, ::Val{2}, ::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        a11 = A[1, 1, indices...]
        a21 = A[2, 1, indices...]
        a12 = A[1, 2, indices...]
        a22 = A[2, 2, indices...]


        b11 = B[1, 1, indices...]
        b21 = B[2, 1, indices...]
        b12 = B[1, 2, indices...]
        b22 = B[2, 2, indices...]

        C[1, 1, indices...] = a11 * b11 + a12 * b21
        C[2, 1, indices...] = a21 * b11 + a22 * b21
        C[1, 2, indices...] = a11 * b12 + a12 * b22
        C[2, 2, indices...] = a21 * b12 + a22 * b22

    end
end



@inline function kernel_Dmatrix_mul!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        a11 = A[1, 1, indices...]
        a21 = A[2, 1, indices...]
        a31 = A[3, 1, indices...]
        a12 = A[1, 2, indices...]
        a22 = A[2, 2, indices...]
        a32 = A[3, 2, indices...]
        a13 = A[1, 3, indices...]
        a23 = A[2, 3, indices...]
        a33 = A[3, 3, indices...]
        b11 = B[1, 1, indices...]
        b21 = B[2, 1, indices...]
        b31 = B[3, 1, indices...]
        b12 = B[1, 2, indices...]
        b22 = B[2, 2, indices...]
        b32 = B[3, 2, indices...]
        b13 = B[1, 3, indices...]
        b23 = B[2, 3, indices...]
        b33 = B[3, 3, indices...]
        C[1, 1, indices...] = a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = a31 * b13 + a32 * b23 + a33 * b33
    end
end


@inline function kernel_Dmatrix_mul!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        a11 = A[1, 1, indices...]
        a21 = A[2, 1, indices...]
        a12 = A[1, 2, indices...]
        a22 = A[2, 2, indices...]

        b11 = B[1, 1, indices...]
        b21 = B[2, 1, indices...]
        b12 = B[1, 2, indices...]
        b22 = B[2, 2, indices...]

        C[1, 1, indices...] = a11 * b11 + a12 * b21
        C[2, 1, indices...] = a21 * b11 + a22 * b21
        C[1, 2, indices...] = a11 * b12 + a12 * b22
        C[2, 2, indices...] = a21 * b12 + a22 * b22

    end
end





#C = A B α + C β
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}, α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul!, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, α, β
    )
    #set_halo!(C)
end


@inline function kernel_Dmatrix_mul!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, α, β) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = β * C[ic, jc, indices...]
            for kc = 1:NC3
                C[ic, jc, indices...] += α * A[ic, kc, indices...] * B[kc, jc, indices...]
            end
        end
    end
end

@inline function kernel_Dmatrix_mul!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer, α, β) where {nw}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        a11 = α * A[1, 1, indices...]
        a21 = α * A[2, 1, indices...]
        a31 = α * A[3, 1, indices...]
        a12 = α * A[1, 2, indices...]
        a22 = α * A[2, 2, indices...]
        a32 = α * A[3, 2, indices...]
        a13 = α * A[1, 3, indices...]
        a23 = α * A[2, 3, indices...]
        a33 = α * A[3, 3, indices...]

        b11 = B[1, 1, indices...]
        b21 = B[2, 1, indices...]
        b31 = B[3, 1, indices...]
        b12 = B[1, 2, indices...]
        b22 = B[2, 2, indices...]
        b32 = B[3, 2, indices...]
        b13 = B[1, 3, indices...]
        b23 = B[2, 3, indices...]
        b33 = B[3, 3, indices...]
        C[1, 1, indices...] = β * C[1, 1, indices...] + a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = β * C[2, 1, indices...] + a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = β * C[3, 1, indices...] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = β * C[1, 2, indices...] + a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = β * C[2, 2, indices...] + a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = β * C[3, 2, indices...] + a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = β * C[1, 3, indices...] + a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = β * C[2, 3, indices...] + a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = β * C[3, 3, indices...] + a31 * b13 + a32 * b23 + a33 * b33
    end

end

@inline function kernel_Dmatrix_mul!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, dindexer, α, β) where {nw}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        a11 = α * A[1, 1, indices...]
        a21 = α * A[2, 1, indices...]
        a12 = α * A[1, 2, indices...]
        a22 = α * A[2, 2, indices...]


        b11 = B[1, 1, indices...]
        b21 = B[2, 1, indices...]

        b12 = B[1, 2, indices...]
        b22 = B[2, 2, indices...]


        C[1, 1, indices...] = β * C[1, 1, indices...] + a11 * b11 + a12 * b21
        C[2, 1, indices...] = β * C[2, 1, indices...] + a21 * b11 + a22 * b21
        C[1, 2, indices...] = β * C[1, 2, indices...] + a11 * b12 + a12 * b22
        C[2, 2, indices...] = β * C[2, 2, indices...] + a21 * b12 + a22 * b22
    end

end





#C = a*x'
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI},
    a::TA, x::Adjoint_Lattice{L}) where {T1,AT1,NC1,nw,NG,TA<:Number,D,DI,L<:LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mulsdagx!, C.A, a, x.data.A, Val(NC1), Val(NG), Val(nw), C.indexer
    )
    #set_halo!(C)
end

@inline function kernel_Dmatrix_mulsdagx!(i, C, a, x, ::Val{NC1}, ::Val{NG}, ::Val{nw}, dindexer) where {NC1,NG,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for ig = 1:NG
        for ic = 1:NC1
            C[ic, ig, indices...] = a * x[ig, ic, indices...]'
        end
    end
    return
end

@inline function kernel_Dmatrix_mulsdagx!(i, C, a, x, ::Val{2}, ::Val{NG}, ::Val{nw}, dindexer) where {NG,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for ig = 1:NG
        C[1, ig, indices...] = a * x[ig, 1, indices...]'
        C[2, ig, indices...] = a * x[ig, 2, indices...]'
    end
    return
end

@inline function kernel_Dmatrix_mulsdagx!(i, C, a, x, ::Val{3}, ::Val{NG}, ::Val{nw}, dindexer) where {NG,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for ig = 1:NG
        C[1, ig, indices...] = a * x[ig, 1, indices...]'
        C[2, ig, indices...] = a * x[ig, 2, indices...]'
        C[3, ig, indices...] = a * x[ig, 3, indices...]'
    end
    return
end

#C = x'*a
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI},
    x::Adjoint_Lattice{L}, a::TA) where {T1,AT1,NC1,nw,NG,TA<:Number,D,DI,L<:LatticeMatrix{D,T1,AT1,NC1,NG,nw,DI}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mulsdagx!, C.A, a, x.data.A, Val(NC1), Val(NG), Val(nw), C.indexer
    )
    #set_halo!(C)
end



#C = A'*B
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{L}, B::LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI}}

    mul_AdagB!(C, A.data, B)
    #set_halo!(C)
end


function mul_AdagB!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::L, B::LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_AdagB!, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer
    )
    #set_halo!(C)
end


@inline function kernel_Dmatrix_mul_AdagB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = 0
            for kc = 1:NC3
                C[ic, jc, indices...] += A[kc, ic, indices...]' * B[kc, jc, indices...]
            end
        end
    end
end


@inline function kernel_Dmatrix_mul_AdagB!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        a11 = A[1, 1, indices...]'
        a12 = A[2, 1, indices...]'

        a21 = A[1, 2, indices...]'
        a22 = A[2, 2, indices...]'


        b11 = B[1, 1, indices...]
        b21 = B[2, 1, indices...]

        b12 = B[1, 2, indices...]
        b22 = B[2, 2, indices...]

        C[1, 1, indices...] = a11 * b11 + a12 * b21
        C[2, 1, indices...] = a21 * b11 + a22 * b21
        C[1, 2, indices...] = a11 * b12 + a12 * b22
        C[2, 2, indices...] = a21 * b12 + a22 * b22
    end
end

@inline function kernel_Dmatrix_mul_AdagB!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        a11 = A[1, 1, indices...]'
        a12 = A[2, 1, indices...]'
        a13 = A[3, 1, indices...]'

        a21 = A[1, 2, indices...]'
        a22 = A[2, 2, indices...]'
        a23 = A[3, 2, indices...]'

        a31 = A[1, 3, indices...]'
        a32 = A[2, 3, indices...]'
        a33 = A[3, 3, indices...]'

        b11 = B[1, 1, indices...]
        b21 = B[2, 1, indices...]
        b31 = B[3, 1, indices...]
        b12 = B[1, 2, indices...]
        b22 = B[2, 2, indices...]
        b32 = B[3, 2, indices...]
        b13 = B[1, 3, indices...]
        b23 = B[2, 3, indices...]
        b33 = B[3, 3, indices...]
        C[1, 1, indices...] = a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = a31 * b13 + a32 * b23 + a33 * b33
    end
end


#C = α*A'*B+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{L}, B::LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI}}

    mul_AdagB!(C, A.data, B, α, β)
    #set_halo!(C)
end


function mul_AdagB!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::L, B::LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}, α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_AdagB!, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_Dmatrix_mul_AdagB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = β * C[ic, jc, indices...]
            for kc = 1:NC3
                C[ic, jc, indices...] += α * A[kc, ic, indices...]' * B[kc, jc, indices...]
            end
        end
    end
end




@inline function kernel_Dmatrix_mul_AdagB!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        a11 = α * A[1, 1, indices...]'
        a12 = α * A[2, 1, indices...]'
        a13 = α * A[3, 1, indices...]'

        a21 = α * A[1, 2, indices...]'
        a22 = α * A[2, 2, indices...]'
        a23 = α * A[3, 2, indices...]'

        a31 = α * A[1, 3, indices...]'
        a32 = α * A[2, 3, indices...]'
        a33 = α * A[3, 3, indices...]'

        b11 = B[1, 1, indices...]
        b21 = B[2, 1, indices...]
        b31 = B[3, 1, indices...]
        b12 = B[1, 2, indices...]
        b22 = B[2, 2, indices...]
        b32 = B[3, 2, indices...]
        b13 = B[1, 3, indices...]
        b23 = B[2, 3, indices...]
        b33 = B[3, 3, indices...]
        C[1, 1, indices...] = β * C[1, 1, indices...] + a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = β * C[2, 1, indices...] + a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = β * C[3, 1, indices...] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = β * C[1, 2, indices...] + a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = β * C[2, 2, indices...] + a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = β * C[3, 2, indices...] + a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = β * C[1, 3, indices...] + a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = β * C[2, 3, indices...] + a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = β * C[3, 3, indices...] + a31 * b13 + a32 * b23 + a33 * b33
    end
end






#C = A*shiftedB
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::Shifted_Lattice{L,D}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}
    #println("C = A*shiftedB $NC1 $NC2 $NC3 ")
    #display(B.data.A[:, :, 2, 2, 2, 2])
    #println("BdataA")
    shift = get_shift(B)
    #for i = 1:prod(C.PN)
    #    kernel_Dmatrix_mul_AshiftB!(i, C.A, A.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift)
    #end
    mul_AshiftB!(C, A, B.data, shift)
    #JACC.parallel_for(
    #    prod(C.PN), kernel_Dmatrix_mul_AshiftB!, C.A, A.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift
    #)
    #set_halo!(C)
end


function mul_AshiftB!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::L, shift) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}
    #println("C = A*shiftedB $NC1 $NC2 $NC3 ")
    #display(B.data.A[:, :, 2, 2, 2, 2])
    #println("BdataA")
    #shift = get_shift(B)

    #for i = 1:prod(C.PN)
    #    kernel_Dmatrix_mul_AshiftB!(i, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift)
    #end


    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_AshiftB!, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift
    )

    #set_halo!(C)
end

function mul_simple!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::Shifted_Lattice{L,D}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}
    shift = get_shift(B)
    mul_simple_AshiftB!(C, A, B.data, shift)
end

function mul_simple_AshiftB!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::L, shift) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}
    @inbounds for i in 1:prod(C.PN)
        kernel_Dmatrix_mul_AshiftB!(i, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift)
    end
end


@inline function kernel_Dmatrix_mul_AshiftB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift) where {NC1,NC2,NC3,nw}
    #println("d $NC1 $NC2 $NC3 dd")
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = 0
            for kc = 1:NC3
                C[ic, jc, indices...] += A[ic, kc, indices...] * B[kc, jc, indices_p...]
            end
        end
    end
end

function mul_shiftAshiftB!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::L, shiftA, shiftB) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftAshiftB!, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shiftA, shiftB
    )
    #set_halo!(C)
end

function mul_shiftAshiftB!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{L1}, B::L, shiftA, shiftB) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftAdagshiftB!, C.A, A.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shiftA, shiftB
    )
    #set_halo!(C)
end

@inline function kernel_Dmatrix_mul_AshiftB!(i, y, A, x, ::Val{3}, ::Val{4}, ::Val{3}, ::Val{nw}, dindexer, shift) where {nw}
    indices = delinearize(dindexer, i, nw)
    #println("dd")
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw
    indices_p = shiftindices(indices, shift)

    @inbounds for ialpha = 1:4
        x1 = x[1, ialpha, indices_p...]
        x2 = x[2, ialpha, indices_p...]
        x3 = x[3, ialpha, indices_p...]


        y[1, ialpha, indices...] =
            A[1, 1, indices...] * x1 +
            A[1, 2, indices...] * x2 +
            A[1, 3, indices...] * x3
        y[2, ialpha, indices...] =
            A[2, 1, indices...] * x1 +
            A[2, 2, indices...] * x2 +
            A[2, 3, indices...] * x3
        y[3, ialpha, indices...] =
            A[3, 1, indices...] * x1 +
            A[3, 2, indices...] * x2 +
            A[3, 3, indices...] * x3

        #if i == 1
        #    println((x1, x2, x3))
        #    println((y[1, ialpha, indices...], y[2, ialpha, indices...], y[3, ialpha, indices...]))
        #end
    end


    #=
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = 0
            for kc = 1:NC3
                C[ic, jc, indices...] += A[ic, kc, indices...] * B[kc, jc, indices_p...]
            end
        end
    end
    =#
end



@inline function kernel_Dmatrix_mul_AshiftB!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer, shift) where {nw}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw
    @inbounds begin
        indices_p = shiftindices(indices, shift)


        a11 = A[1, 1, indices...]
        a21 = A[2, 1, indices...]
        a31 = A[3, 1, indices...]
        a12 = A[1, 2, indices...]
        a22 = A[2, 2, indices...]
        a32 = A[3, 2, indices...]
        a13 = A[1, 3, indices...]
        a23 = A[2, 3, indices...]
        a33 = A[3, 3, indices...]
        b11 = B[1, 1, indices_p...]
        b21 = B[2, 1, indices_p...]
        b31 = B[3, 1, indices_p...]
        b12 = B[1, 2, indices_p...]
        b22 = B[2, 2, indices_p...]
        b32 = B[3, 2, indices_p...]
        b13 = B[1, 3, indices_p...]
        b23 = B[2, 3, indices_p...]
        b33 = B[3, 3, indices_p...]
        C[1, 1, indices...] = a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = a31 * b13 + a32 * b23 + a33 * b33
    end
end

@inline function kernel_Dmatrix_mul_AshiftB!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, dindexer, shift) where {nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds begin
        indices_p = shiftindices(indices, shift)

        a11 = A[1, 1, indices...]
        a21 = A[2, 1, indices...]
        a12 = A[1, 2, indices...]
        a22 = A[2, 2, indices...]

        b11 = B[1, 1, indices_p...]
        b21 = B[2, 1, indices_p...]
        b12 = B[1, 2, indices_p...]
        b22 = B[2, 2, indices_p...]

        C[1, 1, indices...] = a11 * b11 + a12 * b21
        C[2, 1, indices...] = a21 * b11 + a22 * b21
        C[1, 2, indices...] = a11 * b12 + a12 * b22
        C[2, 2, indices...] = a21 * b12 + a22 * b22
    end
end




#C = α A*shiftedB + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::Shifted_Lattice{L,D},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}
    shift = get_shift(B)
    #βin = T1(β)
    #αin = T1(α)
    mul_AshiftB!(C, A, B.data, shift, α, β)
    #JACC.parallel_for(
    #    prod(C.PN), kernel_Dmatrix_mul_AshiftB!, C.A, A.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift, αin, βin
    #)
    #set_halo!(C)
end

function mul_AshiftB!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::L, shift,
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}
    βin = T1(β)
    αin = T1(α)
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_AshiftB!, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift, αin, βin
    )
    #set_halo!(C)
end

function mul_simple!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::Shifted_Lattice{L,D},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}
    shift = get_shift(B)
    mul_simple_AshiftB!(C, A, B.data, shift, α, β)
end

function mul_simple_AshiftB!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::L, shift,
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}
    αin = T1(α)
    βin = T1(β)
    @inbounds for i in 1:prod(C.PN)
        kernel_Dmatrix_mul_AshiftB!(i, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift, αin, βin)
    end
end


@inline function kernel_Dmatrix_mul_AshiftB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)


    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = β * C[ic, jc, indices...]
            for kc = 1:NC3
                C[ic, jc, indices...] += α * A[ic, kc, indices...] * B[kc, jc, indices_p...]
            end
        end
    end
end



@inline function kernel_Dmatrix_mul_AshiftB!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer, shift, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw
    @inbounds begin
        indices_p = shiftindices(indices, shift)


        a11 = A[1, 1, indices...]
        a21 = A[2, 1, indices...]
        a31 = A[3, 1, indices...]
        a12 = A[1, 2, indices...]
        a22 = A[2, 2, indices...]
        a32 = A[3, 2, indices...]
        a13 = A[1, 3, indices...]
        a23 = A[2, 3, indices...]
        a33 = A[3, 3, indices...]

        b11 = B[1, 1, indices_p...]
        b21 = B[2, 1, indices_p...]
        b31 = B[3, 1, indices_p...]
        c11 = a11 * b11 + a12 * b21 + a13 * b31
        c21 = a21 * b11 + a22 * b21 + a23 * b31
        c31 = a31 * b11 + a32 * b21 + a33 * b31

        # ----  j=2 ----
        b12 = B[1, 2, indices_p...]
        b22 = B[2, 2, indices_p...]
        b32 = B[3, 2, indices_p...]
        c12 = a11 * b12 + a12 * b22 + a13 * b32
        c22 = a21 * b12 + a22 * b22 + a23 * b32
        c32 = a31 * b12 + a32 * b22 + a33 * b32

        # ----  j=3 ----
        b13 = B[1, 3, indices_p...]
        b23 = B[2, 3, indices_p...]
        b33 = B[3, 3, indices_p...]
        c13 = a11 * b13 + a12 * b23 + a13 * b33
        c23 = a21 * b13 + a22 * b23 + a23 * b33
        c33 = a31 * b13 + a32 * b23 + a33 * b33

        if iszero(β)
            C[1, 1, indices...] = α * c11
            C[2, 1, indices...] = α * c21
            C[3, 1, indices...] = α * c31
            C[1, 2, indices...] = α * c12
            C[2, 2, indices...] = α * c22
            C[3, 2, indices...] = α * c32
            C[1, 3, indices...] = α * c13
            C[2, 3, indices...] = α * c23
            C[3, 3, indices...] = α * c33
        else
            C[1, 1, indices...] = α * c11 + β * C[1, 1, indices...]
            C[2, 1, indices...] = α * c21 + β * C[2, 1, indices...]
            C[3, 1, indices...] = α * c31 + β * C[3, 1, indices...]
            C[1, 2, indices...] = α * c12 + β * C[1, 2, indices...]
            C[2, 2, indices...] = α * c22 + β * C[2, 2, indices...]
            C[3, 2, indices...] = α * c32 + β * C[3, 2, indices...]
            C[1, 3, indices...] = α * c13 + β * C[1, 3, indices...]
            C[2, 3, indices...] = α * c23 + β * C[2, 3, indices...]
            C[3, 3, indices...] = α * c33 + β * C[3, 3, indices...]
        end


        #=
        a11 = α * A[1, 1, indices...]
        a21 = α * A[2, 1, indices...]
        a31 = α * A[3, 1, indices...]
        a12 = α * A[1, 2, indices...]
        a22 = α * A[2, 2, indices...]
        a32 = α * A[3, 2, indices...]
        a13 = α * A[1, 3, indices...]
        a23 = α * A[2, 3, indices...]
        a33 = α * A[3, 3, indices...]
        b11 = B[1, 1, indices_p...]
        b21 = B[2, 1, indices_p...]
        b31 = B[3, 1, indices_p...]
        b12 = B[1, 2, indices_p...]
        b22 = B[2, 2, indices_p...]
        b32 = B[3, 2, indices_p...]
        b13 = B[1, 3, indices_p...]
        b23 = B[2, 3, indices_p...]
        b33 = B[3, 3, indices_p...]
        C[1, 1, indices...] = β * C[1, 1, indices...] + a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = β * C[2, 1, indices...] + a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = β * C[3, 1, indices...] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = β * C[1, 2, indices...] + a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = β * C[2, 2, indices...] + a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = β * C[3, 2, indices...] + a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = β * C[1, 3, indices...] + a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = β * C[2, 3, indices...] + a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = β * C[3, 3, indices...] + a31 * b13 + a32 * b23 + a33 * b33
        =#
    end


end

@inline function kernel_Dmatrix_mul_AshiftB!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, dindexer, shift, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    @inbounds begin
        indices_p = shiftindices(indices, shift)

        a11 = A[1, 1, indices...]
        a21 = A[2, 1, indices...]
        a12 = A[1, 2, indices...]
        a22 = A[2, 2, indices...]

        b11 = B[1, 1, indices_p...]
        b21 = B[2, 1, indices_p...]
        c11 = a11 * b11 + a12 * b21
        c21 = a21 * b11 + a22 * b21

        b12 = B[1, 2, indices_p...]
        b22 = B[2, 2, indices_p...]
        c12 = a11 * b12 + a12 * b22
        c22 = a21 * b12 + a22 * b22

        if iszero(β)
            C[1, 1, indices...] = α * c11
            C[2, 1, indices...] = α * c21
            C[1, 2, indices...] = α * c12
            C[2, 2, indices...] = α * c22
        else
            C[1, 1, indices...] = α * c11 + β * C[1, 1, indices...]
            C[2, 1, indices...] = α * c21 + β * C[2, 1, indices...]
            C[1, 2, indices...] = α * c12 + β * C[1, 2, indices...]
            C[2, 2, indices...] = α * c22 + β * C[2, 2, indices...]
        end
    end
end






#C = shiftedA'*B
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Shifted_Lattice{L,D}}, B::LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI}}

    shift = get_shift(A)
    mul_shiftAdag_B!(C, A, B, shift)
    #set_halo!(C)
end

function mul_shiftAdag_B!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Shifted_Lattice{L,D}}, B::LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}, shift) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftAdagB!, C.A, A.data.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift
    )
    #set_halo!(C)
end


@inline function kernel_Dmatrix_mul_shiftAdagB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = 0
            for kc = 1:NC3
                C[ic, jc, indices...] += A[kc, ic, indices_p...]' * B[kc, jc, indices...]
            end
        end
    end
end

@inline function kernel_Dmatrix_mul_shiftAdagB!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer, shift) where {nw}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw
    @inbounds begin
        indices_p = shiftindices(indices, shift)


        a11 = A[1, 1, indices_p...]'
        a12 = A[2, 1, indices_p...]'
        a13 = A[3, 1, indices_p...]'
        a21 = A[1, 2, indices_p...]'
        a22 = A[2, 2, indices_p...]'
        a23 = A[3, 2, indices_p...]'
        a31 = A[1, 3, indices_p...]'
        a32 = A[2, 3, indices_p...]'
        a33 = A[3, 3, indices_p...]'

        b11 = B[1, 1, indices...]
        b21 = B[2, 1, indices...]
        b31 = B[3, 1, indices...]
        b12 = B[1, 2, indices...]
        b22 = B[2, 2, indices...]
        b32 = B[3, 2, indices...]
        b13 = B[1, 3, indices...]
        b23 = B[2, 3, indices...]
        b33 = B[3, 3, indices...]
        C[1, 1, indices...] = a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = a31 * b13 + a32 * b23 + a33 * b33
    end
end


#C = α*shiftedA'*B + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Shifted_Lattice{L,D}}, B::LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI}}

    shift = get_shift(A)
    mul_shiftAdag_B!(C, A, B, shift, α, β)
    #set_halo!(C)
end

function mul_shiftAdag_B!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Shifted_Lattice{L,D}}, B::LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}, shift,
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftAdagB!, C.A, A.data.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_Dmatrix_mul_shiftAdagB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = β * C[ic, jc, indices...]
            for kc = 1:NC3
                C[ic, jc, indices...] += α * A[kc, ic, indices_p...]' * B[kc, jc, indices...]
            end
        end
    end
end


@inline function kernel_Dmatrix_mul_shiftAdagB!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, dindexer, shift, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw
    @inbounds begin
        indices_p = shiftindices(indices, shift)


        a11 = α * A[1, 1, indices_p...]'
        a12 = α * A[2, 1, indices_p...]'
        #a13 = α * A[3, 1, indices_p...]'
        a21 = α * A[1, 2, indices_p...]'
        a22 = α * A[2, 2, indices_p...]'
        #a23 = α * A[3, 2, indices_p...]'
        #a31 = α * A[1, 3, indices_p...]'
        #a32 = α * A[2, 3, indices_p...]'
        #a33 = α * A[3, 3, indices_p...]'
        b11 = B[1, 1, indices...]
        b21 = B[2, 1, indices...]
        #b31 = B[3, 1, indices...]
        b12 = B[1, 2, indices...]
        b22 = B[2, 2, indices...]
        #b32 = B[3, 2, indices...]
        #b13 = B[1, 3, indices...]
        #b23 = B[2, 3, indices...]
        #b33 = B[3, 3, indices...]
        C[1, 1, indices...] = β * C[1, 1, indices...] + a11 * b11 + a12 * b21 #+ a13 * b31
        C[2, 1, indices...] = β * C[2, 1, indices...] + a21 * b11 + a22 * b21 #+ a23 * b31
        #C[3, 1, indices...] = β * C[3, 1, indices...] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = β * C[1, 2, indices...] + a11 * b12 + a12 * b22 #+ a13 * b32
        C[2, 2, indices...] = β * C[2, 2, indices...] + a21 * b12 + a22 * b22 #+ a23 * b32
        #C[3, 2, indices...] = β * C[3, 2, indices...] + a31 * b12 + a32 * b22 + a33 * b32
        #C[1, 3, indices...] = β * C[1, 3, indices...] + a11 * b13 + a12 * b23 + a13 * b33
        #C[2, 3, indices...] = β * C[2, 3, indices...] + a21 * b13 + a22 * b23 + a23 * b33
        #C[3, 3, indices...] = β * C[3, 3, indices...] + a31 * b13 + a32 * b23 + a33 * b33
    end


end

@inline function kernel_Dmatrix_mul_shiftAdagB!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer, shift, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw
    @inbounds begin
        indices_p = shiftindices(indices, shift)


        a11 = α * A[1, 1, indices_p...]'
        a12 = α * A[2, 1, indices_p...]'
        a13 = α * A[3, 1, indices_p...]'
        a21 = α * A[1, 2, indices_p...]'
        a22 = α * A[2, 2, indices_p...]'
        a23 = α * A[3, 2, indices_p...]'
        a31 = α * A[1, 3, indices_p...]'
        a32 = α * A[2, 3, indices_p...]'
        a33 = α * A[3, 3, indices_p...]'
        b11 = B[1, 1, indices...]
        b21 = B[2, 1, indices...]
        b31 = B[3, 1, indices...]
        b12 = B[1, 2, indices...]
        b22 = B[2, 2, indices...]
        b32 = B[3, 2, indices...]
        b13 = B[1, 3, indices...]
        b23 = B[2, 3, indices...]
        b33 = B[3, 3, indices...]
        C[1, 1, indices...] = β * C[1, 1, indices...] + a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = β * C[2, 1, indices...] + a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = β * C[3, 1, indices...] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = β * C[1, 2, indices...] + a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = β * C[2, 2, indices...] + a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = β * C[3, 2, indices...] + a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = β * C[1, 3, indices...] + a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = β * C[2, 3, indices...] + a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = β * C[3, 3, indices...] + a31 * b13 + a32 * b23 + a33 * b33
    end


end


#C = shiftedA*B'
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Shifted_Lattice{L1,D}, B::Adjoint_Lattice{L2}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    shift = get_shift(A)
    mul_shiftA_Bdag!(C, A, B, shift)
    #set_halo!(C)
end

function mul_shiftA_Bdag!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Shifted_Lattice{L1,D}, B::Adjoint_Lattice{L2}, shift) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftABdag!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift
    )
    #set_halo!(C)
end


@inline function kernel_Dmatrix_mul_shiftABdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = 0
            for kc = 1:NC3
                C[ic, jc, indices...] += A[ic, kc, indices_p...] * B[jc, kc, indices...]'
            end
        end
    end
end

@inline function kernel_Dmatrix_mul_shiftABdag!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, dindexer, shift) where {nw}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw
    @inbounds begin
        indices_p = shiftindices(indices, shift)


        a11 = A[1, 1, indices_p...]
        a21 = A[2, 1, indices_p...]
        #a31 = A[3, 1, indices_p...]
        a12 = A[1, 2, indices_p...]
        a22 = A[2, 2, indices_p...]
        #a32 = A[3, 2, indices_p...]
        #a13 = A[1, 3, indices_p...]
        #a23 = A[2, 3, indices_p...]
        #a33 = A[3, 3, indices_p...]

        b11 = B[1, 1, indices...]'
        b12 = B[2, 1, indices...]'
        #b13 = B[3, 1, indices...]'
        b21 = B[1, 2, indices...]'
        b22 = B[2, 2, indices...]'
        #b23 = B[3, 2, indices...]'
        #b31 = B[1, 3, indices...]'
        #b32 = B[2, 3, indices...]'
        #b33 = B[3, 3, indices...]'

        C[1, 1, indices...] = a11 * b11 + a12 * b21 #+ a13 * b31
        C[2, 1, indices...] = a21 * b11 + a22 * b21 #+ a23 * b31
        #C[3, 1, indices...] = a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = a11 * b12 + a12 * b22 #+ a13 * b32
        C[2, 2, indices...] = a21 * b12 + a22 * b22 #+ a23 * b32
        #C[3, 2, indices...] = a31 * b12 + a32 * b22 + a33 * b32
        #C[1, 3, indices...] = a11 * b13 + a12 * b23 + a13 * b33
        #C[2, 3, indices...] = a21 * b13 + a22 * b23 + a23 * b33
        #C[3, 3, indices...] = a31 * b13 + a32 * b23 + a33 * b33
    end
end


@inline function kernel_Dmatrix_mul_shiftABdag!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer, shift) where {nw}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw
    @inbounds begin
        indices_p = shiftindices(indices, shift)


        a11 = A[1, 1, indices_p...]
        a21 = A[2, 1, indices_p...]
        a31 = A[3, 1, indices_p...]
        a12 = A[1, 2, indices_p...]
        a22 = A[2, 2, indices_p...]
        a32 = A[3, 2, indices_p...]
        a13 = A[1, 3, indices_p...]
        a23 = A[2, 3, indices_p...]
        a33 = A[3, 3, indices_p...]

        b11 = B[1, 1, indices...]'
        b12 = B[2, 1, indices...]'
        b13 = B[3, 1, indices...]'
        b21 = B[1, 2, indices...]'
        b22 = B[2, 2, indices...]'
        b23 = B[3, 2, indices...]'
        b31 = B[1, 3, indices...]'
        b32 = B[2, 3, indices...]'
        b33 = B[3, 3, indices...]'

        C[1, 1, indices...] = a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = a31 * b13 + a32 * b23 + a33 * b33
    end
end


#C = α*shiftedA*B'+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Shifted_Lattice{L1,D}, B::Adjoint_Lattice{L2},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    shift = get_shift(A)
    mul_shiftA_Bdag!(C, A, B, shift, α, β)
    #set_halo!(C)
end

function mul_shiftA_Bdag!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Shifted_Lattice{L1,D}, B::Adjoint_Lattice{L2}, shift,
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftABdag!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_Dmatrix_mul_shiftABdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = β * C[ic, jc, indices...]
            for kc = 1:NC3
                C[ic, jc, indices...] += α * A[ic, kc, indices_p...] * B[jc, kc, indices...]'
            end
        end
    end
end

@inline function kernel_Dmatrix_mul_shiftABdag!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, dindexer, shift, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw
    @inbounds begin
        indices_p = shiftindices(indices, shift)


        a11 = α * A[1, 1, indices_p...]
        a21 = α * A[2, 1, indices_p...]
        #a31 = α * A[3, 1, indices_p...]
        a12 = α * A[1, 2, indices_p...]
        a22 = α * A[2, 2, indices_p...]
        #a32 = α * A[3, 2, indices_p...]
        #a13 = α * A[1, 3, indices_p...]
        #a23 = α * A[2, 3, indices_p...]
        #a33 = α * A[3, 3, indices_p...]
        b11 = B[1, 1, indices...]'
        b12 = B[2, 1, indices...]'
        #b13 = B[3, 1, indices...]'

        b21 = B[1, 2, indices...]'
        b22 = B[2, 2, indices...]'
        #b23 = B[3, 2, indices...]'

        #b31 = B[1, 3, indices...]'
        #b32 = B[2, 3, indices...]'
        #b33 = B[3, 3, indices...]'

        C[1, 1, indices...] = β * C[1, 1, indices...] + a11 * b11 + a12 * b21 #+ a13 * b31
        C[2, 1, indices...] = β * C[2, 1, indices...] + a21 * b11 + a22 * b21 #+ a23 * b31
        #C[3, 1, indices...] = β * C[3, 1, indices...] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = β * C[1, 2, indices...] + a11 * b12 + a12 * b22 #+ a13 * b32
        C[2, 2, indices...] = β * C[2, 2, indices...] + a21 * b12 + a22 * b22 #+ a23 * b32
        #C[3, 2, indices...] = β * C[3, 2, indices...] + a31 * b12 + a32 * b22 + a33 * b32
        #C[1, 3, indices...] = β * C[1, 3, indices...] + a11 * b13 + a12 * b23 + a13 * b33
        #C[2, 3, indices...] = β * C[2, 3, indices...] + a21 * b13 + a22 * b23 + a23 * b33
        #C[3, 3, indices...] = β * C[3, 3, indices...] + a31 * b13 + a32 * b23 + a33 * b33
    end

end


@inline function kernel_Dmatrix_mul_shiftABdag!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer, shift, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw
    @inbounds begin
        indices_p = shiftindices(indices, shift)


        a11 = α * A[1, 1, indices_p...]
        a21 = α * A[2, 1, indices_p...]
        a31 = α * A[3, 1, indices_p...]
        a12 = α * A[1, 2, indices_p...]
        a22 = α * A[2, 2, indices_p...]
        a32 = α * A[3, 2, indices_p...]
        a13 = α * A[1, 3, indices_p...]
        a23 = α * A[2, 3, indices_p...]
        a33 = α * A[3, 3, indices_p...]
        b11 = B[1, 1, indices...]'
        b12 = B[2, 1, indices...]'
        b13 = B[3, 1, indices...]'

        b21 = B[1, 2, indices...]'
        b22 = B[2, 2, indices...]'
        b23 = B[3, 2, indices...]'

        b31 = B[1, 3, indices...]'
        b32 = B[2, 3, indices...]'
        b33 = B[3, 3, indices...]'

        C[1, 1, indices...] = β * C[1, 1, indices...] + a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = β * C[2, 1, indices...] + a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = β * C[3, 1, indices...] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = β * C[1, 2, indices...] + a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = β * C[2, 2, indices...] + a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = β * C[3, 2, indices...] + a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = β * C[1, 3, indices...] + a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = β * C[2, 3, indices...] + a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = β * C[3, 3, indices...] + a31 * b13 + a32 * b23 + a33 * b33
    end

end


#C = shiftedA'*B'
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Shifted_Lattice{L1,D}}, B::Adjoint_Lattice{L2}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    shift = get_shift(A)
    mul_shiftAdag_Bdag!(C, A, B, shift)
    #set_halo!(C)
end

function mul_shiftAdag_Bdag!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Shifted_Lattice{L1,D}}, B::Adjoint_Lattice{L2}, shift) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftAdagBdag!, C.A, A.data.data.A, B.data.A, Val(NC1),
        Val(NC2), Val(NC3), Val(nw), C.indexer, shift
    )
    #set_halo!(C)
end



@inline function kernel_Dmatrix_mul_shiftAdagBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = 0
            for kc = 1:NC3
                C[ic, jc, indices...] += A[kc, ic, indices_p...]' * B[jc, kc, indices...]'
            end
        end
    end
end



#C = α*shiftedA'*B'+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Shifted_Lattice{L1,D}}, B::Adjoint_Lattice{L2},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    shift = get_shift(A)
    mul_shiftAdag_Bdag!(C, A, B, shift, α, β)
    #set_halo!(C)
end

function mul_shiftAdag_Bdag!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Shifted_Lattice{L1,D}}, B::Adjoint_Lattice{L2}, shift,
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftAdagBdag!, C.A, A.data.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift, α::S, β::S
    )
    #set_halo!(C)
end



@inline function kernel_Dmatrix_mul_shiftAdagBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = β * C[ic, jc, indices...]
            for kc = 1:NC3
                C[ic, jc, indices...] += α * A[kc, ic, indices_p...]' * B[jc, kc, indices...]'
            end
        end
    end
end

@inline function kernel_Dmatrix_mul_shiftAdagBdag!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, dindexer, shift, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw
    @inbounds begin
        indices_p = shiftindices(indices, shift)


        a11 = α * A[1, 1, indices_p...]'
        a12 = α * A[2, 1, indices_p...]'
        #a13 = α * A[3, 1, indices_p...]'
        a21 = α * A[1, 2, indices_p...]'
        a22 = α * A[2, 2, indices_p...]'
        #a23 = α * A[3, 2, indices_p...]'
        #a31 = α * A[1, 3, indices_p...]'
        #a32 = α * A[2, 3, indices_p...]'
        #a33 = α * A[3, 3, indices_p...]'

        b11 = B[1, 1, indices...]'
        b12 = B[2, 1, indices...]'
        #b13 = B[3, 1, indices...]'

        b21 = B[1, 2, indices...]'
        b22 = B[2, 2, indices...]'
        #b23 = B[3, 2, indices...]'

        #b31 = B[1, 3, indices...]'
        #b32 = B[2, 3, indices...]'
        #b33 = B[3, 3, indices...]'

        C[1, 1, indices...] = β * C[1, 1, indices...] + a11 * b11 + a12 * b21 #+ a13 * b31
        C[2, 1, indices...] = β * C[2, 1, indices...] + a21 * b11 + a22 * b21 #+ a23 * b31
        #C[3, 1, indices...] = β * C[3, 1, indices...] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = β * C[1, 2, indices...] + a11 * b12 + a12 * b22 #+ a13 * b32
        C[2, 2, indices...] = β * C[2, 2, indices...] + a21 * b12 + a22 * b22 #+ a23 * b32
        #C[3, 2, indices...] = β * C[3, 2, indices...] + a31 * b12 + a32 * b22 + a33 * b32
        #C[1, 3, indices...] = β * C[1, 3, indices...] + a11 * b13 + a12 * b23 + a13 * b33
        #C[2, 3, indices...] = β * C[2, 3, indices...] + a21 * b13 + a22 * b23 + a23 * b33
        #C[3, 3, indices...] = β * C[3, 3, indices...] + a31 * b13 + a32 * b23 + a33 * b33
    end

end


@inline function kernel_Dmatrix_mul_shiftAdagBdag!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer, shift, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw
    @inbounds begin
        indices_p = shiftindices(indices, shift)


        a11 = α * A[1, 1, indices_p...]'
        a12 = α * A[2, 1, indices_p...]'
        a13 = α * A[3, 1, indices_p...]'
        a21 = α * A[1, 2, indices_p...]'
        a22 = α * A[2, 2, indices_p...]'
        a23 = α * A[3, 2, indices_p...]'
        a31 = α * A[1, 3, indices_p...]'
        a32 = α * A[2, 3, indices_p...]'
        a33 = α * A[3, 3, indices_p...]'

        b11 = B[1, 1, indices...]'
        b12 = B[2, 1, indices...]'
        b13 = B[3, 1, indices...]'

        b21 = B[1, 2, indices...]'
        b22 = B[2, 2, indices...]'
        b23 = B[3, 2, indices...]'

        b31 = B[1, 3, indices...]'
        b32 = B[2, 3, indices...]'
        b33 = B[3, 3, indices...]'

        C[1, 1, indices...] = β * C[1, 1, indices...] + a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = β * C[2, 1, indices...] + a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = β * C[3, 1, indices...] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = β * C[1, 2, indices...] + a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = β * C[2, 2, indices...] + a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = β * C[3, 2, indices...] + a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = β * C[1, 3, indices...] + a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = β * C[2, 3, indices...] + a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = β * C[3, 3, indices...] + a31 * b13 + a32 * b23 + a33 * b33
    end

end


#C = A'*shiftedB
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{L1}, B::Shifted_Lattice{L2,D}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}


    shift = get_shift(B)
    mul_Adag_shiftB!(C, A, B.data, shift)
    #set_halo!(C)
end

function mul_Adag_shiftB!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{L1}, B::L, shift) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_AdagshiftB!, C.A, A.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift
    )
    #set_halo!(C)
end


@inline function kernel_Dmatrix_mul_AdagshiftB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = 0
            for kc = 1:NC3
                C[ic, jc, indices...] += A[kc, ic, indices...]' * B[kc, jc, indices_p...]
            end
        end
    end
end

@inline function kernel_Dmatrix_mul_AdagshiftB!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, dindexer, shift) where {nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds begin
        indices_p = shiftindices(indices, shift)

        a11 = A[1, 1, indices...]'
        a21 = A[2, 1, indices...]'
        a12 = A[1, 2, indices...]'
        a22 = A[2, 2, indices...]'

        b11 = B[1, 1, indices_p...]
        b21 = B[2, 1, indices_p...]
        b12 = B[1, 2, indices_p...]
        b22 = B[2, 2, indices_p...]

        C[1, 1, indices...] = a11 * b11 + a21 * b21
        C[2, 1, indices...] = a12 * b11 + a22 * b21
        C[1, 2, indices...] = a11 * b12 + a21 * b22
        C[2, 2, indices...] = a12 * b12 + a22 * b22
    end
end

@inline function kernel_Dmatrix_mul_AdagshiftB!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer, shift) where {nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds begin
        indices_p = shiftindices(indices, shift)

        a11 = A[1, 1, indices...]'
        a21 = A[2, 1, indices...]'
        a31 = A[3, 1, indices...]'
        a12 = A[1, 2, indices...]'
        a22 = A[2, 2, indices...]'
        a32 = A[3, 2, indices...]'
        a13 = A[1, 3, indices...]'
        a23 = A[2, 3, indices...]'
        a33 = A[3, 3, indices...]'

        b11 = B[1, 1, indices_p...]
        b21 = B[2, 1, indices_p...]
        b31 = B[3, 1, indices_p...]
        b12 = B[1, 2, indices_p...]
        b22 = B[2, 2, indices_p...]
        b32 = B[3, 2, indices_p...]
        b13 = B[1, 3, indices_p...]
        b23 = B[2, 3, indices_p...]
        b33 = B[3, 3, indices_p...]

        C[1, 1, indices...] = a11 * b11 + a21 * b21 + a31 * b31
        C[2, 1, indices...] = a12 * b11 + a22 * b21 + a32 * b31
        C[3, 1, indices...] = a13 * b11 + a23 * b21 + a33 * b31
        C[1, 2, indices...] = a11 * b12 + a21 * b22 + a31 * b32
        C[2, 2, indices...] = a12 * b12 + a22 * b22 + a32 * b32
        C[3, 2, indices...] = a13 * b12 + a23 * b22 + a33 * b32
        C[1, 3, indices...] = a11 * b13 + a21 * b23 + a31 * b33
        C[2, 3, indices...] = a12 * b13 + a22 * b23 + a32 * b33
        C[3, 3, indices...] = a13 * b13 + a23 * b23 + a33 * b33
    end
end

#C = α*A'*shiftedB+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{L1}, B::Shifted_Lattice{L2,D},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}

    shift = get_shift(B)
    mul_Adag_shiftB!(C, A, B.data, shift, α, β)
    #set_halo!(C)
end

function mul_Adag_shiftB!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{L1}, B::L, shift,
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_AdagshiftB!, C.A, A.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_Dmatrix_mul_AdagshiftB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = β * C[ic, jc, indices...]
            for kc = 1:NC3
                C[ic, jc, indices...] += α * A[kc, ic, indices...]' * B[kc, jc, indices_p...]
            end
        end
    end
end

@inline function kernel_Dmatrix_mul_AdagshiftB!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, dindexer, shift, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    @inbounds begin
        indices_p = shiftindices(indices, shift)

        a11 = A[1, 1, indices...]'
        a21 = A[2, 1, indices...]'
        a12 = A[1, 2, indices...]'
        a22 = A[2, 2, indices...]'

        b11 = B[1, 1, indices_p...]
        b21 = B[2, 1, indices_p...]
        b12 = B[1, 2, indices_p...]
        b22 = B[2, 2, indices_p...]

        c11 = a11 * b11 + a21 * b21
        c21 = a12 * b11 + a22 * b21
        c12 = a11 * b12 + a21 * b22
        c22 = a12 * b12 + a22 * b22

        if iszero(β)
            C[1, 1, indices...] = α * c11
            C[2, 1, indices...] = α * c21
            C[1, 2, indices...] = α * c12
            C[2, 2, indices...] = α * c22
        else
            C[1, 1, indices...] = α * c11 + β * C[1, 1, indices...]
            C[2, 1, indices...] = α * c21 + β * C[2, 1, indices...]
            C[1, 2, indices...] = α * c12 + β * C[1, 2, indices...]
            C[2, 2, indices...] = α * c22 + β * C[2, 2, indices...]
        end
    end
end

@inline function kernel_Dmatrix_mul_AdagshiftB!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer, shift, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    @inbounds begin
        indices_p = shiftindices(indices, shift)

        a11 = A[1, 1, indices...]'
        a21 = A[2, 1, indices...]'
        a31 = A[3, 1, indices...]'
        a12 = A[1, 2, indices...]'
        a22 = A[2, 2, indices...]'
        a32 = A[3, 2, indices...]'
        a13 = A[1, 3, indices...]'
        a23 = A[2, 3, indices...]'
        a33 = A[3, 3, indices...]'

        b11 = B[1, 1, indices_p...]
        b21 = B[2, 1, indices_p...]
        b31 = B[3, 1, indices_p...]
        b12 = B[1, 2, indices_p...]
        b22 = B[2, 2, indices_p...]
        b32 = B[3, 2, indices_p...]
        b13 = B[1, 3, indices_p...]
        b23 = B[2, 3, indices_p...]
        b33 = B[3, 3, indices_p...]

        c11 = a11 * b11 + a21 * b21 + a31 * b31
        c21 = a12 * b11 + a22 * b21 + a32 * b31
        c31 = a13 * b11 + a23 * b21 + a33 * b31
        c12 = a11 * b12 + a21 * b22 + a31 * b32
        c22 = a12 * b12 + a22 * b22 + a32 * b32
        c32 = a13 * b12 + a23 * b22 + a33 * b32
        c13 = a11 * b13 + a21 * b23 + a31 * b33
        c23 = a12 * b13 + a22 * b23 + a32 * b33
        c33 = a13 * b13 + a23 * b23 + a33 * b33

        if iszero(β)
            C[1, 1, indices...] = α * c11
            C[2, 1, indices...] = α * c21
            C[3, 1, indices...] = α * c31
            C[1, 2, indices...] = α * c12
            C[2, 2, indices...] = α * c22
            C[3, 2, indices...] = α * c32
            C[1, 3, indices...] = α * c13
            C[2, 3, indices...] = α * c23
            C[3, 3, indices...] = α * c33
        else
            C[1, 1, indices...] = α * c11 + β * C[1, 1, indices...]
            C[2, 1, indices...] = α * c21 + β * C[2, 1, indices...]
            C[3, 1, indices...] = α * c31 + β * C[3, 1, indices...]
            C[1, 2, indices...] = α * c12 + β * C[1, 2, indices...]
            C[2, 2, indices...] = α * c22 + β * C[2, 2, indices...]
            C[3, 2, indices...] = α * c32 + β * C[3, 2, indices...]
            C[1, 3, indices...] = α * c13 + β * C[1, 3, indices...]
            C[2, 3, indices...] = α * c23 + β * C[2, 3, indices...]
            C[3, 3, indices...] = α * c33 + β * C[3, 3, indices...]
        end
    end
end

#C = A*shiftedB'
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::Adjoint_Lattice{Shifted_Lattice{L,D}}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    shift = get_shift(B)
    mul_A_shiftBdag!(C, A, B.data.data, shift)
    #set_halo!(C)
end

function mul_simple!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::Adjoint_Lattice{Shifted_Lattice{L,D}}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}
    shift = get_shift(B)
    mul_simple_A_shiftBdag!(C, A, B.data.data, shift)
end

function mul_A_shiftBdag!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::L, shift) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_AshiftBdag!, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift
    )
    #set_halo!(C)
end

function mul_simple_A_shiftBdag!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::L, shift) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}
    @inbounds for i in 1:prod(C.PN)
        kernel_Dmatrix_mul_AshiftBdag!(i, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift)
    end
end


@inline function kernel_Dmatrix_mul_AshiftBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw
    indices_p = shiftindices(indices, shift)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            #C[ic, jc, indices...] = 0
            C[ic, jc, indices...] = zero(eltype(C))
        end
        for kc = 1:NC3
            b = conj(B[jc, kc, indices_p...])
            for ic = 1:NC1
                C[ic, jc, indices...] += A[ic, kc, indices...] * b
            end
        end
    end
end

@inline function kernel_Dmatrix_mul_AshiftBdag!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer, shift) where {nw}
    indices = delinearize(dindexer, i, nw)

    @inbounds begin
        indices_p = shiftindices(indices, shift)

        a11 = A[1, 1, indices...]
        a21 = A[2, 1, indices...]
        a31 = A[3, 1, indices...]
        a12 = A[1, 2, indices...]
        a22 = A[2, 2, indices...]
        a32 = A[3, 2, indices...]
        a13 = A[1, 3, indices...]
        a23 = A[2, 3, indices...]
        a33 = A[3, 3, indices...]

        b11 = B[1, 1, indices_p...]'
        b21 = B[1, 2, indices_p...]'
        b31 = B[1, 3, indices_p...]'
        C[1, 1, indices...] = a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = a31 * b11 + a32 * b21 + a33 * b31

        # ----  j=2 ----
        b12 = B[2, 1, indices_p...]'
        b22 = B[2, 2, indices_p...]'
        b32 = B[2, 3, indices_p...]'
        C[1, 2, indices...] = a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = a31 * b12 + a32 * b22 + a33 * b32

        # ----  j=3 ----
        b13 = B[3, 1, indices_p...]'
        b23 = B[3, 2, indices_p...]'
        b33 = B[3, 3, indices_p...]'
        C[1, 3, indices...] = a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = a31 * b13 + a32 * b23 + a33 * b33

    end
end

@inline function kernel_Dmatrix_mul_AshiftBdag!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, dindexer, shift) where {nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds begin
        indices_p = shiftindices(indices, shift)

        a11 = A[1, 1, indices...]
        a21 = A[2, 1, indices...]
        a12 = A[1, 2, indices...]
        a22 = A[2, 2, indices...]

        b11 = B[1, 1, indices_p...]'
        b21 = B[1, 2, indices_p...]'
        C[1, 1, indices...] = a11 * b11 + a12 * b21
        C[2, 1, indices...] = a21 * b11 + a22 * b21

        b12 = B[2, 1, indices_p...]'
        b22 = B[2, 2, indices_p...]'
        C[1, 2, indices...] = a11 * b12 + a12 * b22
        C[2, 2, indices...] = a21 * b12 + a22 * b22
    end
end



#C = α*A*shiftedB'+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::Adjoint_Lattice{Shifted_Lattice{L,D}},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    shift = get_shift(B)
    mul_A_shiftBdag!(C, A, B.data.data, shift, α, β)
    #set_halo!(C)
end

function mul_A_shiftBdag!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::L, shift,
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_AshiftBdag!, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift, α::S, β::S
    )
    #set_halo!(C)
end

function mul_simple!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::Adjoint_Lattice{Shifted_Lattice{L,D}},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}
    shift = get_shift(B)
    mul_simple_A_shiftBdag!(C, A, B.data.data, shift, α, β)
end

function mul_simple_A_shiftBdag!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::L, shift,
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}
    αin = T1(α)
    βin = T1(β)
    @inbounds for i in 1:prod(C.PN)
        kernel_Dmatrix_mul_AshiftBdag!(i, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift, αin, βin)
    end
end


@inline function kernel_Dmatrix_mul_AshiftBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = β * C[ic, jc, indices...]
            for kc = 1:NC3
                C[ic, jc, indices...] += α * A[ic, kc, indices...] * B[jc, kc, indices_p...]'
            end
        end
    end
end


@inline function kernel_Dmatrix_mul_AshiftBdag!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer, shift, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        indices_p = shiftindices(indices, shift)

        a11 = A[1, 1, indices...]
        a21 = A[2, 1, indices...]
        a31 = A[3, 1, indices...]
        a12 = A[1, 2, indices...]
        a22 = A[2, 2, indices...]
        a32 = A[3, 2, indices...]
        a13 = A[1, 3, indices...]
        a23 = A[2, 3, indices...]
        a33 = A[3, 3, indices...]

        b11 = B[1, 1, indices_p...]'
        b21 = B[1, 2, indices_p...]'
        b31 = B[1, 3, indices_p...]'
        c11 = a11 * b11 + a12 * b21 + a13 * b31
        c21 = a21 * b11 + a22 * b21 + a23 * b31
        c31 = a31 * b11 + a32 * b21 + a33 * b31

        # ----  j=2 ----
        b12 = B[2, 1, indices_p...]'
        b22 = B[2, 2, indices_p...]'
        b32 = B[2, 3, indices_p...]'
        c12 = a11 * b12 + a12 * b22 + a13 * b32
        c22 = a21 * b12 + a22 * b22 + a23 * b32
        c32 = a31 * b12 + a32 * b22 + a33 * b32

        # ----  j=3 ----
        b13 = B[3, 1, indices_p...]'
        b23 = B[3, 2, indices_p...]'
        b33 = B[3, 3, indices_p...]'
        c13 = a11 * b13 + a12 * b23 + a13 * b33
        c23 = a21 * b13 + a22 * b23 + a23 * b33
        c33 = a31 * b13 + a32 * b23 + a33 * b33

        if iszero(β)
            C[1, 1, indices...] = α * c11
            C[2, 1, indices...] = α * c21
            C[3, 1, indices...] = α * c31
            C[1, 2, indices...] = α * c12
            C[2, 2, indices...] = α * c22
            C[3, 2, indices...] = α * c32
            C[1, 3, indices...] = α * c13
            C[2, 3, indices...] = α * c23
            C[3, 3, indices...] = α * c33
        else
            C[1, 1, indices...] = α * c11 + β * C[1, 1, indices...]
            C[2, 1, indices...] = α * c21 + β * C[2, 1, indices...]
            C[3, 1, indices...] = α * c31 + β * C[3, 1, indices...]
            C[1, 2, indices...] = α * c12 + β * C[1, 2, indices...]
            C[2, 2, indices...] = α * c22 + β * C[2, 2, indices...]
            C[3, 2, indices...] = α * c32 + β * C[3, 2, indices...]
            C[1, 3, indices...] = α * c13 + β * C[1, 3, indices...]
            C[2, 3, indices...] = α * c23 + β * C[2, 3, indices...]
            C[3, 3, indices...] = α * c33 + β * C[3, 3, indices...]
        end

        #=
        a11 = α * A[1, 1, indices...]
        a21 = α * A[2, 1, indices...]
        a31 = α * A[3, 1, indices...]
        a12 = α * A[1, 2, indices...]
        a22 = α * A[2, 2, indices...]
        a32 = α * A[3, 2, indices...]
        a13 = α * A[1, 3, indices...]
        a23 = α * A[2, 3, indices...]
        a33 = α * A[3, 3, indices...]
        b11 = conj(B[1, 1, indices_p...])
        b12 = conj(B[2, 1, indices_p...])
        b13 = conj(B[3, 1, indices_p...])

        b21 = conj(B[1, 2, indices_p...])
        b22 = conj(B[2, 2, indices_p...])
        b23 = conj(B[3, 2, indices_p...])

        b31 = conj(B[1, 3, indices_p...])
        b32 = conj(B[2, 3, indices_p...])
        b33 = conj(B[3, 3, indices_p...])

        C[1, 1, indices...] = β * C[1, 1, indices...] + a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = β * C[2, 1, indices...] + a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = β * C[3, 1, indices...] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = β * C[1, 2, indices...] + a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = β * C[2, 2, indices...] + a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = β * C[3, 2, indices...] + a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = β * C[1, 3, indices...] + a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = β * C[2, 3, indices...] + a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = β * C[3, 3, indices...] + a31 * b13 + a32 * b23 + a33 * b33
        =#
    end

end




#C = A'*shiftedB'
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{L1}, B::Adjoint_Lattice{Shifted_Lattice{L2,D}}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    shift = get_shift(B)
    mul_Adag_shiftBdag!(C, A, B.data.data, shift)
    #set_halo!(C)
end

function mul_Adag_shiftBdag!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{L1}, B::L, shift) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_AdagshiftBdag!, C.A, A.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift
    )
    #set_halo!(C)
end


@inline function kernel_Dmatrix_mul_AdagshiftBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = 0
            for kc = 1:NC3
                C[ic, jc, indices...] += A[kc, ic, indices...]' * B[jc, kc, indices_p...]'
            end
        end
    end
end

@inline function kernel_Dmatrix_mul_AdagshiftBdag!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, dindexer, shift) where {nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds begin
        indices_p = shiftindices(indices, shift)

        a11 = A[1, 1, indices...]'
        a21 = A[2, 1, indices...]'
        a12 = A[1, 2, indices...]'
        a22 = A[2, 2, indices...]'

        b11 = B[1, 1, indices_p...]'
        b12 = B[1, 2, indices_p...]'
        b21 = B[2, 1, indices_p...]'
        b22 = B[2, 2, indices_p...]'

        C[1, 1, indices...] = a11 * b11 + a21 * b12
        C[2, 1, indices...] = a12 * b11 + a22 * b12
        C[1, 2, indices...] = a11 * b21 + a21 * b22
        C[2, 2, indices...] = a12 * b21 + a22 * b22
    end
end

@inline function kernel_Dmatrix_mul_AdagshiftBdag!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer, shift) where {nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds begin
        indices_p = shiftindices(indices, shift)

        a11 = A[1, 1, indices...]'
        a21 = A[2, 1, indices...]'
        a31 = A[3, 1, indices...]'
        a12 = A[1, 2, indices...]'
        a22 = A[2, 2, indices...]'
        a32 = A[3, 2, indices...]'
        a13 = A[1, 3, indices...]'
        a23 = A[2, 3, indices...]'
        a33 = A[3, 3, indices...]'

        b11 = B[1, 1, indices_p...]'
        b12 = B[1, 2, indices_p...]'
        b13 = B[1, 3, indices_p...]'
        b21 = B[2, 1, indices_p...]'
        b22 = B[2, 2, indices_p...]'
        b23 = B[2, 3, indices_p...]'
        b31 = B[3, 1, indices_p...]'
        b32 = B[3, 2, indices_p...]'
        b33 = B[3, 3, indices_p...]'

        C[1, 1, indices...] = a11 * b11 + a21 * b12 + a31 * b13
        C[2, 1, indices...] = a12 * b11 + a22 * b12 + a32 * b13
        C[3, 1, indices...] = a13 * b11 + a23 * b12 + a33 * b13
        C[1, 2, indices...] = a11 * b21 + a21 * b22 + a31 * b23
        C[2, 2, indices...] = a12 * b21 + a22 * b22 + a32 * b23
        C[3, 2, indices...] = a13 * b21 + a23 * b22 + a33 * b23
        C[1, 3, indices...] = a11 * b31 + a21 * b32 + a31 * b33
        C[2, 3, indices...] = a12 * b31 + a22 * b32 + a32 * b33
        C[3, 3, indices...] = a13 * b31 + a23 * b32 + a33 * b33
    end
end

@inline function kernel_Dmatrix_mul_AshiftBdag!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, dindexer, shift, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    @inbounds begin
        indices_p = shiftindices(indices, shift)

        a11 = A[1, 1, indices...]
        a21 = A[2, 1, indices...]
        a12 = A[1, 2, indices...]
        a22 = A[2, 2, indices...]

        b11 = B[1, 1, indices_p...]'
        b21 = B[1, 2, indices_p...]'
        c11 = a11 * b11 + a12 * b21
        c21 = a21 * b11 + a22 * b21

        b12 = B[2, 1, indices_p...]'
        b22 = B[2, 2, indices_p...]'
        c12 = a11 * b12 + a12 * b22
        c22 = a21 * b12 + a22 * b22

        if iszero(β)
            C[1, 1, indices...] = α * c11
            C[2, 1, indices...] = α * c21
            C[1, 2, indices...] = α * c12
            C[2, 2, indices...] = α * c22
        else
            C[1, 1, indices...] = α * c11 + β * C[1, 1, indices...]
            C[2, 1, indices...] = α * c21 + β * C[2, 1, indices...]
            C[1, 2, indices...] = α * c12 + β * C[1, 2, indices...]
            C[2, 2, indices...] = α * c22 + β * C[2, 2, indices...]
        end
    end
end
#C = α*A'*shiftedB'+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{L1}, B::Adjoint_Lattice{Shifted_Lattice{L2,D}},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    shift = get_shift(B)
    mul_Adag_shiftBdag!(C, A, B.data.data, shift, α, β)
    #set_halo!(C)
end

function mul_Adag_shiftBdag!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{L1}, B::L, shift,
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_AdagshiftBdag!, C.A, A.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_Dmatrix_mul_AdagshiftBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = β * C[ic, jc, indices...]
            for kc = 1:NC3
                C[ic, jc, indices...] += α * A[kc, ic, indices...]' * B[jc, kc, indices_p...]'
            end
        end
    end
end

@inline function kernel_Dmatrix_mul_AdagshiftBdag!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, dindexer, shift, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    @inbounds begin
        indices_p = shiftindices(indices, shift)

        a11 = A[1, 1, indices...]'
        a21 = A[2, 1, indices...]'
        a12 = A[1, 2, indices...]'
        a22 = A[2, 2, indices...]'

        b11 = B[1, 1, indices_p...]'
        b12 = B[1, 2, indices_p...]'
        b21 = B[2, 1, indices_p...]'
        b22 = B[2, 2, indices_p...]'

        c11 = a11 * b11 + a21 * b12
        c21 = a12 * b11 + a22 * b12
        c12 = a11 * b21 + a21 * b22
        c22 = a12 * b21 + a22 * b22

        if iszero(β)
            C[1, 1, indices...] = α * c11
            C[2, 1, indices...] = α * c21
            C[1, 2, indices...] = α * c12
            C[2, 2, indices...] = α * c22
        else
            C[1, 1, indices...] = α * c11 + β * C[1, 1, indices...]
            C[2, 1, indices...] = α * c21 + β * C[2, 1, indices...]
            C[1, 2, indices...] = α * c12 + β * C[1, 2, indices...]
            C[2, 2, indices...] = α * c22 + β * C[2, 2, indices...]
        end
    end
end

@inline function kernel_Dmatrix_mul_AdagshiftBdag!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer, shift, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    @inbounds begin
        indices_p = shiftindices(indices, shift)

        a11 = A[1, 1, indices...]'
        a21 = A[2, 1, indices...]'
        a31 = A[3, 1, indices...]'
        a12 = A[1, 2, indices...]'
        a22 = A[2, 2, indices...]'
        a32 = A[3, 2, indices...]'
        a13 = A[1, 3, indices...]'
        a23 = A[2, 3, indices...]'
        a33 = A[3, 3, indices...]'

        b11 = B[1, 1, indices_p...]'
        b12 = B[1, 2, indices_p...]'
        b13 = B[1, 3, indices_p...]'
        b21 = B[2, 1, indices_p...]'
        b22 = B[2, 2, indices_p...]'
        b23 = B[2, 3, indices_p...]'
        b31 = B[3, 1, indices_p...]'
        b32 = B[3, 2, indices_p...]'
        b33 = B[3, 3, indices_p...]'

        c11 = a11 * b11 + a21 * b12 + a31 * b13
        c21 = a12 * b11 + a22 * b12 + a32 * b13
        c31 = a13 * b11 + a23 * b12 + a33 * b13
        c12 = a11 * b21 + a21 * b22 + a31 * b23
        c22 = a12 * b21 + a22 * b22 + a32 * b23
        c32 = a13 * b21 + a23 * b22 + a33 * b23
        c13 = a11 * b31 + a21 * b32 + a31 * b33
        c23 = a12 * b31 + a22 * b32 + a32 * b33
        c33 = a13 * b31 + a23 * b32 + a33 * b33

        if iszero(β)
            C[1, 1, indices...] = α * c11
            C[2, 1, indices...] = α * c21
            C[3, 1, indices...] = α * c31
            C[1, 2, indices...] = α * c12
            C[2, 2, indices...] = α * c22
            C[3, 2, indices...] = α * c32
            C[1, 3, indices...] = α * c13
            C[2, 3, indices...] = α * c23
            C[3, 3, indices...] = α * c33
        else
            C[1, 1, indices...] = α * c11 + β * C[1, 1, indices...]
            C[2, 1, indices...] = α * c21 + β * C[2, 1, indices...]
            C[3, 1, indices...] = α * c31 + β * C[3, 1, indices...]
            C[1, 2, indices...] = α * c12 + β * C[1, 2, indices...]
            C[2, 2, indices...] = α * c22 + β * C[2, 2, indices...]
            C[3, 2, indices...] = α * c32 + β * C[3, 2, indices...]
            C[1, 3, indices...] = α * c13 + β * C[1, 3, indices...]
            C[2, 3, indices...] = α * c23 + β * C[2, 3, indices...]
            C[3, 3, indices...] = α * c33 + β * C[3, 3, indices...]
        end
    end
end



#C = shiftA*shiftedB
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Shifted_Lattice{L1,D}, B::Shifted_Lattice{L2,D}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}

    shiftA = get_shift(A)
    shiftB = get_shift(B)
    mul_shiftA_shiftB!(C, A, B.data, shiftA, shiftB)
    #set_halo!(C)
end

function mul_shiftA_shiftB!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Shifted_Lattice{L1,D}, B::L, shiftA, shiftB) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},L<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftAshiftB!, C.A, A.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shiftA, shiftB
    )
    #set_halo!(C)
end


@inline function kernel_Dmatrix_mul_shiftAshiftB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shiftA, shiftB) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_A = shiftindices(indices, shiftA)

    indices_B = shiftindices(indices, shiftB)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = 0
            for kc = 1:NC3
                C[ic, jc, indices...] += A[ic, kc, indices_A...] * B[kc, jc, indices_B...]
            end
        end
    end
end

#C = α*shiftA*shiftedB+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Shifted_Lattice{L1,D}, B::Shifted_Lattice{L2,D},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}

    shiftA = get_shift(A)
    shiftB = get_shift(B)
    mul_shiftA_shiftB!(C, A, B.data, shiftA, shiftB, α, β)
    #set_halo!(C)
end

function mul_shiftA_shiftB!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Shifted_Lattice{L1,D}, B::L, shiftA, shiftB,
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},L<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftAshiftB!, C.A, A.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shiftA, shiftB, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_Dmatrix_mul_shiftAshiftB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shiftA, shiftB, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    indices_A = shiftindices(indices, shiftA)

    indices_B = shiftindices(indices, shiftB)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = β * C[ic, jc, indices...]
            for kc = 1:NC3
                C[ic, jc, indices...] += α * A[ic, kc, indices_A...] * B[kc, jc, indices_B...]
            end
        end
    end
end


@inline function kernel_Dmatrix_mul_shiftAshiftB!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw},
    dindexer, shiftA, shiftB, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    begin
        indices_A = shiftindices(indices, shiftA)

        indices_B = shiftindices(indices, shiftB)

        a11 = α * A[1, 1, indices_A...]
        a21 = α * A[2, 1, indices_A...]
        a12 = α * A[1, 2, indices_A...]
        a22 = α * A[2, 2, indices_A...]

        b11 = B[1, 1, indices_B...]
        b21 = B[2, 1, indices_B...]
        b12 = B[1, 2, indices_B...]
        b22 = B[2, 2, indices_B...]


        C[1, 1, indices...] = β * C[1, 1, indices...] + a11 * b11 + a12 * b21
        C[2, 1, indices...] = β * C[2, 1, indices...] + a21 * b11 + a22 * b21
        C[1, 2, indices...] = β * C[1, 2, indices...] + a11 * b12 + a12 * b22
        C[2, 2, indices...] = β * C[2, 2, indices...] + a21 * b12 + a22 * b22
    end



end


@inline function kernel_Dmatrix_mul_shiftAshiftB!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw},
    dindexer, shiftA, shiftB, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        indices_A = shiftindices(indices, shiftA)

        indices_B = shiftindices(indices, shiftB)

        a11 = α * A[1, 1, indices_A...]
        a21 = α * A[2, 1, indices_A...]
        a31 = α * A[3, 1, indices_A...]
        a12 = α * A[1, 2, indices_A...]
        a22 = α * A[2, 2, indices_A...]
        a32 = α * A[3, 2, indices_A...]
        a13 = α * A[1, 3, indices_A...]
        a23 = α * A[2, 3, indices_A...]
        a33 = α * A[3, 3, indices_A...]

        b11 = B[1, 1, indices_B...]
        b21 = B[2, 1, indices_B...]
        b31 = B[3, 1, indices_B...]

        b12 = B[1, 2, indices_B...]
        b22 = B[2, 2, indices_B...]
        b32 = B[3, 2, indices_B...]


        b13 = B[1, 3, indices_B...]
        b23 = B[2, 3, indices_B...]
        b33 = B[3, 3, indices_B...]


        C[1, 1, indices...] = β * C[1, 1, indices...] + a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = β * C[2, 1, indices...] + a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = β * C[3, 1, indices...] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = β * C[1, 2, indices...] + a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = β * C[2, 2, indices...] + a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = β * C[3, 2, indices...] + a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = β * C[1, 3, indices...] + a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = β * C[2, 3, indices...] + a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = β * C[3, 3, indices...] + a31 * b13 + a32 * b23 + a33 * b33

    end



end


#C = shiftA'*shiftedB
#C[i,j] = A[k,i]'*B[k,j]
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Shifted_Lattice{L1,D}}, B::Shifted_Lattice{L2,D}) where {D,T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}

    shiftA = get_shift(A)
    shiftB = get_shift(B)
    mul_shiftAdag_shiftB!(C, A, B.data, shiftA, shiftB)
    #set_halo!(C)
end

function mul_shiftAdag_shiftB!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Shifted_Lattice{L1,D}}, B::L, shiftA, shiftB) where {D,T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftAdagshiftB!, C.A, A.data.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shiftA, shiftB
    )
    #set_halo!(C)
end


@inline function kernel_Dmatrix_mul_shiftAdagshiftB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shiftA, shiftB) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_A = shiftindices(indices, shiftA)

    indices_B = shiftindices(indices, shiftB)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = 0
            for kc = 1:NC3
                C[ic, jc, indices...] += A[kc, ic, indices_A...]' * B[kc, jc, indices_B...]
            end
        end
    end
end



#C = α*shiftA'*shiftedB+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Shifted_Lattice{L1,D}}, B::Shifted_Lattice{L2,D},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}

    shiftA = get_shift(A)
    shiftB = get_shift(B)
    mul_shiftAdag_shiftB!(C, A, B.data, shiftA, shiftB, α, β)
    #set_halo!(C)
end

function mul_shiftAdag_shiftB!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Shifted_Lattice{L1,D}}, B::L, shiftA, shiftB,
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftAdagshiftB!, C.A, A.data.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shiftA, shiftB, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_Dmatrix_mul_shiftAdagshiftB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shiftA, shiftB, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    indices_A = shiftindices(indices, shiftA)

    indices_B = shiftindices(indices, shiftB)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = β * C[ic, jc, indices...]
            for kc = 1:NC3
                C[ic, jc, indices...] += α * A[kc, ic, indices_A...]' * B[kc, jc, indices_B...]
            end
        end
    end
end

@inline function kernel_Dmatrix_mul_shiftAdagshiftB!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw},
    dindexer, shiftA, shiftB, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    begin
        indices_A = shiftindices(indices, shiftA)

        indices_B = shiftindices(indices, shiftB)

        a11 = α * A[1, 1, indices_A...]'
        a12 = α * A[2, 1, indices_A...]'
        a21 = α * A[1, 2, indices_A...]'
        a22 = α * A[2, 2, indices_A...]'

        b11 = B[1, 1, indices_B...]
        b21 = B[2, 1, indices_B...]
        b12 = B[1, 2, indices_B...]
        b22 = B[2, 2, indices_B...]


        C[1, 1, indices...] = β * C[1, 1, indices...] + a11 * b11 + a12 * b21
        C[2, 1, indices...] = β * C[2, 1, indices...] + a21 * b11 + a22 * b21
        C[1, 2, indices...] = β * C[1, 2, indices...] + a11 * b12 + a12 * b22
        C[2, 2, indices...] = β * C[2, 2, indices...] + a21 * b12 + a22 * b22
    end



end

@inline function kernel_Dmatrix_mul_shiftAdagshiftB!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw},
    dindexer, shiftA, shiftB) where {nw}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        indices_A = shiftindices(indices, shiftA)

        indices_B = shiftindices(indices, shiftB)

        a11 = A[1, 1, indices_A...]'
        a12 = A[2, 1, indices_A...]'
        a13 = A[3, 1, indices_A...]'
        a21 = A[1, 2, indices_A...]'
        a22 = A[2, 2, indices_A...]'
        a23 = A[3, 2, indices_A...]'
        a31 = A[1, 3, indices_A...]'
        a32 = A[2, 3, indices_A...]'
        a33 = A[3, 3, indices_A...]'

        b11 = B[1, 1, indices_B...]
        b21 = B[2, 1, indices_B...]
        b31 = B[3, 1, indices_B...]

        b12 = B[1, 2, indices_B...]
        b22 = B[2, 2, indices_B...]
        b32 = B[3, 2, indices_B...]


        b13 = B[1, 3, indices_B...]
        b23 = B[2, 3, indices_B...]
        b33 = B[3, 3, indices_B...]


        C[1, 1, indices...] = a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = a31 * b13 + a32 * b23 + a33 * b33

    end



end

@inline function kernel_Dmatrix_mul_shiftAdagshiftB!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw},
    dindexer, shiftA, shiftB, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        indices_A = shiftindices(indices, shiftA)

        indices_B = shiftindices(indices, shiftB)

        a11 = α * A[1, 1, indices_A...]'
        a12 = α * A[2, 1, indices_A...]'
        a13 = α * A[3, 1, indices_A...]'
        a21 = α * A[1, 2, indices_A...]'
        a22 = α * A[2, 2, indices_A...]'
        a23 = α * A[3, 2, indices_A...]'
        a31 = α * A[1, 3, indices_A...]'
        a32 = α * A[2, 3, indices_A...]'
        a33 = α * A[3, 3, indices_A...]'

        b11 = B[1, 1, indices_B...]
        b21 = B[2, 1, indices_B...]
        b31 = B[3, 1, indices_B...]

        b12 = B[1, 2, indices_B...]
        b22 = B[2, 2, indices_B...]
        b32 = B[3, 2, indices_B...]


        b13 = B[1, 3, indices_B...]
        b23 = B[2, 3, indices_B...]
        b33 = B[3, 3, indices_B...]


        C[1, 1, indices...] = β * C[1, 1, indices...] + a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = β * C[2, 1, indices...] + a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = β * C[3, 1, indices...] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = β * C[1, 2, indices...] + a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = β * C[2, 2, indices...] + a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = β * C[3, 2, indices...] + a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = β * C[1, 3, indices...] + a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = β * C[2, 3, indices...] + a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = β * C[3, 3, indices...] + a31 * b13 + a32 * b23 + a33 * b33

    end



end


#C = shiftA*shiftedB'
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Shifted_Lattice{L1,D},
    B::Adjoint_Lattice{Shifted_Lattice{L2,D}}) where {D,T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    shiftA = get_shift(A)
    shiftB = get_shift(B)
    mul_shiftA_shiftBdag!(C, A, B.data.data, shiftA, shiftB)
    #set_halo!(C)
end

function mul_shiftA_shiftBdag!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Shifted_Lattice{L1,D}, B::L, shiftA, shiftB) where {D,T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftAshiftBdag!, C.A, A.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shiftA, shiftB
    )
    #set_halo!(C)
end


@inline function kernel_Dmatrix_mul_shiftAshiftBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shiftA, shiftB) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_A = shiftindices(indices, shiftA)

    indices_B = shiftindices(indices, shiftB)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = 0
            for kc = 1:NC3
                C[ic, jc, indices...] += A[ic, kc, indices_A...] * B[jc, kc, indices_B...]'
            end
        end
    end
end

#C = α* shiftA*shiftedB'+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Shifted_Lattice{L1,D},
    B::Adjoint_Lattice{Shifted_Lattice{L2,D}},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    shiftA = get_shift(A)
    shiftB = get_shift(B)
    #println((shiftA, shiftB))
    mul_shiftA_shiftBdag!(C, A, B.data.data, shiftA, shiftB, α, β)
    #set_halo!(C)
end

function mul_shiftA_shiftBdag!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Shifted_Lattice{L1,D}, B::L, shiftA, shiftB,
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftAshiftBdag!, C.A, A.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shiftA, shiftB, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_Dmatrix_mul_shiftAshiftBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shiftA, shiftB, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    indices_A = shiftindices(indices, shiftA)

    indices_B = shiftindices(indices, shiftB)

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = β * C[ic, jc, indices...]
            for kc = 1:NC3
                C[ic, jc, indices...] += α * A[ic, kc, indices_A...] * B[jc, kc, indices_B...]'
            end
        end
    end
end


@inline function kernel_Dmatrix_mul_shiftAshiftBdag!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw},
    dindexer, shiftA, shiftB, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    begin
        indices_A = shiftindices(indices, shiftA)

        indices_B = shiftindices(indices, shiftB)

        a11 = α * A[1, 1, indices_A...]
        a21 = α * A[2, 1, indices_A...]
        a12 = α * A[1, 2, indices_A...]
        a22 = α * A[2, 2, indices_A...]

        b11 = B[1, 1, indices_B...]'
        b12 = B[2, 1, indices_B...]'
        b21 = B[1, 2, indices_B...]'
        b22 = B[2, 2, indices_B...]'


        C[1, 1, indices...] = β * C[1, 1, indices...] + a11 * b11 + a12 * b21
        C[2, 1, indices...] = β * C[2, 1, indices...] + a21 * b11 + a22 * b21
        C[1, 2, indices...] = β * C[1, 2, indices...] + a11 * b12 + a12 * b22
        C[2, 2, indices...] = β * C[2, 2, indices...] + a21 * b12 + a22 * b22
    end



end


@inline function kernel_Dmatrix_mul_shiftAshiftBdag!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw},
    dindexer, shiftA, shiftB, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        indices_A = shiftindices(indices, shiftA)

        indices_B = shiftindices(indices, shiftB)

        a11 = α * A[1, 1, indices_A...]
        a21 = α * A[2, 1, indices_A...]
        a31 = α * A[3, 1, indices_A...]
        a12 = α * A[1, 2, indices_A...]
        a22 = α * A[2, 2, indices_A...]
        a32 = α * A[3, 2, indices_A...]
        a13 = α * A[1, 3, indices_A...]
        a23 = α * A[2, 3, indices_A...]
        a33 = α * A[3, 3, indices_A...]

        b11 = B[1, 1, indices_B...]'
        b12 = B[2, 1, indices_B...]'
        b13 = B[3, 1, indices_B...]'

        b21 = B[1, 2, indices_B...]'
        b22 = B[2, 2, indices_B...]'
        b23 = B[3, 2, indices_B...]'


        b31 = B[1, 3, indices_B...]'
        b32 = B[2, 3, indices_B...]'
        b33 = B[3, 3, indices_B...]'


        C[1, 1, indices...] = β * C[1, 1, indices...] + a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = β * C[2, 1, indices...] + a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = β * C[3, 1, indices...] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = β * C[1, 2, indices...] + a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = β * C[2, 2, indices...] + a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = β * C[3, 2, indices...] + a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = β * C[1, 3, indices...] + a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = β * C[2, 3, indices...] + a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = β * C[3, 3, indices...] + a31 * b13 + a32 * b23 + a33 * b33

    end



end



#C = shiftA'*shiftedB'
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Shifted_Lattice{L1,D}},
    B::Adjoint_Lattice{Shifted_Lattice{L2,D}}) where {D,T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    shiftA = get_shift(A)
    shiftB = get_shift(B)
    mul_shiftAdag_shiftBdag!(C, A, B.data.data, shiftA, shiftB)
    #set_halo!(C)
end

function mul_shiftAdag_shiftBdag!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Shifted_Lattice{L1,D}},
    B::L, shiftA, shiftB) where {D,T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftAdagshiftBdag!, C.A, A.data.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shiftA, shiftB
    )
    #set_halo!(C)
end


@inline function kernel_Dmatrix_mul_shiftAdagshiftBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shiftA, shiftB) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_A = shiftindices(indices, shiftA)

    indices_B = shiftindices(indices, shiftB)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = 0
            for kc = 1:NC3
                C[ic, jc, indices...] += A[kc, ic, indices_A...]' * B[jc, kc, indices_B...]'
            end
        end
    end
end

#C = α*shiftA'*shiftedB'+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Shifted_Lattice{L1,D}},
    B::Adjoint_Lattice{Shifted_Lattice{L2,D}},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    shiftA = get_shift(A)
    shiftB = get_shift(B)
    mul_shiftAdag_shiftBdag!(C, A, B.data.data, shiftA, shiftB, α, β)
    #set_halo!(C)
end

function mul_shiftAdag_shiftBdag!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Shifted_Lattice{L1,D}},
    B::L, shiftA, shiftB,
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftAdagshiftBdag!, C.A, A.data.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shiftA, shiftB, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_Dmatrix_mul_shiftAdagshiftBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shiftA, shiftB, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    indices_A = shiftindices(indices, shiftA)

    indices_B = shiftindices(indices, shiftB)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = β * C[ic, jc, indices...]
            for kc = 1:NC3
                C[ic, jc, indices...] += α * A[kc, ic, indices_A...]' * B[jc, kc, indices_B...]'
            end
        end
    end
end


@inline function kernel_Dmatrix_mul_shiftAdagshiftBdag!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw},
    dindexer, shiftA, shiftB, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    begin
        indices_A = shiftindices(indices, shiftA)

        indices_B = shiftindices(indices, shiftB)

        a11 = α * A[1, 1, indices_A...]'
        a12 = α * A[2, 1, indices_A...]'
        a21 = α * A[1, 2, indices_A...]'
        a22 = α * A[2, 2, indices_A...]'

        b11 = B[1, 1, indices_B...]'
        b12 = B[2, 1, indices_B...]'
        b21 = B[1, 2, indices_B...]'
        b22 = B[2, 2, indices_B...]'


        C[1, 1, indices...] = β * C[1, 1, indices...] + a11 * b11 + a12 * b21
        C[2, 1, indices...] = β * C[2, 1, indices...] + a21 * b11 + a22 * b21
        C[1, 2, indices...] = β * C[1, 2, indices...] + a11 * b12 + a12 * b22
        C[2, 2, indices...] = β * C[2, 2, indices...] + a21 * b12 + a22 * b22
    end



end


@inline function kernel_Dmatrix_mul_shiftAdagshiftBdag!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw},
    dindexer, shiftA, shiftB, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        indices_A = shiftindices(indices, shiftA)

        indices_B = shiftindices(indices, shiftB)

        a11 = α * A[1, 1, indices_A...]'
        a12 = α * A[2, 1, indices_A...]'
        a13 = α * A[3, 1, indices_A...]'
        a21 = α * A[1, 2, indices_A...]'
        a22 = α * A[2, 2, indices_A...]'
        a23 = α * A[3, 2, indices_A...]'
        a31 = α * A[1, 3, indices_A...]'
        a32 = α * A[2, 3, indices_A...]'
        a33 = α * A[3, 3, indices_A...]'

        b11 = B[1, 1, indices_B...]'
        b12 = B[2, 1, indices_B...]'
        b13 = B[3, 1, indices_B...]'

        b21 = B[1, 2, indices_B...]'
        b22 = B[2, 2, indices_B...]'
        b23 = B[3, 2, indices_B...]'


        b31 = B[1, 3, indices_B...]'
        b32 = B[2, 3, indices_B...]'
        b33 = B[3, 3, indices_B...]'


        C[1, 1, indices...] = β * C[1, 1, indices...] + a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = β * C[2, 1, indices...] + a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = β * C[3, 1, indices...] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = β * C[1, 2, indices...] + a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = β * C[2, 2, indices...] + a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = β * C[3, 2, indices...] + a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = β * C[1, 3, indices...] + a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = β * C[2, 3, indices...] + a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = β * C[3, 3, indices...] + a31 * b13 + a32 * b23 + a33 * b33

    end



end

#C = A*B A is a matrix.
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::TA, B::LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}) where {D,T1,T3,AT1,AT3,NC1,NC2,NC3,nw,DI,TA<:AbstractMatrix}
    N1, N2 = size(A)
    @assert N1 == NC1 "the size mismatch. NC1= $NC1 N1=$N1"
    @assert N2 == NC3 "the size mismatch. NC3= $NC3 N2=$N2"

    At = JACC.array(A)
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mulA!, C.A, At, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer
    )
    #set_halo!(C)
end

function kernel_Dmatrix_mulA!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = zero(eltype(C))
        end

        for kc = 1:NC3
            b = B[kc, jc, indices...]
            for ic = 1:NC1
                C[ic, jc, indices...] += A[ic, kc] * b
            end
        end
    end
end


#C = A*B, B is a matrix.
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T3,AT3,NC1,NC3,nw,DI}, B::TB) where {D,T1,T3,AT1,AT3,NC1,NC2,NC3,nw,DI,TB<:AbstractMatrix}
    N1, N2 = size(B)
    @assert N1 == NC3 "the size mismatch. NC3= $NC3 N1=$N1 "
    @assert N2 == NC2 "the size mismatch. NC3= $NC2 N2=$N2"

    Bt = JACC.array(B)
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mulB!, C.A, A.A, Bt, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer
    )
    #set_halo!(C)
end

function kernel_Dmatrix_mulB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = zero(eltype(C))
        end

        for kc = 1:NC3
            b = B[kc, jc]
            for ic = 1:NC1
                C[ic, jc, indices...] += A[ic, kc, indices...] * b
            end
        end
    end
end

function kernel_Dmatrix_mulB!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds begin
        b11 = B[1, 1]
        b21 = B[2, 1]
        b12 = B[1, 2]
        b22 = B[2, 2]

        a11 = A[1, 1, indices...]
        a21 = A[2, 1, indices...]
        a12 = A[1, 2, indices...]
        a22 = A[2, 2, indices...]

        C[1, 1, indices...] = a11 * b11 + a12 * b21
        C[2, 1, indices...] = a21 * b11 + a22 * b21
        C[1, 2, indices...] = a11 * b12 + a12 * b22
        C[2, 2, indices...] = a21 * b12 + a22 * b22
    end
end

function kernel_Dmatrix_mulB!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds begin
        b11 = B[1, 1]
        b21 = B[2, 1]
        b31 = B[3, 1]
        b12 = B[1, 2]
        b22 = B[2, 2]
        b32 = B[3, 2]
        b13 = B[1, 3]
        b23 = B[2, 3]
        b33 = B[3, 3]

        a11 = A[1, 1, indices...]
        a21 = A[2, 1, indices...]
        a31 = A[3, 1, indices...]
        a12 = A[1, 2, indices...]
        a22 = A[2, 2, indices...]
        a32 = A[3, 2, indices...]
        a13 = A[1, 3, indices...]
        a23 = A[2, 3, indices...]
        a33 = A[3, 3, indices...]

        C[1, 1, indices...] = a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = a31 * b13 + a32 * b23 + a33 * b33
    end
end

#C = A*B'
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::Adjoint_Lattice{L}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    #println("Using Dmatrix mul ABdag")
    #display(A.A[:,:,2,2,2,2])
    #display(B.data.A[:,:,2,2,2,2])

    mul_ABdag!(C, A, B.data)
    #set_halo!(C)
end


function mul_ABdag!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::L) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_ABdag!, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer
    )
    #set_halo!(C)
end

function mul_simple!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::Adjoint_Lattice{L}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}
    mul_simple_ABdag!(C, A, B.data)
end

function mul_simple_ABdag!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::L) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}
    @inbounds for i in 1:prod(C.PN)
        kernel_Dmatrix_mul_ABdag!(i, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer)
    end
end

function mul_simple!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::Adjoint_Lattice{L},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}
    mul_simple_ABdag!(C, A, B.data, α, β)
end

function mul_simple_ABdag!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::L, α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}
    αin = T1(α)
    βin = T1(β)
    @inbounds for i in 1:prod(C.PN)
        kernel_Dmatrix_mul_ABdag!(i, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, αin, βin)
    end
end


@inline function kernel_Dmatrix_mul_ABdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = 0
            for kc = 1:NC3
                C[ic, jc, indices...] += A[ic, kc, indices...] * B[jc, kc, indices...]'
            end
        end
    end
end

@inline function kernel_Dmatrix_mul_ABdag!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        a11 = A[1, 1, indices...]
        a21 = A[2, 1, indices...]
        #a31 = α * A[3, 1, indices...]
        a12 = A[1, 2, indices...]
        a22 = A[2, 2, indices...]
        #a32 = α * A[3, 2, indices...]
        #a13 = α * A[1, 3, indices...]
        #a23 = α * A[2, 3, indices...]
        #a33 = α * A[3, 3, indices...]


        b11 = B[1, 1, indices...]'
        b12 = B[2, 1, indices...]'
        #b13 = B[3, 1, indices...]'
        b21 = B[1, 2, indices...]'
        b22 = B[2, 2, indices...]'
        #b23 = B[3, 2, indices...]'
        #b31 = B[1, 3, indices...]'
        #b32 = B[2, 3, indices...]'
        #b33 = B[3, 3, indices...]'

        C[1, 1, indices...] = a11 * b11 + a12 * b21 #+ a13 * b31
        C[2, 1, indices...] = a21 * b11 + a22 * b21 #+ a23 * b31
        #C[3, 1, indices...] = β * C[3, 1, indices...] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = a11 * b12 + a12 * b22 #+ a13 * b32
        C[2, 2, indices...] = a21 * b12 + a22 * b22 #+ a23 * b32
        #C[3, 2, indices...] = β * C[3, 2, indices...] + a31 * b12 + a32 * b22 + a33 * b32
        #C[1, 3, indices...] = β * C[1, 3, indices...] + a11 * b13 + a12 * b23 + a13 * b33
        #C[2, 3, indices...] = β * C[2, 3, indices...] + a21 * b13 + a22 * b23 + a23 * b33
        #C[3, 3, indices...] = β * C[3, 3, indices...] + a31 * b13 + a32 * b23 + a33 * b33
    end
end

@inline function kernel_Dmatrix_mul_ABdag!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        a11 = A[1, 1, indices...]
        a21 = A[2, 1, indices...]
        a31 = A[3, 1, indices...]
        a12 = A[1, 2, indices...]
        a22 = A[2, 2, indices...]
        a32 = A[3, 2, indices...]
        a13 = A[1, 3, indices...]
        a23 = A[2, 3, indices...]
        a33 = A[3, 3, indices...]


        b11 = B[1, 1, indices...]'
        b12 = B[2, 1, indices...]'
        b13 = B[3, 1, indices...]'
        b21 = B[1, 2, indices...]'
        b22 = B[2, 2, indices...]'
        b23 = B[3, 2, indices...]'
        b31 = B[1, 3, indices...]'
        b32 = B[2, 3, indices...]'
        b33 = B[3, 3, indices...]'

        C[1, 1, indices...] = a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = a31 * b13 + a32 * b23 + a33 * b33
    end
end


#C = α* A*B' + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::Adjoint_Lattice{L},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    mul_ABdag!(C, A, B.data, α, β)
    #set_halo!(C)
end


function mul_ABdag!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::L, α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_ABdag!, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_Dmatrix_mul_ABdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = β * C[ic, jc, indices...]
        end

        for kc = 1:NC3
            b = conj(B[jc, kc, indices...])
            @simd for ic = 1:NC1
                C[ic, jc, indices...] += α * A[ic, kc, indices...] * b#B[jc, kc, indices...]'
            end
        end
    end
end

@inline function kernel_Dmatrix_mul_ABdag!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, dindexer, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        a11 = α * A[1, 1, indices...]
        a21 = α * A[2, 1, indices...]
        #a31 = α * A[3, 1, indices...]
        a12 = α * A[1, 2, indices...]
        a22 = α * A[2, 2, indices...]
        #a32 = α * A[3, 2, indices...]
        #a13 = α * A[1, 3, indices...]
        #a23 = α * A[2, 3, indices...]
        #a33 = α * A[3, 3, indices...]


        b11 = B[1, 1, indices...]'
        b12 = B[2, 1, indices...]'
        #b13 = B[3, 1, indices...]'
        b21 = B[1, 2, indices...]'
        b22 = B[2, 2, indices...]'
        #b23 = B[3, 2, indices...]'
        #b31 = B[1, 3, indices...]'
        #b32 = B[2, 3, indices...]'
        #b33 = B[3, 3, indices...]'

        C[1, 1, indices...] = β * C[1, 1, indices...] + a11 * b11 + a12 * b21 #+ a13 * b31
        C[2, 1, indices...] = β * C[2, 1, indices...] + a21 * b11 + a22 * b21 #+ a23 * b31
        #C[3, 1, indices...] = β * C[3, 1, indices...] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = β * C[1, 2, indices...] + a11 * b12 + a12 * b22 #+ a13 * b32
        C[2, 2, indices...] = β * C[2, 2, indices...] + a21 * b12 + a22 * b22 #+ a23 * b32
        #C[3, 2, indices...] = β * C[3, 2, indices...] + a31 * b12 + a32 * b22 + a33 * b32
        #C[1, 3, indices...] = β * C[1, 3, indices...] + a11 * b13 + a12 * b23 + a13 * b33
        #C[2, 3, indices...] = β * C[2, 3, indices...] + a21 * b13 + a22 * b23 + a23 * b33
        #C[3, 3, indices...] = β * C[3, 3, indices...] + a31 * b13 + a32 * b23 + a33 * b33
    end
end

@inline function kernel_Dmatrix_mul_ABdag!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        a11 = α * A[1, 1, indices...]
        a21 = α * A[2, 1, indices...]
        a31 = α * A[3, 1, indices...]
        a12 = α * A[1, 2, indices...]
        a22 = α * A[2, 2, indices...]
        a32 = α * A[3, 2, indices...]
        a13 = α * A[1, 3, indices...]
        a23 = α * A[2, 3, indices...]
        a33 = α * A[3, 3, indices...]


        b11 = B[1, 1, indices...]'
        b12 = B[2, 1, indices...]'
        b13 = B[3, 1, indices...]'
        b21 = B[1, 2, indices...]'
        b22 = B[2, 2, indices...]'
        b23 = B[3, 2, indices...]'
        b31 = B[1, 3, indices...]'
        b32 = B[2, 3, indices...]'
        b33 = B[3, 3, indices...]'

        C[1, 1, indices...] = β * C[1, 1, indices...] + a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = β * C[2, 1, indices...] + a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = β * C[3, 1, indices...] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = β * C[1, 2, indices...] + a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = β * C[2, 2, indices...] + a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = β * C[3, 2, indices...] + a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = β * C[1, 3, indices...] + a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = β * C[2, 3, indices...] + a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = β * C[3, 3, indices...] + a31 * b13 + a32 * b23 + a33 * b33
    end
end

#C = A'*B'
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{L1}, B::Adjoint_Lattice{L2}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}
    mul_AdagBdag!(C, A.data, B.data)
    #set_halo!(C)
end


function mul_AdagBdag!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::L1, B::L2) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_AdagBdag!, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer
    )
    #set_halo!(C)
end


@inline function kernel_Dmatrix_mul_AdagBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = 0
            for kc = 1:NC3
                C[ic, jc, indices...] += A[kc, ic, indices...]' * B[jc, kc, indices...]'
            end
        end
    end
end

@inline function kernel_Dmatrix_mul_AdagBdag!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        a11 = A[1, 1, indices...]'
        a12 = A[2, 1, indices...]'
        #a13 = A[3, 1, indices...]'
        a21 = A[1, 2, indices...]'
        a22 = A[2, 2, indices...]'
        #a23 = A[3, 2, indices...]'
        #a31 = A[1, 3, indices...]'
        #a32 = A[2, 3, indices...]'
        #a33 = A[3, 3, indices...]'


        b11 = B[1, 1, indices...]'
        b12 = B[2, 1, indices...]'
        #b13 = B[3, 1, indices...]'
        b21 = B[1, 2, indices...]'
        b22 = B[2, 2, indices...]'
        #b23 = B[3, 2, indices...]'
        #b31 = B[1, 3, indices...]'
        #b32 = B[2, 3, indices...]'
        #b33 = B[3, 3, indices...]'

        C[1, 1, indices...] = a11 * b11 + a12 * b21 #+ a13 * b31
        C[2, 1, indices...] = a21 * b11 + a22 * b21 #+ a23 * b31
        #C[3, 1, indices...] = a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = a11 * b12 + a12 * b22 #+ a13 * b32
        C[2, 2, indices...] = a21 * b12 + a22 * b22 #+ a23 * b32
        #C[3, 2, indices...] = a31 * b12 + a32 * b22 + a33 * b32
        #C[1, 3, indices...] = a11 * b13 + a12 * b23 + a13 * b33
        #C[2, 3, indices...] = a21 * b13 + a22 * b23 + a23 * b33
        #C[3, 3, indices...] = a31 * b13 + a32 * b23 + a33 * b33
    end
end


@inline function kernel_Dmatrix_mul_AdagBdag!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        a11 = A[1, 1, indices...]'
        a12 = A[2, 1, indices...]'
        a13 = A[3, 1, indices...]'
        a21 = A[1, 2, indices...]'
        a22 = A[2, 2, indices...]'
        a23 = A[3, 2, indices...]'
        a31 = A[1, 3, indices...]'
        a32 = A[2, 3, indices...]'
        a33 = A[3, 3, indices...]'


        b11 = B[1, 1, indices...]'
        b12 = B[2, 1, indices...]'
        b13 = B[3, 1, indices...]'
        b21 = B[1, 2, indices...]'
        b22 = B[2, 2, indices...]'
        b23 = B[3, 2, indices...]'
        b31 = B[1, 3, indices...]'
        b32 = B[2, 3, indices...]'
        b33 = B[3, 3, indices...]'

        C[1, 1, indices...] = a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = a31 * b13 + a32 * b23 + a33 * b33
    end
end

#C =  α* A'*B' + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{L1}, B::Adjoint_Lattice{L2},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}
    mul_AdagBdag!(C, A.data, B.data, α, β)
    #set_halo!(C)
end


function mul_AdagBdag!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::L1, B::L2, α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_AdagBdag!, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_Dmatrix_mul_AdagBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = β * C[ic, jc, indices...]
            for kc = 1:NC3
                C[ic, jc, indices...] += α * A[kc, ic, indices...]' * B[jc, kc, indices...]'
            end
        end
    end
end

@inline function kernel_Dmatrix_mul_AdagBdag!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, dindexer, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        a11 = α * A[1, 1, indices...]'
        a12 = α * A[2, 1, indices...]'
        #a13 = α * A[3, 1, indices...]'
        a21 = α * A[1, 2, indices...]'
        a22 = α * A[2, 2, indices...]'
        #a23 = α * A[3, 2, indices...]'
        #a31 = α * A[1, 3, indices...]'
        #a32 = α * A[2, 3, indices...]'
        #a33 = α * A[3, 3, indices...]'


        b11 = B[1, 1, indices...]'
        b12 = B[2, 1, indices...]'
        #b13 = B[3, 1, indices...]'
        b21 = B[1, 2, indices...]'
        b22 = B[2, 2, indices...]'
        #b23 = B[3, 2, indices...]'
        #b31 = B[1, 3, indices...]'
        #b32 = B[2, 3, indices...]'
        #b33 = B[3, 3, indices...]'

        C[1, 1, indices...] = β * C[1, 1, indices...] + a11 * b11 + a12 * b21 #+ a13 * b31
        C[2, 1, indices...] = β * C[2, 1, indices...] + a21 * b11 + a22 * b21 #+ a23 * b31
        #C[3, 1, indices...] = β * C[3, 1, indices...] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = β * C[1, 2, indices...] + a11 * b12 + a12 * b22 #+ a13 * b32
        C[2, 2, indices...] = β * C[2, 2, indices...] + a21 * b12 + a22 * b22 #+ a23 * b32
        #C[3, 2, indices...] = β * C[3, 2, indices...] + a31 * b12 + a32 * b22 + a33 * b32
        #C[1, 3, indices...] = β * C[1, 3, indices...] + a11 * b13 + a12 * b23 + a13 * b33
        #C[2, 3, indices...] = β * C[2, 3, indices...] + a21 * b13 + a22 * b23 + a23 * b33
        #C[3, 3, indices...] = β * C[3, 3, indices...] + a31 * b13 + a32 * b23 + a33 * b33
    end
end


@inline function kernel_Dmatrix_mul_AdagBdag!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer, α::S, β::S) where {nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        a11 = α * A[1, 1, indices...]'
        a12 = α * A[2, 1, indices...]'
        a13 = α * A[3, 1, indices...]'
        a21 = α * A[1, 2, indices...]'
        a22 = α * A[2, 2, indices...]'
        a23 = α * A[3, 2, indices...]'
        a31 = α * A[1, 3, indices...]'
        a32 = α * A[2, 3, indices...]'
        a33 = α * A[3, 3, indices...]'


        b11 = B[1, 1, indices...]'
        b12 = B[2, 1, indices...]'
        b13 = B[3, 1, indices...]'
        b21 = B[1, 2, indices...]'
        b22 = B[2, 2, indices...]'
        b23 = B[3, 2, indices...]'
        b31 = B[1, 3, indices...]'
        b32 = B[2, 3, indices...]'
        b33 = B[3, 3, indices...]'

        C[1, 1, indices...] = β * C[1, 1, indices...] + a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = β * C[2, 1, indices...] + a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = β * C[3, 1, indices...] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = β * C[1, 2, indices...] + a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = β * C[2, 2, indices...] + a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = β * C[3, 2, indices...] + a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = β * C[1, 3, indices...] + a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = β * C[2, 3, indices...] + a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = β * C[3, 3, indices...] + a31 * b13 + a32 * b23 + a33 * b33
    end
end
