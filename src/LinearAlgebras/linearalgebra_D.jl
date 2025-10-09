#Overwrite Y with X*a + Y*b, where a and b are scalars. Return Y.
function LinearAlgebra.axpby!(
    a::Number,
    X::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    b::Number,
    Y::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
) where {T1,AT1,NC1,NC2,nw,D,DI}

    JACC.parallel_for(
        prod(Y.PN), kernel_D_axpby!, a, X.A, b, Y.A, Val(NC1), Val(NC2), Val(nw), Y.indexer
    )
end

@inline function kernel_D_axpby!(i, a, X, b, Y, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, dindexer) where {NC1,NC2,nw}
    indices = delinearize(dindexer, i, nw)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            Y[ic, jc, indices...] = a * X[ic, jc, indices...] + b * Y[ic, jc, indices...]
        end
    end
end

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


function expt!(C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}, A::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI}, t::S=one(S)) where {D,T,AT,NC1,NC2,S<:Number,T1,AT1,nw,DI}
    @assert NC1 == NC2 "Matrix exponentiation requires square matrices, but got $(NC1) x $(NC2)."

    JACC.parallel_for(
        prod(C.PN), kernel_4Dexpt!, C.A, A.A, C.indexer, Val(nw), t, Val(NC1)
    )
    return
    #set_halo!(C)
end

@inline function kernel_4Dexpt!(i, C, A, dindexer, ::Val{nw}, t, ::Val{3}) where nw
    indices = delinearize(dindexer, i, nw)
    a11 = A[1, 1, indices...]
    a12 = A[1, 2, indices...]
    a13 = A[1, 3, indices...]
    a21 = A[2, 1, indices...]
    a22 = A[2, 2, indices...]
    a23 = A[2, 3, indices...]
    a31 = A[3, 1, indices...]
    a32 = A[3, 2, indices...]
    a33 = A[3, 3, indices...]

    c11, c12, c13, c21, c22, c23, c31, c32, c33 = exp3x3_pade(a11, a12, a13, a21, a22, a23, a31, a32, a33, t)
    C[1, 1, indices...] = c11
    C[1, 2, indices...] = c12
    C[1, 3, indices...] = c13
    C[2, 1, indices...] = c21
    C[2, 2, indices...] = c22
    C[2, 3, indices...] = c23
    C[3, 1, indices...] = c31
    C[3, 2, indices...] = c32
    C[3, 3, indices...] = c33

end

@inline function kernel_4Dexpt!(i, C, A, dindexer, ::Val{nw}, t, ::Val{2}) where nw
    indices = delinearize(dindexer, i, nw)
    a11 = A[1, 1, indices...]
    a21 = A[2, 1, indices...]
    a12 = A[1, 2, indices...]
    a22 = A[2, 2, indices...]
    c11, c12, c21, c22 = exp2x2_elem(a11, a12, a21, a22, t)

    C[1, 1, indices...] = c11
    C[1, 2, indices...] = c12
    C[2, 1, indices...] = c21
    C[2, 2, indices...] = c22
end



@inline function kernel_4Dexpt!(i, C, A, dindexer, ::Val{nw}, t, ::Val{N}) where {N,nw}
    indices = delinearize(dindexer, i, nw)
    expm_pade13_writeback!(C, A, indices..., t, Val(N))
    #C[:, :, indices...] = expm_pade13(A[:, :, indices...], t)
end

function expt!(C::LatticeMatrix{D,T,AT,NC1,NC1,nw,DI}, TA::LatticeMatrix{D,T1,AT1,Num,1,nw2,DI}, t::S=one(S)) where {D,T,AT,NC1,Num,S<:Number,T1<:Real,AT1,nw,nw2,DI}

    if NC1 > 3
        error("In NC > 3 case, this function should not be used")
    else
        JACC.parallel_for(
            prod(C.PN), kernel_4Dexpt_TA!, C.A, TA.A, C.indexer, Val(nw), t, Val(NC1), Val(nw2)
        )
    end
    return
    #set_halo!(C)
end

function kernel_4Dexpt_TA!(i, uout, A, dindexer, ::Val{nw}, t, ::Val{2}, ::Val{nw2}) where {nw,nw2}
    indices = delinearize(dindexer, i, nw)
    indices2 = delinearize(dindexer, i, nw2)
    #    ixt = ix + nw2
    #    iyt = iy + nw2
    #    izt = iz + nw2
    #    itt = it + nw2
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    c1_0 = A[1, 1, indices2...]
    c2_0 = A[2, 1, indices2...]
    c3_0 = A[3, 1, indices2...]

    #icum = (((it-1)*NX+iz-1)*NY+iy-1)*NX+ix  
    u1 = t * c1_0 / 2
    u2 = t * c2_0 / 2
    u3 = t * c3_0 / 2
    R = sqrt(u1^2 + u2^2 + u3^2) + tinyvalue
    sR = sin(R) / R
    #sR = ifelse(R == 0,1,sR)
    a0 = cos(R)
    a1 = u1 * sR
    a2 = u2 * sR
    a3 = u3 * sR

    uout[1, 1, indices...] = cos(R) + im * a3
    uout[1, 2, indices...] = im * a1 + a2
    uout[2, 1, indices...] = im * a1 - a2
    uout[2, 2, indices...] = cos(R) - im * a3
end


function kernel_4Dexpt_TA!(i, C, A, dindexer, ::Val{nw}, t, ::Val{3}, ::Val{nw2}) where {nw,nw2}
    indices = delinearize(dindexer, i, nw)
    indices2 = delinearize(dindexer, i, nw2)
    T = eltype(C)
    #ixt = ix + nw2
    #iyt = iy + nw2
    #izt = iz + nw2
    #itt = it + nw2
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    c1_0 = A[1, 1, indices2...]
    c2_0 = A[2, 1, indices2...]
    c3_0 = A[3, 1, indices2...]
    c4_0 = A[4, 1, indices2...]
    c5_0 = A[5, 1, indices2...]

    c6_0 = A[6, 1, indices2...]
    c7_0 = A[7, 1, indices2...]
    c8_0 = A[8, 1, indices2...]

    c1 = t * c1_0 * 0.5
    c2 = t * c2_0 * 0.5
    c3 = t * c3_0 * 0.5
    c4 = t * c4_0 * 0.5
    c5 = t * c5_0 * 0.5
    c6 = t * c6_0 * 0.5
    c7 = t * c7_0 * 0.5
    c8 = t * c8_0 * 0.5
    csum = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8
    if csum == 0
        c = Mat3{eltype(C)}(one(eltype(C)))
        C[1, 1, indices...] = c.a11
        C[1, 2, indices...] = c.a12
        C[1, 3, indices...] = c.a13
        C[2, 1, indices...] = c.a21
        C[2, 2, indices...] = c.a22
        C[2, 3, indices...] = c.a23
        C[3, 1, indices...] = c.a31
        C[3, 2, indices...] = c.a32
        C[3, 3, indices...] = c.a33

    end


    #x[1,1,icum] =  c3+sr3i*c8 +im*(  0.0 )
    v1 = c3 + sr3i * c8
    v2 = 0.0
    #x[1,2,icum] =  c1         +im*( -c2   )
    v3 = c1
    v4 = -c2
    #x[1,3,icum] =  c4         +im*(-c5   )
    v5 = c4
    v6 = -c5

    #x[2,1,icum] =  c1         +im*(  c2   )
    v7 = c1
    v8 = c2

    #x[2,2,icum] =  -c3+sr3i*c8+im*(  0.0 )
    v9 = -c3 + sr3i * c8
    v10 = 0.0

    #x[2,3,icum] =  c6         +im*( -c7   )
    v11 = c6
    v12 = -c7

    #x[3,1,icum] =  c4         +im*(  c5   )
    v13 = c4
    v14 = c5

    #x[3,2,icum] =  c6         +im*(  c7   )
    v15 = c6
    v16 = c7
    #x[3,3,icum] =  -sr3i2*c8  +im*(  0.0 )
    v17 = -sr3i2 * c8
    v18 = 0.0


    #c find eigenvalues of v
    trv3 = (v1 + v9 + v17) / 3.0
    cofac =
        v1 * v9 - v3^2 - v4^2 + v1 * v17 - v5^2 - v6^2 + v9 * v17 - v11^2 -
        v12^2
    det =
        v1 * v9 * v17 - v1 * (v11^2 + v12^2) - v9 * (v5^2 + v6^2) -
        v17 * (v3^2 + v4^2) +
        (v5 * (v3 * v11 - v4 * v12) + v6 * (v3 * v12 + v4 * v11)) * 2.0
    p3 = cofac / 3.0 - trv3^2
    q = trv3 * cofac - det - 2.0 * trv3^3
    x = sqrt(-4.0 * p3) + tinyvalue
    arg = q / (x * p3)

    arg = min(1, max(-1, arg))
    theta = acos(arg) / 3.0
    e1 = x * cos(theta) + trv3
    theta = theta + pi23
    e2 = x * cos(theta) + trv3
    #       theta = theta + pi23
    #       e3 = x * cos(theta) + trv3
    e3 = 3.0 * trv3 - e1 - e2

    # solve for eigenvectors

    w1 = v5 * (v9 - e1) - v3 * v11 + v4 * v12
    w2 = -v6 * (v9 - e1) + v4 * v11 + v3 * v12
    w3 = (v1 - e1) * v11 - v3 * v5 - v4 * v6
    w4 = -(v1 - e1) * v12 - v4 * v5 + v3 * v6
    w5 = -(v1 - e1) * (v9 - e1) + v3^2 + v4^2
    w6 = 0.0

    coeff = 1.0 / sqrt(w1^2 + w2^2 + w3^2 + w4^2 + w5^2)


    w1 = w1 * coeff
    w2 = w2 * coeff
    w3 = w3 * coeff
    w4 = w4 * coeff
    w5 = w5 * coeff

    w7 = v5 * (v9 - e2) - v3 * v11 + v4 * v12
    w8 = -v6 * (v9 - e2) + v4 * v11 + v3 * v12
    w9 = (v1 - e2) * v11 - v3 * v5 - v4 * v6
    w10 = -(v1 - e2) * v12 - v4 * v5 + v3 * v6
    w11 = -(v1 - e2) * (v9 - e2) + v3^2 + v4^2
    w12 = 0.0

    coeff = 1.0 / sqrt(w7^2 + w8^2 + w9^2 + w10^2 + w11^2)

    w7 = w7 * coeff
    w8 = w8 * coeff
    w9 = w9 * coeff
    w10 = w10 * coeff
    w11 = w11 * coeff

    w13 = v5 * (v9 - e3) - v3 * v11 + v4 * v12
    w14 = -v6 * (v9 - e3) + v4 * v11 + v3 * v12
    w15 = (v1 - e3) * v11 - v3 * v5 - v4 * v6
    w16 = -(v1 - e3) * v12 - v4 * v5 + v3 * v6
    w17 = -(v1 - e3) * (v9 - e3) + v3^2 + v4^2
    w18 = 0.0

    coeff = 1.0 / sqrt(w13^2 + w14^2 + w15^2 + w16^2 + w17^2)
    w13 = w13 * coeff
    w14 = w14 * coeff
    w15 = w15 * coeff
    w16 = w16 * coeff
    w17 = w17 * coeff

    # construct the projection v
    c1 = cos(e1)
    s1 = sin(e1)
    ww1 = w1 * c1 - w2 * s1
    ww2 = w2 * c1 + w1 * s1
    ww3 = w3 * c1 - w4 * s1
    ww4 = w4 * c1 + w3 * s1
    ww5 = w5 * c1 - w6 * s1
    ww6 = w6 * c1 + w5 * s1

    c2 = cos(e2)
    s2 = sin(e2)
    ww7 = w7 * c2 - w8 * s2
    ww8 = w8 * c2 + w7 * s2
    ww9 = w9 * c2 - w10 * s2
    ww10 = w10 * c2 + w9 * s2
    ww11 = w11 * c2 - w12 * s2
    ww12 = w12 * c2 + w11 * s2

    c3 = cos(e3)
    s3 = sin(e3)
    ww13 = w13 * c3 - w14 * s3
    ww14 = w14 * c3 + w13 * s3
    ww15 = w15 * c3 - w16 * s3
    ww16 = w16 * c3 + w15 * s3
    ww17 = w17 * c3 - w18 * s3
    ww18 = w18 * c3 + w17 * s3


    w = Mat3{T}(w1 + im * w2,
        w3 + im * w4,
        w5 + im * w6,
        w7 + im * w8,
        w9 + im * w10,
        w11 + im * w12,
        w13 + im * w14,
        w15 + im * w16,
        w17 + im * w18)
    ww = Mat3{T}(ww1 + im * ww2,
        ww3 + im * ww4,
        ww5 + im * ww6,
        ww7 + im * ww8,
        ww9 + im * ww10,
        ww11 + im * ww12,
        ww13 + im * ww14,
        ww15 + im * ww16,
        ww17 + im * ww18)
    c = mul3(conjugate3(w), ww)

    C[1, 1, indices...] = c.a11
    C[1, 2, indices...] = c.a12
    C[1, 3, indices...] = c.a13
    C[2, 1, indices...] = c.a21
    C[2, 2, indices...] = c.a22
    C[2, 3, indices...] = c.a23
    C[3, 1, indices...] = c.a31
    C[3, 2, indices...] = c.a32
    C[3, 3, indices...] = c.a33



end


#C = A'*B
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{L}, B::LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_AdagB!, C.A, A.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer
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

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_AdagB!, C.A, A.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, α::S, β::S
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



#C = A*B'
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::Adjoint_Lattice{L}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_ABdag!, C.A, A.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer
    )
    #set_halo!(C)
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

#C = α* A*B' + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::Adjoint_Lattice{L},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_ABdag!, C.A, A.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, α::S, β::S
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
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_AdagBdag!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer
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
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_AdagBdag!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, α::S, β::S
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

function substitute!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI}, A::LatticeMatrix{D,T2,AT2,NC1,NC2,nw,DI}) where {D,T1,T2,AT1,AT2,NC1,NC2,nw,DI}
    JACC.parallel_for(
        prod(C.PN), kernel_4Dsubstitute!, C.A, A.A, Val(NC1), Val(NC2), Val(nw), C.indexer
    )
    #set_halo!(C)
end

@inline function kernel_4Dsubstitute!(i, C, A, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, dindexer) where {NC1,NC2,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = A[ic, jc, indices...]
        end
    end
end

function substitute!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI}, A::Adjoint_Lattice{L}) where {D,T1,T2,AT1,AT2,NC1,NC2,nw,DI,
    L<:LatticeMatrix{D,T2,AT2,NC1,NC2,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_4Dsubstitute_dag!, C.A, A.data.A, Val(NC1), Val(NC2), Val(nw), C.indexer
    )
    #set_halo!(C)
end

@inline function kernel_4Dsubstitute_dag!(i, C, A, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, dindexer) where {NC1,NC2,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = A[jc, ic, indices...]'
        end
    end
end

function substitute!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI}, A::Shifted_Lattice{L,shift}) where {D,T1,T2,AT1,AT2,NC1,NC2,shift,nw,DI,
    L<:LatticeMatrix{D,T2,AT2,NC1,NC2,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_4Dsubstitute_shift!, C.A, A.data.A, Val(NC1), Val(NC2), Val(nw), C.indexer, shift
    )
    #set_halo!(C)
end
export substitute!

@inline function kernel_4Dsubstitute_shift!(i, C, A, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, dindexer, shift) where {NC1,NC2,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)
    #println("indices... = ", (indices...))
    #println("indices... = ", (indices_p...))
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = A[ic, jc, indices_p...]
        end
    end
end

function substitute!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI}, A::Adjoint_Lattice{Shifted_Lattice{L,shift}}) where {D,T1,T2,AT1,AT2,NC1,NC2,shift,nw,DI,
    L<:LatticeMatrix{D,T2,AT2,NC1,NC2,nw,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_4Dsubstitute_shiftdag!, C.A, A.data.data.A, Val(NC1), Val(NC2), Val(nw), C.indexer, shift
    )
    #set_halo!(C)
end
export substitute!

@inline function kernel_4Dsubstitute_shiftdag!(i, C, A, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, dindexer, shift) where {NC1,NC2,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = A[jc, ic, indices_p...]'
        end
    end
end

#C = shiftedA*B
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Shifted_Lattice{L,shift}, B::LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,DI,
    L<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftAB!, C.A, A.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift
    )
    #set_halo!(C)
end


@inline function kernel_Dmatrix_mul_shiftAB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = 0
            for kc = 1:NC3
                C[ic, jc, indices...] += A[ic, kc, indices_p...] * B[kc, jc, indices...]
            end
        end
    end
end

@inline function kernel_Dmatrix_mul_shiftAB!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, dindexer, shift) where {nw}
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

        b11 = B[1, 1, indices...]
        b21 = B[2, 1, indices...]
        #b31 = B[3, 1, indices...]
        b12 = B[1, 2, indices...]
        b22 = B[2, 2, indices...]
        #b32 = B[3, 2, indices...]
        #b13 = B[1, 3, indices...]
        #b23 = B[2, 3, indices...]
        #b33 = B[3, 3, indices...]
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

@inline function kernel_Dmatrix_mul_shiftAB!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer, shift) where {nw}
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


#C = α shiftedA*B + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Shifted_Lattice{L,shift}, B::LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftAB!, C.A, A.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_Dmatrix_mul_shiftAB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = β * C[ic, jc, indices...]
            for kc = 1:NC3
                C[ic, jc, indices...] += α * A[ic, kc, indices_p...] * B[kc, jc, indices...]
            end
        end
    end
end

@inline function kernel_Dmatrix_mul_shiftAB!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer, shift, α::S, β::S) where {nw,S<:Number}
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
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::Shifted_Lattice{L,shift}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,DI,
    L<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}
    #println("C = A*shiftedB $NC1 $NC2 $NC3 ")
    #display(B.data.A[:, :, 2, 2, 2, 2])
    #println("BdataA")
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_AshiftB!, C.A, A.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift
    )
    #set_halo!(C)
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




#C = α A*shiftedB + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::Shifted_Lattice{L,shift},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}
    βin = T1(β)
    αin = T1(α)
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_AshiftB!, C.A, A.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift, αin, βin
    )
    #set_halo!(C)
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







#C = shiftedA'*B
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Shifted_Lattice{L,shift}}, B::LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,DI,
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
    A::Adjoint_Lattice{Shifted_Lattice{L,shift}}, B::LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,S<:Number,DI,
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
    A::Shifted_Lattice{L1,shift}, B::Adjoint_Lattice{L2}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,DI,
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
    A::Shifted_Lattice{L1,shift}, B::Adjoint_Lattice{L2},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,S<:Number,DI,
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
    A::Adjoint_Lattice{Shifted_Lattice{L1,shift}}, B::Adjoint_Lattice{L2}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftAdagBdag!, C.A, A.data.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift
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
    A::Adjoint_Lattice{Shifted_Lattice{L1,shift}}, B::Adjoint_Lattice{L2},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,S<:Number,DI,
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
    A::Adjoint_Lattice{L1}, B::Shifted_Lattice{L2,shift}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_AdagshiftB!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift
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

#C = α*A'*shiftedB+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{L1}, B::Shifted_Lattice{L2,shift},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,S<:Number,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_AdagshiftB!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift, α::S, β::S
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


#C = A*shiftedB'
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::Adjoint_Lattice{Shifted_Lattice{L,shift}}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,DI,
    L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_AshiftBdag!, C.A, A.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift
    )
    #set_halo!(C)
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

#C = α*A*shiftedB'+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}, B::Adjoint_Lattice{Shifted_Lattice{L,shift}},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_AshiftBdag!, C.A, A.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift, α::S, β::S
    )
    #set_halo!(C)
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
    A::Adjoint_Lattice{L1}, B::Adjoint_Lattice{Shifted_Lattice{L2,shift}}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_AdagshiftBdag!, C.A, A.data.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift
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

#C = α*A'*shiftedB'+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{L1}, B::Adjoint_Lattice{Shifted_Lattice{L2,shift}},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,S<:Number,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_AdagshiftBdag!, C.A, A.data.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shift, α::S, β::S
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



#C = shiftA*shiftedB
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Shifted_Lattice{L1,shiftA}, B::Shifted_Lattice{L2,shiftB}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shiftA,shiftB,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftAshiftB!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shiftA, shiftB
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
    A::Shifted_Lattice{L1,shiftA}, B::Shifted_Lattice{L2,shiftB},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shiftA,shiftB,nw,S<:Number,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftAshiftB!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shiftA, shiftB, α::S, β::S
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
function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{Shifted_Lattice{L1,shiftA}}, B::Shifted_Lattice{L2,shiftB}) where {D,T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,shiftA,shiftB,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftAdagshiftB!, C.A, A.data.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shiftA, shiftB
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
    A::Adjoint_Lattice{Shifted_Lattice{L1,shiftA}}, B::Shifted_Lattice{L2,shiftB},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,shiftA,shiftB,nw,S<:Number,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftAdagshiftB!, C.A, A.data.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shiftA, shiftB, α::S, β::S
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
    A::Shifted_Lattice{L1,shiftA},
    B::Adjoint_Lattice{Shifted_Lattice{L2,shiftB}}) where {D,T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,shiftA,shiftB,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftAshiftBdag!, C.A, A.data.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shiftA, shiftB
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
    A::Shifted_Lattice{L1,shiftA},
    B::Adjoint_Lattice{Shifted_Lattice{L2,shiftB}},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,shiftA,shiftB,nw,S<:Number,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    #println((shiftA, shiftB))
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftAshiftBdag!, C.A, A.data.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shiftA, shiftB, α::S, β::S
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
    A::Adjoint_Lattice{Shifted_Lattice{L1,shiftA}},
    B::Adjoint_Lattice{Shifted_Lattice{L2,shiftB}}) where {D,T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,shiftA,shiftB,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftAdagshiftBdag!, C.A, A.data.data.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shiftA, shiftB
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
    A::Adjoint_Lattice{Shifted_Lattice{L1,shiftA}},
    B::Adjoint_Lattice{Shifted_Lattice{L2,shiftB}},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,shiftA,shiftB,nw,S<:Number,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,nw,DI}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_shiftAdagshiftBdag!, C.A, A.data.data.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shiftA, shiftB, α::S, β::S
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



function LinearAlgebra.tr(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI}) where {D,T1,AT1,NC1,NC2,nw,DI}
    @assert NC1 == NC2 "Trace is only defined for square matrices"
    s = JACC.parallel_reduce(prod(C.PN), +, kernel_tr_4D, C.A, Val(NC1), C.indexer, Val(nw); init=zero(eltype(C.A)))::T1
    s = MPI.Allreduce(s, MPI.SUM, C.comm)
    return s
end

@inline _preduce(n, op, kern, A, NC1, dindexer, vnw, init::T) where {T} =
    JACC.parallel_reduce(n, op, kern, A, NC1, dindexer, vnw; init=init)::T

function LinearAlgebra.tr(C::LatticeMatrix{D,T1,AT1,NC1,NC1,nw,DI}) where {D,T1,AT1,NC1,nw,DI}
    s = _preduce(prod(C.PN), +, kernel_tr_4D, C.A, Val(NC1), C.indexer, Val(nw), zero(T1))::T1
    s = MPI.Allreduce(s, MPI.SUM, C.comm)
    return s
end


@inline function kernel_tr_4D(i, A, ::Val{NC1}, dindexer, ::Val{nw}) where {NC1,nw}
    indices = delinearize(dindexer, i, nw)
    s = zero(eltype(A))
    @inbounds for ic = 1:NC1
        s += A[ic, ic, indices...]
    end
    return s
end

@inline _preduce(n, op, kern, A, B, NC1, dindexer, vnw, init::T) where {T} =
    JACC.parallel_reduce(n, op, kern, A, B, NC1, dindexer, vnw; init=init)::T

function LinearAlgebra.tr(C::LatticeMatrix{D,T1,AT1,NC1,NC1,nw,DI}, B::LatticeMatrix{D,T1,AT1,NC1,NC1,nw,DI}) where {D,T1,AT1,NC1,nw,DI}
    s = _preduce(prod(C.PN), +, kernel_tr_4D, C.A, B.A, Val(NC1), C.indexer, Val(nw), zero(T1))::T1
    s = MPI.Allreduce(s, MPI.SUM, C.comm)
    return s
end

@inline function kernel_tr_4D(i, A, B, ::Val{NC1}, dindexer, ::Val{nw}) where {NC1,nw}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw
    s = zero(eltype(A))
    @inbounds for k = 1:NC1
        for k2 = 1:NC1
            s += A[k, k2, indices...] * B[k2, k, indices...]
        end
    end
    return s
end


function LinearAlgebra.dot(A::LatticeMatrix{D,T1,AT1,NC1,1,nw,DI}, B::LatticeMatrix{D,T2,AT2,NC1,1,nw,DI}) where {D,T1<:Real,T2<:Real,AT1,AT2,NC1,nw,DI}
    s = JACC.parallel_reduce(prod(A.PN), +, kernel_dot_real_1,
        A.A, B.A, A.indexer, Val(NC1), Val(nw); init=zero(eltype(A.A)))
    s = MPI.Allreduce(s, MPI.SUM, A.comm)
end

@inline function kernel_dot_real_1(i, A, B, dindexer, ::Val{NC1}, ::Val{nw}) where {NC1,nw}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw
    s = zero(eltype(A))

    @inbounds for ic = 1:NC1
        s += A[ic, 1, indices...] * B[ic, 1, indices...]
    end
    return s
end



#=
function LinearAlgebra.tr(C::LatticeMatrix{D,T1,AT1,3,3}) where {D,T1,AT1}
    s = JACC.parallel_reduce(prod(C.PN), +, kernel_tr_4D_NC3, C.A, C.indexer, Val(nw); init=zero(eltype(C.A)))
end

function kernel_tr_4D_NC3(i1,i2,i3, A, dindexer, nw)
    indices = delinearize(dindexer,i,nw)
    s = zero(eltype(A))
    for ic = 1:3
        s += A[ic, ic, indices...]
    end
    return s
end
=#

function partial_trace(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI}, μ::Int, position::Int=1) where {D,T1,AT1,NC1,NC2,nw,DI}
    s = JACC.parallel_reduce(prod(C.PN), +, kernel_partial_trace_D, C.A, NC1, C.indexer, μ, position, Val(nw); init=zero(eltype(C.A)))
    s = MPI.Allreduce(s, MPI.SUM, C.comm)
    return s
end
export partial_trace

@inline function kernel_partial_trace_D(i, A, NC, dindexer, μ, position, ::Val{nw}) where nw
    indices = delinearize(dindexer, i, nw)

    s = zero(eltype(A))
    if indices[μ] == position + nw
        for ic = 1:NC
            s += A[ic, ic, indices...]
        end
    end
    return s
end

# ========== host side ==========
function normalize_matrix!(C::LatticeMatrix{D,T,AT,NC,NC,nw,DI}) where {D,T,AT,NC,nw,DI}
    if NC == 2
        JACC.parallel_for(prod(C.PN), kernel_normalize_NC2!, C.A, C.indexer, Val(nw))
    elseif NC == 3
        JACC.parallel_for(prod(C.PN), kernel_normalize_NC3!, C.A, C.indexer, Val(nw))
    else
        # Generic: modified Gram–Schmidt per site (unitarize columns)
        JACC.parallel_for(prod(C.PN), kernel_normalize_generic!, C.A, C.indexer, NC, Val(nw))
    end
    #set_halo!(C)
end
export normalize_matrix!


@inline function kernel_normalize_NC2!(i, u, dindexer, ::Val{nw}) where nw
    indices = delinearize(dindexer, i, nw)
    α = u[1, 1, indices...]
    β = u[2, 1, indices...]
    detU = sqrt(abs(α)^2 + abs(β)^2)
    u[1, 1, indices...] = α / detU
    u[2, 1, indices...] = β / detU
    u[1, 2, indices...] = -conj(β) / detU
    u[2, 2, indices...] = conj(α) / detU
end

@inline function kernel_normalize_NC3!(i, u, dindexer, ::Val{nw}) where nw
    indices = delinearize(dindexer, i, nw)
    w1 = 0
    w2 = 0
    @inbounds for ic = 1:3
        w1 += u[2, ic, indices...] * conj(u[1, ic, indices...])
        w2 += u[1, ic, indices...] * conj(u[1, ic, indices...])
    end
    zerock2 = w2
    w1 = -w1 / w2

    x4 = (u[2, 1, indices...]) + w1 * u[1, 1, indices...]
    x5 = (u[2, 2, indices...]) + w1 * u[1, 2, indices...]
    x6 = (u[2, 3, indices...]) + w1 * u[1, 3, indices...]

    w3 = x4 * conj(x4) + x5 * conj(x5) + x6 * conj(x6)

    zerock3 = w3

    u[2, 1, indices...] = x4
    u[2, 2, indices...] = x5
    u[2, 3, indices...] = x6

    w3 = 1 / sqrt(w3)
    w2 = 1 / sqrt(w2)

    u[1, 1, indices...] = u[1, 1, indices...] * w2
    u[1, 2, indices...] = u[1, 2, indices...] * w2
    u[1, 3, indices...] = u[1, 3, indices...] * w2
    u[2, 1, indices...] = u[2, 1, indices...] * w3
    u[2, 2, indices...] = u[2, 2, indices...] * w3
    u[2, 3, indices...] = u[2, 3, indices...] * w3

    aa1 = real(u[1, 1, indices...])
    aa2 = imag(u[1, 1, indices...])
    aa3 = real(u[1, 2, indices...])
    aa4 = imag(u[1, 2, indices...])
    aa5 = real(u[1, 3, indices...])
    aa6 = imag(u[1, 3, indices...])
    aa7 = real(u[2, 1, indices...])
    aa8 = imag(u[2, 1, indices...])
    aa9 = real(u[2, 2, indices...])
    aa10 = imag(u[2, 2, indices...])
    aa11 = real(u[2, 3, indices...])
    aa12 = imag(u[2, 3, indices...])

    aa13 =
        aa3 * aa11 - aa4 * aa12 - aa5 * aa9 + aa6 * aa10
    aa14 =
        aa5 * aa10 + aa6 * aa9 - aa3 * aa12 - aa4 * aa11
    aa15 = aa5 * aa7 - aa6 * aa8 - aa1 * aa11 + aa2 * aa12
    aa16 = aa1 * aa12 + aa2 * aa11 - aa5 * aa8 - aa6 * aa7
    aa17 = aa1 * aa9 - aa2 * aa10 - aa3 * aa7 + aa4 * aa8
    aa18 = aa3 * aa8 + aa4 * aa7 - aa1 * aa10 - aa2 * aa9

    u[3, 1, indices...] = aa13 + im * aa14
    u[3, 2, indices...] = aa15 + im * aa16
    u[3, 3, indices...] = aa17 + im * aa18

end



# ========== device side (generic N) ==========
# Normalize columns in-place to form a unitary (QR with Q-only), per lattice site
@inline function kernel_normalize_generic!(i, u, dindexer, NC, ::Val{nw}) where nw
    # Index decode
    indices = delinearize(dindexer, i, nw)

    # Type helpers
    T = eltype(u)
    rT = real(one(T))
    epsT = sqrt(eps(rT))  # tolerance for near-zero norms

    # Modified Gram–Schmidt over columns j = 1..NC
    @inbounds for j = 1:NC
        # Orthogonalize column j against columns 1..j-1
        for k = 1:j-1
            # inner = ⟨u[:,k], u[:,j]⟩ = sum(conj(u[k]) * u[j])
            inner = zero(T)
            for r = 1:NC
                inner += conj(u[r, k, indices...]) * u[r, j, indices...]
            end
            # u[:,j] -= inner * u[:,k]
            for r = 1:NC
                u[r, j, indices...] -= inner * u[r, k, indices...]
            end
        end

        # Compute 2-norm of column j
        nrm2 = zero(rT)
        for r = 1:NC
            nrm2 += abs2(u[r, j, indices...])
        end
        nrm = sqrt(nrm2)

        # Handle near-zero; fall back to a canonical basis vector
        if nrm < epsT
            # Zero column then set j-th row to 1 (produces consistent unitary completion)
            for r = 1:NC
                u[r, j, indices...] = zero(T)
            end
            u[j, j, indices...] = one(T)
        else
            # Normalize column j
            invn = one(rT) / nrm
            invnT = convert(T, invn)  # keep type stability for Complex/Real T
            for r = 1:NC
                u[r, j, indices...] *= invnT
            end
        end
    end

    # Optional: single re-orthogonalization sweep for improved numerical stability
    # (uncomment if needed)
    # @inbounds for j = 1:NC
    #     for k = 1:j-1
    #         inner = zero(T)
    #         for r = 1:NC
    #             inner += conj(u[r,k,ix,iy,iz,it]) * u[r,j,ix,iy,iz,it]
    #         end
    #         for r = 1:NC
    #             u[r,j,ix,iy,iz,it] -= inner * u[r,k,ix,iy,iz,it]
    #         end
    #     end
    #     nrm2 = zero(rT)
    #     for r = 1:NC
    #         nrm2 += abs2(u[r,j,ix,iy,iz,it])
    #     end
    #     nrm = sqrt(nrm2)
    #     invnT = convert(T, one(rT)/max(nrm, epsT))
    #     for r = 1:NC
    #         u[r,j,ix,iy,iz,it] *= invnT
    #     end
    # end

    return nothing
end

#=
function randomize_matrix!(C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}) where {D,T,AT,NC1,NC2,nw,DI}
    JACC.parallel_for(prod(C.PN), kernel_randomize_4D!, C.A, C.indexer, NC1, NC2)
    #set_halo!(C)
end
export randomize_matrix!

@inline function kernel_randomize_4D!(i1,i2,i3, u, dindexer, NC1, NC2)
    indices = delinearize(dindexer,i,nw)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, indices...] = pcgrand(rng,eltype(u)) - 0.5 + im * (pcgrand(rng,eltype(u)) - 0.5)
        end
    end

end
=#

# Host wrapper: choose a fixed or time-based seed and launch
function randomize_matrix!(C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}) where {D,T,AT,NC1,NC2,nw,DI}
    seed0 = UInt64(0x12345678ABCDEF01)  # or UInt64(time_ns())
    JACC.parallel_for(prod(C.PN), kernel_randomize_4D!, C.A, C.indexer, Val(NC1), Val(NC2), Val(nw), seed0)
    set_halo!(C)
end
export randomize_matrix!

# We split on element type at compile time via Val to avoid dynamic branches.
@inline function kernel_randomize_4D!(i, u, dindexer, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, seed0::UInt64) where {NC1,NC2,nw}
    indices = delinearize(dindexer, i, nw)
    T = eltype(u)

    if T === ComplexF32
        _rand_fill!(Val(:c32), indices, u, Val(NC1), Val(NC2), Val(nw), seed0)
    elseif T === ComplexF64
        _rand_fill!(Val(:c64), indices, u, Val(NC1), Val(NC2), Val(nw), seed0)
    elseif T === Float32
        _rand_fill!(Val(:r32), indices, u, Val(NC1), Val(NC2), Val(nw), seed0)
    elseif T === Float64
        _rand_fill!(Val(:r64), indices, u, Val(NC1), Val(NC2), Val(nw), seed0)
    else
        # If you ever support other types, you can add more specializations.
        # For now, throw a clear error on host side before launching widely.
        @assert false "Unsupported eltype in randomize: $(T)"
    end
    return nothing
end

# --- Specializations (no convert(T, ...) inside) ---

# ComplexF32
@inline function _rand_fill!(::Val{:c32}, indices, u, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, seed0::UInt64) where {NC1,NC2,nw}
    @inbounds for jc = 1:NC2, ic = 1:NC1
        state, inc = mix_seed(indices..., ic, jc, seed0)
        state, r1 = pcg32_step(state, inc)
        state, r2 = pcg32_step(state, inc)
        realv = u01_f32(r1) - 0.5f0
        imagv = u01_f32(r2) - 0.5f0
        u[ic, jc, indices...] = ComplexF32(realv, imagv)
    end
    return nothing
end

# ComplexF64
@inline function _rand_fill!(::Val{:c64}, indices, u, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, seed0::UInt64) where {NC1,NC2,nw}
    @inbounds for jc = 1:NC2, ic = 1:NC1
        state, inc = mix_seed(indices..., ic, jc, seed0)
        state, r1 = pcg32_step(state, inc)
        state, r2 = pcg32_step(state, inc)
        realv = u01_f64(r1, r2) - 0.5
        state, i1 = pcg32_step(state, inc)
        state, i2 = pcg32_step(state, inc)
        imagv = u01_f64(i1, i2) - 0.5
        u[ic, jc, indices...] = ComplexF64(realv, imagv)
    end
    return nothing
end

# Float32
@inline function _rand_fill!(::Val{:r32}, indices, u, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, seed0::UInt64) where {NC1,NC2,nw}
    @inbounds for jc = 1:NC2, ic = 1:NC1
        state, inc = mix_seed(indices..., ic, jc, seed0)
        state, r1 = pcg32_step(state, inc)
        realv = u01_f32(r1) - 0.5f0
        u[ic, jc, indices...] = realv  # already Float32
    end
    return nothing
end

# Float64
@inline function _rand_fill!(::Val{:r64}, indices, u, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, seed0::UInt64) where {NC1,NC2,nw}
    @inbounds for jc = 1:NC2, ic = 1:NC1
        state, inc = mix_seed(indices..., ic, jc, seed0)
        state, r1 = pcg32_step(state, inc)
        state, r2 = pcg32_step(state, inc)
        realv = u01_f64(r1, r2) - 0.5
        u[ic, jc, indices...] = realv  # already Float64
    end
    return nothing
end

function clear_matrix!(C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}) where {D,T,AT,NC1,NC2,nw,DI}
    JACC.parallel_for(prod(C.PN), kernel_clear_4D!, C.A, C.indexer, Val(NC1), Val(NC2), Val(nw))
    set_halo!(C)
end
export clear_matrix!

@inline function kernel_clear_4D!(i, u, dindexer, ::Val{NC1}, ::Val{NC2}, ::Val{nw}) where {NC1,NC2,nw}
    indices = delinearize(dindexer, i, nw)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, indices...] = zero(eltype(u))
        end
    end

end

function makeidentity_matrix!(C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}) where {D,T,AT,NC1,NC2,nw,DI}
    JACC.parallel_for(prod(C.PN), kernel_makeidentity_4D!, C.A, C.indexer, Val(NC1), Val(NC2), Val(nw))
    set_halo!(C)
end
export makeidentity_matrix!


export makeidentity_matrix!

@inline function kernel_makeidentity_4D!(i, u, dindexer, ::Val{NC1}, ::Val{NC2}, ::Val{nw}) where {NC1,NC2,nw}
    indices = delinearize(dindexer, i, nw)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, indices...] = ifelse(ic == jc, one(eltype(u)), zero(eltype(u)))
        end
    end

end


@inline function kernel_makeidentity_4D!(i, u, dindexer, ::Val{3}, ::Val{3}, ::Val{nw}) where {nw}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw
    v1 = one(eltype(u))
    v0 = zero(eltype(u))
    u[1, 1, indices...] = v1
    u[2, 1, indices...] = v0
    u[3, 1, indices...] = v0
    u[1, 2, indices...] = v0
    u[2, 2, indices...] = v1
    u[3, 2, indices...] = v0
    u[1, 3, indices...] = v0
    u[2, 3, indices...] = v0
    u[3, 3, indices...] = v1

end


#C = C+ α*A
function add_matrix!(C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}, A::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI}, α::S=1) where {D,T,T1,AT,AT1,NC1,NC2,nw,S<:Number,DI}
    JACC.parallel_for(prod(C.PN), kernel_add_4D!, C.A, A.A, C.indexer, Val(NC1), Val(NC2), α, Val(nw))
    #set_halo!(C)
end
export add_matrix!

@inline function kernel_add_4D!(i, u, v, dindexer, ::Val{NC1}, ::Val{NC2}, α, ::Val{nw}) where {NC1,NC2,nw}
    indices = delinearize(dindexer, i, nw)
    #println("i = $i ", (indices...))

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, indices...] += α * v[ic, jc, indices...]
        end
    end
    #if i == 1 && NC2 == 4 && NC1 == 3
    #    println("i = $i")
    #    display(u[:, :, indices...])
    #    println("a α = $α")
    #    display(v[:, :, indices...])
    #end
end

#C = C+ α*shiftA
function add_matrix!(C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}, A::Shifted_Lattice{L,shift}, α::S=1) where {D,T,T1,AT,AT1,NC1,NC2,shift,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI}}
    JACC.parallel_for(prod(C.PN), kernel_add_4D_shift!, C.A, A.data.A, C.indexer, Val(NC1), Val(NC2), α, shift, Val(nw))
    #set_halo!(C)
end


@inline function kernel_add_4D_shift!(i, u, v, dindexer, ::Val{NC1}, ::Val{NC2}, α, shift, ::Val{nw}) where {NC1,NC2,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, indices...] += α * v[ic, jc, indices_p...]
        end
    end
end

#C = C+ α*Adag
function add_matrix!(C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}, A::Adjoint_Lattice{L}, α::S=1) where {D,T,T1,AT,AT1,NC1,NC2,nw,S<:Number,DI,L<:LatticeMatrix{D,T1,AT1,NC2,NC1,nw,DI}}
    JACC.parallel_for(prod(C.PN), kernel_add_4D_dag!, C.A, A.data.A, C.indexer, Val(NC1), Val(NC2), α, Val(nw))
    #set_halo!(C)
end

@inline function kernel_add_4D_dag!(i, u, v, dindexer, ::Val{NC1}, ::Val{NC2}, α, ::Val{nw}) where {NC1,NC2,nw}
    indices = delinearize(dindexer, i, nw)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, indices...] += α * v[jc, ic, indices...]'
        end
    end
end

#C = C+ α*shiftAdag
function add_matrix!(C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}, A::Adjoint_Lattice{Shifted_Lattice{L,shift}}, α::S=1) where {D,T,T1,AT,AT1,NC1,NC2,shift,nw,S<:Number,DI,L<:LatticeMatrix{D,T1,AT1,NC2,NC1,nw,DI}}
    JACC.parallel_for(prod(C.PN), kernel_add_4D_shiftdag!, C.A, A.data.data.A, C.indexer, Val(NC1), Val(NC2), α, shift, Val(nw))
    #set_halo!(C)
end


@inline function kernel_add_4D_shiftdag!(i, u, v, dindexer, ::Val{NC1}, ::Val{NC2}, α, shift, ::Val{nw}) where {NC1,NC2,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, indices...] += α * v[jc, ic, indices_p...]'
        end
    end
end

function applyfunction!(C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}, f::Function, variables...) where {D,T,AT,NC1,NC2,nw,DI}
    JACC.parallel_for(prod(C.PN), kernel_apply_function_4D!, C.A, C.indexer, Val(NC1), Val(NC2), Val(nw), f, variables...)
    #set_halo!(C)
end
export applyfunction!

@inline function kernel_apply_function_4D!(i, u, dindexer, ::Val{N1}, ::Val{N2}, ::Val{nw}, f, variables...) where {N1,N2,nw}
    indices = delinearize(dindexer, i, nw)
    At = MMatrix{N1,N2,eltype(u)}(undef)

    @inbounds for jc = 1:N2
        for ic = 1:N1
            At[ic, jc] = u[ic, jc, indices...]
        end
    end
    Aout = f(At, variables...)

    for jc = 1:N2
        for ic = 1:N1
            u[ic, jc, indices...] = Aout[ic, jc]
        end
    end
end

function traceless_antihermitian_add!(C::LatticeMatrix{D,T,AT,NG,1,nw}, factor,
    A::LatticeMatrix{D,T2,AT2,NC,NC,nw2}) where {D,T<:Real,AT,NG,nw,T2,AT2,NC,nw2}
    JACC.parallel_for(prod(C.PN), kernel_4d_Traceless_antihermitian_add!, C.A, A.A, factor, C.indexer, Val(NG), Val(NC), Val(nw), Val(nw2))
end

function kernel_4d_Traceless_antihermitian_add!(i, c, vin, factor, dindexer, ::Val{NG}, ::Val{NC}, ::Val{nw}, ::Val{nw2}) where {NC,NG,nw,nw2}
    error("NC > 3 is not supported in kernel_4d_Traceless_antihermitian_add!")
end

const fac12 = 1 / 2

function kernel_4d_Traceless_antihermitian_add!(i, c, vin, factor, dindexer, ::Val{NG}, ::Val{2}, ::Val{nw}, ::Val{nw2}) where {NG,nw,nw2}
    indices = delinearize(dindexer, i, nw)
    ix2 = ix + nw2
    iy2 = iy + nw2
    iz2 = iz + nw2
    it2 = it + nw2
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    v11 = vin[1, 1, ix2, iy2, iz2, it2]
    v22 = vin[2, 2, ix2, iy2, iz2, it2]

    tri = fac12 * (imag(v11) + imag(v22))

    v12 = vin[1, 2, ix2, iy2, iz2, it2]
    #v13 = vin[1,3,ix,iy,iz,it]
    v21 = vin[2, 1, ix2, iy2, iz2, it2]

    x12 = v12 - conj(v21)

    x21 = -conj(x12)

    y11 = (imag(v11) - tri) * im
    y12 = 0.5 * x12
    y21 = 0.5 * x21
    y22 = (imag(v22) - tri) * im

    c[1, 1, indices...] =
        (imag(y12) + imag(y21)) * factor + c[1, 1, indices...]
    c[2, 1, indices...] =
        (real(y12) - real(y21)) * factor + c[2, 1, indices...]
    c[3, 1, indices...] =
        (imag(y11) - imag(y22)) * factor + c[3, 1, indices...]

end


function kernel_4d_Traceless_antihermitian_add!(i, c, vin, factor, dindexer, ::Val{NG}, ::Val{3}, ::Val{nw}, ::Val{nw2}) where {NG,nw,nw2}
    indices = delinearize(dindexer, i, nw)
    ix2 = ix + nw2
    iy2 = iy + nw2
    iz2 = iz + nw2
    it2 = it + nw2
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    fac13 = 1 / 3


    v11 = vin[1, 1, ix2, iy2, iz2, it2]
    v22 = vin[2, 2, ix2, iy2, iz2, it2]
    v33 = vin[3, 3, ix2, iy2, iz2, it2]

    tri = fac13 * (imag(v11) + imag(v22) + imag(v33))

    #=
    vout[1,1,ix,iy,iz,it] = (imag(v11)-tri)*im
    vout[2,2,ix,iy,iz,it] = (imag(v22)-tri)*im
    vout[3,3,ix,iy,iz,it] = (imag(v33)-tri)*im
    =#
    y11 = (imag(v11) - tri) * im
    y22 = (imag(v22) - tri) * im
    y33 = (imag(v33) - tri) * im

    v12 = vin[1, 2, ix2, iy2, iz2, it2]
    v13 = vin[1, 3, ix2, iy2, iz2, it2]
    v21 = vin[2, 1, ix2, iy2, iz2, it2]
    v23 = vin[2, 3, ix2, iy2, iz2, it2]
    v31 = vin[3, 1, ix2, iy2, iz2, it2]
    v32 = vin[3, 2, ix2, iy2, iz2, it2]

    x12 = v12 - conj(v21)
    x13 = v13 - conj(v31)
    x23 = v23 - conj(v32)

    x21 = -conj(x12)
    x31 = -conj(x13)
    x32 = -conj(x23)

    #=
    vout[1,2,ix,iy,iz,it] = 0.5  * x12
    vout[1,3,ix,iy,iz,it] = 0.5  * x13
    vout[2,1,ix,iy,iz,it] = 0.5  * x21
    vout[2,3,ix,iy,iz,it] = 0.5  * x23
    vout[3,1,ix,iy,iz,it] = 0.5  * x31
    vout[3,2,ix,iy,iz,it] = 0.5  * x32
    =#
    y12 = 0.5 * x12
    y13 = 0.5 * x13
    y21 = 0.5 * x21
    y23 = 0.5 * x23
    y31 = 0.5 * x31
    y32 = 0.5 * x32


    c[1, 1, indices...] =
        (imag(y12) + imag(y21)) * factor + c[1, 1, indices...]
    c[2, 1, indices...] =
        (real(y12) - real(y21)) * factor + c[2, 1, indices...]
    c[3, 1, indices...] =
        (imag(y11) - imag(y22)) * factor + c[3, 1, indices...]
    c[4, 1, indices...] =
        (imag(y13) + imag(y31)) * factor + c[4, 1, indices...]
    c[5, 1, indices...] =
        (real(y13) - real(y31)) * factor + c[5, 1, indices...]

    c[6, 1, indices...] =
        (imag(y23) + imag(y32)) * factor + c[6, 1, indices...]
    c[7, 1, indices...] =
        (real(y23) - real(y32)) * factor + c[7, 1, indices...]
    c[8, 1, indices...] =
        sr3i * (imag(y11) + imag(y22) - 2 * imag(y33)) * factor +
        c[8, 1, indices...]
end