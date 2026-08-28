#Overwrite Y with X*a + Y*b, where a and b are scalars. Return Y.
function LinearAlgebra.axpby!(
    a::Number,
    X::TX,
    b::Number,
    Y::TY,
) where {T1,AT1,NC1,NC2,nw,D,DI,
    TX<:LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},TY<:LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI}}

    _parallel_for_mutating!(Y,
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

@inline function kernel_D_axpby!(i, a, X, b, Y, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)

    Y[1, 1, indices...] = a * X[1, 1, indices...] + b * Y[1, 1, indices...]
    Y[2, 1, indices...] = a * X[2, 1, indices...] + b * Y[2, 1, indices...]
    Y[3, 1, indices...] = a * X[3, 1, indices...] + b * Y[3, 1, indices...]


    Y[1, 2, indices...] = a * X[1, 2, indices...] + b * Y[1, 2, indices...]
    Y[2, 2, indices...] = a * X[2, 2, indices...] + b * Y[2, 2, indices...]
    Y[3, 2, indices...] = a * X[3, 2, indices...] + b * Y[3, 2, indices...]

    Y[1, 3, indices...] = a * X[1, 3, indices...] + b * Y[1, 3, indices...]
    Y[2, 3, indices...] = a * X[2, 3, indices...] + b * Y[2, 3, indices...]
    Y[3, 3, indices...] = a * X[3, 3, indices...] + b * Y[3, 3, indices...]


end

include("mul.jl")




#C = A transpose(B) 
function mul_AtransB!(C::LatticeMatrix{D,T1,AT1,NC1,NC1,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC1,nw,DI}, B::LatticeMatrix{D,T3,AT3,NC1,NC1,nw,DI}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,nw,DI}

    _parallel_for_mutating!(C,
        prod(C.PN), kernel_Dmatrix_mul_AtransB!, C.A, A.A, B.A, Val(NC1), Val(nw), C.indexer
    )
    #set_halo!(C)
end

@inline function kernel_Dmatrix_mul_AtransB!(i, C, A, B, ::Val{NC1}, ::Val{nw}, dindexer) where {NC1,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC1
        for ic = 1:NC1
            C[ic, jc, indices...] = zero(eltype(C))
        end

        for kc = 1:NC1
            b = B[jc, kc, indices...]
            for ic = 1:NC1
                C[ic, jc, indices...] += A[ic, kc, indices...] * b# B[kc, jc, indices...]
            end
        end
    end
end

@inline function kernel_Dmatrix_mul_AtransB!(i, C, A, B, ::Val{3}, ::Val{nw}, dindexer) where {nw}
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
        b21 = B[1, 2, indices...]
        b31 = B[1, 3, indices...]
        b12 = B[2, 1, indices...]
        b22 = B[2, 2, indices...]
        b32 = B[2, 3, indices...]
        b13 = B[3, 1, indices...]
        b23 = B[3, 2, indices...]
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

@inline function kernel_Dmatrix_mul_AtransB!(i, C, A, B, ::Val{2}, ::Val{nw}, dindexer) where {nw}
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
        b21 = B[1, 2, indices...]
        b12 = B[2, 1, indices...]
        b22 = B[2, 2, indices...]

        C[1, 1, indices...] = a11 * b11 + a12 * b21
        C[2, 1, indices...] = a21 * b11 + a22 * b21
        C[1, 2, indices...] = a11 * b12 + a12 * b22
        C[2, 2, indices...] = a21 * b12 + a22 * b22

    end
end







function expt!(C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}, A::Traceless_AntiHermitian{L}, t=1) where {
    D,T,AT,NC1,NC2,T1,AT1,nw,DI,L<:LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI}}

    expt_TA!(C, A.data, t)
    return
    #set_halo!(C)
end

function expt_TA!(C::TC, A::TA, t::S=one(S)) where {
    D,T,AT,NC1,NC2,S<:Number,nw,DI,TC<:LatticeMatrix{D,T,AT,NC1,NC2,nw,DI},TA<:LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}}
    if NC1 == 3 && T <: Complex && S <: Real
        _parallel_for_mutating!(C,
            prod(C.PN), kernel_4Dexpt_TA_su3_ch!,
            C.A, A.A, C.indexer, Val(nw), t,
        )
    else
        traceless_antihermitian!(C, A)
        _parallel_for_mutating!(C,
            prod(C.PN), kernel_4Dexpt_TA!, C.A, C.indexer, Val(nw), t, Val(NC1)
        )
    end
    return
    #set_halo!(C)
end
export expt_TA!

@inline function _writeback_exp_iQ_su3_ch!(
    C, indices,
    q11, q12, q13, q21, q22, q23, q31, q32, q33,
)
    c11, c12, c13, c21, c22, c23, c31, c32, c33 = _exp_iQ_su3_ch(
        q11, q12, q13,
        q21, q22, q23,
        q31, q32, q33,
    )
    @inbounds begin
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
    return nothing
end

@inline function kernel_4Dexpt_TA_su3_ch!(
    i, C, A, dindexer, ::Val{nw}, t,
) where {nw}
    indices = delinearize(dindexer, i, nw)
    RT = typeof(real(zero(eltype(C))))
    zero_r = zero(RT)
    half = one(RT) / RT(2)
    third = one(RT) / RT(3)
    tt = RT(t)

    @inbounds begin
        a11 = A[1, 1, indices...]
        a12 = A[1, 2, indices...]
        a13 = A[1, 3, indices...]
        a21 = A[2, 1, indices...]
        a22 = A[2, 2, indices...]
        a23 = A[2, 3, indices...]
        a31 = A[3, 1, indices...]
        a32 = A[3, 2, indices...]
        a33 = A[3, 3, indices...]
    end

    tri = third * (imag(a11) + imag(a22) + imag(a33))
    y11 = complex(zero_r, imag(a11) - tri)
    y22 = complex(zero_r, imag(a22) - tri)
    y33 = complex(zero_r, imag(a33) - tri)
    y12 = half * (a12 - conj(a21))
    y13 = half * (a13 - conj(a31))
    y23 = half * (a23 - conj(a32))
    y21 = -conj(y12)
    y31 = -conj(y13)
    y32 = -conj(y23)

    # Q = -i*t*Y is Hermitian and exp(tY) = exp(iQ).
    q11 = complex(tt * imag(y11), -tt * real(y11))
    q12 = complex(tt * imag(y12), -tt * real(y12))
    q13 = complex(tt * imag(y13), -tt * real(y13))
    q21 = complex(tt * imag(y21), -tt * real(y21))
    q22 = complex(tt * imag(y22), -tt * real(y22))
    q23 = complex(tt * imag(y23), -tt * real(y23))
    q31 = complex(tt * imag(y31), -tt * real(y31))
    q32 = complex(tt * imag(y32), -tt * real(y32))
    q33 = complex(tt * imag(y33), -tt * real(y33))

    _writeback_exp_iQ_su3_ch!(
        C, indices,
        q11, q12, q13,
        q21, q22, q23,
        q31, q32, q33,
    )
    return nothing
end

@inline function kernel_4Dexpt_TA_basis_su3_ch!(
    i, C, A, dindexer, ::Val{nw}, t, ::Val{nw2},
) where {nw,nw2}
    indices = delinearize(dindexer, i, nw)
    indices2 = delinearize(dindexer, i, nw2)
    RT = typeof(real(zero(eltype(C))))
    zero_r = zero(RT)
    half_t = RT(t) / RT(2)
    inv_sqrt_three = inv(sqrt(RT(3)))

    @inbounds begin
        c1 = half_t * RT(A[1, 1, indices2...])
        c2 = half_t * RT(A[2, 1, indices2...])
        c3 = half_t * RT(A[3, 1, indices2...])
        c4 = half_t * RT(A[4, 1, indices2...])
        c5 = half_t * RT(A[5, 1, indices2...])
        c6 = half_t * RT(A[6, 1, indices2...])
        c7 = half_t * RT(A[7, 1, indices2...])
        c8 = half_t * RT(A[8, 1, indices2...])
    end

    q11 = complex(c3 + inv_sqrt_three * c8, zero_r)
    q12 = complex(c1, -c2)
    q13 = complex(c4, -c5)
    q21 = conj(q12)
    q22 = complex(-c3 + inv_sqrt_three * c8, zero_r)
    q23 = complex(c6, -c7)
    q31 = conj(q13)
    q32 = conj(q23)
    q33 = complex(-RT(2) * inv_sqrt_three * c8, zero_r)

    _writeback_exp_iQ_su3_ch!(
        C, indices,
        q11, q12, q13,
        q21, q22, q23,
        q31, q32, q33,
    )
    return nothing
end

@inline function _writeback_expt3x3_pade!(
    C, indices, a11, a12, a13, a21, a22, a23, a31, a32, a33, t
)
    c11, c12, c13, c21, c22, c23, c31, c32, c33 =
        exp3x3_pade(a11, a12, a13, a21, a22, a23, a31, a32, a33, t)
    C[1, 1, indices...] = c11
    C[1, 2, indices...] = c12
    C[1, 3, indices...] = c13
    C[2, 1, indices...] = c21
    C[2, 2, indices...] = c22
    C[2, 3, indices...] = c23
    C[3, 1, indices...] = c31
    C[3, 2, indices...] = c32
    C[3, 3, indices...] = c33
    return
end

@inline function _writeback_expt3x3_taylor4!(
    C, indices, a11, a12, a13, a21, a22, a23, a31, a32, a33, t
)
    m11 = t * a11
    m12 = t * a12
    m13 = t * a13
    m21 = t * a21
    m22 = t * a22
    m23 = t * a23
    m31 = t * a31
    m32 = t * a32
    m33 = t * a33

    m211 = m11 * m11 + m12 * m21 + m13 * m31
    m212 = m11 * m12 + m12 * m22 + m13 * m32
    m213 = m11 * m13 + m12 * m23 + m13 * m33
    m221 = m21 * m11 + m22 * m21 + m23 * m31
    m222 = m21 * m12 + m22 * m22 + m23 * m32
    m223 = m21 * m13 + m22 * m23 + m23 * m33
    m231 = m31 * m11 + m32 * m21 + m33 * m31
    m232 = m31 * m12 + m32 * m22 + m33 * m32
    m233 = m31 * m13 + m32 * m23 + m33 * m33

    m311 = m211 * m11 + m212 * m21 + m213 * m31
    m312 = m211 * m12 + m212 * m22 + m213 * m32
    m313 = m211 * m13 + m212 * m23 + m213 * m33
    m321 = m221 * m11 + m222 * m21 + m223 * m31
    m322 = m221 * m12 + m222 * m22 + m223 * m32
    m323 = m221 * m13 + m222 * m23 + m223 * m33
    m331 = m231 * m11 + m232 * m21 + m233 * m31
    m332 = m231 * m12 + m232 * m22 + m233 * m32
    m333 = m231 * m13 + m232 * m23 + m233 * m33

    m411 = m311 * m11 + m312 * m21 + m313 * m31
    m412 = m311 * m12 + m312 * m22 + m313 * m32
    m413 = m311 * m13 + m312 * m23 + m313 * m33
    m421 = m321 * m11 + m322 * m21 + m323 * m31
    m422 = m321 * m12 + m322 * m22 + m323 * m32
    m423 = m321 * m13 + m322 * m23 + m323 * m33
    m431 = m331 * m11 + m332 * m21 + m333 * m31
    m432 = m331 * m12 + m332 * m22 + m333 * m32
    m433 = m331 * m13 + m332 * m23 + m333 * m33

    c2 = 0.5
    c3 = 1.0 / 6.0
    c4 = 1.0 / 24.0

    C[1, 1, indices...] = one(eltype(C)) + m11 + c2 * m211 + c3 * m311 + c4 * m411
    C[1, 2, indices...] = m12 + c2 * m212 + c3 * m312 + c4 * m412
    C[1, 3, indices...] = m13 + c2 * m213 + c3 * m313 + c4 * m413
    C[2, 1, indices...] = m21 + c2 * m221 + c3 * m321 + c4 * m421
    C[2, 2, indices...] = one(eltype(C)) + m22 + c2 * m222 + c3 * m322 + c4 * m422
    C[2, 3, indices...] = m23 + c2 * m223 + c3 * m323 + c4 * m423
    C[3, 1, indices...] = m31 + c2 * m231 + c3 * m331 + c4 * m431
    C[3, 2, indices...] = m32 + c2 * m232 + c3 * m332 + c4 * m432
    C[3, 3, indices...] = one(eltype(C)) + m33 + c2 * m233 + c3 * m333 + c4 * m433
    return
end


@inline function kernel_4Dexpt_TA!(i, C, dindexer, ::Val{nw}, t, ::Val{3}) where nw
    indices = delinearize(dindexer, i, nw)

    a11 = C[1, 1, indices...]
    a12 = C[1, 2, indices...]
    a13 = C[1, 3, indices...]
    a21 = C[2, 1, indices...]
    a22 = C[2, 2, indices...]
    a23 = C[2, 3, indices...]
    a31 = C[3, 1, indices...]
    a32 = C[3, 2, indices...]
    a33 = C[3, 3, indices...]

    v11 = C[1, 1, indices...]
    v22 = C[2, 2, indices...]
    v33 = C[3, 3, indices...]
    tri = fac13 * (imag(v11) + imag(v22) + imag(v33))


    y11 = (imag(v11) - tri) * im
    y22 = (imag(v22) - tri) * im
    y33 = (imag(v33) - tri) * im

    v12 = C[1, 2, indices...]
    v13 = C[1, 3, indices...]
    v21 = C[2, 1, indices...]
    v23 = C[2, 3, indices...]
    v31 = C[3, 1, indices...]
    v32 = C[3, 2, indices...]

    x12 = v12 - conj(v21)
    x13 = v13 - conj(v31)
    x23 = v23 - conj(v32)

    x21 = -conj(x12)
    x31 = -conj(x13)
    x32 = -conj(x23)

    y12 = 0.5 * x12
    y13 = 0.5 * x13
    y21 = 0.5 * x21
    y23 = 0.5 * x23
    y31 = 0.5 * x31
    y32 = 0.5 * x32

    # Small-Q branch (cold-start friendly): avoid fragile eigenvalue path.
    qnorm =
        abs(t) * sqrt(real(
            y11 * conj(y11) + y12 * conj(y12) + y13 * conj(y13) +
            y21 * conj(y21) + y22 * conj(y22) + y23 * conj(y23) +
            y31 * conj(y31) + y32 * conj(y32) + y33 * conj(y33)
        ))
    if qnorm <= 1e-6
        _writeback_expt3x3_taylor4!(
            C, indices,
            y11, y12, y13,
            y21, y22, y23,
            y31, y32, y33,
            t
        )
        return
    end

    c1_0 = (imag(y12) + imag(y21))
    c2_0 = (real(y12) - real(y21))
    c3_0 = (imag(y11) - imag(y22))
    c4_0 = (imag(y13) + imag(y31))
    c5_0 = (real(y13) - real(y31))

    c6_0 = (imag(y23) + imag(y32))
    c7_0 = (real(y23) - real(y32))
    c8_0 = sr3i * (imag(y11) + imag(y22) - 2 * imag(y33))

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
        C[1, 1, indices...] = 1
        C[1, 2, indices...] = 0
        C[1, 3, indices...] = 0
        C[2, 1, indices...] = 0
        C[2, 2, indices...] = 1
        C[2, 3, indices...] = 0
        C[3, 1, indices...] = 0
        C[3, 2, indices...] = 0
        C[3, 3, indices...] = 1
        return
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
    if !(isfinite(p3) && isfinite(q)) || p3 >= -tinyvalue
        _writeback_expt3x3_pade!(C, indices, a11, a12, a13, a21, a22, a23, a31, a32, a33, t)
        return
    end
    x = sqrt(-4.0 * p3) + tinyvalue
    denom = x * p3
    if !isfinite(denom) || abs(denom) <= tinyvalue
        _writeback_expt3x3_pade!(C, indices, a11, a12, a13, a21, a22, a23, a31, a32, a33, t)
        return
    end
    arg = q / denom

    if !isfinite(arg)
        _writeback_expt3x3_pade!(C, indices, a11, a12, a13, a21, a22, a23, a31, a32, a33, t)
        return
    end
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

    n1 = w1^2 + w2^2 + w3^2 + w4^2 + w5^2
    if !(isfinite(n1) && n1 > tinyvalue)
        _writeback_expt3x3_pade!(C, indices, a11, a12, a13, a21, a22, a23, a31, a32, a33, t)
        return
    end
    coeff = 1.0 / sqrt(n1)


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

    n2 = w7^2 + w8^2 + w9^2 + w10^2 + w11^2
    if !(isfinite(n2) && n2 > tinyvalue)
        _writeback_expt3x3_pade!(C, indices, a11, a12, a13, a21, a22, a23, a31, a32, a33, t)
        return
    end
    coeff = 1.0 / sqrt(n2)

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

    n3 = w13^2 + w14^2 + w15^2 + w16^2 + w17^2
    if !(isfinite(n3) && n3 > tinyvalue)
        _writeback_expt3x3_pade!(C, indices, a11, a12, a13, a21, a22, a23, a31, a32, a33, t)
        return
    end
    coeff = 1.0 / sqrt(n3)
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

    w11m = w1 + im * w2
    w12m = w3 + im * w4
    w13m = w5 + im * w6
    w21m = w7 + im * w8
    w22m = w9 + im * w10
    w23m = w11 + im * w12
    w31m = w13 + im * w14
    w32m = w15 + im * w16
    w33m = w17 + im * w18

    ww11m = ww1 + im * ww2
    ww12m = ww3 + im * ww4
    ww13m = ww5 + im * ww6
    ww21m = ww7 + im * ww8
    ww22m = ww9 + im * ww10
    ww23m = ww11 + im * ww12
    ww31m = ww13 + im * ww14
    ww32m = ww15 + im * ww16
    ww33m = ww17 + im * ww18

    #mul!(uout, w', ww)

    C[1, 1, indices...] = w11m' * ww11m + w21m' * ww21m + w31m' * ww31m
    C[1, 2, indices...] = w11m' * ww12m + w21m' * ww22m + w31m' * ww32m
    C[1, 3, indices...] = w11m' * ww13m + w21m' * ww23m + w31m' * ww33m
    C[2, 1, indices...] = w12m' * ww11m + w22m' * ww21m + w32m' * ww31m
    C[2, 2, indices...] = w12m' * ww12m + w22m' * ww22m + w32m' * ww32m
    C[2, 3, indices...] = w12m' * ww13m + w22m' * ww23m + w32m' * ww33m
    C[3, 1, indices...] = w13m' * ww11m + w23m' * ww21m + w33m' * ww31m
    C[3, 2, indices...] = w13m' * ww12m + w23m' * ww22m + w33m' * ww32m
    C[3, 3, indices...] = w13m' * ww13m + w23m' * ww23m + w33m' * ww33m


end

@inline function kernel_4Dexpt_TA!(i, C, dindexer, ::Val{nw}, t, ::Val{2}) where nw
    indices = delinearize(dindexer, i, nw)
    v11 = C[1, 1, indices...]
    v22 = C[2, 2, indices...]

    tri = fac12 * (imag(v11) + imag(v22))



    v12 = C[1, 2, indices...]
    #v13 = vin[1,3,ix,iy,iz,it]
    v21 = C[2, 1, indices...]

    x12 = v12 - conj(v21)

    x21 = -conj(x12)

    y11 = (imag(v11) - tri) * im
    y12 = 0.5 * x12
    y21 = 0.5 * x21
    y22 = (imag(v22) - tri) * im

    c1_0 = (imag(y12) + imag(y21))
    c2_0 = (real(y12) - real(y21))
    c3_0 = (imag(y11) - imag(y22))

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

    C[1, 1, indices...] = cos(R) + im * a3
    C[1, 2, indices...] = im * a1 + a2
    C[2, 1, indices...] = im * a1 - a2
    C[2, 2, indices...] = cos(R) - im * a3

end



@inline function kernel_4Dexpt_TA!(i, C, dindexer, ::Val{nw}, t, ::Val{N}) where {N,nw}
    indices = delinearize(dindexer, i, nw)
    expm_pade13_writeback!(C, C, indices..., t, Val(N))
    #C[:, :, indices...] = expm_pade13(A[:, :, indices...], t)
end

function expt!(C::TC, A::TA, t=1) where {D,T,AT,NC1,NC2,T1,AT1,nw,DI,TA<:LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},TC<:LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}}
    @assert NC1 == NC2 "Matrix exponentiation requires square matrices, but got $(NC1) x $(NC2)."

    _parallel_for_mutating!(C,
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

@inline function _su3_xi0_stable(w)
    RT = typeof(w)
    w2 = w * w
    if abs(w) > RT(0.05)
        return sin(w) / w
    end
    return one(RT) - (w2 / RT(6)) *
        (one(RT) - (w2 / RT(20)) * (one(RT) - w2 / RT(42)))
end

@inline function _su3_xi1_stable(w)
    RT = typeof(w)
    w2 = w * w
    if abs(w) > RT(0.05)
        return cos(w) / w2 - sin(w) / (w2 * w)
    end
    return -one(RT) / RT(3) + (w2 / RT(30)) *
        (one(RT) - (w2 / RT(28)) * (one(RT) - w2 / RT(54)))
end

# Morningstar--Peardon analytic coefficients for exp(iH) and their
# derivatives with respect to c1=tr(H^2)/2 and c0=tr(H^3)/3.  Negative c0
# is evaluated through the exact reflection symmetries of Eqs. (34) and (70),
# avoiding the vanishing denominator at the negative degenerate endpoint.
@inline function _su3_exp_coefficients_analytic(c0, c1)
    RT = typeof(c1)
    two = RT(2)
    three = RT(3)
    nine = RT(9)

    reflected = c0 < zero(RT)
    c0abs = abs(c0)
    c0max = two * (c1 / three) * sqrt(c1 / three)
    ratio = clamp(c0abs / c0max, zero(RT), one(RT))
    theta = acos(ratio)
    u = sqrt(c1 / three) * cos(theta / three)
    w = sqrt(c1) * sin(theta / three)
    xi0 = _su3_xi0_stable(w)
    xi1 = _su3_xi1_stable(w)

    emiu = exp(-im * u)
    e2iu = exp(two * im * u)

    h0 =
        (u^2 - w^2) * e2iu +
        emiu * (RT(8) * u^2 * cos(w) + two * im * u * (three * u^2 + w^2) * xi0)
    h1 =
        two * u * e2iu -
        emiu * (two * u * cos(w) - im * (three * u^2 - w^2) * xi0)
    h2 = e2iu - emiu * (cos(w) + three * im * u * xi0)

    denom = nine * u^2 - w^2
    f0 = h0 / denom
    f1 = h1 / denom
    f2 = h2 / denom

    r10 =
        two * (u + im * (u^2 - w^2)) * e2iu +
        two * emiu * (
            RT(4) * u * (two - im * u) * cos(w) +
            im * (nine * u^2 + w^2 - im * u * (three * u^2 + w^2)) * xi0
        )
    r11 =
        two * (one(RT) + two * im * u) * e2iu +
        emiu * (
            -two * (one(RT) - im * u) * cos(w) +
            im * (RT(6) * u + im * (w^2 - three * u^2)) * xi0
        )
    r12 = two * im * e2iu + im * emiu * (cos(w) - three * (one(RT) - im * u) * xi0)
    r20 =
        -two * e2iu +
        two * im * u * emiu * (
            cos(w) + (one(RT) + RT(4) * im * u) * xi0 + three * u^2 * xi1
        )
    r21 =
        -im * emiu * (
            cos(w) + (one(RT) + two * im * u) * xi0 - three * u^2 * xi1
        )
    r22 = emiu * (xi0 - three * im * u * xi1)

    denom2 = two * denom^2
    b10 = (
        two * u * r10 + (three * u^2 - w^2) * r20 -
        two * (RT(15) * u^2 + w^2) * f0
    ) / denom2
    b11 = (
        two * u * r11 + (three * u^2 - w^2) * r21 -
        two * (RT(15) * u^2 + w^2) * f1
    ) / denom2
    b12 = (
        two * u * r12 + (three * u^2 - w^2) * r22 -
        two * (RT(15) * u^2 + w^2) * f2
    ) / denom2
    b20 = (r10 - three * u * r20 - RT(24) * u * f0) / denom2
    b21 = (r11 - three * u * r21 - RT(24) * u * f1) / denom2
    b22 = (r12 - three * u * r22 - RT(24) * u * f2) / denom2

    if reflected
        f0 = conj(f0)
        f1 = -conj(f1)
        f2 = conj(f2)
        b10 = conj(b10)
        b11 = -conj(b11)
        b12 = conj(b12)
        b20 = -conj(b20)
        b21 = conj(b21)
        b22 = -conj(b22)
    end

    return f0, f1, f2, b10, b11, b12, b20, b21, b22
end

@inline function _exp_ta_series_threshold(::Type{RT}) where {RT<:AbstractFloat}
    return RT(64) * sqrt(eps(RT))
end

# For a small X=tA, evaluate t*L_exp(X,C) from the power series of the
# block-matrix exponential.  This is a pullback representative for the trace
# pairing and avoids all invariant-coefficient cancellations at the origin.
@inline function _exp_ta_pullback_taylor!(
    result::MMatrix{N,N,T}, X::MMatrix{N,N,T}, C::MMatrix{N,N,T}, t,
) where {N,T}
    RT = typeof(real(zero(T)))
    power = MMatrix{N,N,T}(undef)
    upper = MMatrix{N,N,T}(undef)
    left = MMatrix{N,N,T}(undef)
    right = MMatrix{N,N,T}(undef)
    nextpower = MMatrix{N,N,T}(undef)

    @inbounds for jc = 1:N, ic = 1:N
        result[ic, jc] = C[ic, jc]
        power[ic, jc] = X[ic, jc]
        upper[ic, jc] = C[ic, jc]
    end

    factorial = one(RT)
    for order = 2:12
        gemm!(left, power, C)
        gemm!(right, upper, X)
        factorial *= RT(order)
        @inbounds for jc = 1:N, ic = 1:N
            upper[ic, jc] = left[ic, jc] + right[ic, jc]
            result[ic, jc] += upper[ic, jc] / factorial
        end
        gemm!(nextpower, power, X)
        @inbounds for jc = 1:N, ic = 1:N
            power[ic, jc] = nextpower[ic, jc]
        end
    end

    @inbounds for jc = 1:N, ic = 1:N
        result[ic, jc] *= t
    end
    return nothing
end

@inline function _store_exp_ta_pullback!(
    output, result::MMatrix{N,N,T}, indices, ::Val{false},
) where {N,T}
    @inbounds for jc = 1:N, ic = 1:N
        output[ic, jc, indices...] = result[ic, jc]
    end
    return nothing
end

@inline function _exp_ta_cotangent_entry(
    cotangent, ic, jc, indices, ::Val{false},
)
    return cotangent[ic, jc, indices...]
end


@inline function _exp_ta_cotangent_entry(
    cotangent, ic, jc, indices, ::Val{true},
)
    return conj(cotangent[jc, ic, indices...])
end

@inline function _store_exp_ta_pullback!(
    output, result::MMatrix{N,N,T}, indices, ::Val{true},
) where {N,T}
    RT = typeof(real(zero(T)))
    trace_imag = zero(RT)
    @inbounds for ic = 1:N
        trace_imag += imag(result[ic, ic])
    end
    trace_imag /= RT(N)
    @inbounds for ic = 1:N
        output[ic, ic, indices...] -= (imag(result[ic, ic]) - trace_imag) * im
    end
    @inbounds for jc = 2:N, ic = 1:jc-1
        value = (result[ic, jc] - conj(result[jc, ic])) / RT(2)
        output[ic, jc, indices...] -= value
        output[jc, ic, indices...] += conj(value)
    end
    return nothing
end

@inline function kernel_exp_ta_pullback_su2!(
    i, output, A, cotangent, dindexer, ::Val{nw}, t, accumulate_ta::Val,
) where {nw}
    indices = delinearize(dindexer, i, nw)
    T = eltype(output)
    RT = typeof(real(zero(T)))
    Q = MMatrix{2,2,T}(undef)
    C = MMatrix{2,2,T}(undef)
    result = MMatrix{2,2,T}(undef)

    a11 = A[1, 1, indices...]
    a12 = A[1, 2, indices...]
    a21 = A[2, 1, indices...]
    a22 = A[2, 2, indices...]
    trace_imag = (imag(a11) + imag(a22)) / RT(2)
    Q[1, 1] = t * (imag(a11) - trace_imag) * im
    Q[2, 2] = t * (imag(a22) - trace_imag) * im
    Q[1, 2] = t * (a12 - conj(a21)) / RT(2)
    Q[2, 1] = -conj(Q[1, 2])
    @inbounds for jc = 1:2, ic = 1:2
        C[ic, jc] = _exp_ta_cotangent_entry(
            cotangent, ic, jc, indices, accumulate_ta,
        )
    end

    trQ2 = Q[1, 1]^2 + Q[1, 2] * Q[2, 1] +
            Q[2, 1] * Q[1, 2] + Q[2, 2]^2
    q2 = max(zero(RT), real(-trQ2 / RT(2)))
    if q2 <= _exp_ta_series_threshold(RT)
        _exp_ta_pullback_taylor!(result, Q, C, t)
    else
        q = sqrt(q2)
        sinq, cosq = sincos(q)
        f = sinq / q
        b0 = f / RT(2)
        b1 = (sinq - q * cosq) / (RT(2) * q^3)
        trCB = zero(T)
        @inbounds for jc = 1:2, ic = 1:2
            bij = b1 * Q[jc, ic]
            ic == jc && (bij += b0)
            trCB += C[ic, jc] * bij
        end
        @inbounds for jc = 1:2, ic = 1:2
            result[ic, jc] = t * (f * C[ic, jc] + trCB * Q[ic, jc])
        end
    end
    _store_exp_ta_pullback!(output, result, indices, accumulate_ta)
    return nothing
end

@inline function kernel_exp_ta_pullback_su3!(
    i, output, A, cotangent, dindexer, ::Val{nw}, t, accumulate_ta::Val,
) where {nw}
    indices = delinearize(dindexer, i, nw)
    T = eltype(output)
    RT = typeof(real(zero(T)))
    Q = MMatrix{3,3,T}(undef)
    H = MMatrix{3,3,T}(undef)
    H2 = MMatrix{3,3,T}(undef)
    C = MMatrix{3,3,T}(undef)
    result = MMatrix{3,3,T}(undef)

    trace_imag = zero(RT)
    @inbounds for ic = 1:3
        trace_imag += imag(A[ic, ic, indices...])
    end
    trace_imag /= RT(3)
    @inbounds for ic = 1:3
        Q[ic, ic] = t * (imag(A[ic, ic, indices...]) - trace_imag) * im
    end
    @inbounds for jc = 2:3, ic = 1:jc-1
        value = t * (A[ic, jc, indices...] - conj(A[jc, ic, indices...])) / RT(2)
        Q[ic, jc] = value
        Q[jc, ic] = -conj(value)
    end
    @inbounds for jc = 1:3, ic = 1:3
        H[ic, jc] = Q[ic, jc] / im
        C[ic, jc] = _exp_ta_cotangent_entry(
            cotangent, ic, jc, indices, accumulate_ta,
        )
    end
    gemm!(H2, H, H)
    c1 = real(H2[1, 1] + H2[2, 2] + H2[3, 3]) / RT(2)

    if c1 <= _exp_ta_series_threshold(RT)
        _exp_ta_pullback_taylor!(result, Q, C, t)
    else
        c0 = real(
            H2[1, 1] * H[1, 1] + H2[1, 2] * H[2, 1] + H2[1, 3] * H[3, 1] +
            H2[2, 1] * H[1, 2] + H2[2, 2] * H[2, 2] + H2[2, 3] * H[3, 2] +
            H2[3, 1] * H[1, 3] + H2[3, 2] * H[2, 3] + H2[3, 3] * H[3, 3]
        ) / RT(3)
        _, f1, f2, b10, b11, b12, b20, b21, b22 =
            _su3_exp_coefficients_analytic(c0, c1)

        B1 = MMatrix{3,3,T}(undef)
        B2 = MMatrix{3,3,T}(undef)
        @inbounds for jc = 1:3, ic = 1:3
            identity_entry = ic == jc ? one(T) : zero(T)
            B1[ic, jc] = b10 * identity_entry + b11 * H[ic, jc] + b12 * H2[ic, jc]
            B2[ic, jc] = b20 * identity_entry + b21 * H[ic, jc] + b22 * H2[ic, jc]
        end

        trCB1 = zero(T)
        trCB2 = zero(T)
        @inbounds for jc = 1:3, ic = 1:3
            trCB1 += C[ic, jc] * B1[jc, ic]
            trCB2 += C[ic, jc] * B2[jc, ic]
        end

        factor = t / im
        @inbounds for jc = 1:3, ic = 1:3
            hc = zero(T)
            ch = zero(T)
            for kc = 1:3
                hc += H[ic, kc] * C[kc, jc]
                ch += C[ic, kc] * H[kc, jc]
            end
            derivative =
                trCB1 * H[ic, jc] + f1 * C[ic, jc] +
                trCB2 * H2[ic, jc] + f2 * (hc + ch)
            result[ic, jc] = factor * derivative
        end
    end
    _store_exp_ta_pullback!(output, result, indices, accumulate_ta)
    return nothing
end

"""
    exp_ta_pullback!(output, cotangent, A, t=1)

Compute a representative of the pullback of `exp(t * TA(A))` with respect to
the bilinear trace pairing, restricted to traceless anti-Hermitian variations
of `A`. The SU(2) and SU(3) site-local kernels use the same JACC path on CPU
and accelerator backends.
"""
function exp_ta_pullback!(
    output::TO,
    cotangent::TC,
    A::TA,
    t::S=1,
) where {
    D,T,AT,NC,nw,DI,S<:Real,
    TO<:LatticeMatrix{D,T,AT,NC,NC,nw,DI},
    TC<:LatticeMatrix{D,T,AT,NC,NC,nw,DI},
    TA<:LatticeMatrix{D,T,AT,NC,NC,nw,DI},
}
    kernel = if NC == 2
        kernel_exp_ta_pullback_su2!
    elseif NC == 3
        kernel_exp_ta_pullback_su3!
    else
        throw(ArgumentError(
            "exp_ta_pullback! is implemented only for SU(2) and SU(3), got NC=$NC",
        ))
    end
    _parallel_for_mutating!(
        output,
        prod(output.PN),
        kernel,
        output.A,
        A.A,
        cotangent.A,
        output.indexer,
        Val(nw),
        t,
        Val(false),
    )
    return nothing
end
export exp_ta_pullback!

#=
function expt_TA!(C::TC, TA::TTA, t::S=one(S)) where {D,T,AT,NC1,
    S<:Number,T1,AT1,nw,nw2,DI,TC<:LatticeMatrix{D,T,AT,NC1,NC1,nw,DI},
    TTA<:LatticeMatrix{D,T1,AT1,NC1,NC1,nw2,DI}}

    if NC1 > 3
        error("In NC > 3 case, this function should not be used")
    else
        _parallel_for_mutating!(C,
            prod(C.PN), kernel_4Dexpt_TA_general!, C.A, TA.A, C.indexer, Val(nw), t, Val(NC1), Val(nw2)
        )
    end
    return
    #set_halo!(C)
end
export expt_TA!

function kernel_4Dexpt_TA_general!(i, uout, A, dindexer, ::Val{nw}, t, ::Val{2}, ::Val{nw2}) where {nw,nw2}
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

    # A is assumed anti-Hermitian P. Use V = -i*A (Hermitian).
    # For z = x + i y,  -i z = y - i x  ⇒ real(V)=imag(A), imag(V)=-real(A)

    # off-diagonal: V12 = (c1 - i c2)/2
    # c1 = 2*real(V12) = 2*imag(A12)
    # c2 = -2*imag(V12) = -2*(-real(A12)) = 2*real(A12)
    c1_0 = 2 * imag(A[1, 2, indices2...])
    c2_0 = 2 * real(A[1, 2, indices2...])

    # diagonal: V11 = c3/2, but V11 = imag(A11) for anti-Hermitian A11 = i*α
    c3_0 = 2 * imag(A[1, 1, indices2...])

    #c1_0 = A[1, 1, indices2...]
    #c2_0 = A[2, 1, indices2...]
    #c3_0 = A[3, 1, indices2...]

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


function kernel_4Dexpt_TA_general!(i, C, A, dindexer, ::Val{nw}, t, ::Val{3}, ::Val{nw2}) where {nw,nw2}
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

    # A is assumed anti-Hermitian (P).  We use V = -i*A (Hermitian).
    # For any complex z = x + i y:
    #   V = -i z = y - i x
    # so real(V) = imag(z), imag(V) = -real(z).

    # --- off-diagonals ---
    # Original (Hermitian V): c1_0 = real(V12), c2_0 = -imag(V12)
    # With V12 = -i*A12:
    #   real(V12) = imag(A12)
    #   imag(V12) = -real(A12)
    # => c1_0 = imag(A12), c2_0 = real(A12)

    c1_0 = imag(A[1, 2, indices2...])
    c2_0 = real(A[1, 2, indices2...])

    c4_0 = imag(A[1, 3, indices2...])
    c5_0 = real(A[1, 3, indices2...])

    c6_0 = imag(A[2, 3, indices2...])
    c7_0 = real(A[2, 3, indices2...])

    # --- diagonals ---
    # For anti-Hermitian A, diagonal is purely imaginary: A11 = i α, etc.
    # V11 = -i*A11 = α (real)
    a = imag(A[1, 1, indices2...])
    b = imag(A[2, 2, indices2...])
    d = imag(A[3, 3, indices2...])   # not used below but good for sanity

    # c3 from difference
    c3_0 = (a - b) / 2

    # c8 from sum (a+b = 2*sr3i*c8)
    c8_0 = (a + b) / (2 * sr3i)
    #=
        c1_0 = A[1, 1, indices2...]
        c2_0 = A[2, 1, indices2...]
        c3_0 = A[3, 1, indices2...]
        c4_0 = A[4, 1, indices2...]
        c5_0 = A[5, 1, indices2...]

        c6_0 = A[6, 1, indices2...]
        c7_0 = A[7, 1, indices2...]
        c8_0 = A[8, 1, indices2...]
        =#

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
=#


function expt!(C::LatticeMatrix{D,T,AT,NC1,NC1,nw,DI}, TA::LatticeMatrix{D,T1,AT1,Num,1,nw2,DI}, t::S=one(S)) where {D,T,AT,NC1,Num,S<:Number,T1<:Real,AT1,nw,nw2,DI}

    if NC1 > 3
        error("In NC > 3 case, this function should not be used")
    elseif NC1 == 3 && T <: Complex && S <: Real
        _parallel_for_mutating!(C,
            prod(C.PN), kernel_4Dexpt_TA_basis_su3_ch!,
            C.A, TA.A, C.indexer, Val(nw), t, Val(nw2),
        )
    else
        _parallel_for_mutating!(C,
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
        return
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


function substitute!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI}, A::LatticeMatrix{D,T2,AT2,NC1,NC2,nw,DI}) where {D,T1,T2,AT1,AT2,NC1,NC2,nw,DI}
    _parallel_for_mutating!(C,
        prod(C.PN), kernel_4Dsubstitute!, C.A, A.A, Val(NC1), Val(NC2), Val(nw), C.indexer
    )
    #set_halo!(C)
    return nothing
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
    _parallel_for_mutating!(C,
        prod(C.PN), kernel_4Dsubstitute_dag!, C.A, A.data.A, Val(NC1), Val(NC2), Val(nw), C.indexer
    )
    #set_halo!(C)
    return nothing
end

@inline function kernel_4Dsubstitute_dag!(i, C, A, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, dindexer) where {NC1,NC2,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = A[jc, ic, indices...]'
        end
    end
end

function substitute!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI}, A::TA) where {D,T1,AT1,NC1,NC2,nw,DI,TA<:AbstractArray}
    n1, n2, nsize... = size(C.A)
    n1A, n2A, nsizeA... = size(A)
    @assert n1 == n1A && n2 == n2A "size of A is wrong!"
    @assert length(nsizeA) == D "dimension of A is wrong!"
    for i = 1:D
        @assert nsize[i] == nsizeA[i] + 2nw "lattice size of A is wrong!"
    end
    At = JACC.array(A)

    _parallel_for_mutating!(C,
        prod(C.PN), kernel_4Dsubstitute_matrix!, C.A, At, Val(NC1), Val(NC2), Val(nw), C.indexer
    )
    #set_halo!(C)
    return nothing
end

@inline function kernel_4Dsubstitute_matrix!(i, C, A, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, dindexer) where {NC1,NC2,nw}
    indices = delinearize(dindexer, i, nw)
    indices_0 = delinearize(dindexer, i, 0)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = A[ic, jc, indices_0...]
        end
    end
end




function substitute!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI}, A::Shifted_Lattice{L,D}) where {D,T1,T2,AT1,AT2,NC1,NC2,nw,DI,
    L<:LatticeMatrix{D,T2,AT2,NC1,NC2,nw,DI}}
    shift = get_shift(A)
    _parallel_for_mutating!(C,
        prod(C.PN), kernel_4Dsubstitute_shift!, C.A, A.data.A, Val(NC1), Val(NC2), Val(nw), C.indexer, shift
    )
    #set_halo!(C)
    return nothing
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

function substitute!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI}, A::Adjoint_Lattice{<:Shifted_Lattice{L,D}}) where {D,T1,T2,AT1,AT2,NC1,NC2,nw,DI,
    L<:LatticeMatrix{D,T2,AT2,NC1,NC2,nw,DI}}
    shift = get_shift(A)
    _parallel_for_mutating!(C,
        prod(C.PN), kernel_4Dsubstitute_shiftdag!, C.A, A.data.data.A, Val(NC1), Val(NC2), Val(nw), C.indexer, shift
    )
    #set_halo!(C)
    return nothing
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
    A::Shifted_Lattice{L,D}, B::LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}}
    shift = get_shift(A)
    mul_shiftA_B!(C, A, B, shift)
    #set_halo!(C)
end

function mul_shiftA_B!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Shifted_Lattice{L,D}, B::LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}, shift) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI,
    L<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}}
    _parallel_for_mutating!(C,
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
    A::Shifted_Lattice{L,D}, B::LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI},
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}}

    shift = get_shift(A)
    mul_shiftA_B!(C, A, B, shift, α, β)
    #set_halo!(C)
end

function mul_shiftA_B!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Shifted_Lattice{L,D}, B::LatticeMatrix{D,T3,AT3,NC3,NC2,nw,DI}, shift,
    α::S, β::S) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI}}
    _parallel_for_mutating!(C,
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







#C = shiftA'*shiftedB
#C[i,j] = A[k,j]'*B[k,i]
function mulT!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{<:Shifted_Lattice{L1,D}}, B::Shifted_Lattice{L2,D}) where {D,T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC2,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC3,NC1,nw,DI}}

    shiftA = get_shift(A)
    shiftB = get_shift(B)
    _parallel_for_mutating!(C,
        prod(C.PN), kernel_Dmatrix_mulT_shiftAdagshiftB!, C.A, A.data.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shiftA, shiftB
    )
    #set_halo!(C)
end
export mulT!


@inline function kernel_Dmatrix_mulT_shiftAdagshiftB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shiftA, shiftB) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_A = shiftindices(indices, shiftA)

    indices_B = shiftindices(indices, shiftB)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = 0
            for kc = 1:NC3
                C[ic, jc, indices...] += A[kc, jc, indices_A...]' * B[kc, ic, indices_B...]
            end
        end
    end
end

@inline function kernel_Dmatrix_mulT_shiftAdagshiftB!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, dindexer, shiftA, shiftB) where {nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds begin
        indices_A = shiftindices(indices, shiftA)
        indices_B = shiftindices(indices, shiftB)

        a11 = A[1, 1, indices_A...]'
        a21 = A[2, 1, indices_A...]'
        a12 = A[1, 2, indices_A...]'
        a22 = A[2, 2, indices_A...]'

        b11 = B[1, 1, indices_B...]
        b21 = B[2, 1, indices_B...]
        b12 = B[1, 2, indices_B...]
        b22 = B[2, 2, indices_B...]

        C[1, 1, indices...] = a11 * b11 + a21 * b21
        C[2, 1, indices...] = a11 * b12 + a21 * b22
        C[1, 2, indices...] = a12 * b11 + a22 * b21
        C[2, 2, indices...] = a12 * b12 + a22 * b22
    end
end

@inline function kernel_Dmatrix_mulT_shiftAdagshiftB!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer, shiftA, shiftB) where {nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds begin
        indices_A = shiftindices(indices, shiftA)
        indices_B = shiftindices(indices, shiftB)

        a11 = A[1, 1, indices_A...]'
        a21 = A[2, 1, indices_A...]'
        a31 = A[3, 1, indices_A...]'
        a12 = A[1, 2, indices_A...]'
        a22 = A[2, 2, indices_A...]'
        a32 = A[3, 2, indices_A...]'
        a13 = A[1, 3, indices_A...]'
        a23 = A[2, 3, indices_A...]'
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

        C[1, 1, indices...] = a11 * b11 + a21 * b21 + a31 * b31
        C[2, 1, indices...] = a11 * b12 + a21 * b22 + a31 * b32
        C[3, 1, indices...] = a11 * b13 + a21 * b23 + a31 * b33
        C[1, 2, indices...] = a12 * b11 + a22 * b21 + a32 * b31
        C[2, 2, indices...] = a12 * b12 + a22 * b22 + a32 * b32
        C[3, 2, indices...] = a12 * b13 + a22 * b23 + a32 * b33
        C[1, 3, indices...] = a13 * b11 + a23 * b21 + a33 * b31
        C[2, 3, indices...] = a13 * b12 + a23 * b22 + a33 * b32
        C[3, 3, indices...] = a13 * b13 + a23 * b23 + a33 * b33
    end
end


#C = shiftA'*B'
#C[i,j] = A[k,j]'*B[i,k]
function mulT!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{<:Shifted_Lattice{L1,D}}, B::Adjoint_Lattice{L2}) where {D,T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,nw,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC2,nw,DI},L2<:LatticeMatrix{D,T3,AT3,NC3,NC1,nw,DI}}

    shiftA = get_shift(A)
    _parallel_for_mutating!(C,
        prod(C.PN), kernel_Dmatrix_mulT_shiftAdagBdag!, C.A, A.data.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer, shiftA
    )
    #set_halo!(C)
end
export mulT!


@inline function kernel_Dmatrix_mulT_shiftAdagBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shiftA) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_A = shiftindices(indices, shiftA)

    #indices_B = shiftindices(indices, shiftB)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = 0
            for kc = 1:NC3
                C[ic, jc, indices...] += A[kc, jc, indices_A...]' * B[ic, kc, indices...]'
            end
        end
    end
end

@inline function kernel_Dmatrix_mulT_shiftAdagBdag!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, dindexer, shiftA) where {nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds begin
        indices_A = shiftindices(indices, shiftA)

        a11 = A[1, 1, indices_A...]'
        a21 = A[2, 1, indices_A...]'
        a12 = A[1, 2, indices_A...]'
        a22 = A[2, 2, indices_A...]'

        b11 = B[1, 1, indices...]'
        b12 = B[1, 2, indices...]'
        b21 = B[2, 1, indices...]'
        b22 = B[2, 2, indices...]'

        C[1, 1, indices...] = a11 * b11 + a21 * b12
        C[2, 1, indices...] = a11 * b21 + a21 * b22
        C[1, 2, indices...] = a12 * b11 + a22 * b12
        C[2, 2, indices...] = a12 * b21 + a22 * b22
    end
end

@inline function kernel_Dmatrix_mulT_shiftAdagBdag!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, dindexer, shiftA) where {nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds begin
        indices_A = shiftindices(indices, shiftA)

        a11 = A[1, 1, indices_A...]'
        a21 = A[2, 1, indices_A...]'
        a31 = A[3, 1, indices_A...]'
        a12 = A[1, 2, indices_A...]'
        a22 = A[2, 2, indices_A...]'
        a32 = A[3, 2, indices_A...]'
        a13 = A[1, 3, indices_A...]'
        a23 = A[2, 3, indices_A...]'
        a33 = A[3, 3, indices_A...]'

        b11 = B[1, 1, indices...]'
        b12 = B[1, 2, indices...]'
        b13 = B[1, 3, indices...]'
        b21 = B[2, 1, indices...]'
        b22 = B[2, 2, indices...]'
        b23 = B[2, 3, indices...]'
        b31 = B[3, 1, indices...]'
        b32 = B[3, 2, indices...]'
        b33 = B[3, 3, indices...]'

        C[1, 1, indices...] = a11 * b11 + a21 * b12 + a31 * b13
        C[2, 1, indices...] = a11 * b21 + a21 * b22 + a31 * b23
        C[3, 1, indices...] = a11 * b31 + a21 * b32 + a31 * b33
        C[1, 2, indices...] = a12 * b11 + a22 * b12 + a32 * b13
        C[2, 2, indices...] = a12 * b21 + a22 * b22 + a32 * b23
        C[3, 2, indices...] = a12 * b31 + a22 * b32 + a32 * b33
        C[1, 3, indices...] = a13 * b11 + a23 * b12 + a33 * b13
        C[2, 3, indices...] = a13 * b21 + a23 * b22 + a33 * b23
        C[3, 3, indices...] = a13 * b31 + a23 * b32 + a33 * b33
    end
end

function LinearAlgebra.tr(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI}) where {D,T1,AT1,NC1,NC2,nw,DI}
    @assert NC1 == NC2 "Trace is only defined for square matrices"
    s = JACC.parallel_reduce(prod(C.PN), kernel_tr_4D, C.A, Val(NC1), C.indexer, Val(nw); init=zero(eltype(C.A)), op=+)::T1
    s = _allreduce_sum(s, C.comm)
    return s
end

@inline _preduce(n, op, kern, A, NC1, dindexer, vnw, init::T) where {T} =
    JACC.parallel_reduce(n, kern, A, NC1, dindexer, vnw; init=init, op)::T


Base.@noinline function LinearAlgebra.tr(C::LatticeMatrix{D,T1,AT1,NC1,NC1,nw,DI}) where {D,T1,AT1,NC1,nw,DI}
    s = _preduce(prod(C.PN), +, kernel_tr_4D, C.A, Val(NC1), C.indexer, Val(nw), zero(T1))::T1
    s = _allreduce_sum(s, C.comm)
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
    JACC.parallel_reduce(n, kern, A, B, NC1, dindexer, vnw; init=init, op)::T

function LinearAlgebra.tr(C::LatticeMatrix{D,T1,AT1,NC1,NC1,nw,DI}, B::LatticeMatrix{D,T1,AT1,NC1,NC1,nw,DI}) where {D,T1,AT1,NC1,nw,DI}
    s = _preduce(prod(C.PN), +, kernel_tr_4D, C.A, B.A, Val(NC1), C.indexer, Val(nw), zero(T1))::T1
    s = _allreduce_sum(s, C.comm)
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
    s = JACC.parallel_reduce(prod(A.PN), kernel_dot_real_1,
        A.A, B.A, A.indexer, Val(NC1), Val(nw); init=zero(eltype(A.A)), op=+)
    s = _allreduce_sum(s, A.comm)
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
    # `position` is a global, one-based coordinate. Only ranks that own the
    # requested hyperplane contribute to the collective reduction.
    local_position = position - C.coords[μ] * C.PN[μ]
    s = if 1 <= local_position <= C.PN[μ]
        JACC.parallel_reduce(prod(C.PN), kernel_partial_trace_D, C.A, NC1,
            C.indexer, μ, local_position, Val(nw); init=zero(eltype(C.A)), op=+)
    else
        zero(eltype(C.A))
    end
    s = _allreduce_sum(s, C.comm)
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

"""
    normalize_matrix!(C)

Project every site of the square lattice matrix `C` onto SU(N) (or SO(N)
for real element types). The NC=2 and NC=3 paths use specialized kernels;
larger matrices use modified Gram–Schmidt followed by a determinant-phase
correction.
"""
function normalize_matrix!(C::LatticeMatrix{D,T,AT,NC,NC,nw,DI}) where {D,T,AT,NC,nw,DI}
    if NC == 2
        _parallel_for_mutating!(C, prod(C.PN), kernel_normalize_NC2!, C.A, C.indexer, Val(nw))
    elseif NC == 3
        _parallel_for_mutating!(C, prod(C.PN), kernel_normalize_NC3!, C.A, C.indexer, Val(nw))
    else
        _parallel_for_mutating!(C, prod(C.PN), kernel_normalize_generic!, C.A, C.indexer, Val(NC), Val(nw))
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
    T = eltype(u)
    w1 = zero(T)
    w2 = real(zero(T))
    @inbounds for ic = 1:3
        u1 = u[1, ic, indices...]
        w1 += u[2, ic, indices...] * conj(u1)
        w2 += abs2(u1)
    end
    w1 = -w1 / w2

    x4 = (u[2, 1, indices...]) + w1 * u[1, 1, indices...]
    x5 = (u[2, 2, indices...]) + w1 * u[1, 2, indices...]
    x6 = (u[2, 3, indices...]) + w1 * u[1, 3, indices...]

    w3 = abs2(x4) + abs2(x5) + abs2(x6)

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



# Modified Gram–Schmidt followed by a determinant-phase correction. The
# fixed-size LU workspace stays local to each CPU/GPU kernel invocation.
@inline function kernel_normalize_generic!(
    i, u, dindexer, ::Val{NC}, ::Val{nw},
) where {NC,nw}
    indices = delinearize(dindexer, i, nw)

    T = eltype(u)
    rT = real(one(T))
    epsT = sqrt(eps(rT))

    @inbounds for j = 1:NC
        for k = 1:j-1
            inner = zero(T)
            for r = 1:NC
                inner += conj(u[r, k, indices...]) * u[r, j, indices...]
            end
            for r = 1:NC
                u[r, j, indices...] -= inner * u[r, k, indices...]
            end
        end

        nrm2 = zero(rT)
        for r = 1:NC
            nrm2 += abs2(u[r, j, indices...])
        end
        nrm = sqrt(nrm2)

        if nrm < epsT
            # Complete a rank-deficient input with the canonical basis vector
            # having the largest component outside the existing span.
            best_row = 1
            best_nrm2 = zero(rT)
            for candidate = 1:NC
                candidate_nrm2 = one(rT)
                for k = 1:j-1
                    candidate_nrm2 -= abs2(u[candidate, k, indices...])
                end
                if candidate_nrm2 > best_nrm2
                    best_nrm2 = candidate_nrm2
                    best_row = candidate
                end
            end

            for r = 1:NC
                value = r == best_row ? one(T) : zero(T)
                for k = 1:j-1
                    value -= conj(u[best_row, k, indices...]) *
                             u[r, k, indices...]
                end
                u[r, j, indices...] = value
            end

            nrm2 = zero(rT)
            for r = 1:NC
                nrm2 += abs2(u[r, j, indices...])
            end
            nrm = sqrt(nrm2)
        end

        invn = one(rT) / nrm
        invnT = convert(T, invn)
        for r = 1:NC
            u[r, j, indices...] *= invnT
        end
    end

    LU = MMatrix{NC,NC,T}(undef)
    pivots = MVector{NC,Int}(undef)
    @inbounds for column = 1:NC
        for row = 1:NC
            LU[row, column] = u[row, column, indices...]
        end
    end
    lu_factor!(LU, pivots)

    determinant = one(T)
    @inbounds for k = 1:NC
        if pivots[k] != k
            determinant = -determinant
        end
        determinant *= LU[k, k]
    end

    determinant_magnitude = abs(determinant)
    if determinant_magnitude > epsT
        phase = conj(determinant) / determinant_magnitude
        @inbounds for row = 1:NC
            u[row, NC, indices...] *= phase
        end
    end

    return nothing
end

#=
function randomize_matrix!(C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}) where {D,T,AT,NC1,NC2,nw,DI}
    _parallel_for_mutating!(C, prod(C.PN), kernel_randomize_4D!, C.A, C.indexer, NC1, NC2)
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

const _DEFAULT_RANDOMIZE_SEED = UInt64(0x12345678ABCDEF01)

@inline _random_real_type(::Type{Float32}) = Float32
@inline _random_real_type(::Type{Float64}) = Float64
@inline _random_real_type(::Type{ComplexF32}) = Float32
@inline _random_real_type(::Type{ComplexF64}) = Float64

function _check_randomize_eltype(::Type{T}) where {T}
    T in (Float32, Float64, ComplexF32, ComplexF64) && return nothing
    throw(ArgumentError("random lattice fills do not support element type $T"))
end

"""
    randomize_matrix!(C, key::RNGStreamKey; rng_algorithm=Philox4x32())
    randomize_matrix!(C; seed=0x12345678ABCDEF01, sweep=0, direction=0,
                      color=0, subgroup=0, rng_algorithm=Philox4x32())

Fill the core of `C` with uniform values in `[-0.5, 0.5)`.  Every site owns
one independent stream keyed by its zero-based global site id, so an explicit
key produces the same global field for every MPI decomposition.  Complex
elements consume independent uniforms for their real and imaginary parts.
"""
function randomize_matrix!(
    C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI},
    key::RNGStreamKey;
    rng_algorithm::SiteRNGAlgorithm=Philox4x32(),
) where {D,T,AT,NC1,NC2,nw,DI}
    _check_randomize_eltype(T)
    _parallel_for_mutating!(
        C,
        prod(C.PN),
        kernel_randomize_global_sites!,
        C.A,
        C.indexer,
        Val(NC1),
        Val(NC2),
        Val(nw),
        C.coords,
        C.PN,
        C.gsize,
        key,
        rng_algorithm,
    )
    set_halo!(C)
    return nothing
end

function randomize_matrix!(
    C::LatticeMatrix;
    seed::Integer=_DEFAULT_RANDOMIZE_SEED,
    sweep::Integer=0,
    direction::Integer=0,
    color::Integer=0,
    subgroup::Integer=0,
    rng_algorithm::SiteRNGAlgorithm=Philox4x32(),
)
    key = RNGStreamKey(seed, sweep, direction, color, subgroup)
    return randomize_matrix!(C, key; rng_algorithm)
end

"""
    randomize_gaussian_matrix!(C, key::RNGStreamKey;
                               sigma=1, rng_algorithm=Philox4x32())
    randomize_gaussian_matrix!(C; sigma=1, seed=..., sweep=0, direction=0,
                               color=0, subgroup=0,
                               rng_algorithm=Philox4x32())

Fill the core of `C` with independent zero-mean normal values of standard
deviation `sigma`, using the same decomposition-independent site streams as
[`randomize_matrix!`](@ref).  Complex elements receive independent normal real
and imaginary components.
"""
function randomize_gaussian_matrix!(
    C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI},
    key::RNGStreamKey;
    sigma::Real=1,
    rng_algorithm::SiteRNGAlgorithm=Philox4x32(),
) where {D,T,AT,NC1,NC2,nw,DI}
    _check_randomize_eltype(T)
    real_type = _random_real_type(T)
    sigma_typed = real_type(sigma)
    sigma_typed >= zero(real_type) || throw(ArgumentError("sigma must be non-negative"))
    _parallel_for_mutating!(
        C,
        prod(C.PN),
        kernel_randomize_gaussian_global_sites!,
        C.A,
        C.indexer,
        Val(NC1),
        Val(NC2),
        Val(nw),
        C.coords,
        C.PN,
        C.gsize,
        key,
        rng_algorithm,
        sigma_typed,
    )
    set_halo!(C)
    return nothing
end

function randomize_gaussian_matrix!(
    C::LatticeMatrix;
    sigma::Real=1,
    seed::Integer=_DEFAULT_RANDOMIZE_SEED,
    sweep::Integer=0,
    direction::Integer=0,
    color::Integer=0,
    subgroup::Integer=0,
    rng_algorithm::SiteRNGAlgorithm=Philox4x32(),
)
    key = RNGStreamKey(seed, sweep, direction, color, subgroup)
    return randomize_gaussian_matrix!(C, key; sigma, rng_algorithm)
end

@inline function _global_site_rng(
    i,
    dindexer,
    coords::NTuple{D,Int},
    local_size::NTuple{D,Int},
    global_size::NTuple{D,Int},
    key,
    algorithm,
) where {D}
    local_indices = delinearize(dindexer, i, 0)
    global_indices = global_site_coordinates(local_indices, coords, local_size)
    global_site = global_site_id(global_indices, global_size)
    return site_rng(key, global_site, algorithm)
end

@inline function kernel_randomize_global_sites!(
    i,
    u,
    dindexer,
    ::Val{NC1},
    ::Val{NC2},
    ::Val{nw},
    coords,
    local_size,
    global_size,
    key,
    algorithm,
) where {NC1,NC2,nw}
    indices = delinearize(dindexer, i, nw)
    rng = _global_site_rng(i, dindexer, coords, local_size, global_size, key, algorithm)
    _uniform_site_fill!(u, indices, Val(NC1), Val(NC2), rng)
    return nothing
end

@inline function _uniform_site_fill!(
    u::AbstractArray{T},
    indices,
    ::Val{NC1},
    ::Val{NC2},
    rng,
) where {T<:Union{Float32,Float64},NC1,NC2}
    @inbounds for jc in 1:NC2, ic in 1:NC1
        rng, value = rand_uniform(rng, T)
        u[ic, jc, indices...] = value - T(0.5)
    end
    return rng
end

@inline function _uniform_site_fill!(
    u::AbstractArray{Complex{T}},
    indices,
    ::Val{NC1},
    ::Val{NC2},
    rng,
) where {T<:Union{Float32,Float64},NC1,NC2}
    @inbounds for jc in 1:NC2, ic in 1:NC1
        rng, real_value = rand_uniform(rng, T)
        rng, imag_value = rand_uniform(rng, T)
        u[ic, jc, indices...] = Complex{T}(real_value - T(0.5), imag_value - T(0.5))
    end
    return rng
end

@inline function kernel_randomize_gaussian_global_sites!(
    i,
    u,
    dindexer,
    ::Val{NC1},
    ::Val{NC2},
    ::Val{nw},
    coords,
    local_size,
    global_size,
    key,
    algorithm,
    sigma,
) where {NC1,NC2,nw}
    indices = delinearize(dindexer, i, nw)
    rng = _global_site_rng(i, dindexer, coords, local_size, global_size, key, algorithm)
    _gaussian_site_fill!(u, indices, Val(NC1), Val(NC2), rng, sigma)
    return nothing
end

@inline function _gaussian_site_fill!(
    u::AbstractArray{T},
    indices,
    ::Val{NC1},
    ::Val{NC2},
    rng,
    sigma::T,
) where {T<:Union{Float32,Float64},NC1,NC2}
    use_spare = false
    spare = zero(T)
    @inbounds for jc in 1:NC2, ic in 1:NC1
        if use_spare
            value = spare
        else
            rng, value, spare = rand_normal_pair(rng, T)
        end
        u[ic, jc, indices...] = sigma * value
        use_spare = !use_spare
    end
    return rng
end

@inline function _gaussian_site_fill!(
    u::AbstractArray{Complex{T}},
    indices,
    ::Val{NC1},
    ::Val{NC2},
    rng,
    sigma::T,
) where {T<:Union{Float32,Float64},NC1,NC2}
    @inbounds for jc in 1:NC2, ic in 1:NC1
        rng, real_value, imag_value = rand_normal_pair(rng, T)
        u[ic, jc, indices...] = Complex{T}(sigma * real_value, sigma * imag_value)
    end
    return rng
end

export randomize_matrix!, randomize_gaussian_matrix!

function clear_matrix!(C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}) where {D,T,AT,NC1,NC2,nw,DI}
    _parallel_for_mutating!(C, prod(C.PN), kernel_clear_4D!, C.A, C.indexer, Val(NC1), Val(NC2), Val(nw))
    set_halo!(C)
    return nothing
end

function clear_matrix!(
    C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI},
    target_even::Bool,
) where {D,T,AT,NC1,NC2,nw,DI}
    _parallel_for_mutating!(C,
        prod(C.PN),
        kernel_clear_evenodd!,
        C.A,
        C.indexer,
        Val(NC1),
        Val(NC2),
        Val(nw),
        C.coords,
        C.PN,
        target_even,
    )
    set_halo!(C)
    return nothing
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

@inline function kernel_clear_evenodd!(
    i,
    u,
    dindexer,
    ::Val{NC1},
    ::Val{NC2},
    ::Val{nw},
    coords::NTuple{D,Int},
    local_size::NTuple{D,Int},
    target_even::Bool,
) where {NC1,NC2,nw,D}
    local_indices = delinearize(dindexer, i, 0)

    if _global_site_is_even(local_indices, coords, local_size) == target_even
        storage_indices = ntuple(d -> local_indices[d] + nw, D)
        @inbounds for jc = 1:NC2
            for ic = 1:NC1
                u[ic, jc, storage_indices...] = zero(eltype(u))
            end
        end
    end

    return nothing
end

function makeidentity_matrix!(C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}) where {D,T,AT,NC1,NC2,nw,DI}
    _parallel_for_mutating!(C, prod(C.PN), kernel_makeidentity_4D!, C.A, C.indexer, Val(NC1), Val(NC2), Val(nw))
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
    _parallel_for_mutating!(C, prod(C.PN), kernel_add_4D!, C.A, A.A, C.indexer, Val(NC1), Val(NC2), α, Val(nw))
    #set_halo!(C)
end
export add_matrix!

function add_matrix_evenodd!(
    C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    target_even::Bool,
    α::S=1,
) where {D,T,T1,AT,AT1,NC1,NC2,nw,S<:Number,DI}
    _parallel_for_mutating!(C,
        prod(C.PN), kernel_add_4D_evenodd!, C.A, A.A, C.indexer,
        Val(NC1), Val(NC2), α, Val(nw), C.coords, C.PN, target_even
    )
    return nothing
end
export add_matrix_evenodd!

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

@inline function kernel_add_4D_evenodd!(
    i, u, v, dindexer, vNC1::Val{NC1}, vNC2::Val{NC2}, α, vnw::Val{nw},
    coords, local_size, target_even::Bool,
) where {NC1,NC2,nw}
    local_indices = delinearize(dindexer, i, 0)
    if _global_site_is_even(local_indices, coords, local_size) == target_even
        kernel_add_4D!(i, u, v, dindexer, vNC1, vNC2, α, vnw)
    end
    return nothing
end

#C = C+ α*shiftA
function add_matrix!(C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}, A::Shifted_Lattice{L,D}, α::S=1) where {D,T,T1,AT,AT1,NC1,NC2,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI}}
    shift = get_shift(A)
    add_matrix_shiftedA!(C, A.data, shift, α)
    #JACC.parallel_for(prod(C.PN), kernel_add_4D_shift!, C.A, A.data.A, C.indexer, Val(NC1), Val(NC2), α, shift, Val(nw))
    #set_halo!(C)
end

function add_matrix_evenodd!(
    C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI},
    A::Shifted_Lattice{L,D},
    target_even::Bool,
    α::S=1,
) where {D,T,T1,AT,AT1,NC1,NC2,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI}}
    shift = get_shift(A)
    _parallel_for_mutating!(C,
        prod(C.PN), kernel_add_4D_shift_evenodd!, C.A, A.data.A, C.indexer,
        Val(NC1), Val(NC2), α, shift, Val(nw), C.coords, C.PN, target_even
    )
    return nothing
end

function add_matrix_shiftedA!(C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}, A::L, shift, α::S=1) where {D,T,T1,AT,AT1,NC1,NC2,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI}}
    _ensure_halo_for_shift!(A, shift)
    _parallel_for_mutating!(C, prod(C.PN), kernel_add_4D_shift!, C.A, A.A, C.indexer, Val(NC1), Val(NC2), α, shift, Val(nw))
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

@inline function kernel_add_4D_shift_evenodd!(
    i, u, v, dindexer, vNC1::Val{NC1}, vNC2::Val{NC2}, α, shift,
    vnw::Val{nw}, coords, local_size, target_even::Bool,
) where {NC1,NC2,nw}
    local_indices = delinearize(dindexer, i, 0)
    if _global_site_is_even(local_indices, coords, local_size) == target_even
        kernel_add_4D_shift!(i, u, v, dindexer, vNC1, vNC2, α, shift, vnw)
    end
    return nothing
end

#C = C+ α*Adag
function add_matrix!(C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}, A::Adjoint_Lattice{L}, α::S=1) where {D,T,T1,AT,AT1,NC1,NC2,nw,S<:Number,DI,L<:LatticeMatrix{D,T1,AT1,NC2,NC1,nw,DI}}
    add_matrix_Adag!(C, A.data, α)
    #set_halo!(C)
end

function add_matrix_evenodd!(
    C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{L},
    target_even::Bool,
    α::S=1,
) where {D,T,T1,AT,AT1,NC1,NC2,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T1,AT1,NC2,NC1,nw,DI}}
    _parallel_for_mutating!(C,
        prod(C.PN), kernel_add_4D_dag_evenodd!, C.A, A.data.A, C.indexer,
        Val(NC1), Val(NC2), α, Val(nw), C.coords, C.PN, target_even
    )
    return nothing
end

function add_matrix_Adag!(C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}, A::L, α::S=1) where {D,T,T1,AT,AT1,NC1,NC2,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T1,AT1,NC2,NC1,nw,DI}}
    _parallel_for_mutating!(C, prod(C.PN), kernel_add_4D_dag!, C.A, A.A, C.indexer, Val(NC1), Val(NC2), α, Val(nw))
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

@inline function kernel_add_4D_dag_evenodd!(
    i, u, v, dindexer, vNC1::Val{NC1}, vNC2::Val{NC2}, α, vnw::Val{nw},
    coords, local_size, target_even::Bool,
) where {NC1,NC2,nw}
    local_indices = delinearize(dindexer, i, 0)
    if _global_site_is_even(local_indices, coords, local_size) == target_even
        kernel_add_4D_dag!(i, u, v, dindexer, vNC1, vNC2, α, vnw)
    end
    return nothing
end

#C = C+ α*shiftAdag
function add_matrix!(C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}, A::Adjoint_Lattice{<:Shifted_Lattice{L,D}}, α::S=1) where {D,T,T1,AT,AT1,NC1,NC2,nw,S<:Number,DI,L<:LatticeMatrix{D,T1,AT1,NC2,NC1,nw,DI}}
    shift = get_shift(A)
    add_matrix_shiftedAdag!(C, A.data.data, shift, α)
    #set_halo!(C)
end

function add_matrix_evenodd!(
    C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI},
    A::Adjoint_Lattice{<:Shifted_Lattice{L,D}},
    target_even::Bool,
    α::S=1,
) where {D,T,T1,AT,AT1,NC1,NC2,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T1,AT1,NC2,NC1,nw,DI}}
    shift = get_shift(A)
    _parallel_for_mutating!(C,
        prod(C.PN), kernel_add_4D_shiftdag_evenodd!, C.A, A.data.data.A,
        C.indexer, Val(NC1), Val(NC2), α, shift, Val(nw), C.coords, C.PN,
        target_even
    )
    return nothing
end

function add_matrix_shiftedAdag!(C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}, A::L, shift, α::S=1) where {D,T,T1,AT,AT1,NC1,NC2,nw,S<:Number,DI,
    L<:LatticeMatrix{D,T1,AT1,NC2,NC1,nw,DI}}
    _ensure_halo_for_shift!(A, shift)
    _parallel_for_mutating!(C, prod(C.PN), kernel_add_4D_shiftdag!, C.A, A.A, C.indexer, Val(NC1), Val(NC2), α, shift, Val(nw))
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

@inline function kernel_add_4D_shiftdag_evenodd!(
    i, u, v, dindexer, vNC1::Val{NC1}, vNC2::Val{NC2}, α, shift,
    vnw::Val{nw}, coords, local_size, target_even::Bool,
) where {NC1,NC2,nw}
    local_indices = delinearize(dindexer, i, 0)
    if _global_site_is_even(local_indices, coords, local_size) == target_even
        kernel_add_4D_shiftdag!(i, u, v, dindexer, vNC1, vNC2, α, shift, vnw)
    end
    return nothing
end

function applyfunction!(C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}, f::Function, variables...) where {D,T,AT,NC1,NC2,nw,DI}
    _parallel_for_mutating!(C, prod(C.PN), kernel_apply_function_4D!, C.A, C.indexer, Val(NC1), Val(NC2), Val(nw), f, variables...)
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

function map_matrix_evenodd!(
    U::LatticeMatrix{D,TU,ATU,NC1,NC2,nw,DI},
    V::LatticeMatrix{D,TV,ATV,NC1,NC2,nw,DI},
    f!,
    target_even::Bool,
) where {D,TU,TV,ATU,ATV,NC1,NC2,nw,DI}
    _parallel_for_mutating!(U,
        prod(U.PN), kernel_map_matrix_evenodd!, U.A, V.A, U.indexer,
        Val(NC1), Val(NC2), Val(nw), U.coords, U.PN, f!, target_even
    )
    set_halo!(U)
    return nothing
end
export map_matrix_evenodd!

@inline function kernel_map_matrix_evenodd!(
    i, u, v, dindexer, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, coords,
    local_size, f!, target_even::Bool,
) where {NC1,NC2,nw}
    local_indices = delinearize(dindexer, i, 0)
    if _global_site_is_even(local_indices, coords, local_size) == target_even
        indices = delinearize(dindexer, i, nw)
        u_local = MMatrix{NC1,NC2,eltype(u)}(undef)
        v_local = MMatrix{NC1,NC2,eltype(v)}(undef)

        @inbounds for jc in 1:NC2
            for ic in 1:NC1
                u_local[ic, jc] = u[ic, jc, indices...]
                v_local[ic, jc] = v[ic, jc, indices...]
            end
        end

        f!(u_local, v_local)

        @inbounds for jc in 1:NC2
            for ic in 1:NC1
                u[ic, jc, indices...] = u_local[ic, jc]
            end
        end
    end
    return nothing
end



function traceless_antihermitian_add!(C::LatticeMatrix{D,T,AT,NC,NC,nw}, factor,
    A::LatticeMatrix{D,T2,AT2,NC,NC,nw2}) where {D,T,AT,nw,T2,AT2,NC,nw2}
    _parallel_for_mutating!(C, prod(C.PN), kernel_4d_Traceless_antihermitian_add_general!, C.A, A.A, factor, C.indexer, Val(NC), Val(nw), Val(nw2))
end
export traceless_antihermitian_add!

function kernel_4d_Traceless_antihermitian_add_general!(i, c, vin, factor, dindexer, ::Val{NC}, ::Val{nw}, ::Val{nw2}) where {NC,nw,nw2}
    indices = delinearize(dindexer, i, nw)
    indices2 = delinearize(dindexer, i, nw2)

    tri = zero(eltype(c))
    for ic = 1:NC
        tri += imag(vin[ic, ic, indices2...])
    end
    tri /= NC

    for k = 1:NC
        c[k, k, indices...] +=
            (imag(vin[k, k, indices2...]) - tri) * im * factor
    end

    for k1 = 1:NC
        for k2 = k1+1:NC
            vv =
                0.5 * (
                    vin[k1, k2, indices2...] -
                    conj(vin[k2, k1, indices2...])
                )
            c[k1, k2, indices...] += vv * factor
            c[k2, k1, indices...] += -conj(vv) * factor
        end
    end

end




function traceless_antihermitian_add!(C::LatticeMatrix{D,T,AT,NG,1,nw}, factor,
    A::LatticeMatrix{D,T2,AT2,NC,NC,nw2}) where {D,T<:Real,AT,NG,nw,T2,AT2,NC,nw2}
    _parallel_for_mutating!(C, prod(C.PN), kernel_4d_Traceless_antihermitian_add!, C.A, A.A, factor, C.indexer, Val(NG), Val(NC), Val(nw), Val(nw2))
end

function kernel_4d_Traceless_antihermitian_add!(i, c, vin, factor, dindexer, ::Val{NG}, ::Val{NC}, ::Val{nw}, ::Val{nw2}) where {NC,NG,nw,nw2}
    error("NC > 3 is not supported in kernel_4d_Traceless_antihermitian_add!")
end

const fac12 = 1 / 2

function kernel_4d_Traceless_antihermitian_add!(i, c, vin, factor, dindexer, ::Val{NG}, ::Val{2}, ::Val{nw}, ::Val{nw2}) where {NG,nw,nw2}
    indices = delinearize(dindexer, i, nw)
    indices2 = delinearize(dindexer, i, nw2)
    #ix2 = ix + nw2
    #iy2 = iy + nw2
    #iz2 = iz + nw2
    #it2 = it + nw2
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    v11 = vin[1, 1, indices2...]
    v22 = vin[2, 2, indices2...]

    tri = fac12 * (imag(v11) + imag(v22))

    v12 = vin[1, 2, indices2...]
    #v13 = vin[1,3,ix,iy,iz,it]
    v21 = vin[2, 1, indices2...]

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
    indices2 = delinearize(dindexer, i, nw2)
    #ix2 = ix + nw2
    #iy2 = iy + nw2
    #iz2 = iz + nw2
    #it2 = it + nw2
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    fac13 = 1 / 3


    v11 = vin[1, 1, indices2...]
    v22 = vin[2, 2, indices2...]
    v33 = vin[3, 3, indices2...]

    tri = fac13 * (imag(v11) + imag(v22) + imag(v33))

    #=
    vout[1,1,ix,iy,iz,it] = (imag(v11)-tri)*im
    vout[2,2,ix,iy,iz,it] = (imag(v22)-tri)*im
    vout[3,3,ix,iy,iz,it] = (imag(v33)-tri)*im
    =#
    y11 = (imag(v11) - tri) * im
    y22 = (imag(v22) - tri) * im
    y33 = (imag(v33) - tri) * im

    v12 = vin[1, 2, indices2...]
    v13 = vin[1, 3, indices2...]
    v21 = vin[2, 1, indices2...]
    v23 = vin[2, 3, indices2...]
    v31 = vin[3, 1, indices2...]
    v32 = vin[3, 2, indices2...]

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

#dot(A,B) = tr(A^+*B)
function LinearAlgebra.dot(A::LatticeMatrix{D,T1,AT1,NC1,NG,nw}, B::LatticeMatrix{D,T2,AT2,NC1,NG,nw}) where {D,NG,T1,T2,AT1,AT2,NC1,nw}
    s = JACC.parallel_reduce(prod(A.PN), kernel_dot_D,
        A.A, B.A, A.indexer, Val(NC1), Val(NG), Val(nw); init=zero(eltype(A.A)), op=+)
    s = _allreduce_sum(s, A.comm)
end

@inline function kernel_dot_D(i, A, B, dindexer, ::Val{NC1}, ::Val{NG}, ::Val{nw}) where {NC1,nw,NG}
    indices = delinearize(dindexer, i, nw)
    s = zero(eltype(A))

    @inbounds for ialpha = 1:NG
        for ic = 1:NC1
            s += conj(A[ic, ialpha, indices...]) * B[ic, ialpha, indices...]
        end
    end
    return s
end


@inline function calc_coefficients_Q(Q)
    @assert size(Q) == (3, 3)
    c0 =
        Q[1, 1] * Q[2, 2] * Q[3, 3] +
        Q[1, 2] * Q[2, 3] * Q[3, 1] +
        Q[1, 3] * Q[2, 1] * Q[3, 2] - Q[1, 3] * Q[2, 2] * Q[3, 1] -
        Q[1, 2] * Q[2, 1] * Q[3, 3] - Q[1, 1] * Q[2, 3] * Q[3, 2]

    c1 = 0.0
    for i = 1:3
        for j = 1:3
            c1 += Q[i, j] * Q[j, i]
        end
    end
    c1 /= 2
    c0max = 2 * (c1 / 3)^(3 / 2)
    θ = acos(c0 / c0max)
    u = sqrt(c1 / 3) * cos(θ / 3)
    w = sqrt(c1) * sin(θ / 3)
    ξ0 = sin(w) / w
    ξ1 = cos(w) / w^2 - sin(w) / w^3

    emiu = exp(-im * u)
    e2iu = exp(2 * im * u)

    h0 = (u^2 - w^2) * e2iu + emiu * (8u^2 * cos(w) + 2 * im * u * (3u^2 + w^2) * ξ0)
    h1 = 2u * e2iu - emiu * (2u * cos(w) - im * (3u^2 - w^2) * ξ0)
    h2 = e2iu - emiu * (cos(w) + 3 * im * u * ξ0)

    denom = 9u^2 - w^2

    f0 = h0 / denom
    f1 = h1 / denom
    f2 = h2 / denom

    r10 =
        2 * (u + im * (u^2 - w^2)) * e2iu +
        2 *
        emiu *
        (4u * (2 - im * u) * cos(w) + im * (9u^2 + w^2 - im * u * (3u^2 + w^2)) * ξ0)
    r11 =
        2 * (1 + 2 * im * u) * e2iu +
        emiu * (-2 * (1 - im * u) * cos(w) + im * (6u + im * (w^2 - 3u^2)) * ξ0)
    r12 = 2 * im * e2iu + im * emiu * (cos(w) - 3 * (1 - im * u) * ξ0)
    r20 = -2 * e2iu + 2 * im * u * emiu * (cos(w) + (1 + 4 * im * u) * ξ0 + 3u^2 * ξ1)
    r21 = -im * emiu * (cos(w) + (1 + 2 * im * u) * ξ0 - 3 * u^2 * ξ1)
    r22 = emiu * (ξ0 - 3 * im * u * ξ1)
    b10 = (2 * u * r10 + (3u^2 - w^2) * r20 - 2 * (15u^2 + w^2) * f0) / (2 * denom^2)

    b11 = (2 * u * r11 + (3u^2 - w^2) * r21 - 2 * (15u^2 + w^2) * f1) / (2 * denom^2)
    b12 = (2 * u * r12 + (3u^2 - w^2) * r22 - 2 * (15u^2 + w^2) * f2) / (2 * denom^2)
    b20 = (r10 - 3 * u * r20 - 24 * u * f0) / (2 * denom^2)
    b21 = (r11 - 3 * u * r21 - 24 * u * f1) / (2 * denom^2)
    b22 = (r12 - 3 * u * r22 - 24 * u * f2) / (2 * denom^2)

    return f0, f1, f2, b10, b11, b12, b20, b21, b22
end


function kernel_construct_Λmatrix_forSTOUT!(i, Λ, δ, Q, U, NC, dindexer, nw, global_buffer)

    indices = delinearize(dindexer, i, nw)
    
    
    @inbounds begin
      
        temp1 = view(global_buffer, :, :, 1,i)
        temp2 = view(global_buffer, :, :, 2,i)
        temp3 = view(global_buffer, :, :, 3,i)
        Qn = view(global_buffer, :, :, 4,i)
        Mn = view(global_buffer, :, :, 5,i)
        Unδn = view(global_buffer, :, :, 6,i)
    
        for ic in 1:3
            for jc in 1:3
                Qn[ic,jc] = Q[ic,jc,indices...]
            end
        end

        #calc_Mmatrix! --> elementwise operation
        trQ2 = 0.0
        for i = 1:3
            for j = 1:3
                trQ2 += Qn[i, j] * Qn[j, i]
            end
        end
    
    
        
        if abs(trQ2) > 1e-18
            Qn ./= im
            #println("Qn b ",Qn)
            f0, f1, f2, b10, b11, b12, b20, b21, b22 = calc_coefficients_Q(Qn)            
            
            for ic in 1:3
                for jc in 1:3
                    Unδn[ic,jc] = 0.0f0 + 0im
                    for k = 1:3
                        Unδn[ic,jc] += U[ic,k, indices...] * δ[k,jc, indices...]
                    end
                end
            end
            
            B1 = temp1
            B1 .= 0
            B2 = temp3
            B2 .= 0
            for i = 1:3
                B1[i, i] = b10
                B2[i, i] = b20
            end
            for j = 1:3
                for i = 1:3
                    B1[i, j] += b11 * Qn[i, j]
                    B2[i, j] += b21 * Qn[i, j]
                    for k = 1:3
                        B1[i, j] += b12 * Qn[i, k] * Qn[k, j]
                        B2[i, j] += b22 * Qn[i, k] * Qn[k, j]
                    end
                end
            end
        

            trB1 = 0.0
            trB2 = 0.0
            for i = 1:3
                for j = 1:3
                    trB1 += Unδn[i, j] * B1[j, i]
                    trB2 += Unδn[i, j] * B2[j, i]
                end
            end

            for j = 1:3
                for i = 1:3
                    Mn[i, j] = trB1 * Qn[i, j] + f1 * Unδn[i, j]
                    for k = 1:3
                        Mn[i, j] +=
                            trB2 * Qn[i, k] * Qn[k, j] +
                            f2 * (Qn[i, k] * Unδn[k, j] + Unδn[i, k] * Qn[k, j])
                    end
                end
            end
            
            for i = 1:3
                for j = 1:3
                    Mn[i, j] /= im
                end
            end
        else
           
            #Mn .= 0
            #mul!(Mn, Un, δn) # --> f1 = 1, to have a well-defined point when Q == 0 for θ =0. 
            for ic in 1:3
                for jc in 1:3
                    Mn[ic,jc] = 0.0f0 + 0im
                    for k = 1:3
                        Mn[ic,jc] += U[ic,k, indices...] * δ[k,jc, indices...]
                    end
                end
            end
            
        end


        #calc_Λmatrix!(Λn, Mn, NC) --> elementwise operation
        temp2 .= 0
        for i = 1:3
            for j = 1:3
                temp2[i, j] = (1 / 2) * (Mn[i,j] - conj(Mn[j,i]))
            end
        end
            
        #trMn = (1 / (6)) * tr(Mn - Mn')
        trMn = 0.0
        for i = 1:3
            trMn += ( Mn[i, i] - conj(Mn[i, i]) ) / 6
        end

        for i = 1:3
            temp2[i, i] += -trMn
        end
        
        for jc = 1:NC
            for ic = 1:NC
                Λ[ic,jc,indices...] = temp2[ic, jc]
            end
        end
    
    end
    
    return
    
end

function construct_Λmatrix_forSTOUT_matrix!(
    Λ::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI},
    δ_current::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI},
    Q::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI},
    u::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI},
) where {D,T,AT,NC1,NC2,nw,DI}

    global_buffer = JACC.zeros(ComplexF64, 3, 3, 6, prod(Λ.PN))

    JACC.parallel_for(prod(Λ.PN), kernel_construct_Λmatrix_forSTOUT!, Λ.A, δ_current.A, Q.A, u.A, NC1, Λ.indexer, nw, global_buffer)
    set_halo!(Λ)
end
export construct_Λmatrix_forSTOUT_matrix!
