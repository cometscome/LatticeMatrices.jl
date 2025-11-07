function numerical_differenciation(f, indices, A::T, params...) where {D,T1,AT1,NC1,NC2,nw,DI,T<:LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI}}
    ϵ = 1e-8
    grad = zeros(T1, NC1, NC2)

    for jc = 1:NC2
        for ic = 1:NC1
            Ap = deepcopy(A)
            Ap.A[ic, jc, indices...] += ϵ
            Am = deepcopy(A)
            Am.A[ic, jc, indices...] -= ϵ
            grad[ic, jc] = (f(Ap, params...) - f(Am, params...)) / (2ϵ)

            Ap = deepcopy(A)
            Ap.A[ic, jc, indices...] += im * ϵ
            Am = deepcopy(A)
            Am.A[ic, jc, indices...] -= im * ϵ
            grad[ic, jc] += im * (f(Ap, params...) - f(Am, params...)) / (2ϵ)
        end
    end

    return grad
end
export numerical_differenciation

function numerical_differenciation(f, indices, A::T, B::T, params...) where {D,T1,AT1,NC1,NC2,nw,DI,T<:LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI}}
    ϵ = 1e-8
    gradA = zeros(T1, NC1, NC2)

    for jc = 1:NC2
        for ic = 1:NC1
            Ap = deepcopy(A)
            Ap.A[ic, jc, indices...] += ϵ
            Am = deepcopy(A)
            Am.A[ic, jc, indices...] -= ϵ
            gradA[ic, jc] = (f(Ap, B, params...) - f(Am, B, params...)) / (2ϵ)

            Ap = deepcopy(A)
            Ap.A[ic, jc, indices...] += im * ϵ
            Am = deepcopy(A)
            Am.A[ic, jc, indices...] -= im * ϵ
            gradA[ic, jc] += im * (f(Ap, B, params...) - f(Am, B, params...)) / (2ϵ)
        end
    end

    ϵ = 1e-8
    gradB = zeros(T1, NC1, NC2)

    for jc = 1:NC2
        for ic = 1:NC1
            Bp = deepcopy(B)
            Bp.A[ic, jc, indices...] += ϵ
            Bm = deepcopy(B)
            Bm.A[ic, jc, indices...] -= ϵ
            gradB[ic, jc] = (f(A, Bp, params...) - f(A, Bm, params...)) / (2ϵ)

            Bp = deepcopy(B)
            Bp.A[ic, jc, indices...] += im * ϵ
            Bm = deepcopy(B)
            Bm.A[ic, jc, indices...] -= im * ϵ
            gradB[ic, jc] += im * (f(A, Bp, params...) - f(A, Bm, params...)) / (2ϵ)
        end
    end

    return gradA, gradB
end
export numerical_differenciation