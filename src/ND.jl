function numerical_differentiation(f, indices, A::T, params...) where {D,T1,AT1,NC1,NC2,nw,DI,T<:LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI}}
    ϵ = 1e-8
    grad = zeros(T1, NC1, NC2)

    for jc = 1:NC2
        for ic = 1:NC1
            Ap = deepcopy(A)
            Ap.A[ic, jc, indices...] += ϵ
            set_halo!(Ap)
            Am = deepcopy(A)
            Am.A[ic, jc, indices...] -= ϵ
            set_halo!(Am)
            grad[ic, jc] = (f(Ap, params...) - f(Am, params...)) / (2ϵ)

            Ap = deepcopy(A)
            Ap.A[ic, jc, indices...] += im * ϵ
            set_halo!(Ap)
            Am = deepcopy(A)
            Am.A[ic, jc, indices...] -= im * ϵ
            set_halo!(Am)
            grad[ic, jc] += im * (f(Ap, params...) - f(Am, params...)) / (2ϵ)
        end
    end

    return grad
end
export numerical_differentiation

function numerical_differentiation(f, indices, A::T, B::T, params...) where {D,T1,AT1,NC1,NC2,nw,DI,T<:LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI}}
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


"""
    numerical_differentiation_U(f, indices, U; params=(), targets=:all, ϵ=1e-8)

Numerical differentiation for functions of multiple LatticeMatrix inputs.

- `U` can be a Tuple or Vector of LatticeMatrix.
- `indices` is the lattice site indices (e.g. (2,2,2,2)).
- For each target k in `targets`, compute a matrix `grad[k]` of size (NC1,NC2):
      grad[k][ic,jc] = d f / dRe(U[k].A[ic,jc,indices...])
                     + i * d f / dIm(U[k].A[ic,jc,indices...])
  using central differences.

Returns:
- If `U` is a Tuple: returns a Tuple of gradients, one per element.
- If `U` is a Vector: returns a Vector of gradients, one per element.

Notes:
- Returns the Wirtinger derivative ∂f/∂U = (df/dx - i df/dy)/2.
"""
function Wiltinger_numerical_derivative(f, indices, U;
    params=(),
    targets=:all,
    ϵ::Real=1e-8)

    # normalize U to a mutable container for easy copying/replacement
    is_tuple = U isa Tuple
    Uvec = is_tuple ? collect(U) : copy(U)

    # which components of U to differentiate
    Ks = targets === :all ? eachindex(Uvec) : targets

    grads = Vector{Any}(undef, length(Uvec))
    for i in eachindex(grads)
        grads[i] = nothing
    end

    # helper to call f with the same container type as the original U
    function callf(Uwork)
        if is_tuple
            return f(Tuple(Uwork)..., params...)
        else
            return f(Uwork, params...)   # If your f expects f(U::Vector, ...) keep this
        end
    end

    # Important:
    # - If your f is defined as f(U, temp, ...) (single U container),
    #   then set callf accordingly. Here we support both common styles:
    #
    #   (A) f(U::Vector, params...)
    #   (B) f(U1, U2, ..., params...)
    #
    # If you always use style (A), we can simplify.

    for k in Ks
        Uk = Uvec[k]
        @assert hasfield(typeof(Uk), :A) "U[$k] must have field `.A`"

        # infer element type and matrix size from Uk.A
        Aarr = Uk.A
        T1 = eltype(Aarr)
        NC1, NC2 = size(Aarr, 1), size(Aarr, 2)

        grad = zeros(T1, NC1, NC2)

        for jc = 1:NC2, ic = 1:NC1
            # --- Re part derivative ---
            Up = deepcopy(Uvec)
            Up[k].A[ic, jc, indices...] += ϵ
            Um = deepcopy(Uvec)
            Um[k].A[ic, jc, indices...] -= ϵ
            dRe = (callf(Up) - callf(Um)) / (2ϵ)

            # --- Im part derivative ---
            Up = deepcopy(Uvec)
            Up[k].A[ic, jc, indices...] += im * ϵ
            Um = deepcopy(Uvec)
            Um[k].A[ic, jc, indices...] -= im * ϵ
            dIm = (callf(Up) - callf(Um)) / (2ϵ)

            # your convention: df/dx + i df/dy
            #grad[ic, jc] = dRe + im * dIm
            # Wiltinger: 
            grad[ic, jc] = (dRe - im * dIm) / 2
        end

        grads[k] = grad
    end

    # return in the same container style
    if is_tuple
        return Tuple(grads)
    else
        return grads
    end
end


export Wiltinger_numerical_derivative
