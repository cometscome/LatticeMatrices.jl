"""
    CGResult

Status returned by [`cg!`](@ref).  `reason` is one of `:converged`,
`:maximum_iterations`, `:nonpositive_curvature`, or `:numerical_breakdown`.
"""
struct CGResult{T<:Real}
    converged::Bool
    iterations::Int
    residual_norm::T
    relative_residual::T
    reason::Symbol
end

export CGResult

@inline function _cg_relative_residual(residual_norm, right_hand_side_norm)
    if iszero(right_hand_side_norm)
        return iszero(residual_norm) ? zero(residual_norm) :
               oftype(residual_norm, Inf)
    end
    return residual_norm / right_hand_side_norm
end

@inline function _cg_result(converged, iterations, residual_norm,
    right_hand_side_norm, reason)
    relative_residual = _cg_relative_residual(
        residual_norm, right_hand_side_norm)
    return CGResult(converged, iterations, residual_norm,
        relative_residual, reason)
end

function _validate_cg_arguments(x, A, b, r, p, Ap, rtol, atol, maxiter)
    fields = (x, b, r, p, Ap)
    names = (:x, :b, :r, :p, :Ap)
    for i in eachindex(fields), j in (i + 1):length(fields)
        fields[i] === fields[j] && throw(ArgumentError(
            "CG fields $(names[i]) and $(names[j]) must not alias"))
    end

    rtol >= zero(rtol) || throw(ArgumentError("rtol must be nonnegative"))
    atol >= zero(atol) || throw(ArgumentError("atol must be nonnegative"))
    isfinite(rtol) || throw(ArgumentError("rtol must be finite"))
    isfinite(atol) || throw(ArgumentError("atol must be finite"))
    maxiter >= 0 || throw(ArgumentError("maxiter must be nonnegative"))
    return nothing
end

"""
    cg!(x, A, b, r, p, Ap; rtol=1e-10, atol=0, maxiter=5000)

Solve `A*x = b` with the conjugate-gradient method, overwriting the initial
guess `x`.  `A` must be Hermitian positive definite and implement
`mul!(output, A, input)`.

The caller owns and supplies all three work fields:

- `r`: residual,
- `p`: search direction,
- `Ap`: operator image of the search direction.

`x`, `b`, `r`, `p`, and `Ap` must have compatible geometry and backend, and
must not share storage.  The contents of the three work fields are unspecified
after return.  No fields are allocated and no output is printed.  The return
value is a [`CGResult`](@ref); non-convergence is reported there rather than
thrown as an exception.

The stopping condition is
`norm(r) <= max(atol, rtol * norm(b))`.  Operator implementations are
responsible for synchronizing any input halo needed by `mul!`; CG itself only
uses the lattice's public linear-algebra operations.
"""
function cg!(x, A, b, r, p, Ap;
    rtol=1e-10, atol=0, maxiter::Integer=5000)

    _validate_cg_arguments(x, A, b, r, p, Ap, rtol, atol, maxiter)

    bnorm_squared = real(dot(b, b))
    if !isfinite(bnorm_squared) || bnorm_squared < zero(bnorm_squared)
        nan_norm = oftype(float(bnorm_squared), NaN)
        return _cg_result(false, 0, nan_norm, nan_norm,
            :numerical_breakdown)
    end
    bnorm = sqrt(bnorm_squared)
    tolerance = max(convert(typeof(bnorm), atol),
        convert(typeof(bnorm), rtol) * bnorm)

    # Ap is also the initial A*x buffer, so standard CG needs only three
    # explicitly supplied work fields rather than four.
    mul!(Ap, A, x)
    axpby!(1, b, 0, r)
    axpby!(-1, Ap, 1, r)
    axpby!(1, r, 0, p)

    rr = real(dot(r, r))
    if !isfinite(rr) || rr < zero(rr)
        nan_norm = oftype(bnorm, NaN)
        return _cg_result(false, 0, nan_norm, bnorm,
            :numerical_breakdown)
    end
    residual_norm = sqrt(rr)
    residual_norm <= tolerance &&
        return _cg_result(true, 0, residual_norm, bnorm, :converged)

    for iteration in 1:maxiter
        mul!(Ap, A, p)
        pAp = real(dot(p, Ap))
        if !isfinite(pAp)
            return _cg_result(false, iteration - 1, residual_norm, bnorm,
                :numerical_breakdown)
        elseif pAp <= zero(pAp)
            return _cg_result(false, iteration - 1, residual_norm, bnorm,
                :nonpositive_curvature)
        end

        alpha = rr / pAp
        axpby!(alpha, p, 1, x)
        axpby!(-alpha, Ap, 1, r)

        rr_new = real(dot(r, r))
        if !isfinite(rr_new) || rr_new < zero(rr_new)
            return _cg_result(false, iteration, oftype(bnorm, NaN), bnorm,
                :numerical_breakdown)
        end
        residual_norm = sqrt(rr_new)
        residual_norm <= tolerance &&
            return _cg_result(true, iteration, residual_norm, bnorm,
                :converged)

        beta = rr_new / rr
        axpby!(1, r, beta, p)
        rr = rr_new
    end

    return _cg_result(false, maxiter, residual_norm, bnorm,
        :maximum_iterations)
end

export cg!

# Compatibility adapter for the original pool-based API.  Old callback
# operators relied on CG eagerly synchronizing the field passed to `mul!`, so
# retain that behavior here without imposing it on the explicit low-level API.
struct _LegacyCGOperator{T}
    parent::T
end

function LinearAlgebra.mul!(y, A::_LegacyCGOperator, x)
    set_halo!(x)
    return mul!(y, A.parent, x)
end

"""
    cg(x, A, b, temps; eps=1e-10, maxsteps=5000, verboselevel=2)

Compatibility interface for the original pool-based CG implementation.
Three work fields are borrowed from `temps` and passed to [`cg!`](@ref).
This method retains the original `eps`/`maxsteps` keywords, returns `nothing`
on convergence, and throws on failure.  New low-level code should call `cg!`
and supply `r`, `p`, and `Ap` explicitly.
"""
function cg(x, A, b, temps;
    eps=1e-10, maxsteps::Integer=5000, verboselevel=2)

    verboselevel >= 3 && begin
        println("--------------------------------------")
        println("cg method")
    end

    r, r_index = get_block(temps)
    p, p_index = get_block(temps)
    Ap, Ap_index = get_block(temps)

    result = try
        cg!(x, _LegacyCGOperator(A), b, r, p, Ap;
            rtol=eps, atol=0, maxiter=maxsteps)
    finally
        unused!(temps, r_index)
        unused!(temps, p_index)
        unused!(temps, Ap_index)
    end

    if result.converged
        set_halo!(x)
        if verboselevel >= 3
            println("Converged at $(result.iterations)-th step. eps: " *
                    "$(result.relative_residual)")
            println("--------------------------------------")
        end
        return nothing
    end

    error("""
    The CG is not converged! with maxsteps = $maxsteps
    residual is $(result.relative_residual)
    reason is $(result.reason)
    maxsteps should be larger.""")
end
