


const shift_1p = (1, 0, 0, 0)
const shift_1m = (-1, 0, 0, 0)
const shift_2p = (0, 1, 0, 0)
const shift_2m = (0, -1, 0, 0)
const shift_3p = (0, 0, 1, 0)
const shift_3m = (0, 0, -1, 0)
const shift_4p = (0, 0, 0, 1)
const shift_4m = (0, 0, 0, -1)
const shifts_p = (shift_1p, shift_2p, shift_3p, shift_4p)
const shifts_m = (shift_1m, shift_2m, shift_3m, shift_4m)

include("WilsonDiracOperator.jl")
include("WilsonDiracCloverOperator.jl")
include("StaggeredDiracOperator.jl")
include("HISQSmearing.jl")
include("HISQDiracOperator.jl")
include("HISQFullSmearing.jl")
include("HISQPullback.jl")


struct DiracOp{T,TF,Dmul,Ddagmul,P}
    U::Vector{T}
    apply::Dmul
    apply_dag::Ddagmul
    p::P
    temps::PreallocatedArray{T}
    phitemps::PreallocatedArray{TF}
end

function DiracOp(U, apply, apply_dag, p, phi; numtemp=4, numphitemp=4)
    T = eltype(U)
    Dmul = typeof(apply)
    Ddagmul = typeof(apply_dag)

    temps = PreallocatedArray(U[1]; num=numtemp, haslabel=false)
    phitemps = PreallocatedArray(phi; num=numphitemp, haslabel=false)
    TF = typeof(phi)


    return DiracOp{T,TF,Dmul,Ddagmul,typeof(p)}(U, apply, apply_dag, p, temps, phitemps)
end
export DiracOp

function LinearAlgebra.mul!(y, D::DiracOp, x)
    ensure_halo!(x)
    temp, ittemp = get_block(D.temps, 4)
    phitemp, itphitemp = get_block(D.phitemps, 4)
    D.apply(y, D.U[1], D.U[2], D.U[3], D.U[4], x, D.p, phitemp, temp)
    unused!(D.temps, ittemp)
    unused!(D.phitemps, itphitemp)
    return y
end



struct AdjointOp{Op}
    op::Op
end
Base.adjoint(D::DiracOp) = AdjointOp(D)
Base.adjoint(A::AdjointOp{<:DiracOp}) = A.op

function LinearAlgebra.mul!(y, A::AdjointOp{<:DiracOp}, x)
    D = A.op
    ensure_halo!(x)
    temp, ittemp = get_block(D.temps, 4)
    phitemp, itphitemp = get_block(D.phitemps, 4)

    D.apply_dag(y, D.U[1], D.U[2], D.U[3], D.U[4], x, D.p, phitemp, temp)

    unused!(D.temps, ittemp)
    unused!(D.phitemps, itphitemp)
    return y
end



"""
    DdagDOp(D, temp)
    DdagDOp(D::DiracOp)

Allocation-free normal operator `D' * D`.  `temp` is caller-owned storage for
`D*x` and must not alias the input or output passed to `mul!`.  A distinct
`DdagDOp` (and `temp`) is required for each concurrent application.

The one-argument `DiracOp` constructor is retained for compatibility.  It
borrows the temporary field from `D.phitemps` for each application, as in the
pre-v1 implementation.  New code should pass `temp` explicitly.
"""
struct DdagDOp{T,F}
    D::T
    temp::F
end

DdagDOp(D::DiracOp) = DdagDOp(D, nothing)

export DdagDOp

Base.adjoint(A::DdagDOp) = A

function LinearAlgebra.mul!(y, A::DdagDOp, x)
    if A.temp === nothing
        temp, temp_index = get_block(A.D.phitemps)
        try
            mul!(temp, A.D, x)
            mul!(y, adjoint(A.D), temp)
            return y
        finally
            unused!(A.D.phitemps, temp_index)
        end
    end

    (A.temp === x || A.temp === y) && throw(ArgumentError(
        "DdagDOp temporary field must not alias its input or output"))
    mul!(A.temp, A.D, x)
    mul!(y, adjoint(A.D), A.temp)
    return y
end


"""
    solve!(x, A::DdagDOp, b, r, p, Ap; kwargs...)

Explicit-workspace convenience alias for [`cg!`](@ref).  The returned value is
a [`CGResult`](@ref).
"""
function solve!(x, A::DdagDOp, b, r, p, Ap; kwargs...)
    return cg!(x, A, b, r, p, Ap; kwargs...)
end

"""
    solve!(x, A::DdagDOp{<:DiracOp}, b; verboselevel=2)

Compatibility interface that borrows CG work fields from `A.D.phitemps` and
returns `nothing` on convergence.  New code should use the explicit-workspace
method and inspect its `CGResult`.
"""
function solve!(x, A::DdagDOp{<:DiracOp}, b; verboselevel=2)
    return cg(x, A, b, A.D.phitemps; verboselevel)
end

export solve!

"""
    pseudofermion_action(D, phi, eta, Deta, r, p, Ap; kwargs...)

Compute `real(dot(phi, (D' * D) \\ phi))`.  `eta` contains the initial guess
and is overwritten by the solution.  `Deta` is the normal-operator temporary;
`r`, `p`, and `Ap` are the three CG work fields.  All five fields are supplied
and owned by the caller.
"""
function pseudofermion_action(D, φ, η, Dη, r, p, Ap; kwargs...)
    normal_operator = DdagDOp(D, Dη)
    result = solve!(η, normal_operator, φ, r, p, Ap; kwargs...)
    result.converged || error(
        "CG failed with reason $(result.reason) after $(result.iterations) iterations; " *
        "relative residual = $(result.relative_residual)")
    return real(dot(φ, η))
end

"""
    pseudofermion_action(D::DiracOp, phi)

Compatibility interface that borrows the solution and work fields from
`D.phitemps`, matching the pre-v1 API.  New code should supply the five work
fields explicitly.
"""
function pseudofermion_action(D::DiracOp, φ)
    η, eta_index = get_block(D.phitemps)
    try
        normal_operator = DdagDOp(D)
        solve!(η, normal_operator, φ)
        return real(dot(φ, η))
    finally
        unused!(D.phitemps, eta_index)
    end
end

export pseudofermion_action




function dSFdU end
export dSFdU
