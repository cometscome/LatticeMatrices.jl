module LatticeMatricesEnzymeExt
using LinearAlgebra
using LatticeMatrices
using Enzyme
using JACC
import LatticeMatrices: Wirtinger_derivative!, toann, DiffArg, NoDiffArg, Enzyme_derivative!, fold_halo_to_core_grad!, dSFdU,
    zero_halo_region!, zero_halo_dim!, fold_halo_dim_to_core_grad!, enzyme_duplicated


include("./AD/AD.jl")
include("./AD/WilsonDiracOperator.jl")
include("./AD/StaggeredDiracOperator.jl")
include("./AD/HISQDiracOperator.jl")
include("./AD/HISQSmearing.jl")
include("./AD/HISQFullSmearing.jl")
include("./AD/WilsonDiracCloverOperator.jl")
include("./AD/DomainwallDiracOperator.jl")

# Convert user-specified arguments into Enzyme annotations.
#
# - DiffArg is mapped to Active (for scalars) or Duplicated (if extended later).
# - NoDiffArg is always mapped to Const.

toann(a::DiffArg) = Enzyme.Active(a.x)
toann(a::NoDiffArg) = Enzyme.Const(a.x)

function _reject_nw0_enzyme_input(x::LatticeMatrix, label::AbstractString)
    x.nw == 0 && throw(ArgumentError(
        "Enzyme_derivative! does not support nw=0 lattice arguments ($label); " *
        "construct every lattice argument and work buffer with nw >= 1."))
    return nothing
end

function _reject_nw0_enzyme_input(xs::Tuple, label::AbstractString)
    for (i, x) in pairs(xs)
        _reject_nw0_enzyme_input(x, "$label[$i]")
    end
    return nothing
end

function _reject_nw0_enzyme_input(xs::AbstractVector, label::AbstractString)
    for (i, x) in pairs(xs)
        (x isa LatticeMatrix || x isa DiffArg || x isa NoDiffArg) || continue
        _reject_nw0_enzyme_input(x, "$label[$i]")
    end
    return nothing
end

_reject_nw0_enzyme_input(x::Union{DiffArg,NoDiffArg}, label::AbstractString) =
    _reject_nw0_enzyme_input(x.x, label)
_reject_nw0_enzyme_input(::Any, ::AbstractString) = nothing

function _validate_enzyme_inputs(inputs::Pair...)
    for (label, value) in inputs
        _reject_nw0_enzyme_input(value, string(label))
    end
    return nothing
end

@static if VERSION >= v"1.12"
    # Julia 1.12 passes the GC roots of immutable composite arguments
    # separately from their inline fields.  Lattice matrices and fixed-size
    # workspaces therefore need Enzyme's mixed-activity ABI.
    @inline _enzyme_workspace(x::AbstractVector) = Tuple(x)
end

@inline _enzyme_workspace(x) = x

@static if VERSION >= v"1.12"
    @inline function enzyme_duplicated(x, dx)
        return Enzyme.MixedDuplicated(x, Ref(dx))
    end
    @inline function enzyme_duplicated(
        x::AbstractVector{<:LatticeMatrix},
        dx::AbstractVector{<:LatticeMatrix},
    )
        length(x) == length(dx) || throw(DimensionMismatch(
            "primal and shadow lattice collections must have equal length"))
        return Enzyme.MixedDuplicated(Tuple(x), Ref(Tuple(dx)))
    end
else
    @inline enzyme_duplicated(x, dx) = Enzyme.Duplicated(x, dx)
end

function _enzyme_workspace_pair(x, dx, label::AbstractString)
    if x === nothing
        dx === nothing || throw(ArgumentError("$label is nothing but its shadow is set"))
        return nothing, nothing
    end
    dx === nothing && throw(ArgumentError("$label is set but its shadow is nothing"))
    return _enzyme_workspace(x), _enzyme_workspace(dx)
end

function _fold_and_zero!(ls::LatticeMatrix)
    for d in length(ls.PN):-1:1
        fold_halo_dim_to_core_grad!(ls, d)
    end
    zero_halo_region!(ls)
    return nothing
end



Enzyme_derivative!(func, U1, U2, U3, U4, dfdU1, dfdU2, dfdU3, dfdU4, temp, dtemp, args...) =
    Enzyme_derivative!(func, U1, U2, U3, U4, dfdU1, dfdU2, dfdU3, dfdU4, args...; temp=temp, dtemp=dtemp)

Enzyme_derivative!(func, U1, U2, U3, U4, dfdU1, dfdU2, dfdU3, dfdU4, temp, dtemp, phitemp, dphitemp, args...) =
    Enzyme_derivative!(func, U1, U2, U3, U4, dfdU1, dfdU2, dfdU3, dfdU4, args...; temp=temp, dtemp=dtemp, phitemp=phitemp, dphitemp=dphitemp)

function Enzyme_derivative!(
    func,
    U::Vector{T},
    dfdU, args...;
    temp=nothing,
    dtemp=nothing
) where T
    # NOTE: Vector U input is not supported. Define a function with U1,U2,U3,U4 args for autodiff.
    error("Enzyme_derivative! does not support Vector U input. Please define a function that takes U1, U2, U3, U4 as separate arguments and run autodiff on that.")
end

function Enzyme_derivative!(
    func,
    U1,
    U2,
    U3,
    U4,
    dfdU1,
    dfdU2,
    dfdU3,
    dfdU4, args...;
    temp=nothing,
    dtemp=nothing,
    phitemp=nothing,
    dphitemp=nothing
)
    _validate_enzyme_inputs(
        "U1" => U1, "U2" => U2, "U3" => U3, "U4" => U4,
        "dfdU1" => dfdU1, "dfdU2" => dfdU2, "dfdU3" => dfdU3, "dfdU4" => dfdU4,
        "args" => args, "temp" => temp, "dtemp" => dtemp,
        "phitemp" => phitemp, "dphitemp" => dphitemp)
    temp, dtemp = _enzyme_workspace_pair(temp, dtemp, "temp")
    phitemp, dphitemp = _enzyme_workspace_pair(phitemp, dphitemp, "phitemp")
    #println("Enzyme_derivative! in LatticeMatrices.jl")
    Enzyme.API.strictAliasing!(false)
    # Primary variables: always differentiated
    annU1 = enzyme_duplicated(U1, dfdU1)
    annU2 = enzyme_duplicated(U2, dfdU2)
    annU3 = enzyme_duplicated(U3, dfdU3)
    annU4 = enzyme_duplicated(U4, dfdU4)

    # Convert additional arguments
    ann_args = map(toann, args)

    if phitemp !== nothing && dphitemp === nothing
        error("phitemp is set but dphitemp is nothing")
    end

    # Call Enzyme
    if temp === nothing && phitemp === nothing
        result = Enzyme.autodiff(
            Reverse,
            Enzyme.Const(func),     # function object is always treated as read-only
            Active,          # return value is a real scalar
            annU1,
            annU2,
            annU3,
            annU4,
            ann_args...
        )
    else
        extra_args = Any[]
        if phitemp !== nothing
            push!(extra_args, enzyme_duplicated(phitemp, dphitemp))
        end
        if temp !== nothing
            push!(extra_args, enzyme_duplicated(temp, dtemp))
        end
        result = Enzyme.autodiff(
            Reverse,
            Enzyme.Const(func),
            Active,
            annU1,
            annU2,
            annU3,
            annU4,
            ann_args...,
            extra_args...
            #ann_args..., DuplicatedNoNeed(temp, dtemp)
        )
    end

    # Halo values are constrained to core values; fold halo gradients back to core.
    _fold_and_zero!(dfdU1)
    _fold_and_zero!(dfdU2)
    _fold_and_zero!(dfdU3)
    _fold_and_zero!(dfdU4)

    # Gradients of Active scalar arguments are returned by Enzyme
    return result
end

function Enzyme_derivative!(
    func,
    U1,
    U2,
    U3,
    dfdU1,
    dfdU2,
    dfdU3, args...;
    temp=nothing,
    dtemp=nothing
)
    _validate_enzyme_inputs(
        "U1" => U1, "U2" => U2, "U3" => U3,
        "dfdU1" => dfdU1, "dfdU2" => dfdU2, "dfdU3" => dfdU3,
        "args" => args, "temp" => temp, "dtemp" => dtemp)
    temp, dtemp = _enzyme_workspace_pair(temp, dtemp, "temp")
    println("Enzyme_derivative! in LatticeMatrices.jl")
    Enzyme.API.strictAliasing!(false)
    # Primary variables: always differentiated
    annU1 = enzyme_duplicated(U1, dfdU1)
    annU2 = enzyme_duplicated(U2, dfdU2)
    annU3 = enzyme_duplicated(U3, dfdU3)

    # Convert additional arguments
    ann_args = map(toann, args)

    # Call Enzyme
    if temp === nothing
        result = Enzyme.autodiff(
            Reverse,
            Enzyme.Const(func),     # function object is always treated as read-only
            Active,          # return value is a real scalar
            annU1,
            annU2,
            annU3,
            ann_args...
        )
    else
        result = Enzyme.autodiff(
            Reverse,
            Enzyme.Const(func),
            Active,
            annU1,
            annU2,
            annU3,
            ann_args..., enzyme_duplicated(temp, dtemp)
            #ann_args..., DuplicatedNoNeed(temp, dtemp)
        )
    end

    # Halo values are constrained to core values; fold halo gradients back to core.
    _fold_and_zero!(dfdU1)
    _fold_and_zero!(dfdU2)
    _fold_and_zero!(dfdU3)

    # Gradients of Active scalar arguments are returned by Enzyme
    return result
end

function Enzyme_derivative!(
    func,
    U1,
    U2,
    dfdU1,
    dfdU2, args...;
    temp=nothing,
    dtemp=nothing
)
    _validate_enzyme_inputs(
        "U1" => U1, "U2" => U2,
        "dfdU1" => dfdU1, "dfdU2" => dfdU2,
        "args" => args, "temp" => temp, "dtemp" => dtemp)
    temp, dtemp = _enzyme_workspace_pair(temp, dtemp, "temp")
    println("Enzyme_derivative! in LatticeMatrices.jl")
    Enzyme.API.strictAliasing!(false)
    # Primary variables: always differentiated
    annU1 = enzyme_duplicated(U1, dfdU1)
    annU2 = enzyme_duplicated(U2, dfdU2)

    # Convert additional arguments
    ann_args = map(toann, args)

    # Call Enzyme
    if temp === nothing
        result = Enzyme.autodiff(
            Reverse,
            Enzyme.Const(func),     # function object is always treated as read-only
            Active,          # return value is a real scalar
            annU1,
            annU2,
            ann_args...
        )
    else
        result = Enzyme.autodiff(
            Reverse,
            Enzyme.Const(func),
            Active,
            annU1,
            annU2,
            ann_args..., enzyme_duplicated(temp, dtemp)
            #ann_args..., DuplicatedNoNeed(temp, dtemp)
        )
    end

    # Halo values are constrained to core values; fold halo gradients back to core.
    _fold_and_zero!(dfdU1)
    _fold_and_zero!(dfdU2)

    # Gradients of Active scalar arguments are returned by Enzyme
    return result
end

function Enzyme_derivative!(
    func,
    U1,
    dfdU1, args...;
    temp=nothing,
    dtemp=nothing
)
    _validate_enzyme_inputs(
        "U1" => U1, "dfdU1" => dfdU1,
        "args" => args, "temp" => temp, "dtemp" => dtemp)
    temp, dtemp = _enzyme_workspace_pair(temp, dtemp, "temp")
    println("Enzyme_derivative! in LatticeMatrices.jl")
    Enzyme.API.strictAliasing!(false)
    # Primary variables: always differentiated
    annU1 = enzyme_duplicated(U1, dfdU1)

    # Convert additional arguments
    ann_args = map(toann, args)

    # Call Enzyme
    if temp === nothing
        result = Enzyme.autodiff(
            Reverse,
            Enzyme.Const(func),     # function object is always treated as read-only
            Active,          # return value is a real scalar
            annU1,
            ann_args...
        )
    else
        result = Enzyme.autodiff(
            Reverse,
            Enzyme.Const(func),
            Active,
            annU1,
            ann_args..., enzyme_duplicated(temp, dtemp)
            #ann_args..., DuplicatedNoNeed(temp, dtemp)
        )
    end

    # Halo values are constrained to core values; fold halo gradients back to core.
    _fold_and_zero!(dfdU1)
    # Gradients of Active scalar arguments are returned by Enzyme
    return result
end

export Enzyme_derivative
#=
function Wirtinger_derivative!(func, U, dfdU, temp=nothing, dtemp=nothing; params...)
    if length(params) > 1
        if temp === nothing
            Enzyme.autodiff(Reverse, Const(func), Active,
                Duplicated(U, dfdU), Enzyme.Const.(params...))
        else
            Enzyme.autodiff(Reverse, Const(func), Active,
                Duplicated(U, dfdU), DuplicatedNoNeed(temp, dtemp), Enzyme.Const.(params...))
        end
    else
        if temp === nothing
            Enzyme.autodiff(Reverse, Const(func), Active,
                Duplicated(U, dfdU))
        else
            Enzyme.autodiff(Reverse, Const(func), Active,
                Duplicated(U, dfdU), DuplicatedNoNeed(temp, dtemp))
        end
    end
    #println("1")
    #display(dfdU[1].A[:, :, 2, 2, 2, 2])
    Wirtinger!.(dfdU)
    #println("2")
    #display(dfdU[1].A[:, :, 2, 2, 2, 2])
end
=#

function g(χ, U1, U2, U3, U4, η, p, apply, phitemp, temp)
    phitemp1 = phitemp[end]
    apply(phitemp1, U1, U2, U3, U4, η, p, phitemp, temp)
    #Dmul!(phitemp1, U1, U2, U3, U4, D, η)
    s = -2 * real(dot(χ, phitemp1))
    return s
end

function dSFdU(dfdU, D::T, φ; numtemp=5) where {T<:DiracOp}
    U = D.U
    U1 = U[1]
    U2 = U[2]
    U3 = U[3]
    U4 = U[4]

    #dfdU, itdfdUtemp = get_block(D.temps, 4)
    dfdU1 = dfdU[1]
    dfdU2 = dfdU[2]
    dfdU3 = dfdU[3]
    dfdU4 = dfdU[4]

    Dη, itDη = get_block(D.phitemps)
    DdagD = DdagDOp(D, Dη)
    phitemp1, itphitemp1 = get_block(D.phitemps)
    η = phitemp1
    cg_r, itcg_r = get_block(D.phitemps)
    cg_p, itcg_p = get_block(D.phitemps)
    cg_Ap, itcg_Ap = get_block(D.phitemps)

    solve!(η, DdagD, φ, cg_r, cg_p, cg_Ap) #η = (DdagD)^-1 φ
    unused!(D.phitemps, itcg_r)
    unused!(D.phitemps, itcg_p)
    unused!(D.phitemps, itcg_Ap)
    unused!(D.phitemps, itDη)
    println("solved")
    set_halo!(η)
    phitemp2, itphitemp2 = get_block(D.phitemps)
    χ = phitemp2
    mul!(χ, D, η)

    #phitemp1, itphitemp1 = get_block(D.phitemps)
    func(U1, U2, U3, U4, χ, η, apply, phitemp, temp) = g(χ, U1, U2, U3, U4, η, D.p, apply, phitemp, temp)

    temp, ittemp = get_block(D.temps, numtemp)
    phitemp, itphitemp = get_block(D.phitemps, numtemp)
    dtemp, itdtemp = get_block(D.temps, numtemp)
    dphitemp, itdphitemp = get_block(D.phitemps, numtemp)

    Enzyme_derivative!(
        func,
        U1,
        U2,
        U3,
        U4,
        dfdU1,
        dfdU2,
        dfdU3,
        dfdU4,
        nodiff(χ), nodiff(η), nodiff(D.apply); temp=temp, dtemp=dtemp, phitemp=phitemp, dphitemp=dphitemp)

    #for μ = 1:4
    #    mul!(dfdU[μ], -2)
    #end

    unused!(D.temps, ittemp)
    unused!(D.temps, itdtemp)
    unused!(D.phitemps, itphitemp)
    unused!(D.phitemps, itdphitemp)
    unused!(D.phitemps, itphitemp1)
    unused!(D.phitemps, itphitemp2)



end

end # module
