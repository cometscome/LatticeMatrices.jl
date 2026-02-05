module LatticeMatricesEnzymeExt
using LinearAlgebra
using LatticeMatrices
using Enzyme
using JACC
import LatticeMatrices: Wiltinger_derivative!, toann, DiffArg, NoDiffArg, Enzyme_derivative!, fold_halo_to_core_grad!, dSFdU,
    zero_halo_region!, zero_halo_dim!, fold_halo_dim_to_core_grad!


include("./AD/AD.jl")

# Convert user-specified arguments into Enzyme annotations.
#
# - DiffArg is mapped to Active (for scalars) or Duplicated (if extended later).
# - NoDiffArg is always mapped to Const.

toann(a::DiffArg) = Enzyme.Active(a.x)
toann(a::NoDiffArg) = Enzyme.Const(a.x)

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
    #println("Enzyme_derivative! in LatticeMatrices.jl")
    Enzyme.API.strictAliasing!(false)
    # Primary variables: always differentiated
    annU1 = Enzyme.Duplicated(U1, dfdU1)
    annU2 = Enzyme.Duplicated(U2, dfdU2)
    annU3 = Enzyme.Duplicated(U3, dfdU3)
    annU4 = Enzyme.Duplicated(U4, dfdU4)

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
            push!(extra_args, Duplicated(phitemp, dphitemp))
        end
        if temp !== nothing
            push!(extra_args, Duplicated(temp, dtemp))
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
    println("Enzyme_derivative! in LatticeMatrices.jl")
    Enzyme.API.strictAliasing!(false)
    # Primary variables: always differentiated
    annU1 = Enzyme.Duplicated(U1, dfdU1)
    annU2 = Enzyme.Duplicated(U2, dfdU2)
    annU3 = Enzyme.Duplicated(U3, dfdU3)

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
            ann_args..., Duplicated(temp, dtemp)
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
    println("Enzyme_derivative! in LatticeMatrices.jl")
    Enzyme.API.strictAliasing!(false)
    # Primary variables: always differentiated
    annU1 = Enzyme.Duplicated(U1, dfdU1)
    annU2 = Enzyme.Duplicated(U2, dfdU2)

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
            ann_args..., Duplicated(temp, dtemp)
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
    println("Enzyme_derivative! in LatticeMatrices.jl")
    Enzyme.API.strictAliasing!(false)
    # Primary variables: always differentiated
    annU1 = Enzyme.Duplicated(U1, dfdU1)

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
            ann_args..., Duplicated(temp, dtemp)
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
function Wiltinger_derivative!(func, U, dfdU, temp=nothing, dtemp=nothing; params...)
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
    Wiltinger!.(dfdU)
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

    DdagD = DdagDOp(D)
    phitemp1, itphitemp1 = get_block(D.phitemps)
    η = phitemp1

    solve!(η, DdagD, φ) #η = (DdagD)^-1 φ
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
