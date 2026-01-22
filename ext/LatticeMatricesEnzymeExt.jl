module LatticeMatricesEnzymeExt
using LinearAlgebra
using LatticeMatrices
using Enzyme
using JACC
import LatticeMatrices: Wiltinger_derivative!, toann, DiffArg, NoDiffArg, Enzyme_derivative!, fold_halo_to_core_grad!

include("./AD/AD.jl")

# Convert user-specified arguments into Enzyme annotations.
#
# - DiffArg is mapped to Active (for scalars) or Duplicated (if extended later).
# - NoDiffArg is always mapped to Const.

toann(a::DiffArg) = Enzyme.Active(a.x)
toann(a::NoDiffArg) = Enzyme.Const(a.x)

Wiltinger_derivative!(func, U, dfdU, temp, dtemp, args...) =
    Wiltinger_derivative!(func, U, dfdU, args...; temp=temp, dtemp=dtemp)

function Wiltinger_derivative!(
    func,
    U,
    dfdU, args...;
    temp=nothing,
    dtemp=nothing
)
    println("Wilttinger_derivative in LatticeMatrices.jl")
    # Primary variable: always differentiated
    annU = Enzyme.Duplicated(U, dfdU)

    # Convert additional arguments
    ann_args = map(toann, args)

    # Call Enzyme
    if temp === nothing
        result = Enzyme.autodiff(
            Reverse,
            Enzyme.Const(func),     # function object is always treated as read-only
            Active,          # return value is a real scalar
            annU,
            ann_args...
        )
    else
        result = Enzyme.autodiff(
            Reverse,
            Enzyme.Const(func),
            Active,
            annU,
            ann_args..., Duplicated(temp, dtemp)
            #ann_args..., DuplicatedNoNeed(temp, dtemp)
        )
    end

    # Convert real/imaginary gradients to Wirtinger derivatives
    Wiltinger!.(dfdU)

    # Gradients of Active scalar arguments are returned by Enzyme
    return result
end

Enzyme_derivative!(func, U1, U2, U3, U4, dfdU1, dfdU2, dfdU3, dfdU4, temp, dtemp, args...) =
    Enzyme_derivative!(func, U1, U2, U3, U4, dfdU1, dfdU2, dfdU3, dfdU4, args...; temp=temp, dtemp=dtemp)

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
    dtemp=nothing
)
    println("Enzyme_derivative! in LatticeMatrices.jl")
    Enzyme.API.strictAliasing!(false)
    # Primary variables: always differentiated
    annU1 = Enzyme.Duplicated(U1, dfdU1)
    annU2 = Enzyme.Duplicated(U2, dfdU2)
    annU3 = Enzyme.Duplicated(U3, dfdU3)
    annU4 = Enzyme.Duplicated(U4, dfdU4)

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
            annU4,
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
            annU4,
            ann_args..., Duplicated(temp, dtemp)
            #ann_args..., DuplicatedNoNeed(temp, dtemp)
        )
    end

    # Halo values are constrained to core values; fold halo gradients back to core.
    fold_halo_to_core_grad!(dfdU1)
    fold_halo_to_core_grad!(dfdU2)
    fold_halo_to_core_grad!(dfdU3)
    fold_halo_to_core_grad!(dfdU4)

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
    fold_halo_to_core_grad!(dfdU1)
    fold_halo_to_core_grad!(dfdU2)
    fold_halo_to_core_grad!(dfdU3)

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
    fold_halo_to_core_grad!(dfdU1)
    fold_halo_to_core_grad!(dfdU2)

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
    fold_halo_to_core_grad!(dfdU1)
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


end # module
