module LatticeMatricesEnzymeExt
using LinearAlgebra
using LatticeMatrices
using Enzyme
using JACC
import LatticeMatrices: Wiltinger_derivative!, toann, DiffArg, NoDiffArg

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
            ann_args..., DuplicatedNoNeed(temp, dtemp)
        )
    end

    # Convert real/imaginary gradients to Wirtinger derivatives
    Wiltinger!.(dfdU)

    # Gradients of Active scalar arguments are returned by Enzyme
    return result
end
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