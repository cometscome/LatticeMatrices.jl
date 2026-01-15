
using LatticeMatrices
using MPI
using LinearAlgebra
using Enzyme
import JACC
JACC.@init_backend
import Enzyme: autodiff, Duplicated, Const
import Enzyme.EnzymeRules: forward, augmented_primal, reverse, FwdConfig, RevConfigWidth,
    Active, needs_primal, needs_shadow, AugmentedReturn,
    overwritten

import Enzyme: Duplicated, Const, Active
import Enzyme.EnzymeRules: augmented_primal, reverse, RevConfig, AugmentedReturn
using InteractiveUtils

Base.@noinline function tr2(z)
    println("tr")
    return real(tr(z))
end


# =
#= 
#function tr2(z)
#    return tr(z)
#end
# tr augmented_primal: forward pass just calls the existing one. No tape needed.
function augmented_primal(cfg::RevConfig,
    ::Const{typeof(tr2)},
    ::Type{<:Active},
    C::Annotation)
    println("ap")
    #error("dd")
    # Forward pass (can omit the needs_primal optimization)
    s = LinearAlgebra.tr(C.val)
    # tr output is scalar, so shadow is nothing and no tape is needed
    return AugmentedReturn(s, nothing, nothing)
end

# Reverse kernel: Ā[ic,ic,indices...] += ds
@inline function kernel_tr_pullback_4D(i, dA, ::Val{NC1}, dindexer, ::Val{nw}, dsval) where {NC1,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for ic = 1:NC1
        dA[ic, ic, indices...] += dsval
    end
    return nothing
end




# tr reverse: add upstream scalar ds to each site's diagonal
function Enzyme.EnzymeRules.reverse(cfg,
    ::Const{typeof(tr2)},
    ds, _tape,
    C) where {D,T1,AT1,NC1,nw,DI}
    #C::Duplicated{<:LatticeMatrix{D,T1,AT1,NC1,NC1,nw,DI}}) where {D,T1,AT1,NC1,nw,DI}
    println("re")
    error("dd")
    # No race: each site writes to distinct elements
    JACC.parallel_for(prod(C.val.PN)) do i
        kernel_tr_pullback_4D(i, C.dval.A, Val(NC1), C.val.indexer, Val(nw), ds.val)
    end
    return (nothing,)
end
=#

# ===== Loss calling your existing tr implementation =====

Base.@noinline function loss(A::AbstractMatrix{<:Real})::Float64
    s = 0.0
    @inbounds for j in axes(A, 2), i in axes(A, 1)
        s += A[i, j]^2
    end
    return s
end

function Enzyme.EnzymeRules.augmented_primal(cfg::RevConfig,
    ::Const{typeof(tr2)},
    ::Type{<:Active},                # Output is Active (flows to Float64 here)
    C::Annotation{T}) where {T<:LatticeMatrix} # Allow Duplicated/MixedDuplicated/Const
    if needs_primal(cfg)
        # Compute only when Enzyme says it needs the primal
        s = tr2(C.val)               # ComplexF64 でOK
        return AugmentedReturn(s, nothing, nothing)
    else
        # Current crash is here: must return Nothing
        return AugmentedReturn(nothing, nothing, nothing)
    end
end


function Enzyme.EnzymeRules.augmented_primal(cfg::RevConfig,
    ::Const{typeof(tr2)},
    ::Type{<:Active},                # Output is Active (flows to Float64 here)
    C::Annotation{<:AbstractMatrix}) # Allow Duplicated/MixedDuplicated/Const
    if needs_primal(cfg)
        # Compute only when Enzyme says it needs the primal
        s = tr(C.val)               # ComplexF64 でOK
        return AugmentedReturn(s, nothing, nothing)
    else
        # Current crash is here: must return Nothing
        return AugmentedReturn(nothing, nothing, nothing)
    end
end

function reverse(cfg::RevConfig,
    ::Const{typeof(tr2)},
    ds::Active, _tape,
    C::Annotation{<:AbstractMatrix})
    @info ">>> tr reverse rule ENTERED" ds = ds.val typeofC = typeof(C.val)
    # Here we only confirm it was called; replace with real implementation later
    n, _ = size(C.val)
    for i in 1:n
        C.dval[i, i] += ds.val
    end
    return (nothing,)
end

@inline function kernel_tr_pullback_4D(i, dA, ::Val{NC1}, dindexer, ::Val{nw}, dsval) where {NC1,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for ic = 1:NC1
        dA[ic, ic, indices...] += dsval
    end
    return nothing
end

function reverse(cfg::RevConfig,
    ::Const{typeof(tr2)},
    ds::Active, _tape,
    C::Annotation{T}) where {T<:LatticeMatrix}
    #@info ">>> tr reverse rule ENTERED" ds = ds.val typeofC = typeof(C.val)
    #@info typeof(C.dval)

    dstruct = C.dval isa Base.RefValue ? C.dval[] : C.dval
    NC = Val(C.val.NC1)
    nw = Val(C.val.nw)

    JACC.parallel_for(
        prod(C.val.PN), kernel_tr_pullback_4D, dstruct.A, NC, C.val.indexer, nw, ds.val
    )
    return (nothing,)
end




#=
function augmented_primal(::RevConfig,
    ::Const{typeof(tr2)},
    ::Type{<:Active},
    C::Duplicated)
    s = LinearAlgebra.tr(C.val)
    @info "[AP] tr called" typeofC = typeof(C.val)
    return AugmentedReturn(s, nothing, nothing)
end

function reverse(::RevConfig,
    ::Const{typeof(tr2)},
    ds::Active, _tape,
    C::Duplicated)
    @info "[REV] tr called" ds = ds.val typeofC = typeof(C.val)
    return (nothing,)
end
=#

function loss7(U, temp)
    dim = 4
    #mul!(C, A, A)
    U1 = U[1]
    U2 = U[2]
    C = temp[1]
    D = temp[2]
    shift_1 = ntuple(i -> ifelse(i == 1, 1, 0), dim)
    shift_2 = ntuple(i -> ifelse(i == 2, 1, 0), dim)
    U2_p = Shifted_Lattice(U2, shift_1)
    U1_p = Shifted_Lattice(U1, shift_2)
    mul!(C, U1, U2_p)
    mul!(D, C, U1_p')
    mul!(C, D, U2')
    return realtrace(C)
end


function test_N(NC, dim)
    NX = 16
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    myrank = MPI.Comm_rank(MPI.COMM_WORLD)
    gsize = ntuple(_ -> NX, dim)
    #gsize = (NX, NY)
    nw = 1
    NG = NC

    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    if length(ARGS) == 0
        n1 = nprocs ÷ 2
        if n1 == 0
            n1 = 1
        end
        PEs = ntuple(i -> ifelse(i == 1, n1, ifelse(i == 2, nprocs ÷ n1, 1)), dim)
        #PEs = (n1, nprocs ÷ n1, 1, 1)
    else
        PEs = Tuple(parse.(Int64, ARGS))
    end
    PEs = PEs[1:dim]
    M1 = LatticeMatrix(NC, NG, dim, gsize, PEs; nw)
    comm = M1.cart

    A1 = rand(ComplexF64, NC, NG, gsize...)

    A2 = rand(ComplexF64, NC, NG, gsize...)
    M2 = LatticeMatrix(A2, dim, PEs; nw)
    dM2 = similar(M2)

    A3 = rand(ComplexF64, NC, NG, gsize...)
    M3 = LatticeMatrix(A3, dim, PEs; nw)

    dM3 = similar(M3)
    f(M) = real(tr(M))

    # Use a real scalar loss (e.g., real ∘ tr)
    #loss(C) = real(tr2(C))

    #display(methods(Enzyme.EnzymeRules.reverse))

    dM3 = similar(M3)  # Gradient buffer: zero A, metadata follows C

    #M3 = rand(ComplexF64, 4, 4)
    #dM3 = zero(M3)
    loss(C) = realtrace(C)  # Key: make it a real scalar
    shift = ntuple(i -> ifelse(i == 1, -1, 0), dim)


    function loss2(A, B, shift)
        C = similar(A)
        B_p = Shifted_Lattice(B, shift)

        mul!(C, A, B_p)
        return realtrace(C)
    end
    function loss3(A)
        C = similar(A)
        mul!(C, A, A)
        return realtrace(C)
    end
    function loss4(A, C, D)
        #mul!(C, A, A)
        traceless_antihermitian!(D, A)
        expt!(C, D, 0.3)
        #mul!(C, D, D)
        return realtrace(C)
    end

    function loss5(A, C)
        #mul!(C, A, A)
        #traceless_antihermitian!(D, A)
        #expt!(C, D, 0.3)
        mul!(C, A, A')
        return realtrace(C)
    end


    function loss6(U1, U2, C, D)
        #mul!(C, A, A)
        shift_1 = ntuple(i -> ifelse(i == 1, 1, 0), dim)
        shift_2 = ntuple(i -> ifelse(i == 2, 1, 0), dim)
        U2_p = Shifted_Lattice(U2, shift_1)
        U1_p = Shifted_Lattice(U1, shift_2)
        mul!(C, U1, U2_p)
        mul!(D, C, U1_p')
        mul!(C, D, U2')
        return realtrace(C)
    end



    #return

    #Enzyme.autodiff(Reverse, loss2, Duplicated(M3, dM3), Duplicated(M2, dM2), Const(shift))
    #Enzyme.autodiff(Reverse, loss3, Duplicated(M3, dM3))
    C = similar(M3)
    D = similar(M3)

    println(loss(M3))
    println(loss2(M3, M2, shift))
    println(loss3(M3))
    println(loss4(M3, C, D))
    println(loss5(M3, C))
    println(loss6(M3, M2, C, D))
    # @code_llvm loss(M3)

    traceless_antihermitian!(M3, M2)

    dC = similar(M3)
    dD = similar(M3)

    #Enzyme.autodiff(Reverse, loss4, Duplicated(M3, dM3), DuplicatedNoNeed(C, dC), DuplicatedNoNeed(D, dD))
    #Enzyme.autodiff(Reverse, loss5, Duplicated(M3, dM3), DuplicatedNoNeed(C, dC))
    #Enzyme.autodiff(Reverse, loss6, Duplicated(M3, dM3), Duplicated(M2, dM2), DuplicatedNoNeed(C, dC), DuplicatedNoNeed(D, dD))
    Ms = (M3, M2)
    dMs = (dM3, dM2)
    temp = (C, D)
    dtemp = (dC, dD)
    Ms = [M3, M2]
    dMs = [dM3, dM2]
    temp = [C, D]
    dtemp = [dC, dD]

    #=
    Enzyme.autodiff(Reverse, loss6,
        Active,
        Duplicated(M3, dM3),
        Duplicated(M2, dM2),
        DuplicatedNoNeed(C, dC),
        DuplicatedNoNeed(D, dD)
    )
        =#




    #return

    Enzyme.autodiff(Reverse, Const(loss7), Active,
        Duplicated(Ms, dMs), DuplicatedNoNeed(temp, dtemp))



    indices = (2, 2, 2, 2)
    #gradA, gradB = numerical_differenciation(loss2, indices, M3, M2, shift)
    #gradA = numerical_differentiation(loss3, indices, M3)
    #gradA = numerical_differentiation(loss4, indices, M3, C, D)
    #gradA = numerical_differentiation(loss5, indices, M3, C)
    gradA, gradB = numerical_differentiation(loss6, indices, M3, M2, C, D)



    println("=== AD gradA vs numerical gradA ===")
    println("auto diff gradA:")
    display(dM3.A[:, :, indices...])
    println("numerical gradA:")
    display(gradA)
    println("===  ===")

    println("=== AD gradB vs numerical gradB ===")
    println("auto diff gradB:")
    display(dM2.A[:, :, indices...])
    println("numerical gradB:")
    display(gradB)


    #substitute!(dM2, dM3)
    #Wiltinger!(dM3)




    #Enzyme.autodiff(Reverse, loss, Duplicated(M3, dM3))
    #display(dM3)
    # dC.A holds complex "real/imag" gradients
    # If you want Wirtinger gradients, convert at the end
    println("d")
    display(dM2.A[:, :, 2, 2, 2, 2])
    GX, GY = real.(dM2.A), imag.(dM2.A)              # ∂L/∂Re(A), ∂L/∂Im(A)
    ∂L_∂A = Complex.(0.5 .* GX, -0.5 .* GY)  # (∂X - i∂Y)/2
    ∂L_∂Aconj = Complex.(0.5 .* GX, 0.5 .* GY)  # (∂X + i∂Y)/2
    println("Wiltinger")
    display(∂L_∂A[:, :, 2, 2, 2, 2])
    #display(dM2.A[:, :, 2, 2, 2, 2])
    #display(transpose(M2.A[:, :, 2, 2, 2, 2]))
    clear_matrix!.(dMs)

    Wiltinger_derivative!(loss7, Ms, dMs; temp, dtemp)
    display(dM2.A[:, :, 2, 2, 2, 2])

    return
    return


    a = Duplicated(M3, dM3)
    println(typeof(a))
    return
    println(real(tr(M3)))
    Enzyme.autodiff(Reverse, f, Active, Duplicated(M3, dM3))


end

function main()
    MPI.Init()

    for NC = 2:4
        println("NC = $NC")
        test_N(NC, 4)
    end

    MPI.Finalize()
end
main()
