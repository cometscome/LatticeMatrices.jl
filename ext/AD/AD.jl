import Enzyme: autodiff, Duplicated, Const
import Enzyme.EnzymeRules: forward, augmented_primal, reverse, FwdConfig, RevConfigWidth,
    Active, needs_primal, needs_shadow, AugmentedReturn,
    overwritten

import Enzyme.EnzymeRules: augmented_primal, reverse, RevConfig



#    GX, GY = real.(dM3.A), imag.(dM3.A)              # ∂L/∂Re(A), ∂L/∂Im(A)
#    ∂L_∂A = Complex.(0.5 .* GX, -0.5 .* GY)  # (∂X - i∂Y)/2
#    ∂L_∂Aconj = Complex.(0.5 .* GX, 0.5 .* GY)  # (∂X + i∂Y)/2
function kernel_Wiltinger!(i, A, dindexer, ::Val{NC1}, ::Val{NC2}, ::Val{nw}) where {NC1,NC2,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            X = real(A[ic, jc, indices...])
            Y = imag(A[ic, jc, indices...])
            #A[jc, ic, indices...] = Complex(0.5 * X, -0.5 * Y)  # ∂/∂A
            A[ic, jc, indices...] = Complex(0.5 * X, -0.5 * Y)  # ∂/∂A
        end
    end
end

function LatticeMatrices.Wiltinger!(C::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}) where {D,T,AT,NC1,NC2,nw,DI}
    JACC.parallel_for(prod(C.PN), kernel_Wiltinger!, C.A, C.indexer, Val(NC1), Val(NC2), Val(nw))
end
export Wiltinger!


Base.@noinline function LatticeMatrices.realtrace(C::T) where {T<:LatticeMatrix}
    return real(LinearAlgebra.tr(C))
end
export realtrace


function Enzyme.EnzymeRules.augmented_primal(cfg::RevConfig,
    ::Const{typeof(realtrace)},
    ::Type{<:Active},
    C::Annotation{T}) where {T<:LatticeMatrix} # Duplicated/MixedDuplicated/Const 
    if needs_primal(cfg)
        s = realtrace(C.val)
        return AugmentedReturn(s, nothing, nothing)
    else
        return AugmentedReturn(nothing, nothing, nothing)
    end
end


@inline function kernel_tr_pullback_4D(i, dA, ::Val{NC1}, dindexer, ::Val{nw}, dsval) where {NC1,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for ic = 1:NC1
        dA[ic, ic, indices...] += dsval
    end
    return nothing
end

function Enzyme.EnzymeRules.reverse(cfg::RevConfig,
    ::Const{typeof(realtrace)},
    ds::Active, _tape,
    C::Annotation{T}) where {T<:LatticeMatrix}
    #s = tr(C.val)
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

function Enzyme.EnzymeRules.augmented_primal(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    ::Type{<:Duplicated},
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:LatticeMatrix},
    B::Annotation{<:LatticeMatrix})
    LinearAlgebra.mul!(C.val, A.val, B.val)
    return AugmentedReturn(C.val, C.dval, nothing)
end

function Enzyme.EnzymeRules.augmented_primal(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    ::Type{<:DuplicatedNoNeed},
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:LatticeMatrix},
    B::Annotation{<:LatticeMatrix})
    LinearAlgebra.mul!(C.val, A.val, B.val)
    return AugmentedReturn(nothing, C.dval, nothing)
end


function Enzyme.EnzymeRules.augmented_primal(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    ::Type{<:Const},
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:LatticeMatrix},
    B::Annotation{<:LatticeMatrix})
    LinearAlgebra.mul!(C.val, A.val, B.val)
    return AugmentedReturn(nothing, nothing, nothing)
end


@inline function kernel_Dmatrix_mulAdagBadd!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC2
        for kc = 1:NC3
            b = B[kc, jc, indices...]
            for ic = 1:NC1
                C[ic, jc, indices...] += conj(A[kc, ic, indices...]) * b# B[kc, jc, indices...]
            end
        end
    end
end

@inline function kernel_Dmatrix_mulABdagadd!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC2
        for kc = 1:NC3
            b = conj(B[jc, kc, indices...])
            for ic = 1:NC1
                C[ic, jc, indices...] += A[ic, kc, indices...] * b# B[kc, jc, indices...]
            end
        end
    end
end



_getshadow(x) = nothing
_getshadow(x::Base.RefValue) = x[]
_getshadow(x::LatticeMatrix) = x
_getshadow(::Nothing) = nothing
_getshadow(::Type) = nothing

function Enzyme.EnzymeRules.reverse(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    dCout, _tape,
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:LatticeMatrix},
    B::Annotation{<:LatticeMatrix})

    dC_struct = _getshadow(C.dval)
    if dC_struct === nothing
        return (nothing, nothing, nothing)
    end
    dCval = dC_struct.A

    dA_struct = _getshadow(A.dval)
    dB_struct = _getshadow(B.dval)
    dAval = (dA_struct === nothing) ? nothing : dA_struct.A
    dBval = (dB_struct === nothing) ? nothing : dB_struct.A

    Aval, Bval = A.val.A, B.val.A

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.NC2)  # (=B.val.NC1)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)

    # ---- ① dA += dC * B'  ----
    if dAval !== nothing
        JACC.parallel_for(
            Nsites, kernel_Dmatrix_mulABdagadd!, dAval, dCval, Bval, NC1, NC2, NC3, nw, idxr
        )
    end

    # ---- ② dB += A' * dC  ----
    if dBval !== nothing
        JACC.parallel_for(
            Nsites, kernel_Dmatrix_mulAdagBadd!, dBval, Aval, dCval, NC1, NC2, NC3, nw, idxr
        )
    end
    #display(dBval[:, :, 2, 2, 2, 2])

    return (nothing, nothing, nothing)
end


function Enzyme.EnzymeRules.reverse(::RevConfig,
    ::Const{typeof(Shifted_Lattice)},
    dB::Active, _tape,
    A::Annotation{<:LatticeMatrix},
    mu::Const, n::Const)

    dAstruct = A.dval isa Base.RefValue ? A.dval[] : A.dval
    dAstruct === nothing && return (nothing, nothing, nothing)

    # Handy handles
    dAval = dAstruct.A
    dBval = dB.val.A
    N1 = Val(A.val.NC1)
    N2 = Val(A.val.NC2)
    nwv = Val(A.val.nw)
    idx = A.val.indexer
    PN = A.val.PN
    Nsites = prod(PN)

    shift = get_shift(A.val)
    shiftp = ntuple(i -> -shift[i], length(shift))


    JACC.parallel_for(Nsites, kernel_add_4D_shift!, dAval, dBval, idx, N1, N2, 1, shiftp, nw)


    return (nothing, nothing, nothing)
end




function Enzyme.EnzymeRules.augmented_primal(
    ::RevConfig,
    ::Const{typeof(traceless_antihermitian!)},
    ::Type{<:Duplicated},
    A::Annotation{<:LatticeMatrix},
    B::Annotation{<:LatticeMatrix}
)
    @info "entered tlah reverse"
    traceless_antihermitian!(A.val, B.val)
    return AugmentedReturn(nothing, A.dval, nothing)
end

# When output A is constant
function Enzyme.EnzymeRules.augmented_primal(
    ::RevConfig,
    ::Const{typeof(traceless_antihermitian!)},
    ::Type{<:Const},
    A::Annotation{<:LatticeMatrix},
    B::Annotation{<:LatticeMatrix}
)
    #@info "entered tlah reverse"
    traceless_antihermitian!(A.val, B.val)
    return AugmentedReturn(nothing, nothing, nothing)
end



function kernel_traceless_antihermitian_add!(i, vout, dA, ::Val{N}, ::Val{nw}, dindexer) where {N,nw}
    indices = delinearize(dindexer, i, nw)
    fac1N = 1 / N
    tri = 0.0
    for k = 1:N
        tri += imag(dA[k, k, indices...])
    end
    tri *= fac1N
    for k = 1:N
        vout[k, k, indices...] +=
            (imag(dA[k, k, indices...]) - tri) * im
    end


    for k1 = 1:N
        for k2 = k1+1:N
            vv =
                0.5 * (
                    dA[k1, k2, indices...] -
                    conj(dA[k2, k1, indices...])
                )
            vout[k1, k2, indices...] += vv
            vout[k2, k1, indices...] += -conj(vv)
        end
    end
end

function Enzyme.EnzymeRules.reverse(::RevConfig,
    ::Const{typeof(traceless_antihermitian!)},
    ::Type{<:Const}, _tape,         # Third arg is the return activity type (Const{Nothing})
    A::Annotation{<:LatticeMatrix},
    B::Annotation{<:LatticeMatrix})

    # Upstream accumulates in the "shadow of output A" (dAout is not passed)
    dA = A.dval
    dA = dA isa Base.RefValue ? dA[] : dA
    dA === nothing && return (nothing, nothing)

    dB = B.dval
    dB = dB isa Base.RefValue ? dB[] : dB
    dB === nothing && return (nothing, nothing)

    NC1 = Val(A.val.NC1)
    nw = Val(A.val.nw)
    idx = A.val.indexer
    Nsites = prod(A.val.PN)

    # dB += Π_ah,0(dA)
    JACC.parallel_for(Nsites, kernel_traceless_antihermitian_add!, dB.A, dA.A, NC1, nw, idx)
    return (nothing, nothing)
end
