import Enzyme: autodiff, Duplicated, Const
import LatticeMatrices: kernel_add_4D_shift!, Adjoint_Lattice, get_shift,
    kernel_Dmatrix_mul_AshiftB!, kernel_Dmatrix_mul_AshiftBdag!, kernel_clear_4D!
import Enzyme.EnzymeRules: forward, augmented_primal, reverse, FwdConfig, RevConfigWidth,
    Active, needs_primal, needs_shadow, AugmentedReturn,
    overwritten

import Enzyme.EnzymeRules: augmented_primal, reverse, RevConfig
using MPI



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

function Enzyme.EnzymeRules.augmented_primal(cfg::RevConfig,
    ::Const{typeof(realtrace)},
    ::Type{<:Const},
    C::Annotation{T}) where {T<:LatticeMatrix}
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

# Reverse rule for realtrace: dC gets identity on the diagonal.
function Enzyme.EnzymeRules.reverse(cfg::RevConfig,
    ::Const{typeof(realtrace)},
    ds::Active, _tape,
    C::Annotation{T}) where {T<:LatticeMatrix}

    dstruct = _getshadow(C.dval)
    dstruct isa LatticeMatrix || return (nothing,)
    NC = Val(C.val.NC1)
    nw = Val(C.val.nw)

    JACC.parallel_for(
        prod(C.val.PN), kernel_tr_pullback_4D, dstruct.A, NC, C.val.indexer, nw, ds.val
    )
    return (nothing,)
end

function Enzyme.EnzymeRules.reverse(cfg::RevConfig,
    ::Const{typeof(realtrace)},
    ::Type{<:Const}, _tape,
    C::Annotation{T}) where {T<:LatticeMatrix}
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

@inline function kernel_Dmatrix_mulAdag_dC_gather!(i, dB, A, dC,
    ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift) where {NC1,NC2,NC3,nw}

    indices = delinearize(dindexer, i, nw)
    # 読む側を indices - shift にする（gather）
    indices_m = shiftindices(indices, ntuple(j -> -shift[j], length(shift)))

    @inbounds for jc = 1:NC2
        for kc = 1:NC3
            acc = zero(eltype(dB))
            for ic = 1:NC1
                acc += conj(A[ic, kc, indices_m...]) * dC[ic, jc, indices_m...]
            end
            dB[kc, jc, indices...] += acc   # 書き込みは indices
        end
    end
end

@inline function kernel_Dmatrix_mulAdagBadd_shift!(i, dB, A, dC, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)
    @inbounds for jc = 1:NC2
        for kc = 1:NC3
            acc = zero(eltype(dB))
            for ic = 1:NC1
                acc += conj(A[ic, kc, indices...]) * dC[ic, jc, indices...]
            end
            dB[kc, jc, indices_p...] += acc
        end
    end
end

@inline function kernel_Dmatrix_mul_dC_A_conj_add_shift!(i, dB, dC, A, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)
    @inbounds for jc = 1:NC2
        for kc = 1:NC3
            acc = zero(eltype(dB))
            for ic = 1:NC1
                acc += conj(dC[ic, jc, indices...]) * A[ic, kc, indices...]
            end
            dB[jc, kc, indices_p...] += acc
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

@inline function kernel_Dmatrix_mulABdagadd_shift!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)
    @inbounds for jc = 1:NC2
        for kc = 1:NC3
            b = conj(B[kc, jc, indices_p...])
            for ic = 1:NC1
                C[ic, jc, indices...] += A[ic, kc, indices...] * b
            end
        end
    end
end

@inline function kernel_Dmatrix_mul_dC_Bdag_shift!(i, dA, dC, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)
    @inbounds for ic = 1:NC1
        for kc = 1:NC3
            acc = zero(eltype(dA))
            for jc = 1:NC2
                acc += dC[ic, jc, indices...] * conj(B[kc, jc, indices_p...])
            end
            dA[ic, kc, indices...] += acc
        end
    end
end

@inline function kernel_Dmatrix_mulABadd!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC2
        for kc = 1:NC3
            b = B[kc, jc, indices...]
            for ic = 1:NC1
                C[ic, jc, indices...] += A[ic, kc, indices...] * b
            end
        end
    end
end

@inline function kernel_Dmatrix_mulABadd_shift!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)
    @inbounds for jc = 1:NC2
        for kc = 1:NC3
            b = B[kc, jc, indices_p...]
            for ic = 1:NC1
                C[ic, jc, indices...] += A[ic, kc, indices...] * b
            end
        end
    end
end

@inline function kernel_Dmatrix_mul_dC_A_conj_add!(i, dB, dC, A, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC2
        for kc = 1:NC3
            acc = zero(eltype(dB))
            for ic = 1:NC1
                acc += conj(dC[ic, jc, indices...]) * A[ic, kc, indices...]
            end
            dB[jc, kc, indices...] += acc
        end
    end
end

@inline function kernel_Dmatrix_mul_Bt_dC_conj_add!(i, dA, B, dC, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for kc = 1:NC3
        for ic = 1:NC1
            acc = zero(eltype(dA))
            for jc = 1:NC2
                acc += conj(dC[ic, jc, indices...]) * B[jc, kc, indices...]
            end
            dA[kc, ic, indices...] += conj(acc)
        end
    end
end

@inline function kernel_Dmatrix_mul_At_dC_conj_add!(i, dB, A, dC, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC2
        for kc = 1:NC3
            acc = zero(eltype(dB))
            for ic = 1:NC1
                acc += conj(dC[ic, jc, indices...]) * A[kc, ic, indices...]
            end
            dB[jc, kc, indices...] += conj(acc)
        end
    end
end




_getshadow(x) = nothing
_getshadow(x::Base.RefValue) = (x[] isa Type ? nothing : x[])
_getshadow(x::LatticeMatrix) = x
_getshadow(::Nothing) = nothing
_getshadow(::Type) = nothing

_getshadow_data(x) = nothing
_getshadow_data(x::Base.RefValue) = _getshadow_data(x[])
_getshadow_data(x::LatticeMatrix) = x
_getshadow_data(x::Shifted_Lattice) = x.data
_getshadow_data(x::Adjoint_Lattice) = _getshadow_data(x.data)

function Enzyme.EnzymeRules.overwritten(::Const{typeof(LinearAlgebra.mul!)},
    C::Annotation{<:LatticeMatrix},
    A::Annotation,
    B::Annotation)
    return (true, false, false)
end

@inline function _getshadow_out(dCout, C::Annotation{<:LatticeMatrix})
    if dCout isa Active
        return _getshadow(dCout.val)
    elseif dCout isa Base.RefValue
        return dCout[]
    elseif dCout === nothing
        return _getshadow(C.dval)
    else
        return dCout
    end
end

@inline function _ad_debug_enabled()
    return get(ENV, "LM_AD_DEBUG", "") == "1"
end

@inline function _debug_mul_context(tag, dCout, C, A, B, dA_struct, dB_struct)
    _ad_debug_enabled() || return nothing
    println(tag, " dCout=", typeof(dCout),
        " C.dval=", typeof(C.dval),
        " A=", typeof(A.val),
        " B=", typeof(B.val),
        " dA=", typeof(dA_struct),
        " dB=", typeof(dB_struct))
    return nothing
end

@inline function _should_zero_dC(dCout)
    return !(dCout === nothing || dCout isa Type)
end

@inline function _zero_shadow!(C::LatticeMatrix)
    JACC.parallel_for(
        prod(C.PN), kernel_clear_4D!, C.A, C.indexer, Val(C.NC1), Val(C.NC2), Val(C.nw)
    )
    return nothing
end


# Reverse rule for mul!(C, A, B) with LatticeMatrix inputs.
function Enzyme.EnzymeRules.reverse(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    dCout, _tape,
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:LatticeMatrix},
    B::Annotation{<:LatticeMatrix})

    #println("entered mul! reverse for LatticeMatrix")
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = _getshadow(A.dval)
    dB_struct = _getshadow(B.dval)
    dAval = (dA_struct === nothing) ? nothing : dA_struct.A
    dBval = (dB_struct === nothing) ? nothing : dB_struct.A

    _debug_mul_context("mul! LatticeMatrix", dCout, C, A, B, dA_struct, dB_struct)

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
    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end

function Enzyme.EnzymeRules.augmented_primal(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    ::Type{<:Duplicated},
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:LatticeMatrix},
    B::Annotation{<:Shifted_Lattice})
    LinearAlgebra.mul!(C.val, A.val, B.val)
    return AugmentedReturn(C.val, C.dval, nothing)
end

function Enzyme.EnzymeRules.augmented_primal(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    ::Type{<:DuplicatedNoNeed},
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:LatticeMatrix},
    B::Annotation{<:Shifted_Lattice})
    LinearAlgebra.mul!(C.val, A.val, B.val)
    return AugmentedReturn(nothing, C.dval, nothing)
end

function Enzyme.EnzymeRules.augmented_primal(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    ::Type{<:Const},
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:LatticeMatrix},
    B::Annotation{<:Shifted_Lattice})
    LinearAlgebra.mul!(C.val, A.val, B.val)
    return AugmentedReturn(nothing, nothing, nothing)
end

function Enzyme.EnzymeRules.augmented_primal(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    ::Type{<:Duplicated},
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:LatticeMatrix},
    B::Annotation{<:Adjoint_Lattice{<:Shifted_Lattice}})
    LinearAlgebra.mul!(C.val, A.val, B.val)
    return AugmentedReturn(C.val, C.dval, nothing)
end

function Enzyme.EnzymeRules.augmented_primal(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    ::Type{<:DuplicatedNoNeed},
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:LatticeMatrix},
    B::Annotation{<:Adjoint_Lattice{<:Shifted_Lattice}})
    LinearAlgebra.mul!(C.val, A.val, B.val)
    return AugmentedReturn(nothing, C.dval, nothing)
end

function Enzyme.EnzymeRules.augmented_primal(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    ::Type{<:Const},
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:LatticeMatrix},
    B::Annotation{<:Adjoint_Lattice{<:Shifted_Lattice}})
    LinearAlgebra.mul!(C.val, A.val, B.val)
    return AugmentedReturn(nothing, nothing, nothing)
end

function Enzyme.EnzymeRules.augmented_primal(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    ::Type{<:Duplicated},
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:Adjoint_Lattice{<:LatticeMatrix}},
    B::Annotation{<:LatticeMatrix})
    LinearAlgebra.mul!(C.val, A.val, B.val)
    return AugmentedReturn(C.val, C.dval, nothing)
end

function Enzyme.EnzymeRules.augmented_primal(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    ::Type{<:DuplicatedNoNeed},
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:Adjoint_Lattice{<:LatticeMatrix}},
    B::Annotation{<:LatticeMatrix})
    LinearAlgebra.mul!(C.val, A.val, B.val)
    return AugmentedReturn(nothing, C.dval, nothing)
end

function Enzyme.EnzymeRules.augmented_primal(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    ::Type{<:Const},
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:Adjoint_Lattice{<:LatticeMatrix}},
    B::Annotation{<:LatticeMatrix})
    LinearAlgebra.mul!(C.val, A.val, B.val)
    return AugmentedReturn(nothing, nothing, nothing)
end

function Enzyme.EnzymeRules.augmented_primal(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    ::Type{<:Duplicated},
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:LatticeMatrix},
    B::Annotation{<:Adjoint_Lattice{<:LatticeMatrix}})
    LinearAlgebra.mul!(C.val, A.val, B.val)
    return AugmentedReturn(C.val, C.dval, nothing)
end

function Enzyme.EnzymeRules.augmented_primal(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    ::Type{<:DuplicatedNoNeed},
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:LatticeMatrix},
    B::Annotation{<:Adjoint_Lattice{<:LatticeMatrix}})
    LinearAlgebra.mul!(C.val, A.val, B.val)
    return AugmentedReturn(nothing, C.dval, nothing)
end

function Enzyme.EnzymeRules.augmented_primal(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    ::Type{<:Const},
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:LatticeMatrix},
    B::Annotation{<:Adjoint_Lattice{<:LatticeMatrix}})
    LinearAlgebra.mul!(C.val, A.val, B.val)
    return AugmentedReturn(nothing, nothing, nothing)
end

function Enzyme.EnzymeRules.augmented_primal(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    ::Type{<:Duplicated},
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:Adjoint_Lattice{<:LatticeMatrix}},
    B::Annotation{<:Adjoint_Lattice{<:LatticeMatrix}})
    LinearAlgebra.mul!(C.val, A.val, B.val)
    return AugmentedReturn(C.val, C.dval, nothing)
end

function Enzyme.EnzymeRules.augmented_primal(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    ::Type{<:DuplicatedNoNeed},
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:Adjoint_Lattice{<:LatticeMatrix}},
    B::Annotation{<:Adjoint_Lattice{<:LatticeMatrix}})
    LinearAlgebra.mul!(C.val, A.val, B.val)
    return AugmentedReturn(nothing, C.dval, nothing)
end

function Enzyme.EnzymeRules.augmented_primal(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    ::Type{<:Const},
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:Adjoint_Lattice{<:LatticeMatrix}},
    B::Annotation{<:Adjoint_Lattice{<:LatticeMatrix}})
    LinearAlgebra.mul!(C.val, A.val, B.val)
    return AugmentedReturn(nothing, nothing, nothing)
end

#=
# Reverse rule for mul!(C, A, Shifted_Lattice).
function Enzyme.EnzymeRules.reverse(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    dCout, _tape,
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:LatticeMatrix},
    B::Annotation{<:Shifted_Lattice})

    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)

    dA_struct = _getshadow(A.dval)
    dB_data = _getshadow_data(B.dval)

    Bdata = B.val.data
    shift = get_shift(B.val)
    shiftp = ntuple(i -> -shift[i], length(shift))

    println("entered mul! reverse for Shifted_Lattice")
    if dA_struct !== nothing
        JACC.parallel_for(
            prod(C.val.PN), kernel_Dmatrix_mul_dC_Bdag_shift!, dA_struct.A, dC_struct.A, Bdata.A,
            Val(C.val.NC1), Val(C.val.NC2), Val(A.val.NC2), Val(C.val.nw), C.val.indexer, shift
        )
    end

    if dB_data !== nothing
        JACC.parallel_for(
            prod(C.val.PN), kernel_Dmatrix_mulAdag_dC_gather!, dB_data.A, A.val.A, dC_struct.A,
            Val(C.val.NC1), Val(C.val.NC2), Val(A.val.NC2), Val(C.val.nw), C.val.indexer, shift
        )
        fold_halo_grad!(dB_data)
        #=
                JACC.parallel_for(
                    prod(C.val.PN), kernel_Dmatrix_mulAdagBadd_shift!, dB_data.A, A.val.A, dC_struct.A,
                    Val(C.val.NC1), Val(C.val.NC2), Val(A.val.NC2), Val(C.val.nw), C.val.indexer, shiftp
                )
                fold_halo_grad!(dB_data)
                =#
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end
=#

# =========================
# Gather kernels
# =========================

# dA[x] += dC[x] * Bdag[x+shift]
@inline function kernel_Dmatrix_mul_dC_Bdag_gather!(i, dA, dC, Bdata,
    ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift) where {NC1,NC2,NC3,nw}

    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)  # = x + shift (forward と同じ)

    @inbounds for jc = 1:NC3            # NC3 = A.NC3 (= B.NC1)
        for ic = 1:NC1
            acc = zero(eltype(dA))
            for kc = 1:NC2              # NC2 = C.NC2 (= B.NC2)
                # (dC * B†)_{ic,jc} = Σ_k dC_{ic,k} * conj(B_{jc,k})
                acc += dC[ic, kc, indices...] * conj(Bdata[jc, kc, indices_p...])
            end
            dA[ic, jc, indices...] += acc
        end
    end
end

# dBdata[y] += A†[y-shift] * dC[y-shift]   (gather)
@inline function kernel_Dmatrix_mul_Adag_dC_gather!(i, dBdata, A, dC,
    ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift) where {NC1,NC2,NC3,nw}

    indices = delinearize(dindexer, i, nw)
    # gather：読む側だけ y-shift
    shiftm = ntuple(j -> -shift[j], length(shift))
    indices_m = shiftindices(indices, shiftm)  # = y - shift

    @inbounds for jc = 1:NC2            # NC2 = B.NC2
        for kc = 1:NC3                  # NC3 = B.NC1 (= A.NC3)
            acc = zero(eltype(dBdata))
            for ic = 1:NC1              # NC1 = A.NC1 (= C.NC1)
                # (A† * dC)_{kc,jc} = Σ_i conj(A_{i,kc}) * dC_{i,jc}
                acc += conj(A[ic, kc, indices_m...]) * dC[ic, jc, indices_m...]
            end
            dBdata[kc, jc, indices...] += acc   # 書き込みは indices (= y)
        end
    end
end


# =========================
# Reverse rule (gather-only)
# =========================

function Enzyme.EnzymeRules.reverse(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    dCout, _tape,
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:LatticeMatrix},
    B::Annotation{<:Shifted_Lattice})

    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)

    dA_struct = _getshadow(A.dval)
    dB_data = _getshadow_data(B.dval)

    Bdata = B.val.data
    shift = get_shift(B.val)

    _debug_mul_context("mul! Shifted_Lattice", dCout, C, A, B, dA_struct, dB_data)

    #println("entered mul! reverse for Shifted_Lattice")
    if dA_struct !== nothing
        # dA += dC * Bdag(shifted)
        JACC.parallel_for(
            prod(C.val.PN), kernel_Dmatrix_mul_dC_Bdag_gather!,
            dA_struct.A, dC_struct.A, Bdata.A,
            Val(C.val.NC1), Val(C.val.NC2), Val(A.val.NC2), Val(C.val.nw),
            C.val.indexer, shift
        )
    end

    if dB_data !== nothing
        # dB_shift(x+shift) += A†(x) * dC(x); Shifted_Lattice reverse shifts back to B.
        JACC.parallel_for(
            prod(C.val.PN), kernel_Dmatrix_mulAdagBadd_shift!, dB_data.A, A.val.A, dC_struct.A,
            Val(C.val.NC1), Val(C.val.NC2), Val(A.val.NC2), Val(C.val.nw), C.val.indexer, shift
        )
        fold_halo_grad!(dB_data)
    end
    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end


# Reverse rule for mul!(C, A, Adjoint{Shifted_Lattice}).
function Enzyme.EnzymeRules.reverse(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    dCout, _tape,
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:LatticeMatrix},
    B::Annotation{<:Adjoint_Lattice{<:Shifted_Lattice}})

    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)

    dA_struct = _getshadow(A.dval)
    dB_data = _getshadow_data(B.dval)

    Bdata = B.val.data.data
    shift = get_shift(B.val)
    _debug_mul_context("mul! Adjoint{Shifted}", dCout, C, A, B, dA_struct, dB_data)
    #println("entered mul! reverse for Adjoint{Shifted_Lattice}")
    if dA_struct !== nothing
        JACC.parallel_for(
            prod(C.val.PN), kernel_Dmatrix_mul_AshiftB!, dA_struct.A, dC_struct.A, Bdata.A,
            Val(C.val.NC1), Val(C.val.NC2), Val(A.val.NC2), Val(C.val.nw), C.val.indexer, shift,
            1, 1
        )
    end

    if dB_data !== nothing
        JACC.parallel_for(
            prod(C.val.PN), kernel_Dmatrix_mul_dC_A_conj_add_shift!, dB_data.A, dC_struct.A, A.val.A,
            Val(C.val.NC1), Val(Bdata.NC1), Val(Bdata.NC2), Val(C.val.nw), C.val.indexer, shift
        )
        fold_halo_grad!(dB_data)
    end
    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end

function Enzyme.EnzymeRules.reverse(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    dCout, _tape,
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:Adjoint_Lattice{<:LatticeMatrix}},
    B::Annotation{<:LatticeMatrix})

    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)

    dA_data = _getshadow_data(A.dval)
    dB_struct = _getshadow(B.dval)

    Adata = A.val.data
    _debug_mul_context("mul! Adjoint left", dCout, C, A, B, dA_data, dB_struct)

    #println("entered mul! reverse for Adjoint_Lattice (left)")
    if dA_data !== nothing
        # dA += B * dC'
        JACC.parallel_for(
            prod(C.val.PN), kernel_Dmatrix_mulABdagadd!, dA_data.A, B.val.A, dC_struct.A,
            Val(B.val.NC1), Val(C.val.NC1), Val(C.val.NC2), Val(C.val.nw), C.val.indexer
        )
    end

    if dB_struct !== nothing
        # dB += A * dC
        JACC.parallel_for(
            prod(C.val.PN), kernel_Dmatrix_mulABadd!, dB_struct.A, Adata.A, dC_struct.A,
            Val(B.val.NC1), Val(C.val.NC2), Val(C.val.NC1), Val(C.val.nw), C.val.indexer
        )
    end
    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end

function Enzyme.EnzymeRules.reverse(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    dCout, _tape,
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:LatticeMatrix},
    B::Annotation{<:Adjoint_Lattice{<:LatticeMatrix}})

    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)

    dA_struct = _getshadow(A.dval)
    dB_data = _getshadow_data(B.dval)

    Bdata = B.val.data
    _debug_mul_context("mul! Adjoint right", dCout, C, A, B, dA_struct, dB_data)

    #println("entered mul! reverse for Adjoint_Lattice (right)")
    if dA_struct !== nothing
        # dA += dC * B
        JACC.parallel_for(
            prod(C.val.PN), kernel_Dmatrix_mulABadd!, dA_struct.A, dC_struct.A, Bdata.A,
            Val(C.val.NC1), Val(C.val.NC2), Val(A.val.NC2), Val(C.val.nw), C.val.indexer
        )
    end

    if dB_data !== nothing
        # dB += dC' * A
        JACC.parallel_for(
            prod(C.val.PN), kernel_Dmatrix_mul_dC_A_conj_add!, dB_data.A, dC_struct.A, A.val.A,
            Val(C.val.NC1), Val(C.val.NC2), Val(A.val.NC2), Val(C.val.nw), C.val.indexer
        )
    end
    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end

function Enzyme.EnzymeRules.reverse(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    dCout, _tape,
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:Adjoint_Lattice{<:LatticeMatrix}},
    B::Annotation{<:Adjoint_Lattice{<:LatticeMatrix}})

    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)

    dA_data = _getshadow_data(A.dval)
    dB_data = _getshadow_data(B.dval)

    Adata = A.val.data
    Bdata = B.val.data
    _debug_mul_context("mul! Adjoint both", dCout, C, A, B, dA_data, dB_data)

    #println("entered mul! reverse for Adjoint_Lattice (both)")
    if dA_data !== nothing
        # dA += B' * dC'
        JACC.parallel_for(
            prod(C.val.PN), kernel_Dmatrix_mul_Bt_dC_conj_add!, dA_data.A, Bdata.A, dC_struct.A,
            Val(C.val.NC1), Val(C.val.NC2), Val(Adata.NC1), Val(C.val.nw), C.val.indexer
        )
    end

    if dB_data !== nothing
        # dB += A' * dC'
        JACC.parallel_for(
            prod(C.val.PN), kernel_Dmatrix_mul_At_dC_conj_add!, dB_data.A, Adata.A, dC_struct.A,
            Val(C.val.NC1), Val(C.val.NC2), Val(Adata.NC1), Val(C.val.nw), C.val.indexer
        )
    end
    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end



@inline function _normalize_shift(shift, D)
    if shift isa NTuple{D,Int}
        return shift
    elseif shift isa NTuple{D,<:Integer}
        return ntuple(i -> Int(shift[i]), D)
    elseif shift isa AbstractVector{<:Integer}
        @assert length(shift) == D "shift length must be $D"
        return ntuple(i -> Int(shift[i]), D)
    elseif shift isa Tuple
        @assert length(shift) == D "shift length must be $D"
        return ntuple(i -> Int(shift[i]), D)
    else
        error("Unsupported shift type: $(typeof(shift)).")
    end
end

@inline function _shift_from_mu_n(mu, n, D)
    return ntuple(i -> i == mu ? n : 0, D)
end

function _reverse_shifted_lattice!(dB::Active, A::Annotation{<:LatticeMatrix}, shift)
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

    D = length(PN)
    shiftp = ntuple(i -> -shift[i], D)

    JACC.parallel_for(Nsites, kernel_add_4D_shift!, dAval, dBval, idx, N1, N2, 1, shiftp, nwv)

    # Fold halo gradients back to the core (self-neighbor) or exchange with MPI.
    for d in 1:D
        rankM, rankP = A.val.nbr[d]
        if rankM == A.val.myrank && rankP == A.val.myrank
            JACC.parallel_for(
                Nsites, kernel_fold_halo_dim!, dAval, idx, N1, N2, nwv, PN[d], Val(d), A.val.phases[d]
            )
        else
            reverse_exchange_dim!(dAstruct, d)
        end
    end

    return (nothing, nothing, nothing)
end

# Fold halo gradients back into the core, or exchange with MPI as needed.
function fold_halo_grad!(ls::LatticeMatrix)
    PN = ls.PN
    Nsites = prod(PN)
    N1 = Val(ls.NC1)
    N2 = Val(ls.NC2)
    nwv = Val(ls.nw)
    idx = ls.indexer

    for d in 1:length(PN)
        rankM, rankP = ls.nbr[d]
        if rankM == ls.myrank && rankP == ls.myrank
            JACC.parallel_for(
                Nsites, kernel_fold_halo_dim!, ls.A, idx, N1, N2, nwv, PN[d], Val(d), ls.phases[d]
            )
        else
            reverse_exchange_dim!(ls, d)
        end
    end
    return nothing
end


# Reverse rule for Shifted_Lattice constructor with shift tuple.
function Enzyme.EnzymeRules.reverse(::RevConfig,
    ::Const{typeof(Shifted_Lattice)},
    dB::Active, _tape,
    A::Annotation{<:LatticeMatrix},
    shift_in::Const)

    D = length(A.val.PN)
    shift = _normalize_shift(shift_in.val, D)
    return _reverse_shifted_lattice!(dB, A, shift)
end

# Reverse rule for Shifted_Lattice constructor with (mu, n).
function Enzyme.EnzymeRules.reverse(::RevConfig,
    ::Const{typeof(Shifted_Lattice)},
    dB::Active, _tape,
    A::Annotation{<:LatticeMatrix},
    mu::Const, n::Const)

    D = length(A.val.PN)
    muval = Int(mu.val)
    nval = Int(n.val)
    shift = _shift_from_mu_n(muval, nval, D)
    return _reverse_shifted_lattice!(dB, A, shift)
end


@inline function _replace_index(indices, dim, newval)
    return ntuple(i -> i == dim ? newval : indices[i], length(indices))
end

@inline function kernel_fold_halo_dim!(i, dA, dindexer, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, pn_d, ::Val{d}, phase) where {NC1,NC2,nw,d}
    indices = delinearize(dindexer, i, nw)
    id = indices[d]
    phase_conj = conj(phase)

    if id <= nw
        idxg = _replace_index(indices, d, id + pn_d)
        @inbounds for jc = 1:NC2
            for ic = 1:NC1
                dA[ic, jc, indices...] += phase_conj * dA[ic, jc, idxg...]
            end
        end
    elseif id > pn_d + nw
        idxg = _replace_index(indices, d, id - pn_d)
        @inbounds for jc = 1:NC2
            for ic = 1:NC1
                dA[ic, jc, indices...] += phase_conj * dA[ic, jc, idxg...]
            end
        end
    end
end

@inline function _mul_phase_conj!(buf, phase)
    phase_conj = conj(phase)
    JACC.parallel_for(length(buf)) do i
        buf[i] *= phase_conj
    end
end

@inline function _add_buffer!(dst, src)
    JACC.parallel_for(length(src)) do i
        dst[i] += src[i]
    end
end

function reverse_exchange_dim!(ls::LatticeMatrix{D}, d::Int) where {D}
    iSM, iRM = 4d - 3, 4d - 2
    iSP, iRP = 4d - 1, 4d

    bufSM, bufRM = ls.buf[iSM], ls.buf[iRM]
    bufSP, bufRP = ls.buf[iSP], ls.buf[iRP]

    rankM, rankP = ls.nbr[d]
    me = ls.myrank
    reqs = MPI.Request[]

    if rankM != me
        copy!(bufSM, LatticeMatrices._ghostMatrix(ls.A, ls.nw, d, :minus))
        if ls.coords[d] == 0
            _mul_phase_conj!(bufSM, ls.phases[d])
        end
        push!(reqs, MPI.Isend(bufSM, rankM, d + D, ls.cart))
        push!(reqs, MPI.Irecv!(bufRM, rankM, d, ls.cart))
    end

    if rankP != me
        copy!(bufSP, LatticeMatrices._ghostMatrix(ls.A, ls.nw, d, :plus))
        if ls.coords[d] == ls.dims[d] - 1
            _mul_phase_conj!(bufSP, ls.phases[d])
        end
        push!(reqs, MPI.Isend(bufSP, rankP, d, ls.cart))
        push!(reqs, MPI.Irecv!(bufRP, rankP, d + D, ls.cart))
    end

    isempty(reqs) || MPI.Waitall!(reqs)

    if rankM != me
        faceM = LatticeMatrices._faceMatrix(ls.A, ls.nw, d, :minus)
        _add_buffer!(faceM, bufRM)
    end
    if rankP != me
        faceP = LatticeMatrices._faceMatrix(ls.A, ls.nw, d, :plus)
        _add_buffer!(faceP, bufRP)
    end
    return nothing
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
