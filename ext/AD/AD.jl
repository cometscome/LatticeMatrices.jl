import Enzyme.EnzymeRules: augmented_primal, reverse, RevConfig, AugmentedReturn, needs_primal, needs_shadow
import LatticeMatrices: add_matrix!, add_matrix_Adag!, add_matrix_shiftedA!, add_matrix_shiftedAdag!, kernel_add_4D!, kernel_add_4D_dag!, kernel_add_4D_shift!, Adjoint_Lattice, get_shift,
    kernel_Dmatrix_mul_AshiftB!, kernel_Dmatrix_mul_AshiftBdag!, kernel_clear_4D!,
    mul_ABdag!, mul_A_shiftBdag!, mul_AshiftB!, mul_shiftAshiftB!, substitute!, AbstractLattice, expt!, expt_TA!, clear_matrix!, set_halo!,
    fold_halo_dim_to_core_grad!, Staggered_Lattice, staggered_eta_halo
using PreallocatedArrays
using MPI

const ER = Enzyme.EnzymeRules

const _LM_AD_TRACE = get(ENV, "LM_AD_TRACE", "0") == "1"
@inline _adtrace(msg) = (_LM_AD_TRACE ? println("[LM-AD] " * msg) : nothing)
const _LM_TRAP_MUL_REVERSE_FALLBACK = get(ENV, "LM_TRAP_MUL_REVERSE_FALLBACK", "0") == "1"
const _LM_AD_SYNC_DEBUG = get(ENV, "LM_AD_SYNC_DEBUG", "0") == "1"
const _LM_AD_FOLD_SCATTER = get(ENV, "LM_AD_FOLD_SCATTER", "1") == "1"

@inline function _lm_ad_maybe_sync()
    _LM_AD_SYNC_DEBUG || return nothing
    try
        @eval import CUDA
        CUDA.synchronize()
    catch
    end
    return nothing
end

@inline function _lm_ad_maybe_fold_shift_scatter!(dX_struct, shift)
    _LM_AD_FOLD_SCATTER || return nothing
    dX_struct isa LatticeMatrix || return nothing
    for d in 1:length(dX_struct.PN)
        if shift[d] != 0
            fold_halo_dim_to_core_grad!(dX_struct, d)
        end
    end
    return nothing
end

const _LM_TRAP_JACC_AD = get(ENV, "LM_TRAP_JACC_AD", "0") == "1"
@static if _LM_TRAP_JACC_AD
    function ER.augmented_primal(
        cfg::ER.RevConfig,
        ::ER.Const{typeof(JACC.parallel_for)},
        ::Type{RT},
        args...,
    ) where {RT}
        error("Enzyme leaked into JACC.parallel_for (augmented_primal). argtypes=$(map(typeof, args))")
    end

    function ER.reverse(
        cfg::ER.RevConfig,
        ::ER.Const{typeof(JACC.parallel_for)},
        ::Type{RT},
        args...,
    ) where {RT}
        error("Enzyme leaked into JACC.parallel_for (reverse). argtypes=$(map(typeof, args))")
    end
end

@inline function _staggered_eta_halo0(indices, ::Val{μ}, nw) where {μ}
    μ == 0 && return 1
    return staggered_eta_halo(indices, μ, nw)
end



# Allow Shifted/Adjoint wrappers to carry mixed activity when Enzyme supports it.
@static if isdefined(Enzyme, :MixedDuplicated)
    if isdefined(Enzyme.EnzymeRules, :activity)
        import Enzyme.EnzymeRules: activity
        activity(::Type{Shifted_Lattice{D,Dim}}) where {D,Dim} = Enzyme.MixedDuplicated
        activity(::Type{Adjoint_Lattice{D}}) where {D} = Enzyme.MixedDuplicated
    end
    if isdefined(Enzyme, :activity)
        import Enzyme: activity
        activity(::Type{Shifted_Lattice{D,Dim}}) where {D,Dim} = Enzyme.MixedDuplicated
        activity(::Type{Adjoint_Lattice{D}}) where {D} = Enzyme.MixedDuplicated
    end
end

# Extract the primal lattice matrix from either an Annotation or a primal value.
@inline _primal_of(x) = x
@inline _primal_of(x::ER.Annotation) = x.val

# Extract the shadow lattice matrix from an Annotation (Duplicated/MixedDuplicated).
@inline _shadow_of(ann::ER.Annotation) = _getshadow(ann.dval)

# Return a zero cotangent for Active scalar arguments; otherwise return nothing.
@inline _zero_cotangent(::Any) = nothing
@inline _zero_cotangent(x::ER.Active{T}) where {T} = zero(T)




function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_AshiftB!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:LatticeMatrix},
    shift::RT2,
) where {RT,RT2}
    _adtrace("augmented_primal: mul_AshiftB!(C,A,B,shift)")
    #println("mul_AshiftB!: augmented_primal")
    # Forward: C = A * shift(B)
    mul_AshiftB!(C.val, A.val, B.val, shift.val)

    # Always tape A and parent(B) primals to survive workspace reuse.
    tapeA_obj, itA = get_block(A.val.temps)
    tapeA_obj .= A.val.A
    tapeA = (tapeA_obj, itA)
    #tapeA_obj = deepcopy(A.val.A)
    #tapeA = (tapeA_obj, nothing)

    tapeB_obj, itB = get_block(B.val.temps)
    tapeB_obj .= B.val.A
    tapeB = (tapeB_obj, itB)
    #tapeB_obj = deepcopy(B.val.A)
    #tapeB = (tapeB_obj, nothing)

    tape_shift = shift.val
    if get(ENV, "LM_DEBUG_MULASHIFTB", "") == "1"
        println("mul_AshiftB! augmented_primal: itA=$(tapeA[2]) itB=$(tapeB[2]) A_id=$(objectid(tapeA_obj)) B_id=$(objectid(tapeB_obj)) shift.val=$(tape_shift)")
    end

    tape = (tapeA, tapeB, tape_shift)
    RetT = ER.augmented_rule_return_type(cfg, RT, tape)
    return RetT(nothing, nothing, tape)
end




function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_AshiftB!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Duplicated{<:LatticeMatrix},
    shift::RT,
) where {RT}
    do_dB = false
    s = _getshadow(B.dval)
    do_dB = (s isa LatticeMatrix)
    return _rev_mul_AshiftB!(cfg, dCout, tape, C, A, B, shift; do_dB=do_dB)
end
#=
function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_AshiftB!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:LatticeMatrix},
    shift::RT,
) where {RT}
    return _rev_mul_AshiftB!(cfg, dCout, tape, C, A, B, shift; do_dB=false)
end
=#

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_AshiftB!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B,
    shift::RT,
) where {RT}
    # 「B に dval がある」かつ「shadow が LatticeMatrix を返す」なら dB を更新
    do_dB = false
    if hasproperty(B, :dval)
        s = _getshadow(getproperty(B, :dval))
        do_dB = (s isa LatticeMatrix)
    end
    return _rev_mul_AshiftB!(cfg, dCout, tape, C, A, B, shift; do_dB=do_dB)
end

# C = β*C + α*A*shift(B)
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_AshiftB!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:LatticeMatrix},
    shift,
    α::S1,
    β::S2,
) where {RT,S1,S2}
    _adtrace("augmented_primal: mul_AshiftB!(C,A,B,shift,alpha,beta)")
    RealRt = eltype(RT)
    shiftval = hasproperty(shift, :val) ? shift.val : shift
    αval = hasproperty(α, :val) ? α.val : α
    βval = hasproperty(β, :val) ? β.val : β

    primal_ret = mul_AshiftB!(C.val, A.val, B.val, shiftval, αval, βval)

    tapeA_obj, itA = get_block(A.val.temps)
    tapeA_obj .= A.val.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.temps)
    tapeB_obj .= B.val.A
    tapeB = (tapeB_obj, itB)

    tape = (tapeA, tapeB, shiftval, αval)
    primal = ER.needs_primal(cfg) ? convert(RealRt, primal_ret) : nothing
    shadow = ER.needs_shadow(cfg) ? convert(RealRt, nothing) : nothing
    RetT = ER.augmented_rule_return_type(cfg, RT, tape)
    return RetT(primal, shadow, tape)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_AshiftB!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Duplicated{<:LatticeMatrix},
    shift,
    α::S1,
    β::S2,
) where {S1,S2}
    dα = _zero_cotangent(α)
    dβ = _zero_cotangent(β)

    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing, nothing, dα, dβ)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    s = _getshadow(B.dval)
    dBval = (s isa LatticeMatrix) ? s.A : nothing

    tapeA, tapeB, tape_shift, tape_α = tape
    Aval = (tapeA === nothing) ? A.val.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.NC2)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)
    sh = tape_shift
    fac = conj(tape_α)

    if dAval !== nothing && Bval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mul_dA_from_dC_Bdag_shift_scaled!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr, sh, fac
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulAdagBadd_scatter_shift_scaled!,
            dBval, Aval, dCval,
            NC1, NC2, NC3, nw, idxr, sh, fac
        )
        _lm_ad_maybe_fold_shift_scatter!(s, sh)
    end

    if tapeA !== nothing
        unused!(A.val.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing, nothing, dα, dβ)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_AshiftB!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B,
    shift,
    α::S1,
    β::S2,
) where {S1,S2}
    dα = _zero_cotangent(α)
    dβ = _zero_cotangent(β)

    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing, nothing, dα, dβ)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = (hasproperty(B, :dval) ? _getshadow(B.dval) : nothing)
    dBval = (dB_struct isa LatticeMatrix) ? dB_struct.A : nothing

    tapeA, tapeB, tape_shift, tape_α = tape
    Aval = (tapeA === nothing) ? A.val.A : tapeA[1]
    Bval = if tapeB === nothing
        hasproperty(B, :val) ? B.val.A : nothing
    else
        tapeB[1]
    end

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.NC2)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)
    sh = tape_shift
    fac = conj(tape_α)

    if dAval !== nothing && Bval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mul_dA_from_dC_Bdag_shift_scaled!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr, sh, fac
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulAdagBadd_scatter_shift_scaled!,
            dBval, Aval, dCval,
            NC1, NC2, NC3, nw, idxr, sh, fac
        )
        _lm_ad_maybe_fold_shift_scatter!(dB_struct, sh)
    end

    if tapeA !== nothing
        unused!(A.val.temps, tapeA[2])
    end
    if tapeB !== nothing && hasproperty(B, :val)
        unused!(B.val.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing, nothing, dα, dβ)
end

# C = β*C + α*(A * shift(B)) where B is Shifted_Lattice wrapper.
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:Shifted_Lattice},
    α::S1,
    β::S2,
) where {RT,S1,S2}
    RealRt = eltype(RT)
    shift = get_shift(B.val)
    αval = hasproperty(α, :val) ? α.val : α
    βval = hasproperty(β, :val) ? β.val : β
    primal_ret = mul_AshiftB!(C.val, A.val, B.val.data, shift, αval, βval)

    tapeA_obj, itA = get_block(A.val.temps)
    tapeA_obj .= A.val.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.data.temps)
    tapeB_obj .= B.val.data.A
    tapeB = (tapeB_obj, itB)

    tape = (tapeA, tapeB, shift, αval)
    primal = ER.needs_primal(cfg) ? convert(RealRt, primal_ret) : nothing
    shadow = ER.needs_shadow(cfg) ? convert(RealRt, nothing) : nothing
    RetT = ER.augmented_rule_return_type(cfg, RT, tape)
    return RetT(primal, shadow, tape)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:Shifted_Lattice},
    α::S1,
    β::S2,
) where {S1,S2}
    dα = _zero_cotangent(α)
    dβ = _zero_cotangent(β)

    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing, dα, dβ)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = hasproperty(B, :dval) ? _getshadow_data(B.dval) : nothing
    dBval = (dB_struct isa LatticeMatrix) ? dB_struct.A : nothing

    tapeA, tapeB, tape_shift, tape_α = tape
    Aval = (tapeA === nothing) ? A.val.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.data.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.NC2)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)
    sh = tape_shift
    fac = conj(tape_α)

    if dAval !== nothing && Bval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mul_dA_from_dC_Bdag_shift_scaled!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr, sh, fac
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulAdagBadd_scatter_shift_scaled!,
            dBval, Aval, dCval,
            NC1, NC2, NC3, nw, idxr, sh, fac
        )
        _lm_ad_maybe_sync()
        _lm_ad_maybe_fold_shift_scatter!(dB_struct, sh)
    end

    if tapeA !== nothing
        unused!(A.val.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.data.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing, dα, dβ)
end

# C = A * shift(B) where B is a Shifted_Lattice wrapper.
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:Shifted_Lattice},
) where {RT}
    shift = get_shift(B.val)
    mul_AshiftB!(C.val, A.val, B.val.data, shift)

    tapeA_obj, itA = get_block(A.val.temps)
    tapeA_obj .= A.val.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.data.temps)
    tapeB_obj .= B.val.data.A
    tapeB = (tapeB_obj, itB)

    return ER.AugmentedReturn(nothing, nothing, (tapeA, tapeB, shift))
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:Shifted_Lattice},
)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = hasproperty(B, :dval) ? _getshadow_data(B.dval) : nothing
    dBval = (dB_struct isa LatticeMatrix) ? dB_struct.A : nothing
    if get(ENV, "LM_DEBUG_SHIFT_MUL", "") == "1"
        println("mul! reverse (Shifted_Lattice): B.val=", typeof(B.val),
            " dval=", hasproperty(B, :dval) ? typeof(B.dval) : "no dval",
            " dB_struct=", typeof(dB_struct),
            " dC max=", maximum(abs, dCval),
            " dB max(before)=", dBval === nothing ? "none" : maximum(abs, dBval))
    end

    tapeA, tapeB, tape_shift = tape
    Aval = (tapeA === nothing) ? A.val.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.data.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.NC2)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)
    sh = tape_shift

    if dAval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mul_dA_from_dC_Bdag_shift!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr, sh
        )
        _lm_ad_maybe_sync()
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulAdagBadd_scatter_shift!,
            dBval, Aval, dCval,
            NC1, NC2, NC3, nw, idxr, sh
        )
        _lm_ad_maybe_sync()
        _lm_ad_maybe_fold_shift_scatter!(dB_struct, sh)
    end
    if get(ENV, "LM_DEBUG_SHIFT_MUL", "") == "1"
        println("mul! reverse (Shifted_Lattice): dB max(after)=", dBval === nothing ? "none" : maximum(abs, dBval))
    end

    if tapeA !== nothing
        unused!(A.val.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.data.temps, tapeB[2])
    end

    if _should_zero_dC(dCout)
        _zero_shadow!(dC_struct)
    end
    return (nothing, nothing, nothing)
end

# C = A * B' where B is Adjoint_Lattice wrapper.
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:Adjoint_Lattice},
) where {RT}
    mul_ABdag!(C.val, A.val, B.val.data)

    tapeA_obj, itA = get_block(A.val.temps)
    tapeA_obj .= A.val.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.data.temps)
    tapeB_obj .= B.val.data.A
    tapeB = (tapeB_obj, itB)

    return ER.AugmentedReturn(nothing, nothing, (tapeA, tapeB))
end

function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    ::Type{RT},
    C::ER.Annotation{T},
    A::ER.Annotation{T},
    B::ER.Annotation{T},
    α::S1,
    β::S2,
) where {T<:LatticeMatrix,RT,S1,S2}
    αval = hasproperty(α, :val) ? α.val : α
    βval = hasproperty(β, :val) ? β.val : β
    primal_ret = mul!(C.val, A.val, B.val, αval, βval)

    tapeA_obj, it_tapeA = get_block(A.val.temps)
    tapeA_obj .= A.val.A
    tapeA = (tapeA_obj, it_tapeA)

    tapeB_obj, it_tapeB = get_block(B.val.temps)
    tapeB_obj .= B.val.A
    tapeB = (tapeB_obj, it_tapeB)

    tape = (tapeA, tapeB, αval)
    RetT = ER.augmented_rule_return_type(cfg, RT, tape)
    primal = ER.needs_primal(cfg) ? primal_ret : nothing
    shadow = ER.needs_shadow(cfg) ? nothing : nothing
    return RetT(primal, shadow, tape)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:Adjoint_Lattice},
)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = hasproperty(B, :dval) ? _getshadow_data(B.dval) : nothing
    dBval = (dB_struct isa LatticeMatrix) ? dB_struct.A : nothing

    tapeA, tapeB = tape
    Aval = (tapeA === nothing) ? A.val.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.data.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.NC2)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)

    if dAval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulACadd!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulCdagAadd!,
            dBval, dCval, Aval,
            NC2, NC1, NC3, nw, idxr
        )
    end

    if tapeA !== nothing
        unused!(A.val.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.data.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end

# C = β*C + α*A*B' where B is Adjoint_Lattice wrapper.
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:Adjoint_Lattice},
    α::S1,
    β::S2,
) where {RT,S1,S2}
    αval = hasproperty(α, :val) ? α.val : α
    βval = hasproperty(β, :val) ? β.val : β
    primal_ret = mul_ABdag!(C.val, A.val, B.val.data, αval, βval)

    tapeA_obj, itA = get_block(A.val.temps)
    tapeA_obj .= A.val.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.data.temps)
    tapeB_obj .= B.val.data.A
    tapeB = (tapeB_obj, itB)

    tape = (tapeA, tapeB, αval)
    RetT = ER.augmented_rule_return_type(cfg, RT, tape)
    primal = ER.needs_primal(cfg) ? primal_ret : nothing
    shadow = ER.needs_shadow(cfg) ? nothing : nothing
    return RetT(primal, shadow, tape)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:Adjoint_Lattice},
    α::S1,
    β::S2,
) where {S1,S2}
    dα = _zero_cotangent(α)
    dβ = _zero_cotangent(β)

    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing, dα, dβ)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = hasproperty(B, :dval) ? _getshadow_data(B.dval) : nothing
    dBval = (dB_struct isa LatticeMatrix) ? dB_struct.A : nothing

    tapeA, tapeB, tape_α = tape
    Aval = (tapeA === nothing) ? A.val.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.data.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.NC2)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)
    fac = conj(tape_α)

    if dAval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulACadd_scaled!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr, fac
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulCdagAadd_scaled!,
            dBval, dCval, Aval,
            NC2, NC1, NC3, nw, idxr, fac
        )
    end

    if tapeA !== nothing
        unused!(A.val.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.data.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing, dα, dβ)
end

# C = A * (shifted B)'
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:Adjoint_Lattice{<:Shifted_Lattice}},
) where {RT}
    shift = get_shift(B.val)
    mul_A_shiftBdag!(C.val, A.val, B.val.data.data, shift)

    tapeA_obj, itA = get_block(A.val.temps)
    tapeA_obj .= A.val.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.data.data.temps)
    tapeB_obj .= B.val.data.data.A
    tapeB = (tapeB_obj, itB)

    return ER.AugmentedReturn(nothing, nothing, (tapeA, tapeB, shift))
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:Adjoint_Lattice{<:Shifted_Lattice}},
)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = hasproperty(B, :dval) ? _getshadow_data(B.dval) : nothing
    dBval = (dB_struct isa LatticeMatrix) ? dB_struct.A : nothing

    tapeA, tapeB, tape_shift = tape
    Aval = (tapeA === nothing) ? A.val.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.data.data.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.NC2)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)
    sh = tape_shift

    if dAval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulACadd_shift!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr, sh
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulCdagAadd_scatter_shift!,
            dBval, dCval, Aval,
            NC2, NC1, NC3, nw, idxr, sh
        )
        _lm_ad_maybe_fold_shift_scatter!(dB_struct, sh)
    end

    if tapeA !== nothing
        unused!(A.val.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.data.data.temps, tapeB[2])
    end

    if _should_zero_dC(dCout)
        _zero_shadow!(dC_struct)
    end
    return (nothing, nothing, nothing)
end

# C = β*C + α*A*(shifted B)'
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:Adjoint_Lattice{<:Shifted_Lattice}},
    α::S1,
    β::S2,
) where {RT,S1,S2}
    αval = hasproperty(α, :val) ? α.val : α
    βval = hasproperty(β, :val) ? β.val : β
    shift = get_shift(B.val)
    primal_ret = mul_A_shiftBdag!(C.val, A.val, B.val.data.data, shift, αval, βval)

    tapeA_obj, itA = get_block(A.val.temps)
    tapeA_obj .= A.val.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.data.data.temps)
    tapeB_obj .= B.val.data.data.A
    tapeB = (tapeB_obj, itB)

    tape = (tapeA, tapeB, shift, αval)
    RetT = ER.augmented_rule_return_type(cfg, RT, tape)
    primal = ER.needs_primal(cfg) ? primal_ret : nothing
    shadow = ER.needs_shadow(cfg) ? nothing : nothing
    return RetT(primal, shadow, tape)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:Adjoint_Lattice{<:Shifted_Lattice}},
    α::S1,
    β::S2,
) where {S1,S2}
    dα = _zero_cotangent(α)
    dβ = _zero_cotangent(β)

    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing, dα, dβ)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = hasproperty(B, :dval) ? _getshadow_data(B.dval) : nothing
    dBval = (dB_struct isa LatticeMatrix) ? dB_struct.A : nothing

    tapeA, tapeB, tape_shift, tape_α = tape
    Aval = (tapeA === nothing) ? A.val.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.data.data.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.NC2)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)
    sh = tape_shift
    fac = conj(tape_α)

    if dAval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulACadd_shift_scaled!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr, sh, fac
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulCdagAadd_scatter_shift_scaled!,
            dBval, dCval, Aval,
            NC2, NC1, NC3, nw, idxr, sh, fac
        )
        _lm_ad_maybe_fold_shift_scatter!(dB_struct, sh)
    end

    if tapeA !== nothing
        unused!(A.val.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.data.data.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing, dα, dβ)
end

# 共通の本体（Bのwrapper型で do_dB を切り替える）
function _rev_mul_AshiftB!(
    cfg::ER.RevConfig,
    dCout, tape,
    C, A, B, shift;
    do_dB::Bool,
)
    _adtrace("reverse-core: mul_AshiftB! do_dB=$(do_dB)")

    if get(ENV, "LM_DEBUG_MULASHIFTB", "") == "1"
        println("mul_AshiftB! reverse: A=$(typeof(A)) B=$(typeof(B)) do_dB=$(do_dB)")
        if hasproperty(A, :val)
            println("  A.val = $(typeof(getproperty(A, :val)))")
        end
        if hasproperty(A, :dval)
            dval = getproperty(A, :dval)
            println("  A.dval = $(typeof(dval)) shadow = $(typeof(_getshadow(dval)))")
        end
        if hasproperty(B, :val)
            println("  B.val = $(typeof(getproperty(B, :val)))")
        end
        if hasproperty(B, :dval)
            dval = getproperty(B, :dval)
            println("  B.dval = $(typeof(dval)) shadow = $(typeof(_getshadow(dval)))")
        end

        println("  shift.val = ", hasproperty(shift, :val) ? shift.val : shift)
    end

    # Fetch dC (output adjoint)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing, nothing)
    dCval = dC_struct.A

    # Fetch dA buffer
    dA_struct = hasproperty(A, :dval) ? _getshadow(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    # Fetch dB buffer（必要なときだけ）
    dB_struct = do_dB ? _getshadow(B.dval) : nothing
    dBval = (do_dB && (dB_struct isa LatticeMatrix)) ? dB_struct.A : nothing

    # Unpack tapes
    tapeA, tapeB, tape_shift = tape
    Aval = (tapeA === nothing) ? A.val.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.A : tapeB[1]

    # Context
    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.NC2)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)
    sh = tape_shift

    # dA += dC * (B[x+sh])†
    if dAval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mul_dA_from_dC_Bdag_shift!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr, sh
        )
    end

    # dB[x+sh] += (A[x])† * dC
    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulAdagBadd_scatter_shift!,
            dBval, Aval, dCval,
            NC1, NC2, NC3, nw, idxr, sh
        )
        _lm_ad_maybe_sync()
        _lm_ad_maybe_fold_shift_scatter!(dB_struct, sh)
    end

    # Release tape blocks（早期 return があっても必ずやりたいなら try/finally 化推奨）
    if tapeA !== nothing
        unused!(A.val.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.temps, tapeB[2])
    end

    if _should_zero_dC(dCout)
        #ow = ER.overwritten(cfg)
        #if !isempty(ow) && ow[1]
        _zero_shadow!(dC_struct)
        #end
    end
    return (nothing, nothing, nothing, nothing)
end

# C = shiftA * shiftB
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_shiftAshiftB!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:LatticeMatrix},
    shiftA::RT2,
    shiftB::RT3,
) where {RT,RT2,RT3}
    _adtrace("augmented_primal: mul_shiftAshiftB!(C,A,B,shiftA,shiftB)")
    shiftA_val = hasproperty(shiftA, :val) ? shiftA.val : shiftA
    shiftB_val = hasproperty(shiftB, :val) ? shiftB.val : shiftB
    mul_shiftAshiftB!(C.val, A.val, B.val, shiftA_val, shiftB_val)

    tapeA_obj, itA = get_block(A.val.temps)
    tapeA_obj .= A.val.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.temps)
    tapeB_obj .= B.val.A
    tapeB = (tapeB_obj, itB)

    tape = (tapeA, tapeB, shiftA_val, shiftB_val)
    RetT = ER.augmented_rule_return_type(cfg, RT, tape)
    return RetT(nothing, nothing, tape)
end

function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_shiftAshiftB!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice{<:LatticeMatrix}},
    B::ER.Annotation{<:LatticeMatrix},
    shiftA::RT2,
    shiftB::RT3,
) where {RT,RT2,RT3}
    _adtrace("augmented_primal: mul_shiftAshiftB!(C,Adjoint(A),B,shiftA,shiftB)")
    shiftA_val = hasproperty(shiftA, :val) ? shiftA.val : shiftA
    shiftB_val = hasproperty(shiftB, :val) ? shiftB.val : shiftB
    mul_shiftAshiftB!(C.val, A.val, B.val, shiftA_val, shiftB_val)

    tapeA_obj, itA = get_block(A.val.data.temps)
    tapeA_obj .= A.val.data.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.temps)
    tapeB_obj .= B.val.A
    tapeB = (tapeB_obj, itB)

    tape = (tapeA, tapeB, shiftA_val, shiftB_val)
    RetT = ER.augmented_rule_return_type(cfg, RT, tape)
    return RetT(nothing, nothing, tape)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_shiftAshiftB!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Duplicated{<:LatticeMatrix},
    shiftA::RT,
    shiftB::RTB,
) where {RT,RTB}
    do_dB = false
    s = _getshadow(B.dval)
    do_dB = (s isa LatticeMatrix)
    return _rev_mul_shiftAshiftB!(cfg, dCout, tape, C, A, B, shiftA, shiftB; do_dB=do_dB)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_shiftAshiftB!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice{<:LatticeMatrix}},
    B::ER.Duplicated{<:LatticeMatrix},
    shiftA::RT,
    shiftB::RTB,
) where {RT,RTB}
    do_dB = false
    s = _getshadow(B.dval)
    do_dB = (s isa LatticeMatrix)
    return _rev_mul_shiftAdagshiftB!(cfg, dCout, tape, C, A, B, shiftA, shiftB; do_dB=do_dB)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_shiftAshiftB!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B,
    shiftA::RT,
    shiftB::RTB,
) where {RT,RTB}
    do_dB = false
    if hasproperty(B, :dval)
        s = _getshadow(getproperty(B, :dval))
        do_dB = (s isa LatticeMatrix)
    end
    return _rev_mul_shiftAshiftB!(cfg, dCout, tape, C, A, B, shiftA, shiftB; do_dB=do_dB)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_shiftAshiftB!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice{<:LatticeMatrix}},
    B,
    shiftA::RT,
    shiftB::RTB,
) where {RT,RTB}
    do_dB = false
    if hasproperty(B, :dval)
        s = _getshadow(getproperty(B, :dval))
        do_dB = (s isa LatticeMatrix)
    end
    return _rev_mul_shiftAdagshiftB!(cfg, dCout, tape, C, A, B, shiftA, shiftB; do_dB=do_dB)
end

function _rev_mul_shiftAshiftB!(
    cfg::ER.RevConfig,
    dCout, tape,
    C, A, B, shiftA, shiftB;
    do_dB::Bool,
)
    _adtrace("reverse-core: mul_shiftAshiftB! do_dB=$(do_dB)")
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = do_dB ? _getshadow(B.dval) : nothing
    dBval = (do_dB && (dB_struct isa LatticeMatrix)) ? dB_struct.A : nothing

    tapeA, tapeB, tape_shiftA, tape_shiftB = tape
    Aval = (tapeA === nothing) ? A.val.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.NC2)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)
    shA = tape_shiftA
    shB = tape_shiftB

    if dAval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mul_dA_from_dC_Bdag_shiftAshiftB_scatter!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr, shA, shB
        )
        _lm_ad_maybe_sync()
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulAdagBadd_scatter_shiftAshiftB!,
            dBval, Aval, dCval,
            NC1, NC2, NC3, nw, idxr, shA, shB
        )
        _lm_ad_maybe_sync()
        _lm_ad_maybe_fold_shift_scatter!(dB_struct, shB)
    end

    if tapeA !== nothing
        unused!(A.val.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.temps, tapeB[2])
    end

    if _should_zero_dC(dCout)
        _zero_shadow!(dC_struct)
    end
    return (nothing, nothing, nothing, nothing, nothing)
end

function _rev_mul_shiftAdagshiftB!(
    cfg::ER.RevConfig,
    dCout, tape,
    C, A, B, shiftA, shiftB;
    do_dB::Bool,
)
    _adtrace("reverse-core: mul_shiftAdagshiftB! do_dB=$(do_dB)")
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = _getshadow_data(A.dval)
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = do_dB ? _getshadow(B.dval) : nothing
    dBval = (do_dB && (dB_struct isa LatticeMatrix)) ? dB_struct.A : nothing

    tapeA, tapeB, tape_shiftA, tape_shiftB = tape
    Aval = (tapeA === nothing) ? A.val.data.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.data.NC1)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)
    shA = tape_shiftA
    shB = tape_shiftB

    if dAval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mul_dAdag_from_dC_B_shiftAshiftB_scatter!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr, shA, shB
        )
        _lm_ad_maybe_sync()
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulAdagBadd_scatter_shiftAshiftB_adagA!,
            dBval, Aval, dCval,
            NC1, NC2, NC3, nw, idxr, shA, shB
        )
        _lm_ad_maybe_sync()
        _lm_ad_maybe_fold_shift_scatter!(dB_struct, shB)
    end

    if tapeA !== nothing
        unused!(A.val.data.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.temps, tapeB[2])
    end

    if _should_zero_dC(dCout)
        _zero_shadow!(dC_struct)
    end
    return (nothing, nothing, nothing, nothing, nothing)
end

# C = A * B'
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_ABdag!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:LatticeMatrix},
) where {RT}
    _adtrace("augmented_primal: mul_ABdag!(C,A,B)")
    mul_ABdag!(C.val, A.val, B.val)

    tapeA_obj, itA = get_block(A.val.temps)
    tapeA_obj .= A.val.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.temps)
    tapeB_obj .= B.val.A
    tapeB = (tapeB_obj, itB)

    tape = (tapeA, tapeB)
    RetT = ER.augmented_rule_return_type(cfg, RT, tape)
    return RetT(nothing, nothing, tape)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_ABdag!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Duplicated{<:LatticeMatrix},
)
    s = _getshadow(B.dval)
    do_dB = (s isa LatticeMatrix)
    return _rev_mul_ABdag!(cfg, dCout, tape, C, A, B; do_dB=do_dB)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_ABdag!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B,
)
    do_dB = false
    if hasproperty(B, :dval)
        s = _getshadow(getproperty(B, :dval))
        do_dB = (s isa LatticeMatrix)
    end
    return _rev_mul_ABdag!(cfg, dCout, tape, C, A, B; do_dB=do_dB)
end

# C = β*C + α*A*B'
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_ABdag!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:LatticeMatrix},
    α::S1,
    β::S2,
) where {RT,S1,S2}
    _adtrace("augmented_primal: mul_ABdag!(C,A,B,alpha,beta)")
    αval = hasproperty(α, :val) ? α.val : α
    βval = hasproperty(β, :val) ? β.val : β
    primal_ret = mul_ABdag!(C.val, A.val, B.val, αval, βval)

    tapeA_obj, itA = get_block(A.val.temps)
    tapeA_obj .= A.val.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.temps)
    tapeB_obj .= B.val.A
    tapeB = (tapeB_obj, itB)

    tape = (tapeA, tapeB, αval)
    RetT = ER.augmented_rule_return_type(cfg, RT, tape)
    primal = ER.needs_primal(cfg) ? primal_ret : nothing
    shadow = ER.needs_shadow(cfg) ? nothing : nothing
    return RetT(primal, shadow, tape)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_ABdag!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Duplicated{<:LatticeMatrix},
    α::S1,
    β::S2,
) where {S1,S2}
    dα = _zero_cotangent(α)
    dβ = _zero_cotangent(β)

    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing, dα, dβ)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    s = _getshadow(B.dval)
    dBval = (s isa LatticeMatrix) ? s.A : nothing

    tapeA, tapeB, tape_α = tape
    Aval = (tapeA === nothing) ? A.val.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.NC2)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)
    fac = conj(tape_α)

    if dAval !== nothing && Bval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulACadd_scaled!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr, fac
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulCdagAadd_scaled!,
            dBval, dCval, Aval,
            NC2, NC1, NC3, nw, idxr, fac
        )
    end

    if tapeA !== nothing
        unused!(A.val.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing, dα, dβ)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_ABdag!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B,
    α::S1,
    β::S2,
) where {S1,S2}
    dα = _zero_cotangent(α)
    dβ = _zero_cotangent(β)

    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing, dα, dβ)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = (hasproperty(B, :dval) ? _getshadow(B.dval) : nothing)
    dBval = (dB_struct isa LatticeMatrix) ? dB_struct.A : nothing

    tapeA, tapeB, tape_α = tape
    Aval = (tapeA === nothing) ? A.val.A : tapeA[1]
    Bval = if tapeB === nothing
        hasproperty(B, :val) ? B.val.A : nothing
    else
        tapeB[1]
    end

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.NC2)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)
    fac = conj(tape_α)

    if dAval !== nothing && Bval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulACadd_scaled!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr, fac
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulCdagAadd_scaled!,
            dBval, dCval, Aval,
            NC2, NC1, NC3, nw, idxr, fac
        )
    end

    if tapeA !== nothing
        unused!(A.val.temps, tapeA[2])
    end
    if tapeB !== nothing && hasproperty(B, :val)
        unused!(B.val.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing, dα, dβ)
end

function _rev_mul_ABdag!(
    cfg::ER.RevConfig,
    dCout, tape,
    C, A, B;
    do_dB::Bool,
)
    _adtrace("reverse-core: mul_ABdag! do_dB=$(do_dB)")
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = do_dB ? _getshadow(B.dval) : nothing
    dBval = (do_dB && (dB_struct isa LatticeMatrix)) ? dB_struct.A : nothing

    tapeA, tapeB = tape
    Aval = (tapeA === nothing) ? A.val.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.NC2)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)

    if dAval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulACadd!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr
        )
        _lm_ad_maybe_sync()
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulCdagAadd!,
            dBval, dCval, Aval,
            NC2, NC1, NC3, nw, idxr
        )
        _lm_ad_maybe_sync()
    end

    if tapeA !== nothing
        unused!(A.val.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end

# C = A * (shifted B)'
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_A_shiftBdag!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:LatticeMatrix},
    shift::RT2,
) where {RT,RT2}
    _adtrace("augmented_primal: mul_A_shiftBdag!(C,A,B,shift)")
    mul_A_shiftBdag!(C.val, A.val, B.val, shift.val)

    tapeA_obj, itA = get_block(A.val.temps)
    tapeA_obj .= A.val.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.temps)
    tapeB_obj .= B.val.A
    tapeB = (tapeB_obj, itB)

    tape_shift = shift.val
    return ER.AugmentedReturn(nothing, nothing, (tapeA, tapeB, tape_shift))
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_A_shiftBdag!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Duplicated{<:LatticeMatrix},
    shift::RT,
) where {RT}
    s = _getshadow(B.dval)
    do_dB = (s isa LatticeMatrix)
    return _rev_mul_A_shiftBdag!(cfg, dCout, tape, C, A, B, shift; do_dB=do_dB)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_A_shiftBdag!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B,
    shift::RT,
) where {RT}
    do_dB = false
    if hasproperty(B, :dval)
        s = _getshadow(getproperty(B, :dval))
        do_dB = (s isa LatticeMatrix)
    end
    return _rev_mul_A_shiftBdag!(cfg, dCout, tape, C, A, B, shift; do_dB=do_dB)
end

# C = β*C + α*A*shift(B)'
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_A_shiftBdag!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:LatticeMatrix},
    shift,
    α::S1,
    β::S2,
) where {RT,S1,S2}
    _adtrace("augmented_primal: mul_A_shiftBdag!(C,A,B,shift,alpha,beta)")
    RealRt = eltype(RT)
    shiftval = hasproperty(shift, :val) ? shift.val : shift
    αval = hasproperty(α, :val) ? α.val : α
    βval = hasproperty(β, :val) ? β.val : β
    primal_ret = mul_A_shiftBdag!(C.val, A.val, B.val, shiftval, αval, βval)

    tapeA_obj, itA = get_block(A.val.temps)
    tapeA_obj .= A.val.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.temps)
    tapeB_obj .= B.val.A
    tapeB = (tapeB_obj, itB)

    tape = (tapeA, tapeB, shiftval, αval)
    primal = ER.needs_primal(cfg) ? convert(RealRt, primal_ret) : nothing
    shadow = ER.needs_shadow(cfg) ? convert(RealRt, nothing) : nothing
    RetT = ER.augmented_rule_return_type(cfg, RT, tape)
    return RetT(primal, shadow, tape)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_A_shiftBdag!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Duplicated{<:LatticeMatrix},
    shift,
    α::S1,
    β::S2,
) where {S1,S2}
    dα = _zero_cotangent(α)
    dβ = _zero_cotangent(β)

    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing, nothing, dα, dβ)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    s = _getshadow(B.dval)
    dBval = (s isa LatticeMatrix) ? s.A : nothing

    tapeA, tapeB, tape_shift, tape_α = tape
    Aval = (tapeA === nothing) ? A.val.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.NC2)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)
    sh = tape_shift
    fac = conj(tape_α)

    if dAval !== nothing && Bval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulACadd_shift_scaled!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr, sh, fac
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulCdagAadd_scatter_shift_scaled!,
            dBval, dCval, Aval,
            NC2, NC1, NC3, nw, idxr, sh, fac
        )
        _lm_ad_maybe_fold_shift_scatter!(s, sh)
    end

    if tapeA !== nothing
        unused!(A.val.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing, nothing, dα, dβ)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_A_shiftBdag!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B,
    shift,
    α::S1,
    β::S2,
) where {S1,S2}
    dα = _zero_cotangent(α)
    dβ = _zero_cotangent(β)

    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing, nothing, dα, dβ)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = (hasproperty(B, :dval) ? _getshadow(B.dval) : nothing)
    dBval = (dB_struct isa LatticeMatrix) ? dB_struct.A : nothing

    tapeA, tapeB, tape_shift, tape_α = tape
    Aval = (tapeA === nothing) ? A.val.A : tapeA[1]
    Bval = if tapeB === nothing
        hasproperty(B, :val) ? B.val.A : nothing
    else
        tapeB[1]
    end

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.NC2)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)
    sh = tape_shift
    fac = conj(tape_α)

    if dAval !== nothing && Bval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulACadd_shift_scaled!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr, sh, fac
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulCdagAadd_scatter_shift_scaled!,
            dBval, dCval, Aval,
            NC2, NC1, NC3, nw, idxr, sh, fac
        )
        _lm_ad_maybe_fold_shift_scatter!(dB_struct, sh)
    end

    if tapeA !== nothing
        unused!(A.val.temps, tapeA[2])
    end
    if tapeB !== nothing && hasproperty(B, :val)
        unused!(B.val.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing, nothing, dα, dβ)
end

function _rev_mul_A_shiftBdag!(
    cfg::ER.RevConfig,
    dCout, tape,
    C, A, B, shift;
    do_dB::Bool,
)
    _adtrace("reverse-core: mul_A_shiftBdag! do_dB=$(do_dB)")
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = do_dB ? _getshadow(B.dval) : nothing
    dBval = (do_dB && (dB_struct isa LatticeMatrix)) ? dB_struct.A : nothing

    tapeA, tapeB, tape_shift = tape
    Aval = (tapeA === nothing) ? A.val.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.NC2)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)
    sh = tape_shift

    if dAval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulACadd_shift!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr, sh
        )
        _lm_ad_maybe_sync()
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulCdagAadd_scatter_shift!,
            dBval, dCval, Aval,
            NC2, NC1, NC3, nw, idxr, sh
        )
        _lm_ad_maybe_sync()
        _lm_ad_maybe_fold_shift_scatter!(dB_struct, sh)
    end

    if tapeA !== nothing
        unused!(A.val.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing, nothing)
end

# add_matrix_shiftedA! (C += α * shift(A))
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(add_matrix_shiftedA!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    shift,
    α::S
) where {RT,S}
    αval = hasproperty(α, :val) ? α.val : α
    shiftval = hasproperty(shift, :val) ? shift.val : shift
    add_matrix_shiftedA!(C.val, A.val, shiftval, αval)
    return ER.AugmentedReturn(nothing, nothing, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(add_matrix_shiftedA!)},
    dCout, _tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    shift,
    α::S,
) where {S}
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing, nothing)
    dCval = dC_struct.A
    if get(ENV, "LM_DEBUG_ADD_SHIFT", "") == "1"
        println("add_matrix_shiftedA! reverse: shift=", hasproperty(shift, :val) ? shift.val : shift,
            " dC max=", maximum(abs, Array(dCval)))
    end

    dA_struct = hasproperty(A, :dval) ? _getshadow(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing
    if dAval !== nothing
        αval = hasproperty(α, :val) ? α.val : α
        shiftval = hasproperty(shift, :val) ? shift.val : shift
        JACC.parallel_for(
            prod(C.val.PN),
            kernel_add_4D_shift_scatter!,
            dAval, dCval, C.val.indexer,
            Val(C.val.NC1), Val(C.val.NC2),
            conj(αval), shiftval, Val(C.val.nw)
        )
        for d in 1:length(dA_struct.PN)
            if shiftval[d] != 0
                fold_halo_dim_to_core_grad!(dA_struct, d)
            end
        end
        if get(ENV, "LM_DEBUG_ADD_SHIFT", "") == "1"
            println("add_matrix_shiftedA! reverse: dA max=", maximum(abs, Array(dAval)))
        end
    end

    return (nothing, nothing, nothing, nothing)
end

@inline function kernel_add_4D_shift_scatter!(i, u, v, dindexer, ::Val{NC1}, ::Val{NC2}, α, shift, ::Val{nw}) where {NC1,NC2,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, indices_p...] += α * v[ic, jc, indices...]
        end
    end
end

# add_matrix_shiftedAdag! (C += α * shift(A†))
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(add_matrix_shiftedAdag!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    shift,
    α::S
) where {RT,S}
    αval = hasproperty(α, :val) ? α.val : α
    shiftval = hasproperty(shift, :val) ? shift.val : shift
    add_matrix_shiftedAdag!(C.val, A.val, shiftval, αval)
    return ER.AugmentedReturn(nothing, nothing, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(add_matrix_shiftedAdag!)},
    dCout, _tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    shift,
    α::S,
) where {S}
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing
    if dAval !== nothing
        αval = hasproperty(α, :val) ? α.val : α
        shiftval = hasproperty(shift, :val) ? shift.val : shift
        JACC.parallel_for(
            prod(C.val.PN),
            kernel_add_4D_shiftdag_scatter!,
            dAval, dCval, C.val.indexer,
            Val(C.val.NC2), Val(C.val.NC1),
            conj(αval), shiftval, Val(C.val.nw)
        )
    end

    return (nothing, nothing, nothing, nothing)
end

@inline function kernel_add_4D_shiftdag_scatter!(i, u, v, dindexer, ::Val{NC2}, ::Val{NC1}, α, shift, ::Val{nw}) where {NC2,NC1,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            u[jc, ic, indices_p...] += α * v[ic, jc, indices...]'
        end
    end
end

# add_matrix_Adag! (C += α * A†)
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(add_matrix_Adag!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    α::S,
) where {RT,S}
    RealRt = eltype(RT)
    αval = hasproperty(α, :val) ? α.val : α
    primal_ret = add_matrix_Adag!(C.val, A.val, αval)
    primal = ER.needs_primal(cfg) ? convert(RealRt, primal_ret) : nothing
    shadow = ER.needs_shadow(cfg) ? convert(RealRt, nothing) : nothing
    cache = nothing::Any
    RetT = ER.augmented_rule_return_type(cfg, RT, cache)
    return RetT(primal, shadow, cache)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(add_matrix_Adag!)},
    dCout, _tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    α::S,
) where {S}
    dα = _zero_cotangent(α)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, dα)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing
    if dAval !== nothing
        αval = hasproperty(α, :val) ? α.val : α
        JACC.parallel_for(
            prod(C.val.PN),
            kernel_add_4D_dag!,
            dAval, dCval, C.val.indexer,
            Val(C.val.NC2), Val(C.val.NC1),
            conj(αval), Val(C.val.nw)
        )
    end

    return (nothing, nothing, dα)
end

# add_matrix! (C += α * A)
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(add_matrix!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    α::S,
) where {RT,S}
    RealRt = eltype(RT)
    αval = hasproperty(α, :val) ? α.val : α
    primal_ret = add_matrix!(C.val, A.val, αval)
    primal = ER.needs_primal(cfg) ? convert(RealRt, primal_ret) : nothing
    shadow = ER.needs_shadow(cfg) ? convert(RealRt, nothing) : nothing
    cache = nothing::Any
    RetT = ER.augmented_rule_return_type(cfg, RT, cache)
    return RetT(primal, shadow, cache)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(add_matrix!)},
    dCout, _tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    α::S,
) where {S}
    dα = _zero_cotangent(α)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, dα)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing
    if dAval !== nothing
        αval = hasproperty(α, :val) ? α.val : α
        JACC.parallel_for(
            prod(C.val.PN),
            kernel_add_4D!,
            dAval, dCval, C.val.indexer,
            Val(C.val.NC1), Val(C.val.NC2),
            conj(αval), Val(C.val.nw)
        )
    end

    return (nothing, nothing, dα)
end


# clear_matrix!
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(clear_matrix!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
) where {RT}
    clear_matrix!(C.val)
    return ER.AugmentedReturn(nothing, nothing, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(clear_matrix!)},
    dCout, _tape,
    C::ER.Annotation{<:LatticeMatrix},
)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing,)

    _zero_shadow!(dC_struct)
    return (nothing,)
end

# set_halo!
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(set_halo!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
) where {RT}
    set_halo!(C.val)
    return ER.AugmentedReturn(nothing, nothing, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(set_halo!)},
    dCout, _tape,
    C::ER.Annotation{<:LatticeMatrix},
)

    if get(ENV, "LM_DEBUG_SET_HALO", "") == "1"
        println("set_halo! reverse called; dCout=", typeof(dCout))
    end
    if get(ENV, "LM_DEBUG_SET_HALO", "") == "1"
        println("set_halo! reverse: C.val=", typeof(C.val),
            " dval=", hasproperty(C, :dval) ? typeof(C.dval) : "no dval",
            " shadow=", typeof(_getshadow(C.dval)))
    end

    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing,)

    dbg_idx = get(ENV, "LM_DEBUG_SET_HALO_IDX", "")
    if !isempty(dbg_idx)
        parts = split(dbg_idx, ",")
        if length(parts) == length(dC_struct.PN)
            idx = ntuple(i -> parse(Int, strip(parts[i])), length(parts))
            idx_ghost_plus = (dC_struct.PN[1] + 2 * dC_struct.nw, idx[2:end]...)
            println("set_halo! reverse dbg nbr=", dC_struct.nbr, " myrank=", dC_struct.myrank, " phases=", dC_struct.phases)
            println("set_halo! reverse dbg core(before)=", dC_struct.A[:, :, idx...])
            println("set_halo! reverse dbg ghost_plus(before)=", dC_struct.A[:, :, idx_ghost_plus...])
        end
    end

    if get(ENV, "LM_DEBUG_SET_HALO", "") == "1"
        println("set_halo! reverse dC max(before)=", maximum(abs, dC_struct.A))
    end
    for d in length(dC_struct.PN):-1:1
        fold_halo_dim_to_core_grad_phase!(dC_struct, d, C.val.phases[d])
    end
    zero_halo_region!(dC_struct)
    if get(ENV, "LM_DEBUG_SET_HALO", "") == "1"
        println("set_halo! reverse dC max(after)=", maximum(abs, dC_struct.A))
    end
    if !isempty(dbg_idx)
        parts = split(dbg_idx, ",")
        if length(parts) == length(dC_struct.PN)
            idx = ntuple(i -> parse(Int, strip(parts[i])), length(parts))
            idx_ghost_plus = (dC_struct.PN[1] + 2 * dC_struct.nw, idx[2:end]...)
            println("set_halo! reverse dbg core(after)=", dC_struct.A[:, :, idx...])
            println("set_halo! reverse dbg ghost_plus(after)=", dC_struct.A[:, :, idx_ghost_plus...])
        end
    end
    #=
        if get(ENV, "LM_DEBUG_SET_HALO", "") == "1"
            println("set_halo! reverse called; dCout=", typeof(dCout))
        end
        dC_struct = _getshadow_out(dCout, C)
        dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
        if get(ENV, "LM_DEBUG_SET_HALO", "") == "1"
            println("set_halo! reverse dC_struct=", typeof(dC_struct))
        end
        dC_struct === nothing && return (nothing,)

        fold_halo_to_core_grad!(dC_struct)
        zero_halo_region!(dC_struct)
        =#
    return (nothing,)
end

# substitute! (C = A)
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(substitute!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
) where {RT}
    substitute!(C.val, A.val)
    return ER.AugmentedReturn(nothing, nothing, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(substitute!)},
    dCout, _tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing)
    dCval = dC_struct.A

    dA_struct = _getshadow(A.dval)
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing
    if dAval !== nothing
        if get(ENV, "LM_DEBUG_SUBSTITUTE", "") == "1"
            println("substitute! reverse: dC max=", maximum(abs, dCval),
                " dA max(before)=", maximum(abs, dAval))
        end
        α = one(eltype(dCval))
        JACC.parallel_for(
            prod(C.val.PN),
            kernel_add_4D!,
            dAval, dCval, C.val.indexer,
            Val(C.val.NC1), Val(C.val.NC2),
            α, Val(C.val.nw)
        )
        if get(ENV, "LM_DEBUG_SUBSTITUTE", "") == "1"
            println("substitute! reverse: dA max(after)=", maximum(abs, dAval))
        end
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing)
end

# substitute! (C = shift(A))
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(substitute!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Shifted_Lattice},
) where {RT}
    substitute!(C.val, A.val)
    return ER.AugmentedReturn(nothing, nothing, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(substitute!)},
    dCout, _tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Shifted_Lattice},
)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing)
    dCval = dC_struct.A

    dA_struct = _getshadow_data(A.dval)
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing
    if dAval !== nothing
        shiftval = get_shift(A.val)
        α = one(eltype(dCval))
        JACC.parallel_for(
            prod(C.val.PN),
            kernel_add_4D_shift_scatter!,
            dAval, dCval, C.val.indexer,
            Val(C.val.NC1), Val(C.val.NC2),
            α, shiftval, Val(C.val.nw)
        )
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing)
end

# substitute! (C = A†)
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(substitute!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice},
) where {RT}
    substitute!(C.val, A.val)
    return ER.AugmentedReturn(nothing, nothing, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(substitute!)},
    dCout, _tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice},
)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing)
    dCval = dC_struct.A

    dA_struct = _getshadow_data(A.dval)
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing
    if dAval !== nothing
        α = one(eltype(dCval))
        JACC.parallel_for(
            prod(C.val.PN),
            kernel_add_4D_dag!,
            dAval, dCval, C.val.indexer,
            Val(C.val.NC2), Val(C.val.NC1),
            α, Val(C.val.nw)
        )
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing)
end

# substitute! (C = shift(A†))
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(substitute!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice{<:Shifted_Lattice}},
) where {RT}
    substitute!(C.val, A.val)
    return ER.AugmentedReturn(nothing, nothing, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(substitute!)},
    dCout, _tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice{<:Shifted_Lattice}},
)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing)
    dCval = dC_struct.A

    dA_struct = _getshadow_data(A.dval)
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing
    if dAval !== nothing
        shiftval = get_shift(A.val)
        α = one(eltype(dCval))
        JACC.parallel_for(
            prod(C.val.PN),
            kernel_add_4D_shiftdag_scatter!,
            dAval, dCval, C.val.indexer,
            Val(C.val.NC2), Val(C.val.NC1),
            α, shiftval, Val(C.val.nw)
        )
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing)
end



const _expt_ta_eps_q = 1e-18
const fac13 = 1 / 3
#const _LM_DISABLE_EXPT_REVERSE = get(ENV, "LM_DISABLE_EXPT_REVERSE", "0") == "1"

function _expt_ta_su3_reverse_diag_stats(Aval, dindexer, nsites::Int, nw, t)
    min_c1 = Inf
    max_c1 = -Inf
    min_abs_denom = Inf
    max_abs_arg_raw = 0.0
    clamp_count = 0
    small_c1_count = 0

    @inbounds for i = 1:nsites
        indices = delinearize(dindexer, i, nw)

        a11 = Aval[1, 1, indices...]
        a21 = Aval[2, 1, indices...]
        a31 = Aval[3, 1, indices...]
        a12 = Aval[1, 2, indices...]
        a22 = Aval[2, 2, indices...]
        a32 = Aval[3, 2, indices...]
        a13 = Aval[1, 3, indices...]
        a23 = Aval[2, 3, indices...]
        a33 = Aval[3, 3, indices...]

        tri = fac13 * (imag(a11) + imag(a22) + imag(a33))
        y11 = (imag(a11) - tri) * im
        y22 = (imag(a22) - tri) * im
        y33 = (imag(a33) - tri) * im
        x12 = a12 - conj(a21)
        x13 = a13 - conj(a31)
        x23 = a23 - conj(a32)
        y12 = 0.5 * x12
        y13 = 0.5 * x13
        y21 = -0.5 * conj(x12)
        y23 = 0.5 * x23
        y31 = -0.5 * conj(x13)
        y32 = -0.5 * conj(x23)

        q11 = -im * t * y11
        q12 = -im * t * y12
        q13 = -im * t * y13
        q21 = -im * t * y21
        q22 = -im * t * y22
        q23 = -im * t * y23
        q31 = -im * t * y31
        q32 = -im * t * y32
        q33 = -im * t * y33

        q11s = q11 * q11 + q12 * q21 + q13 * q31
        q12s = q11 * q12 + q12 * q22 + q13 * q32
        q13s = q11 * q13 + q12 * q23 + q13 * q33
        q21s = q21 * q11 + q22 * q21 + q23 * q31
        q22s = q21 * q12 + q22 * q22 + q23 * q32
        q23s = q21 * q13 + q22 * q23 + q23 * q33
        q31s = q31 * q11 + q32 * q21 + q33 * q31
        q32s = q31 * q12 + q32 * q22 + q33 * q32
        q33s = q31 * q13 + q32 * q23 + q33 * q33

        trQ2 = q11s + q22s + q33s
        c1 = real(0.5 * trQ2)
        min_c1 = min(min_c1, c1)
        max_c1 = max(max_c1, c1)

        if c1 <= _expt_ta_eps_q
            small_c1_count += 1
            continue
        end

        q11c = q11s * q11 + q12s * q21 + q13s * q31
        q22c = q21s * q12 + q22s * q22 + q23s * q32
        q33c = q31s * q13 + q32s * q23 + q33s * q33
        c0 = real((q11c + q22c + q33c) / 3)

        p = c1 / 3
        sqrtp = sqrt(p)
        denom = p * sqrtp
        min_abs_denom = min(min_abs_denom, abs(denom))

        arg_raw = c0 / 2 / denom
        max_abs_arg_raw = max(max_abs_arg_raw, abs(arg_raw))
        if abs(arg_raw) > 1
            clamp_count += 1
        end
    end

    return (
        min_c1=min_c1,
        max_c1=max_c1,
        min_abs_denom=min_abs_denom,
        max_abs_arg_raw=max_abs_arg_raw,
        clamp_count=clamp_count,
        small_c1_count=small_c1_count,
        nsites=nsites,
    )
end

const _expt_ta_su3_diag_seq = Ref(0)

@inline function _append_expt_ta_su3_diag_tsv!(filepath::AbstractString, rec)
    write_header = !isfile(filepath)
    open(filepath, "a") do io
        if write_header
            println(io, "seq\tt\tnsites\tsmall_c1\tclamp\tmin_c1\tmax_c1\tmin_abs_denom\tmax_abs_arg_raw")
        end
        println(
            io,
            string(rec.seq), '\t',
            string(rec.t), '\t',
            string(rec.nsites), '\t',
            string(rec.small_c1), '\t',
            string(rec.clamp), '\t',
            string(rec.min_c1), '\t',
            string(rec.max_c1), '\t',
            string(rec.min_abs_denom), '\t',
            string(rec.max_abs_arg_raw),
        )
    end
    return nothing
end

function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(expt!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    t::S,
) where {RT,S}
    tval = hasproperty(t, :val) ? t.val : t
    expt!(C.val, A.val, tval)

    tapeA_obj, itA = get_block(A.val.temps)
    tapeA_obj .= A.val.A
    tapeA = (tapeA_obj, itA)

    tapeC_obj, itC = get_block(C.val.temps)
    tapeC_obj .= C.val.A
    tapeC = (tapeC_obj, itC)

    return ER.AugmentedReturn(nothing, nothing, (tapeA, tapeC))
end

#=
function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(expt!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    t::S,
) where {S}


    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = _getshadow(A.dval)
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing
    dAval === nothing && return (nothing, nothing, nothing)

    tapeA = (tape === nothing) ? nothing : tape[1]
    tapeC = (tape === nothing) ? nothing : tape[2]
    Aval = (tapeA === nothing) ? A.val.A : tapeA[1]
    Cval = (tapeC === nothing) ? C.val.A : tapeC[1]

    tval = hasproperty(t, :val) ? t.val : t

    dt = nothing
    if t isa Active
        init = zero(real(zero(eltype(dCval))))
        dt_local = JACC.parallel_reduce(
            prod(C.val.PN),
            kernel_expt_TA_dt!,
            dCval, Cval, Aval,
            C.val.indexer, Val(C.val.NC1), Val(C.val.nw);
            init=init, op=+
        )
        dt = MPI.Allreduce(dt_local, MPI.SUM, C.val.comm)
    end

    if C.val.NC1 == 2 && C.val.NC2 == 2
        JACC.parallel_for(
            prod(C.val.PN),
            kernel_expt_TA_rev_su2!,
            dAval, dCval, Aval,
            C.val.indexer, Val(C.val.nw),
            tval, _expt_ta_eps_q
        )
    elseif C.val.NC1 == 3 && C.val.NC2 == 3
        if get(ENV, "LM_EXPT_TA_SU3_DIAG", "0") == "1"
            stats_local = _expt_ta_su3_reverse_diag_stats(
                Aval, C.val.indexer, prod(C.val.PN), C.val.nw, tval
            )
            comm = C.val.comm
            stats = (
                min_c1=MPI.Allreduce(stats_local.min_c1, MPI.MIN, comm),
                max_c1=MPI.Allreduce(stats_local.max_c1, MPI.MAX, comm),
                min_abs_denom=MPI.Allreduce(stats_local.min_abs_denom, MPI.MIN, comm),
                max_abs_arg_raw=MPI.Allreduce(stats_local.max_abs_arg_raw, MPI.MAX, comm),
                clamp_count=MPI.Allreduce(stats_local.clamp_count, MPI.SUM, comm),
                small_c1_count=MPI.Allreduce(stats_local.small_c1_count, MPI.SUM, comm),
                nsites=MPI.Allreduce(stats_local.nsites, MPI.SUM, comm),
            )
            rank = MPI.Comm_rank(comm)
            if rank == 0
                _expt_ta_su3_diag_seq[] += 1
                seq = _expt_ta_su3_diag_seq[]
                println(
                    "expt_TA_rev_su3 diag: " *
                    "seq=$(seq) t=$(tval) " *
                    "small_c1=$(stats.small_c1_count)/$(stats.nsites) " *
                    "clamp=$(stats.clamp_count)/$(stats.nsites) " *
                    "min_c1=$(stats.min_c1) max_c1=$(stats.max_c1) " *
                    "min_abs_denom=$(stats.min_abs_denom) " *
                    "max_abs_arg_raw=$(stats.max_abs_arg_raw)"
                )
                diag_file = get(ENV, "LM_EXPT_TA_SU3_DIAG_FILE", "")
                if !isempty(diag_file)
                    _append_expt_ta_su3_diag_tsv!(
                        diag_file,
                        (
                            seq=seq,
                            t=tval,
                            nsites=stats.nsites,
                            small_c1=stats.small_c1_count,
                            clamp=stats.clamp_count,
                            min_c1=stats.min_c1,
                            max_c1=stats.max_c1,
                            min_abs_denom=stats.min_abs_denom,
                            max_abs_arg_raw=stats.max_abs_arg_raw,
                        ),
                    )
                end
            end
        end
        JACC.parallel_for(
            prod(C.val.PN),
            kernel_expt_TA_rev_su3!,
            dAval, dCval, Aval,
            C.val.indexer, Val(C.val.nw),
            tval, _expt_ta_eps_q
        )
    else
        error("expt! reverse is only implemented for NC=2 or NC=3.")
    end

    if tapeA !== nothing
        unused!(A.val.temps, tapeA[2])
    end
    if tapeC !== nothing
        unused!(C.val.temps, tapeC[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, dt)
end
=#

function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(expt_TA!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    t::S,
) where {RT,S}
    tval = hasproperty(t, :val) ? t.val : t
    expt_TA!(C.val, A.val, tval)

    tapeA_obj, itA = get_block(A.val.temps)
    tapeA_obj .= A.val.A
    tapeA = (tapeA_obj, itA)

    tapeC_obj, itC = get_block(C.val.temps)
    tapeC_obj .= C.val.A
    tapeC = (tapeC_obj, itC)

    return ER.AugmentedReturn(nothing, nothing, (tapeA, tapeC))
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(expt_TA!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    t::S,
) where {S}
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = _getshadow(A.dval)
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing
    dAval === nothing && return (nothing, nothing, nothing)

    tapeA = (tape === nothing) ? nothing : tape[1]
    tapeC = (tape === nothing) ? nothing : tape[2]
    Aval = (tapeA === nothing) ? A.val.A : tapeA[1]
    Cval = (tapeC === nothing) ? C.val.A : tapeC[1]

    tval = hasproperty(t, :val) ? t.val : t

    dt = nothing
    if t isa Active
        init = zero(real(zero(eltype(dCval))))
        dt_local = JACC.parallel_reduce(
            prod(C.val.PN),
            kernel_expt_TA_dt!,
            dCval, Cval, Aval,
            C.val.indexer, Val(C.val.NC1), Val(C.val.nw);
            init=init, op=+
        )
        dt = MPI.Allreduce(dt_local, MPI.SUM, C.val.comm)
    end

    if C.val.NC1 == 2 && C.val.NC2 == 2
        JACC.parallel_for(
            prod(C.val.PN),
            kernel_expt_TA_rev_su2!,
            dAval, dCval, Aval,
            C.val.indexer, Val(C.val.nw),
            tval, _expt_ta_eps_q
        )
    elseif C.val.NC1 == 3 && C.val.NC2 == 3
        JACC.parallel_for(
            prod(C.val.PN),
            kernel_expt_TA_rev_su3!,
            dAval, dCval, Aval,
            C.val.indexer, Val(C.val.nw),
            tval, _expt_ta_eps_q
        )
    else
        error("expt_TA! reverse is only implemented for NC=2 or NC=3.")
    end

    if tapeA !== nothing
        unused!(A.val.temps, tapeA[2])
    end
    if tapeC !== nothing
        unused!(C.val.temps, tapeC[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, dt)
end


# dt = real(sum(conj(dC) .* (TA * C)))
@inline function kernel_expt_TA_dt!(i, dC, C, A, dindexer, ::Val{2}, ::Val{nw}) where {nw}
    indices = delinearize(dindexer, i, nw)

    a11 = A[1, 1, indices...]
    a12 = A[1, 2, indices...]
    a21 = A[2, 1, indices...]
    a22 = A[2, 2, indices...]

    tri = 0.5 * (imag(a11) + imag(a22))
    x12 = a12 - conj(a21)
    x21 = -conj(x12)

    x11 = (imag(a11) - tri) * im
    x22 = (imag(a22) - tri) * im
    x12 = 0.5 * x12
    x21 = 0.5 * x21

    c11 = C[1, 1, indices...]
    c12 = C[1, 2, indices...]
    c21 = C[2, 1, indices...]
    c22 = C[2, 2, indices...]

    y11 = x11 * c11 + x12 * c21
    y12 = x11 * c12 + x12 * c22
    y21 = x21 * c11 + x22 * c21
    y22 = x21 * c12 + x22 * c22

    dc11 = dC[1, 1, indices...]
    dc12 = dC[1, 2, indices...]
    dc21 = dC[2, 1, indices...]
    dc22 = dC[2, 2, indices...]

    acc = zero(real(zero(eltype(C))))
    acc += real(conj(dc11) * y11)
    acc += real(conj(dc12) * y12)
    acc += real(conj(dc21) * y21)
    acc += real(conj(dc22) * y22)
    return acc
end

@inline function kernel_expt_TA_dt!(i, dC, C, A, dindexer, ::Val{3}, ::Val{nw}) where {nw}
    indices = delinearize(dindexer, i, nw)

    a11 = A[1, 1, indices...]
    a21 = A[2, 1, indices...]
    a31 = A[3, 1, indices...]
    a12 = A[1, 2, indices...]
    a22 = A[2, 2, indices...]
    a32 = A[3, 2, indices...]
    a13 = A[1, 3, indices...]
    a23 = A[2, 3, indices...]
    a33 = A[3, 3, indices...]

    tri = fac13 * (imag(a11) + imag(a22) + imag(a33))

    x11 = (imag(a11) - tri) * im
    x22 = (imag(a22) - tri) * im
    x33 = (imag(a33) - tri) * im

    x12 = a12 - conj(a21)
    x13 = a13 - conj(a31)
    x23 = a23 - conj(a32)
    x21 = -conj(x12)
    x31 = -conj(x13)
    x32 = -conj(x23)

    x12 *= 0.5
    x13 *= 0.5
    x21 *= 0.5
    x23 *= 0.5
    x31 *= 0.5
    x32 *= 0.5

    c11 = C[1, 1, indices...]
    c12 = C[1, 2, indices...]
    c13 = C[1, 3, indices...]
    c21 = C[2, 1, indices...]
    c22 = C[2, 2, indices...]
    c23 = C[2, 3, indices...]
    c31 = C[3, 1, indices...]
    c32 = C[3, 2, indices...]
    c33 = C[3, 3, indices...]

    y11 = x11 * c11 + x12 * c21 + x13 * c31
    y12 = x11 * c12 + x12 * c22 + x13 * c32
    y13 = x11 * c13 + x12 * c23 + x13 * c33

    y21 = x21 * c11 + x22 * c21 + x23 * c31
    y22 = x21 * c12 + x22 * c22 + x23 * c32
    y23 = x21 * c13 + x22 * c23 + x23 * c33

    y31 = x31 * c11 + x32 * c21 + x33 * c31
    y32 = x31 * c12 + x32 * c22 + x33 * c32
    y33 = x31 * c13 + x32 * c23 + x33 * c33

    dc11 = dC[1, 1, indices...]
    dc12 = dC[1, 2, indices...]
    dc13 = dC[1, 3, indices...]
    dc21 = dC[2, 1, indices...]
    dc22 = dC[2, 2, indices...]
    dc23 = dC[2, 3, indices...]
    dc31 = dC[3, 1, indices...]
    dc32 = dC[3, 2, indices...]
    dc33 = dC[3, 3, indices...]

    acc = zero(real(zero(eltype(C))))
    acc += real(conj(dc11) * y11)
    acc += real(conj(dc12) * y12)
    acc += real(conj(dc13) * y13)
    acc += real(conj(dc21) * y21)
    acc += real(conj(dc22) * y22)
    acc += real(conj(dc23) * y23)
    acc += real(conj(dc31) * y31)
    acc += real(conj(dc32) * y32)
    acc += real(conj(dc33) * y33)
    return acc
end

# dA[ic,kc] += sum_j dC[ic,j] * conj(B[kc,j])   (with shift on B indices)
@inline function kernel_Dmatrix_mul_dA_from_dC_Bdag_shift!(
    i, dA, dC, B,
    ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift
) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)

    @inbounds for kc = 1:NC3
        for jc = 1:NC2
            b = conj(B[kc, jc, indices_p...])
            for ic = 1:NC1
                dA[ic, kc, indices...] += dC[ic, jc, indices...] * b
            end
        end
    end
    return nothing
end

@inline function kernel_Dmatrix_mul_dA_from_dC_Bdag_shift_scaled!(
    i, dA, dC, B,
    ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift, fac
) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)

    @inbounds for kc = 1:NC3
        for jc = 1:NC2
            b = fac * conj(B[kc, jc, indices_p...])
            for ic = 1:NC1
                dA[ic, kc, indices...] += dC[ic, jc, indices...] * b
            end
        end
    end
    return nothing
end

@inline function kernel_Dmatrix_mul_dA_from_dC_Bdag_shiftAshiftB_scatter!(
    i, dA, dC, B,
    ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shiftA, shiftB
) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_A = shiftindices(indices, shiftA)
    indices_B = shiftindices(indices, shiftB)

    @inbounds for kc = 1:NC3
        for jc = 1:NC2
            b = conj(B[kc, jc, indices_B...])
            for ic = 1:NC1
                dA[ic, kc, indices_A...] += dC[ic, jc, indices...] * b
            end
        end
    end
    return nothing
end

@inline function kernel_Dmatrix_mul_dAdag_from_dC_B_shiftAshiftB_scatter!(
    i, dA, dC, B,
    ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shiftA, shiftB
) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_A = shiftindices(indices, shiftA)
    indices_B = shiftindices(indices, shiftB)

    @inbounds for kc = 1:NC3
        for jc = 1:NC2
            b = B[kc, jc, indices_B...]
            for ic = 1:NC1
                dA[kc, ic, indices_A...] += conj(dC[ic, jc, indices...]) * b
            end
        end
    end
    return nothing
end

@inline function kernel_expt_TA_rev_su2!(i, dA, dC, A, dindexer, ::Val{nw}, t, eps_Q) where {nw}
    indices = delinearize(dindexer, i, nw)

    a11 = A[1, 1, indices...]
    a12 = A[1, 2, indices...]
    a21 = A[2, 1, indices...]
    a22 = A[2, 2, indices...]

    tri = 0.5 * (imag(a11) + imag(a22))
    y11 = (imag(a11) - tri) * im
    y22 = (imag(a22) - tri) * im
    x12 = a12 - conj(a21)
    y12 = 0.5 * x12
    y21 = -0.5 * conj(x12)

    q11 = t * y11
    q12 = t * y12
    q21 = t * y21
    q22 = t * y22

    trQ2 = q11 * q11 + q12 * q21 + q21 * q12 + q22 * q22

    c11 = dC[1, 1, indices...]
    c12 = dC[1, 2, indices...]
    c21 = dC[2, 1, indices...]
    c22 = dC[2, 2, indices...]

    if abs(trQ2) <= eps_Q
        dq11 = t * c11
        dq12 = t * c12
        dq21 = t * c21
        dq22 = t * c22
    else
        q = sqrt(-0.5 * trQ2)
        f = sin(q) / q
        b0 = 0.5 * f
        b1 = (sin(q) - q * cos(q)) / (2 * q^3)

        b11 = b0 + b1 * q11
        b12 = b1 * q12
        b21 = b1 * q21
        b22 = b0 + b1 * q22

        trsum = c11 * b11 + c12 * b21 + c21 * b12 + c22 * b22

        dqs11 = f * c11 + trsum * q11
        dqs12 = f * c12 + trsum * q12
        dqs21 = f * c21 + trsum * q21
        dqs22 = f * c22 + trsum * q22

        dq11 = t * dqs11
        dq12 = t * dqs12
        dq21 = t * dqs21
        dq22 = t * dqs22
    end

    tri_dq = 0.5 * (imag(dq11) + imag(dq22))
    dA[1, 1, indices...] -= (imag(dq11) - tri_dq) * im
    dA[2, 2, indices...] -= (imag(dq22) - tri_dq) * im

    vv = 0.5 * (dq12 - conj(dq21))
    dA[1, 2, indices...] -= vv
    dA[2, 1, indices...] += conj(vv)
    return nothing
end

@inline function _su3_coeffs_from_c(c0, c1)
    if c1 <= _expt_ta_eps_q
        return (one(c0), im, -0.5 + zero(c0))
    end

    p = c1 / 3
    sqrtp = sqrt(p)
    q = c0 / 2
    denom = p * sqrtp
    arg = q / denom
    arg = min(one(arg), max(-one(arg), arg))
    theta = acos(arg)

    r = 2 * sqrtp
    l1 = r * cos(theta / 3)
    l2 = r * cos((theta + 2 * pi) / 3)
    l3 = r * cos((theta + 4 * pi) / 3)

    den1 = (l1 - l2) * (l1 - l3)
    den2 = (l2 - l1) * (l2 - l3)
    den3 = (l3 - l1) * (l3 - l2)

    e1 = exp(im * l1)
    e2 = exp(im * l2)
    e3 = exp(im * l3)

    f2 = e1 / den1 + e2 / den2 + e3 / den3
    f1 = -(e1 * (l2 + l3) / den1 + e2 * (l1 + l3) / den2 + e3 * (l1 + l2) / den3)
    f0 = e1 * (l2 * l3) / den1 + e2 * (l1 * l3) / den2 + e3 * (l1 * l2) / den3

    return f0, f1, f2
end

@inline function _su3_coeffs_and_derivs(c0, c1)
    f0, f1, f2 = _su3_coeffs_from_c(c0, c1)

    dc1 = 1e-8 * (1 + abs(c1))
    dc0 = 1e-8 * (1 + abs(c0))

    f0p, f1p, f2p = _su3_coeffs_from_c(c0, c1 + dc1)
    f0m, f1m, f2m = _su3_coeffs_from_c(c0, c1 - dc1)
    b10 = (f0p - f0m) / (2 * dc1)
    b11 = (f1p - f1m) / (2 * dc1)
    b12 = (f2p - f2m) / (2 * dc1)

    f0p, f1p, f2p = _su3_coeffs_from_c(c0 + dc0, c1)
    f0m, f1m, f2m = _su3_coeffs_from_c(c0 - dc0, c1)
    b20 = (f0p - f0m) / (2 * dc0)
    b21 = (f1p - f1m) / (2 * dc0)
    b22 = (f2p - f2m) / (2 * dc0)

    return f0, f1, f2, b10, b11, b12, b20, b21, b22
end

@inline function _su3_forward_pade_fallback_from_y(
    y11, y12, y13, y21, y22, y23, y31, y32, y33, t
)
    sr3i = 1 / sqrt(3.0)
    sr3i2 = 2 / sqrt(3.0)
    pi23 = 2 * pi / 3
    tiny = 1e-100

    c1_0 = imag(y12) + imag(y21)
    c2_0 = real(y12) - real(y21)
    c3_0 = imag(y11) - imag(y22)
    c4_0 = imag(y13) + imag(y31)
    c5_0 = real(y13) - real(y31)
    c6_0 = imag(y23) + imag(y32)
    c7_0 = real(y23) - real(y32)
    c8_0 = sr3i * (imag(y11) + imag(y22) - 2 * imag(y33))

    c1 = t * c1_0 * 0.5
    c2 = t * c2_0 * 0.5
    c3 = t * c3_0 * 0.5
    c4 = t * c4_0 * 0.5
    c5 = t * c5_0 * 0.5
    c6 = t * c6_0 * 0.5
    c7 = t * c7_0 * 0.5
    c8 = t * c8_0 * 0.5
    csum = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8
    csum == 0 && return true

    v1 = c3 + sr3i * c8
    v3 = c1
    v4 = -c2
    v5 = c4
    v6 = -c5
    v9 = -c3 + sr3i * c8
    v11 = c6
    v12 = -c7
    v17 = -sr3i2 * c8

    trv3 = (v1 + v9 + v17) / 3.0
    cofac =
        v1 * v9 - v3^2 - v4^2 + v1 * v17 - v5^2 - v6^2 + v9 * v17 - v11^2 -
        v12^2
    det =
        v1 * v9 * v17 - v1 * (v11^2 + v12^2) - v9 * (v5^2 + v6^2) -
        v17 * (v3^2 + v4^2) +
        (v5 * (v3 * v11 - v4 * v12) + v6 * (v3 * v12 + v4 * v11)) * 2.0
    p3 = cofac / 3.0 - trv3^2
    q = trv3 * cofac - det - 2.0 * trv3^3
    (!(isfinite(p3) && isfinite(q)) || p3 >= -tiny) && return true

    x = sqrt(-4.0 * p3) + tiny
    denom = x * p3
    (!isfinite(denom) || abs(denom) <= tiny) && return true

    arg = q / denom
    !isfinite(arg) && return true
    arg = min(1, max(-1, arg))
    theta = acos(arg) / 3.0
    e1 = x * cos(theta) + trv3
    theta = theta + pi23
    e2 = x * cos(theta) + trv3
    e3 = 3.0 * trv3 - e1 - e2

    w1 = v5 * (v9 - e1) - v3 * v11 + v4 * v12
    w2 = -v6 * (v9 - e1) + v4 * v11 + v3 * v12
    w3 = (v1 - e1) * v11 - v3 * v5 - v4 * v6
    w4 = -(v1 - e1) * v12 - v4 * v5 + v3 * v6
    w5 = -(v1 - e1) * (v9 - e1) + v3^2 + v4^2
    n1 = w1^2 + w2^2 + w3^2 + w4^2 + w5^2
    (!(isfinite(n1) && n1 > tiny)) && return true

    w7 = v5 * (v9 - e2) - v3 * v11 + v4 * v12
    w8 = -v6 * (v9 - e2) + v4 * v11 + v3 * v12
    w9 = (v1 - e2) * v11 - v3 * v5 - v4 * v6
    w10 = -(v1 - e2) * v12 - v4 * v5 + v3 * v6
    w11 = -(v1 - e2) * (v9 - e2) + v3^2 + v4^2
    n2 = w7^2 + w8^2 + w9^2 + w10^2 + w11^2
    (!(isfinite(n2) && n2 > tiny)) && return true

    w13 = v5 * (v9 - e3) - v3 * v11 + v4 * v12
    w14 = -v6 * (v9 - e3) + v4 * v11 + v3 * v12
    w15 = (v1 - e3) * v11 - v3 * v5 - v4 * v6
    w16 = -(v1 - e3) * v12 - v4 * v5 + v3 * v6
    w17 = -(v1 - e3) * (v9 - e3) + v3^2 + v4^2
    n3 = w13^2 + w14^2 + w15^2 + w16^2 + w17^2
    (!(isfinite(n3) && n3 > tiny)) && return true

    return false
end

@inline function _exp3x3_pade_from_raw_A(
    a11, a12, a13, a21, a22, a23, a31, a32, a33, t
)
    tri = fac13 * (imag(a11) + imag(a22) + imag(a33))
    y11 = (imag(a11) - tri) * im
    y22 = (imag(a22) - tri) * im
    y33 = (imag(a33) - tri) * im
    x12 = a12 - conj(a21)
    x13 = a13 - conj(a31)
    x23 = a23 - conj(a32)
    y12 = 0.5 * x12
    y13 = 0.5 * x13
    y21 = -0.5 * conj(x12)
    y23 = 0.5 * x23
    y31 = -0.5 * conj(x13)
    y32 = -0.5 * conj(x23)

    return LatticeMatrices.exp3x3_pade(
        y11, y12, y13,
        y21, y22, y23,
        y31, y32, y33,
        t,
    )
end

@inline function _exp3x3_taylor4_from_raw_A(
    a11, a12, a13, a21, a22, a23, a31, a32, a33, t
)
    tri = fac13 * (imag(a11) + imag(a22) + imag(a33))
    y11 = (imag(a11) - tri) * im
    y22 = (imag(a22) - tri) * im
    y33 = (imag(a33) - tri) * im
    x12 = a12 - conj(a21)
    x13 = a13 - conj(a31)
    x23 = a23 - conj(a32)
    y12 = 0.5 * x12
    y13 = 0.5 * x13
    y21 = -0.5 * conj(x12)
    y23 = 0.5 * x23
    y31 = -0.5 * conj(x13)
    y32 = -0.5 * conj(x23)

    m11 = t * y11
    m12 = t * y12
    m13 = t * y13
    m21 = t * y21
    m22 = t * y22
    m23 = t * y23
    m31 = t * y31
    m32 = t * y32
    m33 = t * y33

    m211 = m11 * m11 + m12 * m21 + m13 * m31
    m212 = m11 * m12 + m12 * m22 + m13 * m32
    m213 = m11 * m13 + m12 * m23 + m13 * m33
    m221 = m21 * m11 + m22 * m21 + m23 * m31
    m222 = m21 * m12 + m22 * m22 + m23 * m32
    m223 = m21 * m13 + m22 * m23 + m23 * m33
    m231 = m31 * m11 + m32 * m21 + m33 * m31
    m232 = m31 * m12 + m32 * m22 + m33 * m32
    m233 = m31 * m13 + m32 * m23 + m33 * m33

    m311 = m211 * m11 + m212 * m21 + m213 * m31
    m312 = m211 * m12 + m212 * m22 + m213 * m32
    m313 = m211 * m13 + m212 * m23 + m213 * m33
    m321 = m221 * m11 + m222 * m21 + m223 * m31
    m322 = m221 * m12 + m222 * m22 + m223 * m32
    m323 = m221 * m13 + m222 * m23 + m223 * m33
    m331 = m231 * m11 + m232 * m21 + m233 * m31
    m332 = m231 * m12 + m232 * m22 + m233 * m32
    m333 = m231 * m13 + m232 * m23 + m233 * m33

    m411 = m311 * m11 + m312 * m21 + m313 * m31
    m412 = m311 * m12 + m312 * m22 + m313 * m32
    m413 = m311 * m13 + m312 * m23 + m313 * m33
    m421 = m321 * m11 + m322 * m21 + m323 * m31
    m422 = m321 * m12 + m322 * m22 + m323 * m32
    m423 = m321 * m13 + m322 * m23 + m323 * m33
    m431 = m331 * m11 + m332 * m21 + m333 * m31
    m432 = m331 * m12 + m332 * m22 + m333 * m32
    m433 = m331 * m13 + m332 * m23 + m333 * m33

    c2 = 0.5
    c3 = 1.0 / 6.0
    c4 = 1.0 / 24.0
    return (
        one(a11) + m11 + c2 * m211 + c3 * m311 + c4 * m411,
        m12 + c2 * m212 + c3 * m312 + c4 * m412,
        m13 + c2 * m213 + c3 * m313 + c4 * m413,
        m21 + c2 * m221 + c3 * m321 + c4 * m421,
        one(a11) + m22 + c2 * m222 + c3 * m322 + c4 * m422,
        m23 + c2 * m223 + c3 * m323 + c4 * m423,
        m31 + c2 * m231 + c3 * m331 + c4 * m431,
        m32 + c2 * m232 + c3 * m332 + c4 * m432,
        one(a11) + m33 + c2 * m233 + c3 * m333 + c4 * m433,
    )
end

@inline function _expt_ta_rev_su3_pade_fd!(
    dA, dC, indices, a11, a12, a13, a21, a22, a23, a31, a32, a33, t
)
    c11 = dC[1, 1, indices...]
    c12 = dC[1, 2, indices...]
    c13 = dC[1, 3, indices...]
    c21 = dC[2, 1, indices...]
    c22 = dC[2, 2, indices...]
    c23 = dC[2, 3, indices...]
    c31 = dC[3, 1, indices...]
    c32 = dC[3, 2, indices...]
    c33 = dC[3, 3, indices...]

    basescale = max(
        abs(a11), abs(a12), abs(a13),
        abs(a21), abs(a22), abs(a23),
        abs(a31), abs(a32), abs(a33),
        abs(t), 1.0,
    )
    epsfd = 1e-8 * basescale

    basis = (
        # diagonal generators
        (im, 0, 0, 0, -im, 0, 0, 0, 0),
        (im / sqrt(3.0), 0, 0, 0, im / sqrt(3.0), 0, 0, 0, -2im / sqrt(3.0)),
        # off-diagonal (12)
        (0, 1, 0, -1, 0, 0, 0, 0, 0),
        (0, im, 0, im, 0, 0, 0, 0, 0),
        # off-diagonal (13)
        (0, 0, 1, 0, 0, 0, -1, 0, 0),
        (0, 0, im, 0, 0, 0, im, 0, 0),
        # off-diagonal (23)
        (0, 0, 0, 0, 0, 1, 0, -1, 0),
        (0, 0, 0, 0, 0, im, 0, im, 0),
    )

    for b in basis
        bp11, bp12, bp13, bp21, bp22, bp23, bp31, bp32, bp33 = b

        fplus = _exp3x3_taylor4_from_raw_A(
            a11 + epsfd * bp11, a12 + epsfd * bp12, a13 + epsfd * bp13,
            a21 + epsfd * bp21, a22 + epsfd * bp22, a23 + epsfd * bp23,
            a31 + epsfd * bp31, a32 + epsfd * bp32, a33 + epsfd * bp33, t
        )
        fminus = _exp3x3_taylor4_from_raw_A(
            a11 - epsfd * bp11, a12 - epsfd * bp12, a13 - epsfd * bp13,
            a21 - epsfd * bp21, a22 - epsfd * bp22, a23 - epsfd * bp23,
            a31 - epsfd * bp31, a32 - epsfd * bp32, a33 - epsfd * bp33, t
        )

        df11 = (fplus[1] - fminus[1]) / (2 * epsfd)
        df12 = (fplus[2] - fminus[2]) / (2 * epsfd)
        df13 = (fplus[3] - fminus[3]) / (2 * epsfd)
        df21 = (fplus[4] - fminus[4]) / (2 * epsfd)
        df22 = (fplus[5] - fminus[5]) / (2 * epsfd)
        df23 = (fplus[6] - fminus[6]) / (2 * epsfd)
        df31 = (fplus[7] - fminus[7]) / (2 * epsfd)
        df32 = (fplus[8] - fminus[8]) / (2 * epsfd)
        df33 = (fplus[9] - fminus[9]) / (2 * epsfd)

        g = real(
            conj(c11) * df11 + conj(c12) * df12 + conj(c13) * df13 +
            conj(c21) * df21 + conj(c22) * df22 + conj(c23) * df23 +
            conj(c31) * df31 + conj(c32) * df32 + conj(c33) * df33
        )

        dA[1, 1, indices...] -= g * bp11
        dA[1, 2, indices...] -= g * bp12
        dA[1, 3, indices...] -= g * bp13
        dA[2, 1, indices...] -= g * bp21
        dA[2, 2, indices...] -= g * bp22
        dA[2, 3, indices...] -= g * bp23
        dA[3, 1, indices...] -= g * bp31
        dA[3, 2, indices...] -= g * bp32
        dA[3, 3, indices...] -= g * bp33
    end

    return nothing
end

@inline function kernel_expt_TA_rev_su3!(i, dA, dC, A, dindexer, ::Val{nw}, t, eps_Q) where {nw}
    indices = delinearize(dindexer, i, nw)

    a11 = A[1, 1, indices...]
    a21 = A[2, 1, indices...]
    a31 = A[3, 1, indices...]
    a12 = A[1, 2, indices...]
    a22 = A[2, 2, indices...]
    a32 = A[3, 2, indices...]
    a13 = A[1, 3, indices...]
    a23 = A[2, 3, indices...]
    a33 = A[3, 3, indices...]

    tri = fac13 * (imag(a11) + imag(a22) + imag(a33))

    y11 = (imag(a11) - tri) * im
    y22 = (imag(a22) - tri) * im
    y33 = (imag(a33) - tri) * im

    x12 = a12 - conj(a21)
    x13 = a13 - conj(a31)
    x23 = a23 - conj(a32)

    y12 = 0.5 * x12
    y13 = 0.5 * x13
    y21 = -0.5 * conj(x12)
    y23 = 0.5 * x23
    y31 = -0.5 * conj(x13)
    y32 = -0.5 * conj(x23)

    qnorm =
        abs(t) * sqrt(real(
            y11 * conj(y11) + y12 * conj(y12) + y13 * conj(y13) +
            y21 * conj(y21) + y22 * conj(y22) + y23 * conj(y23) +
            y31 * conj(y31) + y32 * conj(y32) + y33 * conj(y33)
        ))
    if qnorm <= 1e-6
        _expt_ta_rev_su3_pade_fd!(
            dA, dC, indices, a11, a12, a13, a21, a22, a23, a31, a32, a33, t
        )
        return nothing
    end

    q11 = -im * t * y11
    q12 = -im * t * y12
    q13 = -im * t * y13
    q21 = -im * t * y21
    q22 = -im * t * y22
    q23 = -im * t * y23
    q31 = -im * t * y31
    q32 = -im * t * y32
    q33 = -im * t * y33

    q11s = q11 * q11 + q12 * q21 + q13 * q31
    q12s = q11 * q12 + q12 * q22 + q13 * q32
    q13s = q11 * q13 + q12 * q23 + q13 * q33
    q21s = q21 * q11 + q22 * q21 + q23 * q31
    q22s = q21 * q12 + q22 * q22 + q23 * q32
    q23s = q21 * q13 + q22 * q23 + q23 * q33
    q31s = q31 * q11 + q32 * q21 + q33 * q31
    q32s = q31 * q12 + q32 * q22 + q33 * q32
    q33s = q31 * q13 + q32 * q23 + q33 * q33

    trQ2 = q11s + q22s + q33s
    c1 = 0.5 * trQ2

    q11c = q11s * q11 + q12s * q21 + q13s * q31
    q22c = q21s * q12 + q22s * q22 + q23s * q32
    q33c = q31s * q13 + q32s * q23 + q33s * q33
    trQ3 = q11c + q22c + q33c
    c0 = trQ3 / 3

    c0r = real(c0)
    c1r = real(c1)
    f0, f1, f2, b10, b11, b12, b20, b21, b22 = _su3_coeffs_and_derivs(c0r, c1r)

    b1_11 = b10 + b11 * q11 + b12 * q11s
    b1_12 = b11 * q12 + b12 * q12s
    b1_13 = b11 * q13 + b12 * q13s
    b1_21 = b11 * q21 + b12 * q21s
    b1_22 = b10 + b11 * q22 + b12 * q22s
    b1_23 = b11 * q23 + b12 * q23s
    b1_31 = b11 * q31 + b12 * q31s
    b1_32 = b11 * q32 + b12 * q32s
    b1_33 = b10 + b11 * q33 + b12 * q33s

    b2_11 = b20 + b21 * q11 + b22 * q11s
    b2_12 = b21 * q12 + b22 * q12s
    b2_13 = b21 * q13 + b22 * q13s
    b2_21 = b21 * q21 + b22 * q21s
    b2_22 = b20 + b21 * q22 + b22 * q22s
    b2_23 = b21 * q23 + b22 * q23s
    b2_31 = b21 * q31 + b22 * q31s
    b2_32 = b21 * q32 + b22 * q32s
    b2_33 = b20 + b21 * q33 + b22 * q33s

    c11 = dC[1, 1, indices...]
    c12 = dC[1, 2, indices...]
    c13 = dC[1, 3, indices...]
    c21 = dC[2, 1, indices...]
    c22 = dC[2, 2, indices...]
    c23 = dC[2, 3, indices...]
    c31 = dC[3, 1, indices...]
    c32 = dC[3, 2, indices...]
    c33 = dC[3, 3, indices...]

    trCB1 =
        c11 * b1_11 + c12 * b1_21 + c13 * b1_31 +
        c21 * b1_12 + c22 * b1_22 + c23 * b1_32 +
        c31 * b1_13 + c32 * b1_23 + c33 * b1_33

    trCB2 =
        c11 * b2_11 + c12 * b2_21 + c13 * b2_31 +
        c21 * b2_12 + c22 * b2_22 + c23 * b2_32 +
        c31 * b2_13 + c32 * b2_23 + c33 * b2_33

    qc11 = q11 * c11 + q12 * c21 + q13 * c31
    qc12 = q11 * c12 + q12 * c22 + q13 * c32
    qc13 = q11 * c13 + q12 * c23 + q13 * c33
    qc21 = q21 * c11 + q22 * c21 + q23 * c31
    qc22 = q21 * c12 + q22 * c22 + q23 * c32
    qc23 = q21 * c13 + q22 * c23 + q23 * c33
    qc31 = q31 * c11 + q32 * c21 + q33 * c31
    qc32 = q31 * c12 + q32 * c22 + q33 * c32
    qc33 = q31 * c13 + q32 * c23 + q33 * c33

    cq11 = c11 * q11 + c12 * q21 + c13 * q31
    cq12 = c11 * q12 + c12 * q22 + c13 * q32
    cq13 = c11 * q13 + c12 * q23 + c13 * q33
    cq21 = c21 * q11 + c22 * q21 + c23 * q31
    cq22 = c21 * q12 + c22 * q22 + c23 * q32
    cq23 = c21 * q13 + c22 * q23 + c23 * q33
    cq31 = c31 * q11 + c32 * q21 + c33 * q31
    cq32 = c31 * q12 + c32 * q22 + c33 * q32
    cq33 = c31 * q13 + c32 * q23 + c33 * q33

    dq11 = f1 * c11 + f2 * (qc11 + cq11) + trCB1 * q11 + trCB2 * q11s
    dq12 = f1 * c12 + f2 * (qc12 + cq12) + trCB1 * q12 + trCB2 * q12s
    dq13 = f1 * c13 + f2 * (qc13 + cq13) + trCB1 * q13 + trCB2 * q13s
    dq21 = f1 * c21 + f2 * (qc21 + cq21) + trCB1 * q21 + trCB2 * q21s
    dq22 = f1 * c22 + f2 * (qc22 + cq22) + trCB1 * q22 + trCB2 * q22s
    dq23 = f1 * c23 + f2 * (qc23 + cq23) + trCB1 * q23 + trCB2 * q23s
    dq31 = f1 * c31 + f2 * (qc31 + cq31) + trCB1 * q31 + trCB2 * q31s
    dq32 = f1 * c32 + f2 * (qc32 + cq32) + trCB1 * q32 + trCB2 * q32s
    dq33 = f1 * c33 + f2 * (qc33 + cq33) + trCB1 * q33 + trCB2 * q33s

    dq11 *= im
    dq12 *= im
    dq13 *= im
    dq21 *= im
    dq22 *= im
    dq23 *= im
    dq31 *= im
    dq32 *= im
    dq33 *= im

    tri_dq = fac13 * (imag(dq11) + imag(dq22) + imag(dq33))

    z11 = (imag(dq11) - tri_dq) * im
    z22 = (imag(dq22) - tri_dq) * im
    z33 = (imag(dq33) - tri_dq) * im

    xx12 = dq12 - conj(dq21)
    xx13 = dq13 - conj(dq31)
    xx23 = dq23 - conj(dq32)

    z12 = 0.5 * xx12
    z13 = 0.5 * xx13
    z21 = -0.5 * conj(xx12)
    z23 = 0.5 * xx23
    z31 = -0.5 * conj(xx13)
    z32 = -0.5 * conj(xx23)

    dA[1, 1, indices...] -= t * z11
    dA[1, 2, indices...] -= t * z12
    dA[1, 3, indices...] -= t * z13
    dA[2, 1, indices...] -= t * z21
    dA[2, 2, indices...] -= t * z22
    dA[2, 3, indices...] -= t * z23
    dA[3, 1, indices...] -= t * z31
    dA[3, 2, indices...] -= t * z32
    dA[3, 3, indices...] -= t * z33

    return nothing
end


Base.@noinline function LatticeMatrices.realtrace(C::T) where {T<:LatticeMatrix}
    return real(LinearAlgebra.tr(C))
end
export realtrace


# -------------------------
# augmented_primal
# -------------------------
function Enzyme.EnzymeRules.augmented_primal(cfg::RevConfig,
    ::Const{typeof(realtrace)},
    ::Type{<:Active},
    C::Annotation{T}) where {T<:LatticeMatrix}
    #println("realtrace: augmented_primal")

    #s = realtrace(C.val)
    # Enzyme の要求に合わせて返す
    primal = Enzyme.EnzymeRules.needs_primal(cfg) ? realtrace(C.val) : nothing

    # realtrace の shadow は通常不要（dret から来る）ので nothing でOK
    shadow = Enzyme.EnzymeRules.needs_shadow(cfg) ? zero(Float64) : nothing  # 迷うならこの行ごと `nothing` でも可

    return Enzyme.EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

# Reverse rule for realtrace: dC gets identity on the diagonal.
function Enzyme.EnzymeRules.reverse(cfg::RevConfig,
    ::Const{typeof(realtrace)},
    ds::Active, _tape,
    C::Annotation{T}) where {T<:LatticeMatrix}
    #println("realtrace: reverse")

    dstruct = _getshadow(C.dval)
    dstruct isa LatticeMatrix || return (nothing,)
    NC = Val(C.val.NC1)
    nw = Val(C.val.nw)

    JACC.parallel_for(
        prod(C.val.PN), kernel_tr_pullback_4D, dstruct.A, NC, C.val.indexer, nw, ds.val
    )

    return (nothing,)
end

@inline function kernel_tr_pullback_4D(i, dA, ::Val{NC1}, dindexer, ::Val{nw}, dsval) where {NC1,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for ic = 1:NC1
        dA[ic, ic, indices...] += dsval
    end
    return nothing
end

function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.dot)},
    ::Type{RT},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:LatticeMatrix},
) where {RT}
    primal = ER.needs_primal(cfg) ? LinearAlgebra.dot(A.val, B.val) : nothing
    shadow = ER.needs_shadow(cfg) ? zero(eltype(A.val.A)) : nothing
    return ER.AugmentedReturn(primal, shadow, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.dot)},
    ds::ER.Active, _tape,
    A,
    B,
)
    dsval = hasproperty(ds, :val) ? ds.val : ds
    dA_struct = hasproperty(A, :dval) ? _getshadow(getproperty(A, :dval)) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing
    dB_struct = hasproperty(B, :dval) ? _getshadow(getproperty(B, :dval)) : nothing
    dBval = (dB_struct isa LatticeMatrix) ? dB_struct.A : nothing

    if dAval !== nothing
        JACC.parallel_for(
            prod(A.val.PN),
            kernel_dot_pullback_dA!,
            dAval, B.val.A, A.val.indexer,
            Val(A.val.NC1), Val(A.val.NC2), Val(A.val.nw),
            conj(dsval)
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            prod(B.val.PN),
            kernel_dot_pullback_dB!,
            dBval, A.val.A, B.val.indexer,
            Val(B.val.NC1), Val(B.val.NC2), Val(B.val.nw),
            dsval
        )
    end

    return (nothing, nothing)
end

function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(LatticeMatrices.shift_L)},
    ::Type{RT},
    B::ER.Const{<:LatticeMatrix},
    sh::NTuple{Dim,Int},
) where {RT,Dim}
    primal = ER.needs_primal(cfg) ? LatticeMatrices.shift_L(B.val, sh) : nothing
    shadow = ER.needs_shadow(cfg) ? nothing : nothing
    return ER.AugmentedReturn(primal, shadow, nothing)
end

function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(LatticeMatrices.shift_L)},
    ::Type{RT},
    B::ER.Annotation{<:LatticeMatrix},
    sh::NTuple{Dim,Int},
) where {RT,Dim}
    if get(ENV, "LM_DEBUG_SHIFT_L", "") == "1"
        println("shift_L augmented_primal (Annotation): B.val=", typeof(B.val),
            " dval=", hasproperty(B, :dval) ? typeof(B.dval) : "no dval",
            " sh=", sh)
    end
    primal = ER.needs_primal(cfg) ? LatticeMatrices.shift_L(B.val, sh) : nothing
    shadow = if ER.needs_shadow(cfg)
        dB = _getshadow(B.dval)
        dB isa LatticeMatrix ? LatticeMatrices.shift_L(dB, sh) : nothing
    else
        nothing
    end
    return ER.AugmentedReturn(primal, shadow, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(LatticeMatrices.shift_L)},
    _dout, _tape,
    B::ER.Const{<:LatticeMatrix},
    sh::NTuple{Dim,Int},
) where {Dim}
    return (nothing, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(LatticeMatrices.shift_L)},
    _dout, _tape,
    B::ER.Annotation{<:LatticeMatrix},
    sh::NTuple{Dim,Int},
) where {Dim}
    if get(ENV, "LM_DEBUG_SHIFT_L", "") == "1"
        println("shift_L reverse (Annotation): B.val=", typeof(B.val),
            " dval=", hasproperty(B, :dval) ? typeof(B.dval) : "no dval",
            " sh=", sh)
    end
    return (nothing, nothing)
end

function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(LatticeMatrices.Shifted_Lattice)},
    ::Type{RT},
    data::ER.Const{<:LatticeMatrix},
    shift::NTuple{Dim,Int},
    ::Val{Dim},
) where {RT,Dim}
    primal = ER.needs_primal(cfg) ? LatticeMatrices.Shifted_Lattice(data.val, shift, Val(Dim)) : nothing
    return ER.AugmentedReturn(primal, nothing, nothing)
end

function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(LatticeMatrices.Shifted_Lattice)},
    ::Type{RT},
    data::ER.Annotation{<:LatticeMatrix},
    shift::NTuple{Dim,Int},
    ::Val{Dim},
) where {RT,Dim}
    if get(ENV, "LM_DEBUG_SHIFT_L", "") == "1"
        println("Shifted_Lattice augmented_primal (Annotation, Val): data.val=", typeof(data.val),
            " dval=", hasproperty(data, :dval) ? typeof(data.dval) : "no dval",
            " shift=", shift)
    end
    primal = ER.needs_primal(cfg) ? LatticeMatrices.Shifted_Lattice(data.val, shift, Val(Dim)) : nothing
    shadow = if ER.needs_shadow(cfg)
        ddata = _getshadow(data.dval)
        ddata isa LatticeMatrix ? LatticeMatrices.Shifted_Lattice(ddata, shift, Val(Dim)) : nothing
    else
        nothing
    end
    return ER.AugmentedReturn(primal, shadow, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(LatticeMatrices.Shifted_Lattice)},
    _dout, _tape,
    data::ER.Const{<:LatticeMatrix},
    shift::NTuple{Dim,Int},
    ::Val{Dim},
) where {Dim}
    return (nothing, nothing, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(LatticeMatrices.Shifted_Lattice)},
    _dout, _tape,
    data::ER.Annotation{<:LatticeMatrix},
    shift::NTuple{Dim,Int},
    ::Val{Dim},
) where {Dim}
    return (nothing, nothing, nothing)
end

function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(LatticeMatrices.Shifted_Lattice)},
    ::Type{RT},
    data::ER.Const{<:LatticeMatrix},
    shift_in,
) where {RT}
    primal = ER.needs_primal(cfg) ? LatticeMatrices.Shifted_Lattice(data.val, shift_in) : nothing
    return ER.AugmentedReturn(primal, nothing, nothing)
end

function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(LatticeMatrices.Shifted_Lattice)},
    ::Type{RT},
    data::ER.Annotation{<:LatticeMatrix},
    shift_in,
) where {RT}
    if get(ENV, "LM_DEBUG_SHIFT_L", "") == "1"
        println("Shifted_Lattice augmented_primal (Annotation): data.val=", typeof(data.val),
            " dval=", hasproperty(data, :dval) ? typeof(data.dval) : "no dval",
            " shift_in=", shift_in)
    end
    primal = ER.needs_primal(cfg) ? LatticeMatrices.Shifted_Lattice(data.val, shift_in) : nothing
    shadow = if ER.needs_shadow(cfg)
        ddata = _getshadow(data.dval)
        ddata isa LatticeMatrix ? LatticeMatrices.Shifted_Lattice(ddata, shift_in) : nothing
    else
        nothing
    end
    return ER.AugmentedReturn(primal, shadow, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(LatticeMatrices.Shifted_Lattice)},
    _dout, _tape,
    data::ER.Const{<:LatticeMatrix},
    shift_in,
)
    return (nothing, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(LatticeMatrices.Shifted_Lattice)},
    _dout, _tape,
    data::ER.Annotation{<:LatticeMatrix},
    shift_in,
)
    return (nothing, nothing)
end

@inline function kernel_dot_pullback_dA!(i, dA, B, dindexer, ::Val{NC1}, ::Val{NG}, ::Val{nw}, dsconj) where {NC1,NG,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for ialpha = 1:NG
        for ic = 1:NC1
            dA[ic, ialpha, indices...] += B[ic, ialpha, indices...] * dsconj
        end
    end
    return nothing
end

@inline function kernel_dot_pullback_dB!(i, dB, A, dindexer, ::Val{NC1}, ::Val{NG}, ::Val{nw}, dsval) where {NC1,NG,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for ialpha = 1:NG
        for ic = 1:NC1
            dB[ic, ialpha, indices...] += A[ic, ialpha, indices...] * dsval
        end
    end
    return nothing
end

#=
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(real)},
    ::Type{RT},
    x::ER.Annotation{<:Complex},
) where {RT}
    xval = _primal_of(x)
    primal = ER.needs_primal(cfg) ? real(xval) : nothing
    shadow = ER.needs_shadow(cfg) ? zero(real(one(xval))) : nothing
    return ER.AugmentedReturn(primal, shadow, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(real)},
    ds::ER.Active, _tape,
    x::ER.Annotation{<:Complex},
)
    dsval = hasproperty(ds, :val) ? ds.val : ds
    if x isa ER.Const
        return (nothing,)
    elseif x isa ER.Duplicated
        x.dval += complex(dsval)
        return (nothing,)
    else
        return (complex(dsval),)
    end
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(real)},
    ds::ER.Const, _tape,
    x::ER.Annotation{<:Complex},
)
    return (nothing,)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(real)},
    ::Type{RT},
    ds, _tape,
    x::ER.Annotation{<:Complex},
) where {RT}
    if ds isa ER.Active
        dsval = hasproperty(ds, :val) ? ds.val : ds
        if x isa ER.Const
            return (nothing,)
        elseif x isa ER.Duplicated
            x.dval += complex(dsval)
            return (nothing,)
        else
            return (complex(dsval),)
        end
    end
    return (nothing,)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(real)},
    ::Type{RT},
    _tape,
    x::ER.Annotation{<:Complex},
) where {RT}
    return (nothing,)
end
=#


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

function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    ::Type{RT},
    C::ER.Annotation{T},
    A::ER.Annotation{T},
    B::ER.Annotation{T},
) where {T<:LatticeMatrix,RT}
    _adtrace("augmented_primal: mul!(C,A,B) [LatticeMatrix]")

    # Perform the forward (primal) computation
    mul!(C.val, A.val, B.val)

    # Check which arguments may be overwritten before reverse
    # Argument order: (C, A, B)
    ow = ER.overwritten(cfg)
    #println("mul!: augmented_primal")
    #println(ow)


    tapeA_obj, it_tapeA = get_block(A.val.temps)
    tapeA_obj .= A.val.A
    tapeA = (tapeA_obj, it_tapeA)

    tapeB_obj, it_tapeB = get_block(B.val.temps)
    tapeB_obj .= B.val.A
    tapeB = (tapeB_obj, it_tapeB)


    # mul! returns nothing → primal must be nothing
    # No shadow needed; gradients flow via C.dval
    tape = (tapeA, tapeB)
    RetT = ER.augmented_rule_return_type(cfg, RT, tape)
    return RetT(nothing, nothing, tape)
end

function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:AbstractMatrix},
    B::ER.Annotation{<:LatticeMatrix},
) where {RT}
    _adtrace("augmented_primal: mul!(C,AbstractMatrix,A/B lattice)")

    mul!(C.val, A.val, B.val)

    tapeA = copy(A.val)
    tapeB_obj, it_tapeB = get_block(B.val.temps)
    tapeB_obj .= B.val.A
    tapeB = (tapeB_obj, it_tapeB)

    tape = (tapeA, tapeB)
    RetT = ER.augmented_rule_return_type(cfg, RT, tape)
    return RetT(nothing, nothing, tape)
end

function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:AbstractMatrix},
) where {RT}
    _adtrace("augmented_primal: mul!(C,LatticeMatrix,AbstractMatrix)")

    mul!(C.val, A.val, B.val)

    tapeA_obj, it_tapeA = get_block(A.val.temps)
    tapeA_obj .= A.val.A
    tapeA = (tapeA_obj, it_tapeA)

    tapeB = copy(B.val)

    return ER.AugmentedReturn(nothing, nothing, (tapeA, tapeB))
end

function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:LatticeMatrix},
    α::S1,
    β::S2,
) where {RT,S1,S2}
    _adtrace("augmented_primal: mul!(C,A,B,alpha,beta) [LatticeMatrix]")

    αval = hasproperty(α, :val) ? α.val : α
    βval = hasproperty(β, :val) ? β.val : β
    mul!(C.val, A.val, B.val, αval, βval)

    tapeA_obj, it_tapeA = get_block(A.val.temps)
    tapeA_obj .= A.val.A
    tapeA = (tapeA_obj, it_tapeA)

    tapeB_obj, it_tapeB = get_block(B.val.temps)
    tapeB_obj .= B.val.A
    tapeB = (tapeB_obj, it_tapeB)

    tape = (tapeA, tapeB, αval)
    RetT = ER.augmented_rule_return_type(cfg, RT, tape)
    return RetT(nothing, nothing, tape)
end


# Reverse rule for mul!(C, A, B) with LatticeMatrix inputs.
function Enzyme.EnzymeRules.reverse(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    dCout, _tape,
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:LatticeMatrix},
    B::Annotation{<:LatticeMatrix})
    _adtrace("reverse: mul!(C,A,B) [LatticeMatrix]")

    #println("mul!: reverse")

    #println("entered mul! reverse for LatticeMatrix")
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = _getshadow(A.dval)
    dB_struct = _getshadow(B.dval)
    dAval = (dA_struct === nothing) ? nothing : dA_struct.A
    dBval = (dB_struct === nothing) ? nothing : dB_struct.A

    tapeA, tapeB = _tape
    # Use taped values if present, otherwise current values
    if tapeA === nothing
        Aval = A.val.A
    else
        Aval = tapeA[1]
    end
    #Aval = (tapeA === nothing) ? A.val : tapeA
    if tapeB === nothing
        Bval = B.val.A
    else
        Bval = tapeB[1]
    end
    #Bval = (tapeB === nothing) ? B.val : tapeB
    #Aval, Bval = A.val.A, B.val.A

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.NC2)  # (=B.val.NC1)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)

    # ---- ① dA += dC * B'  ----
    if dAval isa AbstractArray
        JACC.parallel_for(
            Nsites, kernel_Dmatrix_mulABdagadd!, dAval, dCval, Bval, NC1, NC2, NC3, nw, idxr
        )
    end

    # ---- ② dB += A' * dC  ----
    if dBval isa AbstractArray
        JACC.parallel_for(
            Nsites, kernel_Dmatrix_mulAdagBadd!, dBval, Aval, dCval, NC1, NC2, NC3, nw, idxr
        )
    end
    if tapeA !== nothing
        unused!(A.val.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, _tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:LatticeMatrix},
    α::S1,
    β::S2,
) where {S1,S2}
    _adtrace("reverse: mul!(C,A,B,alpha,beta) [LatticeMatrix]")
    dα = _zero_cotangent(α)
    dβ = _zero_cotangent(β)

    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing, dα, dβ)
    dCval = dC_struct.A

    dA_struct = _getshadow(A.dval)
    dB_struct = _getshadow(B.dval)
    dAval = (dA_struct === nothing) ? nothing : dA_struct.A
    dBval = (dB_struct === nothing) ? nothing : dB_struct.A

    tapeA, tapeB, tape_α = _tape
    Aval = (tapeA === nothing) ? A.val.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.NC2)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)
    fac = conj(tape_α)

    if dAval isa AbstractArray
        JACC.parallel_for(
            Nsites, kernel_Dmatrix_mulABdagadd_scaled!,
            dAval, dCval, Bval, NC1, NC2, NC3, nw, idxr, fac
        )
        _lm_ad_maybe_sync()
    end

    if dBval isa AbstractArray
        JACC.parallel_for(
            Nsites, kernel_Dmatrix_mulAdagBadd_scaled!,
            dBval, Aval, dCval, NC1, NC2, NC3, nw, idxr, fac
        )
        _lm_ad_maybe_sync()
    end
    if tapeA !== nothing
        unused!(A.val.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing, dα, dβ)
end

function Enzyme.EnzymeRules.reverse(cfg::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    dCout, _tape,
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:AbstractMatrix},
    B::Annotation{<:LatticeMatrix},
)
    _adtrace("reverse: mul!(C,AbstractMatrix,LatticeMatrix)")
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dB_struct = _getshadow(B.dval)
    dBval = (dB_struct === nothing) ? nothing : dB_struct.A

    dAval = nothing
    if hasproperty(A, :dval)
        dAval_raw = getproperty(A, :dval)
        dAval_raw = dAval_raw isa Base.RefValue ? dAval_raw[] : dAval_raw
        dAval = dAval_raw isa AbstractMatrix ? dAval_raw : nothing
    end

    tapeA, tapeB = _tape
    Aval = (tapeA === nothing) ? A.val : tapeA
    Bval = (tapeB === nothing) ? B.val.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(B.val.NC1)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)

    if dBval isa AbstractArray
        At = JACC.array(Aval)
        JACC.parallel_for(
            Nsites, kernel_Dmatrix_mulAdagBadd_matrix!, dBval, At, dCval, NC1, NC2, NC3, nw, idxr
        )
        _lm_ad_maybe_sync()
    end

    if dAval isa AbstractMatrix && (Bval isa AbstractArray)
        _accum_dA_from_dC_Bdag!(dAval, dCval, Bval, idxr, C.val.PN, NC1, NC2, NC3, nw)
    end

    if tapeB !== nothing
        unused!(B.val.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end

function Enzyme.EnzymeRules.reverse(cfg::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    dCout, _tape,
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:LatticeMatrix},
    B::Annotation{<:AbstractMatrix},
)
    _adtrace("reverse: mul!(C,LatticeMatrix,AbstractMatrix)")
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = _getshadow(A.dval)
    dAval = (dA_struct === nothing) ? nothing : dA_struct.A

    dBval = nothing
    if hasproperty(B, :dval)
        dB_raw = getproperty(B, :dval)
        dB_raw = dB_raw isa Base.RefValue ? dB_raw[] : dB_raw
        dBval = dB_raw isa AbstractMatrix ? dB_raw : nothing
    end

    tapeA, tapeB = _tape
    Aval = (tapeA === nothing) ? A.val.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val : tapeB

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.NC2)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)

    if dAval isa AbstractArray
        Bt = JACC.array(Bval)
        JACC.parallel_for(
            Nsites, kernel_Dmatrix_mulACadd_matrix!, dAval, dCval, Bt, NC1, NC2, NC3, nw, idxr
        )
        _lm_ad_maybe_sync()
    end

    if dBval isa AbstractMatrix && (Aval isa AbstractArray)
        _accum_dB_from_Adag_dC_matrix!(dBval, Aval, dCval, idxr, C.val.PN, NC1, NC2, NC3, nw)
    end

    if tapeA !== nothing
        unused!(A.val.temps, tapeA[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end

function Enzyme.EnzymeRules.reverse(
    cfg::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    dCout, _tape,
    C::Annotation{<:LatticeMatrix},
    A,
    B,
)
    _adtrace("reverse: mul! FALLBACK A=$(typeof(A)) B=$(typeof(B))")
    if _LM_TRAP_MUL_REVERSE_FALLBACK
        error("LatticeMatrices AD fallback reached for LinearAlgebra.mul! reverse: A=$(typeof(A)) B=$(typeof(B))")
    end
    return nothing, nothing, nothing
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

@inline function kernel_Dmatrix_mulABdagadd_scaled!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, fac) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC2
        for kc = 1:NC3
            b = conj(B[jc, kc, indices...])
            for ic = 1:NC1
                C[ic, jc, indices...] += fac * A[ic, kc, indices...] * b
            end
        end
    end
end

@inline function kernel_Dmatrix_mulACadd_matrix!(i, dA, dC, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC2
        for kc = 1:NC3
            b = conj(B[kc, jc])
            for ic = 1:NC1
                dA[ic, kc, indices...] += dC[ic, jc, indices...] * b
            end
        end
    end
end

@inline function kernel_Dmatrix_mulAdagBadd_matrix!(i, dB, A, dC, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC2
        for kc = 1:NC3
            acc = zero(eltype(dB))
            for ic = 1:NC1
                acc += conj(A[ic, kc]) * dC[ic, jc, indices...]
            end
            dB[kc, jc, indices...] += acc
        end
    end
end

function _accum_dB_from_Adag_dC_matrix!(dB, A, dC, dindexer, PN, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}) where {NC1,NC2,NC3,nw}
    Nsites = prod(PN)
    for i in 1:Nsites
        indices = delinearize(dindexer, i, nw)
        @inbounds for jc = 1:NC2
            for kc = 1:NC3
                acc = zero(eltype(dB))
                for ic = 1:NC1
                    acc += conj(A[ic, kc, indices...]) * dC[ic, jc, indices...]
                end
                dB[kc, jc] += acc
            end
        end
    end
    return nothing
end

function _accum_dA_from_dC_Bdag!(dA, dC, B, dindexer, PN, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}) where {NC1,NC2,NC3,nw}
    Nsites = prod(PN)
    for i in 1:Nsites
        indices = delinearize(dindexer, i, nw)
        @inbounds for jc = 1:NC2
            for kc = 1:NC3
                bconj = conj(B[kc, jc, indices...])
                for ic = 1:NC1
                    dA[ic, kc] += dC[ic, jc, indices...] * bconj
                end
            end
        end
    end
    return nothing
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

@inline function kernel_Dmatrix_mulAdagBadd_scaled!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, fac) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC2
        for kc = 1:NC3
            b = B[kc, jc, indices...]
            for ic = 1:NC1
                C[ic, jc, indices...] += fac * conj(A[kc, ic, indices...]) * b
            end
        end
    end
end

@inline function _should_zero_dC(dCout)
    return true
end

@inline function _zero_shadow!(C::LatticeMatrix)
    JACC.parallel_for(
        prod(C.PN), kernel_clear_4D!, C.A, C.indexer, Val(C.NC1), Val(C.NC2), Val(C.nw)
    )
    return nothing
end


@inline function kernel_Dmatrix_mulACadd!(i, dA, dC, B,
    ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer
) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for kc = 1:NC3
        for jc = 1:NC2
            b = B[jc, kc, indices...]
            for ic = 1:NC1
                dA[ic, kc, indices...] += dC[ic, jc, indices...] * b
            end
        end
    end
end

@inline function kernel_Dmatrix_mulCdagAadd!(i, dB, dC, A,
    ::Val{NC2}, ::Val{NC1}, ::Val{NC3}, ::Val{nw}, dindexer
) where {NC2,NC1,NC3,nw}
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

@inline function kernel_Dmatrix_mulACadd_scaled!(i, dA, dC, B,
    ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, fac
) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for kc = 1:NC3
        for jc = 1:NC2
            b = B[jc, kc, indices...]
            for ic = 1:NC1
                dA[ic, kc, indices...] += fac * dC[ic, jc, indices...] * b
            end
        end
    end
end

@inline function kernel_Dmatrix_mulCdagAadd_scaled!(i, dB, dC, A,
    ::Val{NC2}, ::Val{NC1}, ::Val{NC3}, ::Val{nw}, dindexer, fac
) where {NC2,NC1,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC2
        for kc = 1:NC3
            acc = zero(eltype(dB))
            for ic = 1:NC1
                acc += fac * conj(dC[ic, jc, indices...]) * A[ic, kc, indices...]
            end
            dB[jc, kc, indices...] += acc
        end
    end
end



@inline function _replace_index(indices, dim, newval)
    return ntuple(i -> i == dim ? newval : indices[i], length(indices))
end

@inline function kernel_fold_halo_dim_to_core!(i, dA, dindexer, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, pn_d, ::Val{d}, phase) where {NC1,NC2,nw,d}
    indices = delinearize(dindexer, i, nw)
    id = indices[d]
    phase_conj = conj(phase)

    if id <= 2 * nw
        idxg = _replace_index(indices, d, id + pn_d)
        @inbounds for jc = 1:NC2
            for ic = 1:NC1
                dA[ic, jc, indices...] += phase_conj * dA[ic, jc, idxg...]
            end
        end
    elseif id > pn_d
        idxg = _replace_index(indices, d, id - pn_d)
        @inbounds for jc = 1:NC2
            for ic = 1:NC1
                dA[ic, jc, indices...] += phase_conj * dA[ic, jc, idxg...]
            end
        end
    end
end

@inline function kernel_add_slab!(i, face, buf, dindexer, ::Val{NC1}, ::Val{NC2}, phase) where {NC1,NC2}
    indices = delinearize(dindexer, i)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            face[ic, jc, indices...] += phase * buf[ic, jc, indices...]
        end
    end
    return nothing
end

function fold_halo_to_core_grad!(ls::LatticeMatrix)
    (ls.A === nothing || !(ls.A isa AbstractArray)) && return nothing
    PN = ls.PN
    N1 = Val(ls.NC1)
    N2 = Val(ls.NC2)
    nwv = Val(ls.nw)
    idx = ls.indexer

    for d in 1:length(PN)
        fold_halo_dim_to_core_grad!(ls, d)
    end
    return nothing
end

function fold_halo_dim_to_core_grad!(ls::LatticeMatrix, d::Int)
    (ls.A === nothing || !(ls.A isa AbstractArray)) && return nothing
    PN = ls.PN
    N1 = Val(ls.NC1)
    N2 = Val(ls.NC2)
    nwv = Val(ls.nw)
    idx = ls.indexer

    rankM, rankP = ls.nbr[d]
    if rankM == ls.myrank && rankP == ls.myrank
        dims_s = ntuple(i -> (i == d ? ls.nw : PN[i] + 2 * ls.nw), length(PN))
        slab_indexer = LatticeMatrices.DIndexer(dims_s)
        nsites = prod(dims_s)
        phase_conj = conj(ls.phases[d])

        face_minus = LatticeMatrices._faceMatrix(ls.A, ls.nw, d, :minus)
        ghost_plus = LatticeMatrices._ghostMatrix(ls.A, ls.nw, d, :plus)
        JACC.parallel_for(nsites, kernel_add_slab!, face_minus, ghost_plus, slab_indexer, N1, N2, phase_conj)

        face_plus = LatticeMatrices._faceMatrix(ls.A, ls.nw, d, :plus)
        ghost_minus = LatticeMatrices._ghostMatrix(ls.A, ls.nw, d, :minus)
        JACC.parallel_for(nsites, kernel_add_slab!, face_plus, ghost_minus, slab_indexer, N1, N2, phase_conj)
        return nothing
    end

    # MPI reverse: send ghost grads back, receive face grads, then accumulate.
    iSM, iRM = 4d - 3, 4d - 2
    iSP, iRP = 4d - 1, 4d
    bufSM, bufRM = ls.buf[iSM], ls.buf[iRM]
    bufSP, bufRP = ls.buf[iSP], ls.buf[iRP]

    copy!(bufSM, LatticeMatrices._ghostMatrix(ls.A, ls.nw, d, :minus))
    copy!(bufSP, LatticeMatrices._ghostMatrix(ls.A, ls.nw, d, :plus))

    reqs = MPI.Request[]
    push!(reqs, MPI.Isend(bufSM, rankM, d, ls.cart))
    push!(reqs, MPI.Isend(bufSP, rankP, d + length(PN), ls.cart))
    push!(reqs, MPI.Irecv!(bufRM, rankM, d + length(PN), ls.cart))
    push!(reqs, MPI.Irecv!(bufRP, rankP, d, ls.cart))
    MPI.Waitall!(reqs)

    dims_s = ntuple(i -> (i == d ? ls.nw : PN[i] + 2 * ls.nw), length(PN))
    slab_indexer = LatticeMatrices.DIndexer(dims_s)
    nsites = prod(dims_s)

    phase_minus = (ls.coords[d] == 0) ? conj(ls.phases[d]) : one(eltype(ls.A))
    face_minus = LatticeMatrices._faceMatrix(ls.A, ls.nw, d, :minus)
    JACC.parallel_for(nsites, kernel_add_slab!, face_minus, bufRM, slab_indexer, N1, N2, phase_minus)

    phase_plus = (ls.coords[d] == ls.dims[d] - 1) ? conj(ls.phases[d]) : one(eltype(ls.A))
    face_plus = LatticeMatrices._faceMatrix(ls.A, ls.nw, d, :plus)
    JACC.parallel_for(nsites, kernel_add_slab!, face_plus, bufRP, slab_indexer, N1, N2, phase_plus)
    return nothing
end

function fold_halo_dim_to_core_grad_phase!(ls::LatticeMatrix, d::Int, phase)
    (ls.A === nothing || !(ls.A isa AbstractArray)) && return nothing
    PN = ls.PN
    N1 = Val(ls.NC1)
    N2 = Val(ls.NC2)
    nwv = Val(ls.nw)
    idx = ls.indexer

    rankM, rankP = ls.nbr[d]
    if rankM == ls.myrank && rankP == ls.myrank
        dims_s = ntuple(i -> (i == d ? ls.nw : PN[i] + 2 * ls.nw), length(PN))
        slab_indexer = LatticeMatrices.DIndexer(dims_s)
        nsites = prod(dims_s)
        phase_conj = conj(phase)

        face_minus = LatticeMatrices._faceMatrix(ls.A, ls.nw, d, :minus)
        ghost_plus = LatticeMatrices._ghostMatrix(ls.A, ls.nw, d, :plus)
        JACC.parallel_for(nsites, kernel_add_slab!, face_minus, ghost_plus, slab_indexer, N1, N2, phase_conj)

        face_plus = LatticeMatrices._faceMatrix(ls.A, ls.nw, d, :plus)
        ghost_minus = LatticeMatrices._ghostMatrix(ls.A, ls.nw, d, :minus)
        JACC.parallel_for(nsites, kernel_add_slab!, face_plus, ghost_minus, slab_indexer, N1, N2, phase_conj)
        return nothing
    end

    iSM, iRM = 4d - 3, 4d - 2
    iSP, iRP = 4d - 1, 4d
    bufSM, bufRM = ls.buf[iSM], ls.buf[iRM]
    bufSP, bufRP = ls.buf[iSP], ls.buf[iRP]

    copy!(bufSM, LatticeMatrices._ghostMatrix(ls.A, ls.nw, d, :minus))
    copy!(bufSP, LatticeMatrices._ghostMatrix(ls.A, ls.nw, d, :plus))

    # NOTE: reverse of exchange_dim! (gradient) with phase from primal
    MPI.Waitall([
        MPI.Irecv!(bufRM, ls.nbr[d][1], iRP, ls.comm),
        MPI.Irecv!(bufRP, ls.nbr[d][2], iRM, ls.comm),
        MPI.Isend(bufSM, ls.nbr[d][1], iSM, ls.comm),
        MPI.Isend(bufSP, ls.nbr[d][2], iSP, ls.comm),
    ])

    phase_conj = conj(phase)
    face_minus = LatticeMatrices._faceMatrix(ls.A, ls.nw, d, :minus)
    face_plus = LatticeMatrices._faceMatrix(ls.A, ls.nw, d, :plus)
    @inbounds face_minus .+= phase_conj .* bufRM
    @inbounds face_plus .+= phase_conj .* bufRP
    return nothing
end

@inline function kernel_zero_ghost_slab!(i, slab, dindexer, ::Val{NC1}, ::Val{NC2}) where {NC1,NC2}
    indices = delinearize(dindexer, i)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            slab[ic, jc, indices...] = zero(eltype(slab))
        end
    end
    return nothing
end

function zero_halo_region!(ls::LatticeMatrix)
    (ls.A === nothing || !(ls.A isa AbstractArray)) && return nothing
    D = length(ls.PN)
    for d in 1:D
        zero_halo_dim!(ls, d)
    end
    return nothing
end

function zero_halo_dim!(ls::LatticeMatrix, d::Int)
    (ls.A === nothing || !(ls.A isa AbstractArray)) && return nothing
    D = length(ls.PN)
    dims_s = ntuple(i -> (i == d ? ls.nw : ls.PN[i] + 2 * ls.nw), D)
    slab_indexer = LatticeMatrices.DIndexer(dims_s)
    nsites = prod(dims_s)

    slab = LatticeMatrices._ghostMatrix(ls.A, ls.nw, d, :minus)
    JACC.parallel_for(nsites, kernel_zero_ghost_slab!, slab, slab_indexer, Val(ls.NC1), Val(ls.NC2))

    slab = LatticeMatrices._ghostMatrix(ls.A, ls.nw, d, :plus)
    JACC.parallel_for(nsites, kernel_zero_ghost_slab!, slab, slab_indexer, Val(ls.NC1), Val(ls.NC2))
    return nothing
end
export zero_halo_region!


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

@inline function kernel_Dmatrix_mulACdagadd_shift!(i, dA, dC, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)
    @inbounds for kc = 1:NC3
        for jc = 1:NC2
            b = conj(B[kc, jc, indices_p...])
            for ic = 1:NC1
                dA[ic, kc, indices...] += dC[ic, jc, indices...] * b
            end
        end
    end
end

@inline function kernel_Dmatrix_mulACadd_shift!(i, dA, dC, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)
    @inbounds for kc = 1:NC3
        for jc = 1:NC2
            b = B[jc, kc, indices_p...]
            for ic = 1:NC1
                dA[ic, kc, indices...] += dC[ic, jc, indices...] * b
            end
        end
    end
end

@inline function kernel_Dmatrix_mulACadd_shift_scaled!(i, dA, dC, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift, fac) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)
    @inbounds for kc = 1:NC3
        for jc = 1:NC2
            b = fac * B[jc, kc, indices_p...]
            for ic = 1:NC1
                dA[ic, kc, indices...] += dC[ic, jc, indices...] * b
            end
        end
    end
end

@inline function kernel_Dmatrix_mulAdagBadd_scatter_shift!(i, dB, A, dC, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift) where {NC1,NC2,NC3,nw}
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

@inline function kernel_Dmatrix_mulAdagBadd_scatter_shift_scaled!(i, dB, A, dC, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift, fac) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)
    @inbounds for jc = 1:NC2
        for kc = 1:NC3
            acc = zero(eltype(dB))
            for ic = 1:NC1
                acc += conj(A[ic, kc, indices...]) * dC[ic, jc, indices...]
            end
            dB[kc, jc, indices_p...] += fac * acc
        end
    end
end

@inline function kernel_Dmatrix_mulAdagBadd_scatter_shiftAshiftB!(i, dB, A, dC, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shiftA, shiftB) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_A = shiftindices(indices, shiftA)
    indices_B = shiftindices(indices, shiftB)
    @inbounds for jc = 1:NC2
        for kc = 1:NC3
            acc = zero(eltype(dB))
            for ic = 1:NC1
                acc += conj(A[ic, kc, indices_A...]) * dC[ic, jc, indices...]
            end
            dB[kc, jc, indices_B...] += acc
        end
    end
end

@inline function kernel_Dmatrix_mulAdagBadd_scatter_shiftAshiftB_adagA!(i, dB, A, dC, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shiftA, shiftB) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_A = shiftindices(indices, shiftA)
    indices_B = shiftindices(indices, shiftB)
    @inbounds for jc = 1:NC2
        for kc = 1:NC3
            acc = zero(eltype(dB))
            for ic = 1:NC1
                acc += conj(A[kc, ic, indices_A...]) * dC[ic, jc, indices...]
            end
            dB[kc, jc, indices_B...] += acc
        end
    end
end

@inline function kernel_Dmatrix_mulCdagAadd_scatter_shift!(i, dB, dC, A, ::Val{NC2}, ::Val{NC1}, ::Val{NC3}, ::Val{nw}, dindexer, shift) where {NC2,NC1,NC3,nw}
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

@inline function kernel_Dmatrix_mulCdagAadd_scatter_shift_scaled!(i, dB, dC, A, ::Val{NC2}, ::Val{NC1}, ::Val{NC3}, ::Val{nw}, dindexer, shift, fac) where {NC2,NC1,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)
    @inbounds for jc = 1:NC2
        for kc = 1:NC3
            acc = zero(eltype(dB))
            for ic = 1:NC1
                acc += conj(dC[ic, jc, indices...]) * A[ic, kc, indices...]
            end
            dB[jc, kc, indices_p...] += fac * acc
        end
    end
end


# C = stA * B
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Staggered_Lattice},
    B::ER.Annotation{<:LatticeMatrix},
) where {RT}
    mul!(C.val, A.val, B.val)

    tapeA_obj, itA = get_block(A.val.data.temps)
    tapeA_obj .= A.val.data.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.temps)
    tapeB_obj .= B.val.A
    tapeB = (tapeB_obj, itB)

    return ER.AugmentedReturn(nothing, nothing, (tapeA, tapeB))
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Staggered_Lattice{TA,μA}},
    B::ER.Duplicated{<:LatticeMatrix},
) where {TA,μA}
    do_dB = false
    s = _getshadow(B.dval)
    do_dB = (s isa LatticeMatrix)
    return _rev_mul_stAB!(cfg, dCout, tape, C, A, B, Val(μA); do_dB=do_dB)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Staggered_Lattice{TA,μA}},
    B,
) where {TA,μA}
    do_dB = false
    if hasproperty(B, :dval)
        s = _getshadow(getproperty(B, :dval))
        do_dB = (s isa LatticeMatrix)
    end
    return _rev_mul_stAB!(cfg, dCout, tape, C, A, B, Val(μA); do_dB=do_dB)
end

function _rev_mul_stAB!(
    cfg::ER.RevConfig,
    dCout, tape,
    C, A, B, μA;
    do_dB::Bool,
)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow_data(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = do_dB ? _getshadow(B.dval) : nothing
    dBval = (do_dB && (dB_struct isa LatticeMatrix)) ? dB_struct.A : nothing

    tapeA, tapeB = tape
    Aval = (tapeA === nothing) ? A.val.data.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.data.NC2)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)

    if dAval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulABdagadd_eta!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr, μA
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulAdagBadd_eta!,
            dBval, Aval, dCval,
            NC1, NC2, NC3, nw, idxr, μA
        )
    end

    if tapeA !== nothing
        unused!(A.val.data.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end

# C = A * stB
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:Staggered_Lattice},
) where {RT}
    mul!(C.val, A.val, B.val)

    tapeA_obj, itA = get_block(A.val.temps)
    tapeA_obj .= A.val.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.data.temps)
    tapeB_obj .= B.val.data.A
    tapeB = (tapeB_obj, itB)

    return ER.AugmentedReturn(nothing, nothing, (tapeA, tapeB))
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Duplicated{<:Staggered_Lattice{TB,μB}},
) where {TB,μB}
    do_dB = false
    s = _getshadow_data(B.dval)
    do_dB = (s isa LatticeMatrix)
    return _rev_mul_AstB!(cfg, dCout, tape, C, A, B, Val(μB); do_dB=do_dB)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:Staggered_Lattice{TB,μB}},
) where {TB,μB}
    do_dB = false
    if hasproperty(B, :dval)
        s = _getshadow_data(getproperty(B, :dval))
        do_dB = (s isa LatticeMatrix)
    end
    return _rev_mul_AstB!(cfg, dCout, tape, C, A, B, Val(μB); do_dB=do_dB)
end

function _rev_mul_AstB!(
    cfg::ER.RevConfig,
    dCout, tape,
    C, A, B, μB;
    do_dB::Bool,
)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = do_dB ? _getshadow_data(B.dval) : nothing
    dBval = (do_dB && (dB_struct isa LatticeMatrix)) ? dB_struct.A : nothing

    tapeA, tapeB = tape
    Aval = (tapeA === nothing) ? A.val.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.data.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.NC2)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)

    if dAval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulABdagadd_eta2!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr, Val(0), μB
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulAdagBadd_eta2!,
            dBval, Aval, dCval,
            NC1, NC2, NC3, nw, idxr, Val(0), μB
        )
    end

    if tapeA !== nothing
        unused!(A.val.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.data.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end

# C = stA * stB
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Staggered_Lattice},
    B::ER.Annotation{<:Staggered_Lattice},
) where {RT}
    mul!(C.val, A.val, B.val)

    tapeA_obj, itA = get_block(A.val.data.temps)
    tapeA_obj .= A.val.data.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.data.temps)
    tapeB_obj .= B.val.data.A
    tapeB = (tapeB_obj, itB)

    return ER.AugmentedReturn(nothing, nothing, (tapeA, tapeB))
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Staggered_Lattice{TA,μA}},
    B::ER.Duplicated{<:Staggered_Lattice{TB,μB}},
) where {TA,μA,TB,μB}
    do_dB = false
    s = _getshadow_data(B.dval)
    do_dB = (s isa LatticeMatrix)
    return _rev_mul_stAstB!(cfg, dCout, tape, C, A, B, Val(μA), Val(μB); do_dB=do_dB)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Staggered_Lattice{TA,μA}},
    B::ER.Annotation{<:Staggered_Lattice{TB,μB}},
) where {TA,μA,TB,μB}
    do_dB = false
    if hasproperty(B, :dval)
        s = _getshadow_data(getproperty(B, :dval))
        do_dB = (s isa LatticeMatrix)
    end
    return _rev_mul_stAstB!(cfg, dCout, tape, C, A, B, Val(μA), Val(μB); do_dB=do_dB)
end

function _rev_mul_stAstB!(
    cfg::ER.RevConfig,
    dCout, tape,
    C, A, B, μA, μB;
    do_dB::Bool,
)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow_data(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = do_dB ? _getshadow_data(B.dval) : nothing
    dBval = (do_dB && (dB_struct isa LatticeMatrix)) ? dB_struct.A : nothing

    tapeA, tapeB = tape
    Aval = (tapeA === nothing) ? A.val.data.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.data.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.data.NC2)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)

    if dAval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulABdagadd_eta2!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr, μA, μB
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulAdagBadd_eta2!,
            dBval, Aval, dCval,
            NC1, NC2, NC3, nw, idxr, μA, μB
        )
    end

    if tapeA !== nothing
        unused!(A.val.data.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.data.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end

# C = stA' * B
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice{<:Staggered_Lattice}},
    B::ER.Annotation{<:LatticeMatrix},
) where {RT}
    mul!(C.val, A.val, B.val)

    tapeA_obj, itA = get_block(A.val.data.data.temps)
    tapeA_obj .= A.val.data.data.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.temps)
    tapeB_obj .= B.val.A
    tapeB = (tapeB_obj, itB)

    return ER.AugmentedReturn(nothing, nothing, (tapeA, tapeB))
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice{<:Staggered_Lattice{TA,μA}}},
    B::ER.Duplicated{<:LatticeMatrix},
) where {TA,μA}
    do_dB = false
    s = _getshadow(B.dval)
    do_dB = (s isa LatticeMatrix)
    return _rev_mul_stAdagB!(cfg, dCout, tape, C, A, B, Val(μA), Val(0); do_dB=do_dB)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice{<:Staggered_Lattice{TA,μA}}},
    B,
) where {TA,μA}
    do_dB = false
    if hasproperty(B, :dval)
        s = _getshadow(getproperty(B, :dval))
        do_dB = (s isa LatticeMatrix)
    end
    return _rev_mul_stAdagB!(cfg, dCout, tape, C, A, B, Val(μA), Val(0); do_dB=do_dB)
end

# C = A' * stB
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice{<:LatticeMatrix}},
    B::ER.Annotation{<:Staggered_Lattice},
) where {RT}
    mul!(C.val, A.val, B.val)

    tapeA_obj, itA = get_block(A.val.data.temps)
    tapeA_obj .= A.val.data.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.data.temps)
    tapeB_obj .= B.val.data.A
    tapeB = (tapeB_obj, itB)

    return ER.AugmentedReturn(nothing, nothing, (tapeA, tapeB))
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice{<:LatticeMatrix}},
    B::ER.Duplicated{<:Staggered_Lattice{TB,μB}},
) where {TB,μB}
    do_dB = false
    s = _getshadow_data(B.dval)
    do_dB = (s isa LatticeMatrix)
    return _rev_mul_AdagstB!(cfg, dCout, tape, C, A, B, Val(0), Val(μB); do_dB=do_dB)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice{<:LatticeMatrix}},
    B::ER.Annotation{<:Staggered_Lattice{TB,μB}},
) where {TB,μB}
    do_dB = false
    if hasproperty(B, :dval)
        s = _getshadow_data(getproperty(B, :dval))
        do_dB = (s isa LatticeMatrix)
    end
    return _rev_mul_AdagstB!(cfg, dCout, tape, C, A, B, Val(0), Val(μB); do_dB=do_dB)
end

# C = stA' * stB
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice{<:Staggered_Lattice}},
    B::ER.Annotation{<:Staggered_Lattice},
) where {RT}
    mul!(C.val, A.val, B.val)

    tapeA_obj, itA = get_block(A.val.data.data.temps)
    tapeA_obj .= A.val.data.data.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.data.temps)
    tapeB_obj .= B.val.data.A
    tapeB = (tapeB_obj, itB)

    return ER.AugmentedReturn(nothing, nothing, (tapeA, tapeB))
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice{<:Staggered_Lattice{TA,μA}}},
    B::ER.Duplicated{<:Staggered_Lattice{TB,μB}},
) where {TA,μA,TB,μB}
    do_dB = false
    s = _getshadow_data(B.dval)
    do_dB = (s isa LatticeMatrix)
    return _rev_mul_stAdagstB!(cfg, dCout, tape, C, A, B, Val(μA), Val(μB); do_dB=do_dB)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice{<:Staggered_Lattice{TA,μA}}},
    B::ER.Annotation{<:Staggered_Lattice{TB,μB}},
) where {TA,μA,TB,μB}
    do_dB = false
    if hasproperty(B, :dval)
        s = _getshadow_data(getproperty(B, :dval))
        do_dB = (s isa LatticeMatrix)
    end
    return _rev_mul_stAdagstB!(cfg, dCout, tape, C, A, B, Val(μA), Val(μB); do_dB=do_dB)
end

function _rev_mul_stAdagB!(
    cfg::ER.RevConfig,
    dCout, tape,
    C, A, B, μA, μB;
    do_dB::Bool,
)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow_data(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = do_dB ? _getshadow(B.dval) : nothing
    dBval = (do_dB && (dB_struct isa LatticeMatrix)) ? dB_struct.A : nothing

    tapeA, tapeB = tape
    Aval = (tapeA === nothing) ? A.val.data.data.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(B.val.NC1)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)

    if dAval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulB_Cdagadd_eta2!,
            dAval, Bval, dCval,
            NC1, NC2, NC3, nw, idxr, μA, μB
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulA_Cadd_eta2!,
            dBval, Aval, dCval,
            NC1, NC2, NC3, nw, idxr, μA, μB
        )
    end

    if tapeA !== nothing
        unused!(A.val.data.data.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end

function _rev_mul_AdagstB!(
    cfg::ER.RevConfig,
    dCout, tape,
    C, A, B, μA, μB;
    do_dB::Bool,
)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow_data(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = do_dB ? _getshadow_data(B.dval) : nothing
    dBval = (do_dB && (dB_struct isa LatticeMatrix)) ? dB_struct.A : nothing

    tapeA, tapeB = tape
    Aval = (tapeA === nothing) ? A.val.data.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.data.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(B.val.data.NC1)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)

    if dAval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulB_Cdagadd_eta2!,
            dAval, Bval, dCval,
            NC1, NC2, NC3, nw, idxr, μA, μB
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulA_Cadd_eta2!,
            dBval, Aval, dCval,
            NC1, NC2, NC3, nw, idxr, μA, μB
        )
    end

    if tapeA !== nothing
        unused!(A.val.data.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.data.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end

function _rev_mul_stAdagstB!(
    cfg::ER.RevConfig,
    dCout, tape,
    C, A, B, μA, μB;
    do_dB::Bool,
)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow_data(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = do_dB ? _getshadow_data(B.dval) : nothing
    dBval = (do_dB && (dB_struct isa LatticeMatrix)) ? dB_struct.A : nothing

    tapeA, tapeB = tape
    Aval = (tapeA === nothing) ? A.val.data.data.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.data.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(B.val.data.NC1)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)

    if dAval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulB_Cdagadd_eta2!,
            dAval, Bval, dCval,
            NC1, NC2, NC3, nw, idxr, μA, μB
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulA_Cadd_eta2!,
            dBval, Aval, dCval,
            NC1, NC2, NC3, nw, idxr, μA, μB
        )
    end

    if tapeA !== nothing
        unused!(A.val.data.data.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.data.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end

# C = stA * B'
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Staggered_Lattice},
    B::ER.Annotation{<:Adjoint_Lattice{<:LatticeMatrix}},
) where {RT}
    mul!(C.val, A.val, B.val)

    tapeA_obj, itA = get_block(A.val.data.temps)
    tapeA_obj .= A.val.data.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.data.temps)
    tapeB_obj .= B.val.data.A
    tapeB = (tapeB_obj, itB)

    return ER.AugmentedReturn(nothing, nothing, (tapeA, tapeB))
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Staggered_Lattice{TA,μA}},
    B::ER.Duplicated{<:Adjoint_Lattice{<:LatticeMatrix}},
) where {TA,μA}
    do_dB = false
    s = _getshadow_data(B.dval)
    do_dB = (s isa LatticeMatrix)
    return _rev_mul_stABdag!(cfg, dCout, tape, C, A, B, Val(μA), Val(0); do_dB=do_dB)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Staggered_Lattice{TA,μA}},
    B::ER.Annotation{<:Adjoint_Lattice{<:LatticeMatrix}},
) where {TA,μA}
    do_dB = false
    if hasproperty(B, :dval)
        s = _getshadow_data(getproperty(B, :dval))
        do_dB = (s isa LatticeMatrix)
    end
    return _rev_mul_stABdag!(cfg, dCout, tape, C, A, B, Val(μA), Val(0); do_dB=do_dB)
end

# C = A * stB'
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:Adjoint_Lattice{<:Staggered_Lattice}},
) where {RT}
    mul!(C.val, A.val, B.val)

    tapeA_obj, itA = get_block(A.val.temps)
    tapeA_obj .= A.val.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.data.data.temps)
    tapeB_obj .= B.val.data.data.A
    tapeB = (tapeB_obj, itB)

    return ER.AugmentedReturn(nothing, nothing, (tapeA, tapeB))
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Duplicated{<:Adjoint_Lattice{<:Staggered_Lattice{TB,μB}}},
) where {TB,μB}
    do_dB = false
    s = _getshadow_data(B.dval)
    do_dB = (s isa LatticeMatrix)
    return _rev_mul_AstBdag!(cfg, dCout, tape, C, A, B, Val(0), Val(μB); do_dB=do_dB)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:Adjoint_Lattice{<:Staggered_Lattice{TB,μB}}},
) where {TB,μB}
    do_dB = false
    if hasproperty(B, :dval)
        s = _getshadow_data(getproperty(B, :dval))
        do_dB = (s isa LatticeMatrix)
    end
    return _rev_mul_AstBdag!(cfg, dCout, tape, C, A, B, Val(0), Val(μB); do_dB=do_dB)
end

# C = stA * stB'
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Staggered_Lattice},
    B::ER.Annotation{<:Adjoint_Lattice{<:Staggered_Lattice}},
) where {RT}
    mul!(C.val, A.val, B.val)

    tapeA_obj, itA = get_block(A.val.data.temps)
    tapeA_obj .= A.val.data.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.data.data.temps)
    tapeB_obj .= B.val.data.data.A
    tapeB = (tapeB_obj, itB)

    return ER.AugmentedReturn(nothing, nothing, (tapeA, tapeB))
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Staggered_Lattice{TA,μA}},
    B::ER.Duplicated{<:Adjoint_Lattice{<:Staggered_Lattice{TB,μB}}},
) where {TA,μA,TB,μB}
    do_dB = false
    s = _getshadow_data(B.dval)
    do_dB = (s isa LatticeMatrix)
    return _rev_mul_stAstBdag!(cfg, dCout, tape, C, A, B, Val(μA), Val(μB); do_dB=do_dB)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Staggered_Lattice{TA,μA}},
    B::ER.Annotation{<:Adjoint_Lattice{<:Staggered_Lattice{TB,μB}}},
) where {TA,μA,TB,μB}
    do_dB = false
    if hasproperty(B, :dval)
        s = _getshadow_data(getproperty(B, :dval))
        do_dB = (s isa LatticeMatrix)
    end
    return _rev_mul_stAstBdag!(cfg, dCout, tape, C, A, B, Val(μA), Val(μB); do_dB=do_dB)
end

function _rev_mul_stABdag!(
    cfg::ER.RevConfig,
    dCout, tape,
    C, A, B, μA, μB;
    do_dB::Bool,
)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow_data(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = do_dB ? _getshadow_data(B.dval) : nothing
    dBval = (do_dB && (dB_struct isa LatticeMatrix)) ? dB_struct.A : nothing

    tapeA, tapeB = tape
    Aval = (tapeA === nothing) ? A.val.data.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.data.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.data.NC2)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)

    if dAval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulACadd_eta2!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr, μA, μB
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulCdagAadd_eta2!,
            dBval, dCval, Aval,
            NC2, NC1, NC3, nw, idxr, μA, μB
        )
    end

    if tapeA !== nothing
        unused!(A.val.data.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.data.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end

function _rev_mul_AstBdag!(
    cfg::ER.RevConfig,
    dCout, tape,
    C, A, B, μA, μB;
    do_dB::Bool,
)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = do_dB ? _getshadow_data(B.dval) : nothing
    dBval = (do_dB && (dB_struct isa LatticeMatrix)) ? dB_struct.A : nothing

    tapeA, tapeB = tape
    Aval = (tapeA === nothing) ? A.val.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.data.data.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.NC2)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)

    if dAval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulACadd_eta2!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr, μA, μB
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulCdagAadd_eta2!,
            dBval, dCval, Aval,
            NC2, NC1, NC3, nw, idxr, μA, μB
        )
    end

    if tapeA !== nothing
        unused!(A.val.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.data.data.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end

function _rev_mul_stAstBdag!(
    cfg::ER.RevConfig,
    dCout, tape,
    C, A, B, μA, μB;
    do_dB::Bool,
)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow_data(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = do_dB ? _getshadow_data(B.dval) : nothing
    dBval = (do_dB && (dB_struct isa LatticeMatrix)) ? dB_struct.A : nothing

    tapeA, tapeB = tape
    Aval = (tapeA === nothing) ? A.val.data.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.data.data.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.data.NC2)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)

    if dAval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulACadd_eta2!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr, μA, μB
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulCdagAadd_eta2!,
            dBval, dCval, Aval,
            NC2, NC1, NC3, nw, idxr, μA, μB
        )
    end

    if tapeA !== nothing
        unused!(A.val.data.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.data.data.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end

# C = stA' * B'
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice{<:Staggered_Lattice}},
    B::ER.Annotation{<:Adjoint_Lattice{<:LatticeMatrix}},
) where {RT}
    mul!(C.val, A.val, B.val)

    tapeA_obj, itA = get_block(A.val.data.data.temps)
    tapeA_obj .= A.val.data.data.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.data.temps)
    tapeB_obj .= B.val.data.A
    tapeB = (tapeB_obj, itB)

    return ER.AugmentedReturn(nothing, nothing, (tapeA, tapeB))
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice{<:Staggered_Lattice{TA,μA}}},
    B::ER.Duplicated{<:Adjoint_Lattice{<:LatticeMatrix}},
) where {TA,μA}
    do_dB = false
    s = _getshadow_data(B.dval)
    do_dB = (s isa LatticeMatrix)
    return _rev_mul_stAdagBdag!(cfg, dCout, tape, C, A, B, Val(μA), Val(0); do_dB=do_dB)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice{<:Staggered_Lattice{TA,μA}}},
    B::ER.Annotation{<:Adjoint_Lattice{<:LatticeMatrix}},
) where {TA,μA}
    do_dB = false
    if hasproperty(B, :dval)
        s = _getshadow_data(getproperty(B, :dval))
        do_dB = (s isa LatticeMatrix)
    end
    return _rev_mul_stAdagBdag!(cfg, dCout, tape, C, A, B, Val(μA), Val(0); do_dB=do_dB)
end

# C = A' * stB'
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice{<:LatticeMatrix}},
    B::ER.Annotation{<:Adjoint_Lattice{<:Staggered_Lattice}},
) where {RT}
    mul!(C.val, A.val, B.val)

    tapeA_obj, itA = get_block(A.val.data.temps)
    tapeA_obj .= A.val.data.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.data.data.temps)
    tapeB_obj .= B.val.data.data.A
    tapeB = (tapeB_obj, itB)

    return ER.AugmentedReturn(nothing, nothing, (tapeA, tapeB))
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice{<:LatticeMatrix}},
    B::ER.Duplicated{<:Adjoint_Lattice{<:Staggered_Lattice{TB,μB}}},
) where {TB,μB}
    do_dB = false
    s = _getshadow_data(B.dval)
    do_dB = (s isa LatticeMatrix)
    return _rev_mul_AdagstBdag!(cfg, dCout, tape, C, A, B, Val(0), Val(μB); do_dB=do_dB)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice{<:LatticeMatrix}},
    B::ER.Annotation{<:Adjoint_Lattice{<:Staggered_Lattice{TB,μB}}},
) where {TB,μB}
    do_dB = false
    if hasproperty(B, :dval)
        s = _getshadow_data(getproperty(B, :dval))
        do_dB = (s isa LatticeMatrix)
    end
    return _rev_mul_AdagstBdag!(cfg, dCout, tape, C, A, B, Val(0), Val(μB); do_dB=do_dB)
end

# C = stA' * stB'
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice{<:Staggered_Lattice}},
    B::ER.Annotation{<:Adjoint_Lattice{<:Staggered_Lattice}},
) where {RT}
    mul!(C.val, A.val, B.val)

    tapeA_obj, itA = get_block(A.val.data.data.temps)
    tapeA_obj .= A.val.data.data.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.data.data.temps)
    tapeB_obj .= B.val.data.data.A
    tapeB = (tapeB_obj, itB)

    return ER.AugmentedReturn(nothing, nothing, (tapeA, tapeB))
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice{<:Staggered_Lattice{TA,μA}}},
    B::ER.Duplicated{<:Adjoint_Lattice{<:Staggered_Lattice{TB,μB}}},
) where {TA,μA,TB,μB}
    do_dB = false
    s = _getshadow_data(B.dval)
    do_dB = (s isa LatticeMatrix)
    return _rev_mul_stAdagstBdag!(cfg, dCout, tape, C, A, B, Val(μA), Val(μB); do_dB=do_dB)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice{<:Staggered_Lattice{TA,μA}}},
    B::ER.Annotation{<:Adjoint_Lattice{<:Staggered_Lattice{TB,μB}}},
) where {TA,μA,TB,μB}
    do_dB = false
    if hasproperty(B, :dval)
        s = _getshadow_data(getproperty(B, :dval))
        do_dB = (s isa LatticeMatrix)
    end
    return _rev_mul_stAdagstBdag!(cfg, dCout, tape, C, A, B, Val(μA), Val(μB); do_dB=do_dB)
end

function _rev_mul_stAdagBdag!(
    cfg::ER.RevConfig,
    dCout, tape,
    C, A, B, μA, μB;
    do_dB::Bool,
)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow_data(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = do_dB ? _getshadow_data(B.dval) : nothing
    dBval = (do_dB && (dB_struct isa LatticeMatrix)) ? dB_struct.A : nothing

    tapeA, tapeB = tape
    Aval = (tapeA === nothing) ? A.val.data.data.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.data.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.data.data.NC1)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)

    if dAval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulBdag_Cdagadd_eta2!,
            dAval, Bval, dCval,
            NC1, NC2, NC3, nw, idxr, μA, μB
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulCdag_Adagadd_eta2!,
            dBval, Aval, dCval,
            NC2, NC1, NC3, nw, idxr, μA, μB
        )
    end

    if tapeA !== nothing
        unused!(A.val.data.data.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.data.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end

function _rev_mul_AdagstBdag!(
    cfg::ER.RevConfig,
    dCout, tape,
    C, A, B, μA, μB;
    do_dB::Bool,
)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow_data(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = do_dB ? _getshadow_data(B.dval) : nothing
    dBval = (do_dB && (dB_struct isa LatticeMatrix)) ? dB_struct.A : nothing

    tapeA, tapeB = tape
    Aval = (tapeA === nothing) ? A.val.data.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.data.data.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.data.NC1)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)

    if dAval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulBdag_Cdagadd_eta2!,
            dAval, Bval, dCval,
            NC1, NC2, NC3, nw, idxr, μA, μB
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulCdag_Adagadd_eta2!,
            dBval, Aval, dCval,
            NC2, NC1, NC3, nw, idxr, μA, μB
        )
    end

    if tapeA !== nothing
        unused!(A.val.data.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.data.data.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end

function _rev_mul_stAdagstBdag!(
    cfg::ER.RevConfig,
    dCout, tape,
    C, A, B, μA, μB;
    do_dB::Bool,
)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _getshadow_data(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = do_dB ? _getshadow_data(B.dval) : nothing
    dBval = (do_dB && (dB_struct isa LatticeMatrix)) ? dB_struct.A : nothing

    tapeA, tapeB = tape
    Aval = (tapeA === nothing) ? A.val.data.data.A : tapeA[1]
    Bval = (tapeB === nothing) ? B.val.data.data.A : tapeB[1]

    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.data.data.NC1)
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)

    if dAval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulBdag_Cdagadd_eta2!,
            dAval, Bval, dCval,
            NC1, NC2, NC3, nw, idxr, μA, μB
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulCdag_Adagadd_eta2!,
            dBval, Aval, dCval,
            NC2, NC1, NC3, nw, idxr, μA, μB
        )
    end

    if tapeA !== nothing
        unused!(A.val.data.data.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B.val.data.data.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end
