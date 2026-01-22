import Enzyme.EnzymeRules: augmented_primal, reverse, RevConfig, AugmentedReturn, needs_primal, needs_shadow
import LatticeMatrices: add_matrix!, add_matrix_shiftedA!, kernel_add_4D!, kernel_add_4D_dag!, kernel_add_4D_shift!, Adjoint_Lattice, get_shift,
    kernel_Dmatrix_mul_AshiftB!, kernel_Dmatrix_mul_AshiftBdag!, kernel_clear_4D!,
    mul_ABdag!, mul_A_shiftBdag!, mul_AshiftB!, substitute!, AbstractLattice
using PreallocatedArrays

const ER = Enzyme.EnzymeRules

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




function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_AshiftB!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:LatticeMatrix},
    shift::RT2,
) where {RT,RT2}
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

    return ER.AugmentedReturn(nothing, nothing, (tapeA, tapeB, tape_shift))
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

# 共通の本体（Bのwrapper型で do_dB を切り替える）
function _rev_mul_AshiftB!(
    cfg::ER.RevConfig,
    dCout, tape,
    C, A, B, shift;
    do_dB::Bool,
)

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
    dA_struct = _getshadow(A.dval)
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
        fold_halo_to_core_grad!(dB_struct)
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

# C = A * B'
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_ABdag!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    B::ER.Annotation{<:LatticeMatrix},
) where {RT}
    mul_ABdag!(C.val, A.val, B.val)

    tapeA_obj, itA = get_block(A.val.temps)
    tapeA_obj .= A.val.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B.val.temps)
    tapeB_obj .= B.val.A
    tapeB = (tapeB_obj, itB)

    return ER.AugmentedReturn(nothing, nothing, (tapeA, tapeB))
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

function _rev_mul_ABdag!(
    cfg::ER.RevConfig,
    dCout, tape,
    C, A, B;
    do_dB::Bool,
)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = _getshadow(A.dval)
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

function _rev_mul_A_shiftBdag!(
    cfg::ER.RevConfig,
    dCout, tape,
    C, A, B, shift;
    do_dB::Bool,
)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = _getshadow(A.dval)
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
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulCdagAadd_scatter_shift!,
            dBval, dCval, Aval,
            NC2, NC1, NC3, nw, idxr, sh
        )
        fold_halo_to_core_grad!(dB_struct)
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

    dA_struct = _getshadow(A.dval)
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
        fold_halo_to_core_grad!(dA_struct)
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

# add_matrix! (C += α * A)
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(add_matrix!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    α::S,
) where {RT,S}
    αval = hasproperty(α, :val) ? α.val : α
    add_matrix!(C.val, A.val, αval)
    return ER.AugmentedReturn(nothing, nothing, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(add_matrix!)},
    dCout, _tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    α::S,
) where {S}
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = _getshadow(A.dval)
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

    return (nothing, nothing, nothing)
end

# add_matrix! (C += α * A†)
function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(add_matrix!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice},
    α::S,
) where {RT,S}
    αval = hasproperty(α, :val) ? α.val : α
    add_matrix!(C.val, A.val, αval)
    return ER.AugmentedReturn(nothing, nothing, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(add_matrix!)},
    dCout, _tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:Adjoint_Lattice},
    α::S,
) where {S}
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = _getshadow_data(A.dval)
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

    return (nothing, nothing, nothing)
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


struct Adjoint_Lattice_Ann{A} <: AbstractLattice
    data::A  # Enzyme annotation (Duplicated/MixedDuplicated/Const)
end

# Normal (non-AD) wrapper
Base.@noinline adj_L(B::LatticeMatrix) = Adjoint_Lattice(B)

# AD-time wrapper
Base.@noinline adj_L(B::ER.Annotation) = Adjoint_Lattice_Ann(B)

Base.@noinline function Base.adjoint(data::ER.Annotation)
    return Adjoint_Lattice_Ann(data)
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
    return ER.AugmentedReturn(nothing, nothing, (tapeA, tapeB))
end


# Reverse rule for mul!(C, A, B) with LatticeMatrix inputs.
function Enzyme.EnzymeRules.reverse(::RevConfig,
    ::Const{typeof(LinearAlgebra.mul!)},
    dCout, _tape,
    C::Annotation{<:LatticeMatrix},
    A::Annotation{<:LatticeMatrix},
    B::Annotation{<:LatticeMatrix})

    println("mul!: reverse")

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

@inline function _should_zero_dC(dCout)
    return dCout !== nothing
end

@inline function _zero_shadow!(C::LatticeMatrix)
    JACC.parallel_for(
        prod(C.PN), kernel_clear_4D!, C.A, C.indexer, Val(C.NC1), Val(C.NC2), Val(C.nw)
    )
    return nothing
end


function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.mul!)},
    ::Type{RT},
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    Badj::ER.Annotation{<:Adjoint_Lattice_Ann},
) where {RT}

    # Forward computation (primal)
    LinearAlgebra.mul!(C.val, A.val, Badj.val)

    # Tape A
    tapeA_obj, itA = get_block(A.val.temps)
    tapeA_obj .= A.val.A
    tapeA = (tapeA_obj, itA)

    # Tape parent(B) primal, accessible through annotation
    parentB = Badj.val.data
    tapeB_obj, itB = get_block(parentB.val.temps)
    tapeB_obj .= parentB.val.A
    tapeB = (tapeB_obj, itB)

    return ER.AugmentedReturn(nothing, nothing, (tapeA, tapeB))
end

function ER.reverse(::RevConfig,
    ::ER.Const{typeof(LinearAlgebra.mul!)},
    dCout, tape,
    C::ER.Annotation{<:LatticeMatrix},
    A::ER.Annotation{<:LatticeMatrix},
    Badj::ER.Annotation{<:Adjoint_Lattice_Ann},
)

    # Fetch dC (output adjoint)
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    # Fetch dA buffer
    dA_struct = _getshadow(A.dval)
    dAval = (dA_struct === nothing) ? nothing : dA_struct.A

    # Fetch dB buffer (IMPORTANT: accumulate into parent B, not into Badj)
    parentB = Badj.val.data                    # Annotation
    dB_struct = _getshadow(parentB.dval)
    dBval = (dB_struct === nothing) ? nothing : dB_struct.A

    # Unpack tapes
    tapeA, tapeB = tape
    Aval = (tapeA === nothing) ? A.val.A : tapeA[1]
    Bval = (tapeB === nothing) ? parentB.val.A : tapeB[1]

    # Context
    NC1 = Val(C.val.NC1)
    NC2 = Val(C.val.NC2)
    NC3 = Val(A.val.NC2)   # A is NC1×NC3, B is NC2×NC3, C is NC1×NC2
    nw = Val(C.val.nw)
    idxr = C.val.indexer
    Nsites = prod(C.val.PN)

    # (1) dA += dC * B
    if dAval isa AbstractArray
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulACadd!,   # new kernel
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr
        )
    end

    # (2) dB += (dC)† * A
    if dBval isa AbstractArray
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulCdagAadd!,  # new kernel
            dBval, dCval, Aval,
            NC2, NC1, NC3, nw, idxr       # note dims: dB is NC2×NC3
        )
    end

    # Release tapes
    if tapeA !== nothing
        unused!(A.val.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(parentB.val.temps, tapeB[2])
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
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

function LinearAlgebra.mul!(C::LatticeMatrix{D,T1,AT1,NC1,NC2,nw,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,nw,DI},
    Badj::Adjoint_Lattice_Ann,
) where {D,T1,T2,AT1,AT2,NC1,NC2,NC3,nw,DI}

    parentB = Badj.data.val  # unwrap primal from annotation

    JACC.parallel_for(
        prod(C.PN),
        kernel_Dmatrix_mul_ABdag!,
        C.A, A.A, parentB.A,
        Val(NC1), Val(NC2), Val(NC3), Val(nw), C.indexer
    )
    return C
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

function fold_halo_to_core_grad!(ls::LatticeMatrix)
    (ls.A === nothing || !(ls.A isa AbstractArray)) && return nothing
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
                Nsites, kernel_fold_halo_dim_to_core!, ls.A, idx, N1, N2, nwv, PN[d], Val(d), ls.phases[d]
            )
        end
    end
    return nothing
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
