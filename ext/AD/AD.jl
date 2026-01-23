import Enzyme.EnzymeRules: augmented_primal, reverse, RevConfig, AugmentedReturn, needs_primal, needs_shadow
import LatticeMatrices: add_matrix!, add_matrix_Adag!, add_matrix_shiftedA!, add_matrix_shiftedAdag!, kernel_add_4D!, kernel_add_4D_dag!, kernel_add_4D_shift!, Adjoint_Lattice, get_shift,
    kernel_Dmatrix_mul_AshiftB!, kernel_Dmatrix_mul_AshiftBdag!, kernel_clear_4D!,
    mul_ABdag!, mul_A_shiftBdag!, mul_AshiftB!, substitute!, AbstractLattice, expt_TA!, clear_matrix!, set_halo!
using PreallocatedArrays
using MPI

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

    dA_struct = _getshadow(A.dval)
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
        fold_halo_to_core_grad!(dA_struct)
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
    αval = hasproperty(α, :val) ? α.val : α
    add_matrix_Adag!(C.val, A.val, αval)
    return ER.AugmentedReturn(nothing, nothing, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(add_matrix_Adag!)},
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
            kernel_add_4D_dag!,
            dAval, dCval, C.val.indexer,
            Val(C.val.NC2), Val(C.val.NC1),
            conj(αval), Val(C.val.nw)
        )
    end

    return (nothing, nothing, nothing)
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
    dC_struct = _getshadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _getshadow(C.dval))
    dC_struct === nothing && return (nothing,)

    fold_halo_to_core_grad!(dC_struct)
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
        α = one(eltype(dCval))
        JACC.parallel_for(
            prod(C.val.PN),
            kernel_add_4D!,
            dAval, dCval, C.val.indexer,
            Val(C.val.NC1), Val(C.val.NC2),
            α, Val(C.val.nw)
        )
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
        fold_halo_to_core_grad!(dA_struct)
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
        fold_halo_to_core_grad!(dA_struct)
    end

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing)
end



const _expt_ta_eps_q = 1e-18
const fac13 = 1 / 3

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
