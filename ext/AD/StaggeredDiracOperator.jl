import LatticeMatrices: StaggeredDiracOperator4D,
    kernel_StaggeredDiracOperator4D!, kernel_add_4D!,
    staggered_link_pullback!, _kernel_staggered_link_pullback_direction!

@inline function _staggered_operator_shadow(operator)
    hasproperty(operator, :dval) || return nothing
    shadow = getproperty(operator, :dval)
    shadow isa Base.RefValue && (shadow = shadow[])
    return shadow isa StaggeredDiracOperator4D ? shadow : nothing
end

# Enzyme's structural shadow does not retain usable boundary phases. Rebuild a
# lattice view with primal metadata before exchanging the output cotangent.
@inline function _staggered_shadow_lattice(
    shadow::T, primal::T,
) where {T<:LatticeMatrix}
    return T(
        primal.nw, primal.phases, primal.NC1, primal.NC2, primal.gsize,
        primal.cart, primal.coords, primal.dims, primal.nbr,
        shadow.A, shadow.buf, shadow.buf_host, shadow.shift_buf_host,
        primal.mpi_transport,
        primal.myrank, primal.PN, primal.comm, primal.indexer,
        shadow.temps, shadow.halo_epoch,
    )
end

function ER.augmented_primal(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.mul!)},
    ::Type{RT},
    result::ER.Annotation{<:LatticeMatrix},
    operator::ER.Annotation{<:StaggeredDiracOperator4D},
    psi::ER.Annotation{<:LatticeMatrix},
) where RT
    result.val.nw == 0 && throw(ArgumentError(
        "Enzyme differentiation of StaggeredDiracOperator4D requires nw >= 1"))
    primal_return = LinearAlgebra.mul!(result.val, operator.val, psi.val)
    tape = nothing
    primal = ER.needs_primal(cfg) ? primal_return : nothing
    shadow = ER.needs_shadow(cfg) ? _getshadow(result.dval) : nothing
    RetT = ER.augmented_rule_return_type(cfg, RT, tape)
    return RetT(primal, shadow, tape)
end

function ER.reverse(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.mul!)},
    dresult_out, _tape,
    result::ER.Annotation{<:LatticeMatrix},
    operator::ER.Annotation{<:StaggeredDiracOperator4D},
    psi::ER.Annotation{<:LatticeMatrix},
)
    dresult = _getshadow_out(dresult_out, result)
    dresult isa LatticeMatrix || (dresult = _getshadow(result.dval))
    dresult isa LatticeMatrix || return (nothing, nothing, nothing)

    # Link and fermion pullbacks read the neighboring output cotangent.
    zero_halo_region!(dresult)
    set_halo!(_staggered_shadow_lattice(dresult, result.val))

    operator_shadow = _staggered_operator_shadow(operator)
    if operator_shadow !== nothing
        dU = operator_shadow.U
        length(dU) == 4 || throw(ArgumentError(
            "StaggeredDiracOperator4D shadow must contain four link fields"))
        all(link -> link isa LatticeMatrix, dU) || throw(ArgumentError(
            "StaggeredDiracOperator4D link shadows must be LatticeMatrix objects"))

        staggered_link_pullback!(
            dU, operator.val.U, dresult, psi.val,
        )
    end

    dpsi = hasproperty(psi, :dval) ? _getshadow(psi.dval) : nothing
    if dpsi isa LatticeMatrix
        temporary, temporary_index = get_block(psi.val.temps)
        U1, U2, U3, U4 = operator.val.U
        JACC.parallel_for(
            prod(psi.val.PN),
            kernel_StaggeredDiracOperator4D!,
            temporary, U1.A, U2.A, U3.A, U4.A,
            operator.val.mass, -one(operator.val.mass) / 2, dresult.A,
            Val(psi.val.NC1), Val(psi.val.nw), psi.val.indexer,
            psi.val.coords, psi.val.PN,
        )
        JACC.parallel_for(
            prod(psi.val.PN),
            kernel_add_4D!,
            dpsi.A, temporary, dpsi.indexer,
            Val(dpsi.NC1), Val(dpsi.NC2), one(eltype(dpsi.A)), Val(dpsi.nw),
        )
        unused!(psi.val.temps, temporary_index)
        mark_halo_dirty!(dpsi)
    end

    _zero_shadow!(dresult)
    zero_halo_region!(dresult)
    return (nothing, nothing, nothing)
end
