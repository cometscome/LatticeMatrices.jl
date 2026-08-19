import LatticeMatrices: WilsonDiracOperator4D, mark_halo_dirty!,
    kernel_adjoint_WilsonDiracOperator4D!, kernel_add_4D!, mul_op,
    _wilson_halfspin_link_pullback_direction3!

@inline function _wilson_operator_shadow(operator)
    hasproperty(operator, :dval) || return nothing
    shadow = getproperty(operator, :dval)
    shadow isa Base.RefValue && (shadow = shadow[])
    return shadow isa WilsonDiracOperator4D ? shadow : nothing
end

# Enzyme's structural shadow of a LatticeMatrix contains cotangents for numeric
# metadata as well as for `A`; in particular its boundary phases are zeros.
# Halo exchange must use the primal lattice metadata while writing the shadow
# storage.
@inline function _wilson_shadow_lattice(shadow::T, primal::T) where {T<:LatticeMatrix}
    return T(
        primal.nw, primal.phases, primal.NC1, primal.NC2, primal.gsize,
        primal.cart, primal.coords, primal.dims, primal.nbr,
        shadow.A, shadow.buf, shadow.buf_host, shadow.shift_buf_host,
        primal.mpi_transport,
        primal.myrank, primal.PN, primal.comm, primal.indexer,
        shadow.temps, shadow.halo_epoch,
    )
end

@inline function _kernel_wilson_link_pullback_direction!(
    dU, dresult, psi, indices, indices_plus, coefficient,
    ::Val{NC}, op_plus, op_minus,
) where NC
    @inbounds for row in 1:NC
        minus_psi = mul_op(op_minus, psi, row, indices)
        for col in 1:NC
            plus_psi = mul_op(op_plus, psi, col, indices_plus)
            value = zero(eltype(dU))
            for spin in 1:4
                # Forward hop:
                #   result_row(x) += coefficient * U[row,col](x) *
                #                    (1-gamma_mu)psi_col(x+mu)
                value += dresult[row, spin, indices...] * conj(plus_psi[spin])

                # Backward hop at x+mu:
                #   result_col(x+mu) += coefficient * conj(U[row,col](x)) *
                #                       (1+gamma_mu)psi_row(x)
                value += minus_psi[spin] *
                         conj(dresult[col, spin, indices_plus...])
            end
            dU[row, col, indices...] += coefficient * value
        end
    end
    return nothing
end

@inline function _kernel_wilson_link_pullback_direction!(
    dU, dresult, psi, indices, indices_plus, coefficient,
    ::Val{3}, ::LatticeMatrices.Oneγ{-1,MU}, ::LatticeMatrices.Oneγ{1,MU},
) where MU
    return _wilson_halfspin_link_pullback_direction3!(
        dU, dresult, psi, indices, indices_plus, coefficient, Val(MU))
end

@inline function _kernel_wilson_link_pullback!(
    site, dU1, dU2, dU3, dU4, dresult, psi, coefficient,
    ::Val{NC}, ::Val{nw}, indexer,
) where {NC,nw}
    indices = delinearize(indexer, site, nw)
    indices_1p = shiftindices(indices, LatticeMatrices.shift_1p)
    indices_2p = shiftindices(indices, LatticeMatrices.shift_2p)
    indices_3p = shiftindices(indices, LatticeMatrices.shift_3p)
    indices_4p = shiftindices(indices, LatticeMatrices.shift_4p)

    # Keep the four directions statically dispatched so this kernel remains
    # compilable by the accelerator backends supported through JACC.
    _kernel_wilson_link_pullback_direction!(
        dU1, dresult, psi, indices, indices_1p, coefficient,
        Val(NC), LatticeMatrices.oneminusγ1, LatticeMatrices.oneplusγ1,
    )
    _kernel_wilson_link_pullback_direction!(
        dU2, dresult, psi, indices, indices_2p, coefficient,
        Val(NC), LatticeMatrices.oneminusγ2, LatticeMatrices.oneplusγ2,
    )
    _kernel_wilson_link_pullback_direction!(
        dU3, dresult, psi, indices, indices_3p, coefficient,
        Val(NC), LatticeMatrices.oneminusγ3, LatticeMatrices.oneplusγ3,
    )
    _kernel_wilson_link_pullback_direction!(
        dU4, dresult, psi, indices, indices_4p, coefficient,
        Val(NC), LatticeMatrices.oneminusγ4, LatticeMatrices.oneplusγ4,
    )
    return nothing
end

function ER.augmented_primal(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.mul!)},
    ::Type{RT},
    result::ER.Annotation{<:LatticeMatrix},
    operator::ER.Annotation{<:WilsonDiracOperator4D},
    psi::ER.Annotation{<:LatticeMatrix},
) where RT
    result.val.nw == 0 && throw(ArgumentError(
        "Enzyme differentiation of WilsonDiracOperator4D requires nw >= 1"))
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
    operator::ER.Annotation{<:WilsonDiracOperator4D},
    psi::ER.Annotation{<:LatticeMatrix},
)
    dresult = _getshadow_out(dresult_out, result)
    dresult isa LatticeMatrix || (dresult = _getshadow(result.dval))
    dresult isa LatticeMatrix || return (nothing, nothing, nothing)

    # The reverse kernels read neighboring output cotangents.  Synchronize
    # explicitly because lower-level Enzyme rules write shadow arrays directly
    # and therefore do not advance their halo epoch.
    zero_halo_region!(dresult)
    set_halo!(_wilson_shadow_lattice(dresult, result.val))

    operator_shadow = _wilson_operator_shadow(operator)
    if operator_shadow !== nothing
        dU = operator_shadow.U
        length(dU) == 4 || throw(ArgumentError(
            "WilsonDiracOperator4D shadow must contain four link fields"))
        all(link -> link isa LatticeMatrix, dU) || throw(ArgumentError(
            "WilsonDiracOperator4D link shadows must be LatticeMatrix objects"))

        JACC.parallel_for(
            prod(result.val.PN),
            _kernel_wilson_link_pullback!,
            dU[1].A, dU[2].A, dU[3].A, dU[4].A,
            dresult.A, psi.val.A, -operator.val.κ,
            Val(result.val.NC1), Val(result.val.nw), result.val.indexer,
        )
        mark_halo_dirty!.(dU)
    end

    dpsi = hasproperty(psi, :dval) ? _getshadow(psi.dval) : nothing
    if dpsi isa LatticeMatrix
        temporary, temporary_index = get_block(psi.val.temps)
        U1, U2, U3, U4 = operator.val.U
        JACC.parallel_for(
            prod(psi.val.PN),
            kernel_adjoint_WilsonDiracOperator4D!,
            temporary, U1.A, U2.A, U3.A, U4.A, operator.val.κ, dresult.A,
            Val(psi.val.NC1), Val(psi.val.nw), psi.val.indexer,
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
