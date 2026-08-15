import LatticeMatrices: HISQDiracOperator4D, mark_halo_dirty!,
    kernel_HISQDiracOperator4D!, kernel_add_4D!,
    staggered_eta_global_halo, hisq_long_shifts_p

@inline function _hisq_operator_shadow(operator)
    hasproperty(operator, :dval) || return nothing
    shadow = getproperty(operator, :dval)
    shadow isa Base.RefValue && (shadow = shadow[])
    return shadow isa HISQDiracOperator4D ? shadow : nothing
end

@inline function _kernel_hisq_link_pullback!(
    site,
    dX1, dX2, dX3, dX4, dL1, dL2, dL3, dL4,
    dresult, psi, fat_coefficient, long_coefficient,
    ::Val{NC}, ::Val{nw}, indexer, mpi_coordinates, local_size,
) where {NC,nw}
    x = delinearize(indexer, site, nw)
    x1p = shiftindices(x, LatticeMatrices.shift_1p)
    x2p = shiftindices(x, LatticeMatrices.shift_2p)
    x3p = shiftindices(x, LatticeMatrices.shift_3p)
    x4p = shiftindices(x, LatticeMatrices.shift_4p)
    x1p3 = shiftindices(x, hisq_long_shifts_p[1])
    x2p3 = shiftindices(x, hisq_long_shifts_p[2])
    x3p3 = shiftindices(x, hisq_long_shifts_p[3])
    x4p3 = shiftindices(x, hisq_long_shifts_p[4])

    eta2 = staggered_eta_global_halo(
        x, 2, nw, mpi_coordinates, local_size)
    eta3 = staggered_eta_global_halo(
        x, 3, nw, mpi_coordinates, local_size)
    eta4 = staggered_eta_global_halo(
        x, 4, nw, mpi_coordinates, local_size)

    _kernel_staggered_link_pullback_direction!(
        dX1, dresult, psi, x, x1p, fat_coefficient, 1, Val(NC))
    _kernel_staggered_link_pullback_direction!(
        dX2, dresult, psi, x, x2p, fat_coefficient, eta2, Val(NC))
    _kernel_staggered_link_pullback_direction!(
        dX3, dresult, psi, x, x3p, fat_coefficient, eta3, Val(NC))
    _kernel_staggered_link_pullback_direction!(
        dX4, dresult, psi, x, x4p, fat_coefficient, eta4, Val(NC))

    _kernel_staggered_link_pullback_direction!(
        dL1, dresult, psi, x, x1p3, long_coefficient, 1, Val(NC))
    _kernel_staggered_link_pullback_direction!(
        dL2, dresult, psi, x, x2p3, long_coefficient, eta2, Val(NC))
    _kernel_staggered_link_pullback_direction!(
        dL3, dresult, psi, x, x3p3, long_coefficient, eta3, Val(NC))
    _kernel_staggered_link_pullback_direction!(
        dL4, dresult, psi, x, x4p3, long_coefficient, eta4, Val(NC))
    return nothing
end

function ER.augmented_primal(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.mul!)},
    ::Type{RT},
    result::ER.Annotation{<:LatticeMatrix},
    operator::ER.Annotation{<:HISQDiracOperator4D},
    psi::ER.Annotation{<:LatticeMatrix},
) where RT
    result.val.nw < 3 && throw(ArgumentError(
        "Enzyme differentiation of HISQDiracOperator4D requires nw >= 3"))
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
    operator::ER.Annotation{<:HISQDiracOperator4D},
    psi::ER.Annotation{<:LatticeMatrix},
)
    dresult = _getshadow_out(dresult_out, result)
    dresult isa LatticeMatrix || (dresult = _getshadow(result.dval))
    dresult isa LatticeMatrix || return (nothing, nothing, nothing)

    # Both the one-link and three-link pullbacks read neighboring output
    # cotangents. Rebuild the shadow view with the primal boundary phases.
    zero_halo_region!(dresult)
    set_halo!(_staggered_shadow_lattice(dresult, result.val))

    fat_coefficient = one(operator.val.mass) / 2
    long_coefficient =
        -(one(operator.val.naik_epsilon) + operator.val.naik_epsilon) / 48

    operator_shadow = _hisq_operator_shadow(operator)
    if operator_shadow !== nothing
        dX = operator_shadow.links.fat_links
        dL = operator_shadow.links.long_links
        length(dX) == 4 && length(dL) == 4 || throw(ArgumentError(
            "HISQDiracOperator4D shadow must contain four X and four L fields"))
        all(link -> link isa LatticeMatrix, dX) &&
            all(link -> link isa LatticeMatrix, dL) || throw(ArgumentError(
            "HISQDiracOperator4D link shadows must be LatticeMatrix objects"))

        JACC.parallel_for(
            prod(result.val.PN),
            _kernel_hisq_link_pullback!,
            dX[1].A, dX[2].A, dX[3].A, dX[4].A,
            dL[1].A, dL[2].A, dL[3].A, dL[4].A,
            dresult.A, psi.val.A, fat_coefficient, long_coefficient,
            Val(result.val.NC1), Val(result.val.nw), result.val.indexer,
            result.val.coords, result.val.PN,
        )
        mark_halo_dirty!.(dX)
        mark_halo_dirty!.(dL)
    end

    dpsi = hasproperty(psi, :dval) ? _getshadow(psi.dval) : nothing
    if dpsi isa LatticeMatrix
        temporary, temporary_index = get_block(psi.val.temps)
        X = operator.val.links.fat_links
        L = operator.val.links.long_links
        JACC.parallel_for(
            prod(psi.val.PN),
            kernel_HISQDiracOperator4D!,
            temporary,
            X[1].A, X[2].A, X[3].A, X[4].A,
            L[1].A, L[2].A, L[3].A, L[4].A,
            operator.val.mass, -fat_coefficient, -long_coefficient,
            dresult.A, Val(psi.val.NC1), Val(psi.val.nw),
            psi.val.indexer, psi.val.coords, psi.val.PN,
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
