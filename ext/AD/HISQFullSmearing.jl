import LatticeMatrices: HISQDiracCache4D,
    hisq_project_u3!, hisq_naik_links!, mul_cached_hisq!,
    mark_halo_dirty!, HaloEpoch, _record_hisq_cache_state!,
    _hisq_project_un_pullback_accumulate!

@inline function _kernel_hisq_naik_pullback!(
    combined_index, dW1, dW2, dW3, dW4,
    dL1, dL2, dL3, dL4, W1, W2, W3, W4,
    volume, ::Val{NC}, ::Val{nw}, indexer,
) where {NC,nw}
    site_index, row, column, axis = _hisq_pullback_element_indices(
        combined_index, volume, Val(NC))
    target = delinearize(indexer, site_index, nw)
    T = eltype(dW1)
    gradient = _hisq_fat7_path_pullback_element(
        dL1, dL2, dL3, dL4, W1, W2, W3, W4,
        target, (axis, axis, axis), one(T), axis,
        axis, row, column, T, Val(NC))
    _hisq_add_pullback_element!(
        dW1, dW2, dW3, dW4, axis, row, column, target, gradient)
    return nothing
end

function ER.augmented_primal(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(hisq_project_u3!)},
    ::Type{RT},
    projected_links::ER.Annotation{<:Union{AbstractVector,Tuple}},
    fat_links::ER.Annotation{<:Union{AbstractVector,Tuple}},
) where RT
    primal_return = hisq_project_u3!(projected_links.val, fat_links.val)
    tape = nothing
    primal = ER.needs_primal(cfg) ? primal_return : nothing
    shadow = ER.needs_shadow(cfg) ?
        _hisq_smearing_vector_shadow(projected_links) : nothing
    RetT = ER.augmented_rule_return_type(cfg, RT, tape)
    return RetT(primal, shadow, tape)
end

function ER.reverse(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(hisq_project_u3!)},
    _dresult_out, _tape,
    projected_links::ER.Annotation{<:Union{AbstractVector,Tuple}},
    fat_links::ER.Annotation{<:Union{AbstractVector,Tuple}},
)
    dprojected = _hisq_smearing_vector_shadow(projected_links)
    dprojected isa Union{AbstractVector,Tuple} || return (nothing, nothing)
    dfat = _hisq_smearing_vector_shadow(fat_links)
    _hisq_project_un_pullback!(dfat, dprojected, fat_links.val)
    return (nothing, nothing)
end

function _hisq_project_un_pullback!(dfat, dprojected, fat_links)
    length(dprojected) == 4 &&
        all(link -> link isa LatticeMatrix, dprojected) ||
        throw(ArgumentError(
            "HISQ U(N) projection output shadow must contain four lattice fields"))
    if dfat isa Union{AbstractVector,Tuple}
        length(dfat) == 4 && all(link -> link isa LatticeMatrix, dfat) ||
            throw(ArgumentError(
                "HISQ U(N) projection input shadow must contain four lattice fields"))
        _hisq_project_un_pullback_accumulate!(
            dfat, dprojected, fat_links)
    end
    for link in dprojected
        _zero_shadow!(link)
        zero_halo_region!(link)
    end
    return nothing
end


function ER.augmented_primal(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(hisq_naik_links!)},
    ::Type{RT},
    long_links::ER.Annotation{<:Union{AbstractVector,Tuple}},
    reunitarized_links::ER.Annotation{<:Union{AbstractVector,Tuple}},
) where RT
    reunitarized_links.val[1].nw < 2 && throw(ArgumentError(
        "Enzyme differentiation of hisq_naik_links! requires nw >= 2"))
    primal_return = hisq_naik_links!(
        long_links.val, reunitarized_links.val)
    tape = nothing
    primal = ER.needs_primal(cfg) ? primal_return : nothing
    shadow = ER.needs_shadow(cfg) ?
        _hisq_smearing_vector_shadow(long_links) : nothing
    RetT = ER.augmented_rule_return_type(cfg, RT, tape)
    return RetT(primal, shadow, tape)
end

function ER.reverse(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(hisq_naik_links!)},
    _dresult_out, _tape,
    long_links::ER.Annotation{<:Union{AbstractVector,Tuple}},
    reunitarized_links::ER.Annotation{<:Union{AbstractVector,Tuple}},
)
    dlong = _hisq_smearing_vector_shadow(long_links)
    dlong isa Union{AbstractVector,Tuple} || return (nothing, nothing)
    dinput = _hisq_smearing_vector_shadow(reunitarized_links)
    _hisq_naik_pullback!(
        dinput, dlong, reunitarized_links.val, long_links.val)
    return (nothing, nothing)
end

function _hisq_naik_pullback!(dinput, dlong, reunitarized_links, long_links)
    length(dlong) == 4 && all(link -> link isa LatticeMatrix, dlong) ||
        throw(ArgumentError(
            "hisq_naik_links! output shadow must contain four lattice fields"))
    for mu in 1:4
        zero_halo_region!(dlong[mu])
        set_halo!(_staggered_shadow_lattice(dlong[mu], long_links[mu]))
    end

    if dinput isa Union{AbstractVector,Tuple}
        length(dinput) == 4 && all(link -> link isa LatticeMatrix, dinput) ||
            throw(ArgumentError(
                "hisq_naik_links! input shadow must contain four lattice fields"))
        W = reunitarized_links
        volume = prod(W[1].PN)
        NC = W[1].NC1
        _hisq_parallel_for(
            4 * NC * NC * volume, _kernel_hisq_naik_pullback!,
            dinput[1].A, dinput[2].A, dinput[3].A, dinput[4].A,
            dlong[1].A, dlong[2].A, dlong[3].A, dlong[4].A,
            W[1].A, W[2].A, W[3].A, W[4].A,
            volume, Val(NC), Val(W[1].nw), W[1].indexer)
        mark_halo_dirty!.(dinput)
    end
    for link in dlong
        _zero_shadow!(link)
        zero_halo_region!(link)
    end
    return nothing
end

@inline function _hisq_cache_scratch_lattice(
    array, primal::T,
) where {T<:LatticeMatrix}
    return T(
        primal.nw, primal.phases, primal.NC1, primal.NC2, primal.gsize,
        primal.cart, primal.coords, primal.dims, primal.nbr,
        array, primal.buf, primal.buf_host, primal.shift_buf_host,
        primal.mpi_transport,
        primal.myrank, primal.PN, primal.comm, primal.indexer,
        primal.temps, HaloEpoch(),
    )
end

function _reserve_hisq_cache_vector_scratch(fields)
    blocks = ntuple(mu -> get_block(fields[mu].temps), Val(4))
    scratch = ntuple(
        mu -> _hisq_cache_scratch_lattice(blocks[mu][1], fields[mu]),
        Val(4))
    indices = ntuple(mu -> blocks[mu][2], Val(4))
    return scratch, indices
end

function _reserve_hisq_cache_pullback_scratch(cache::HISQDiracCache4D)
    dFat, fat_indices = _reserve_hisq_cache_vector_scratch(cache.fat_links)
    dLong, long_indices = _reserve_hisq_cache_vector_scratch(cache.long_links)
    dReunit, reunit_indices =
        _reserve_hisq_cache_vector_scratch(cache.reunitarized_links)
    return (dFat, dLong, dReunit),
           (fat_indices, long_indices, reunit_indices)
end

function _release_hisq_cache_pullback_scratch!(
    cache::HISQDiracCache4D, indices,
)
    collections = (
        cache.fat_links, cache.long_links, cache.reunitarized_links)
    for collection_index in 1:3, mu in 1:4
        unused!(
            collections[collection_index][mu].temps,
            indices[collection_index][mu])
    end
    return nothing
end

@inline function _explicit_hisq_link_shadow(annotation)
    hasproperty(annotation, :dval) || return nothing
    shadow = _getshadow(getproperty(annotation, :dval))
    return shadow isa LatticeMatrix ? shadow : nothing
end

function ER.augmented_primal(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_cached_hisq!)},
    ::Type{RT},
    result::ER.Annotation{<:LatticeMatrix},
    cache::ER.Annotation{<:HISQDiracCache4D},
    U1::ER.Annotation{<:LatticeMatrix},
    U2::ER.Annotation{<:LatticeMatrix},
    U3::ER.Annotation{<:LatticeMatrix},
    U4::ER.Annotation{<:LatticeMatrix},
    psi::ER.Annotation{<:LatticeMatrix},
) where RT
    result.val.nw < 3 && throw(ArgumentError(
        "Enzyme differentiation of mul_cached_hisq! requires nw >= 3"))
    primal_return = mul_cached_hisq!(
        result.val, cache.val, U1.val, U2.val, U3.val, U4.val, psi.val)

    link_shadows = (
        _explicit_hisq_link_shadow(U1),
        _explicit_hisq_link_shadow(U2),
        _explicit_hisq_link_shadow(U3),
        _explicit_hisq_link_shadow(U4),
    )
    any_link_active = any(link -> link isa LatticeMatrix, link_shadows)
    all_links_active = all(link -> link isa LatticeMatrix, link_shadows)
    any_link_active == all_links_active || throw(ArgumentError(
        "mul_cached_hisq! requires all four thin links to be active together"))
    scratch_tape = all_links_active ?
        _reserve_hisq_cache_pullback_scratch(cache.val) : nothing

    primal = ER.needs_primal(cfg) ? primal_return : nothing
    shadow = ER.needs_shadow(cfg) ? _getshadow(result.dval) : nothing
    RetT = ER.augmented_rule_return_type(cfg, RT, scratch_tape)
    return RetT(primal, shadow, scratch_tape)
end

function ER.reverse(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_cached_hisq!)},
    dresult_out, scratch_tape,
    result::ER.Annotation{<:LatticeMatrix},
    cache::ER.Annotation{<:HISQDiracCache4D},
    U1::ER.Annotation{<:LatticeMatrix},
    U2::ER.Annotation{<:LatticeMatrix},
    U3::ER.Annotation{<:LatticeMatrix},
    U4::ER.Annotation{<:LatticeMatrix},
    psi::ER.Annotation{<:LatticeMatrix},
)
    dresult = _getshadow_out(dresult_out, result)
    dresult isa LatticeMatrix || (dresult = _getshadow(result.dval))
    if !(dresult isa LatticeMatrix)
        scratch_tape === nothing ||
            _release_hisq_cache_pullback_scratch!(
                cache.val, scratch_tape[2])
        return (nothing, nothing, nothing, nothing, nothing, nothing, nothing)
    end

    zero_halo_region!(dresult)
    set_halo!(_staggered_shadow_lattice(dresult, result.val))

    dU = (
        _explicit_hisq_link_shadow(U1),
        _explicit_hisq_link_shadow(U2),
        _explicit_hisq_link_shadow(U3),
        _explicit_hisq_link_shadow(U4),
    )
    links_active = all(link -> link isa LatticeMatrix, dU)
    U = (U1.val, U2.val, U3.val, U4.val)
    primal = cache.val

    if links_active
        for mu in 1:4
            dU[mu].A === U[mu].A && throw(ArgumentError(
                "mul_cached_hisq! thin-link shadow aliases its primal link"))
        end

        (dFat, dLong, dReunit), scratch_indices = scratch_tape
        for fields in (dFat, dLong, dReunit), field in fields
            _zero_shadow!(field)
            zero_halo_region!(field)
        end

        operator = primal.operator
        fat_coefficient = one(operator.mass) / 2
        long_coefficient =
            -(one(operator.naik_epsilon) + operator.naik_epsilon) / 48
        JACC.parallel_for(
            prod(result.val.PN), _kernel_hisq_link_pullback!,
            dFat[1].A, dFat[2].A, dFat[3].A, dFat[4].A,
            dLong[1].A, dLong[2].A, dLong[3].A, dLong[4].A,
            dresult.A, psi.val.A, fat_coefficient, long_coefficient,
            Val(result.val.NC1), Val(result.val.nw), result.val.indexer,
            result.val.coords, result.val.PN,
        )
        mark_halo_dirty!.(dFat)
        mark_halo_dirty!.(dLong)

        _hisq_naik_pullback!(
            dReunit, dLong, primal.reunitarized_links, primal.long_links)

        real_type = typeof(real(zero(eltype(U1.val.A))))
        epsilon = convert(real_type, operator.naik_epsilon)
        level2_coefficients = (
            one(real_type) + epsilon / 8,
            one(real_type) / 16,
            one(real_type) / 64,
            one(real_type) / 384,
            -one(real_type) / 8,
        )
        _hisq_fat7_pullback!(
            dReunit, dFat, primal.reunitarized_links, primal.fat_links,
            level2_coefficients, "mul_cached_hisq! level-2 Fat7")

        # dFat has been consumed and cleared above, so reuse it for the
        # level-1-link cotangent produced by the projection pullback.
        _hisq_project_un_pullback!(
            dFat, dReunit, primal.level1_links)

        level1_coefficients = (
            one(real_type) / 8,
            one(real_type) / 16,
            one(real_type) / 64,
            one(real_type) / 384,
            zero(real_type),
        )
        _hisq_fat7_pullback!(
            dU, dFat, U, primal.level1_links,
            level1_coefficients, "mul_cached_hisq! level-1 Fat7")

        _release_hisq_cache_pullback_scratch!(primal, scratch_indices)
    end

    dpsi = hasproperty(psi, :dval) ? _getshadow(psi.dval) : nothing
    if dpsi isa LatticeMatrix
        temporary, temporary_index = get_block(psi.val.temps)
        operator = primal.operator
        X = primal.fat_links
        L = primal.long_links
        fat_coefficient = one(operator.mass) / 2
        long_coefficient =
            -(one(operator.naik_epsilon) + operator.naik_epsilon) / 48
        JACC.parallel_for(
            prod(psi.val.PN), kernel_HISQDiracOperator4D!,
            temporary,
            X[1].A, X[2].A, X[3].A, X[4].A,
            L[1].A, L[2].A, L[3].A, L[4].A,
            operator.mass, -fat_coefficient, -long_coefficient,
            dresult.A, Val(psi.val.NC1), Val(psi.val.nw),
            psi.val.indexer, psi.val.coords, psi.val.PN,
        )
        JACC.parallel_for(
            prod(psi.val.PN), kernel_add_4D!,
            dpsi.A, temporary, dpsi.indexer,
            Val(dpsi.NC1), Val(dpsi.NC2),
            one(eltype(dpsi.A)), Val(dpsi.nw),
        )
        unused!(psi.val.temps, temporary_index)
        mark_halo_dirty!(dpsi)
    end

    _zero_shadow!(dresult)
    zero_halo_region!(dresult)
    _record_hisq_cache_state!(primal, U)
    return (nothing, nothing, nothing, nothing, nothing, nothing, nothing)
end
