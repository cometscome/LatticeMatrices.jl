@inline function _kernel_sun_phase_correct!(
    site_index, output, ::Val{2}, ::Val{nw}, indexer,
) where nw
    site = delinearize(indexer, site_index, nw)
    determinant =
        output[1, 1, site...] * output[2, 2, site...] -
        output[1, 2, site...] * output[2, 1, site...]
    RT = typeof(real(determinant))
    angle = atan(imag(determinant), real(determinant))
    sine, cosine = sincos(-angle / RT(2))
    phase = complex(cosine, sine)
    @inbounds for column in 1:2, row in 1:2
        output[row, column, site...] *= phase
    end
    return nothing
end

@inline function _kernel_sun_phase_correct!(
    site_index, output, ::Val{3}, ::Val{nw}, indexer,
) where nw
    site = delinearize(indexer, site_index, nw)
    a11 = output[1, 1, site...]
    a12 = output[1, 2, site...]
    a13 = output[1, 3, site...]
    a21 = output[2, 1, site...]
    a22 = output[2, 2, site...]
    a23 = output[2, 3, site...]
    a31 = output[3, 1, site...]
    a32 = output[3, 2, site...]
    a33 = output[3, 3, site...]
    determinant =
        a11 * (a22 * a33 - a23 * a32) -
        a12 * (a21 * a33 - a23 * a31) +
        a13 * (a21 * a32 - a22 * a31)
    RT = typeof(real(determinant))
    angle = atan(imag(determinant), real(determinant))
    sine, cosine = sincos(-angle / RT(3))
    phase = complex(cosine, sine)
    @inbounds for column in 1:3, row in 1:3
        output[row, column, site...] *= phase
    end
    return nothing
end

"""
    _sun_polar_project_field!(output, input)

Project each nonsingular input matrix first to U(N) by polar decomposition,
then apply the principal determinant phase to obtain SU(N).  The branch is
explicit and deterministic.  Its analytic pullback is defined away from the
negative-real determinant branch cut.
"""
function _sun_polar_project_field!(output, input)
    NC = input.NC1
    NC in (2, 3) || throw(ArgumentError(
        "principal-branch SU(N) polar projection supports only N=2 or N=3"))
    _nhyp_project_field!(output, input; sync=false)
    _parallel_for_mutating!(
        output, prod(output.PN), _kernel_sun_phase_correct!,
        output.A, Val(NC), Val(output.nw), output.indexer)
    return output
end

@inline function _sun_identity_matrix!(matrix, ::Val{NC}) where NC
    @inbounds for column in 1:NC, row in 1:NC
        matrix[row, column] = ifelse(
            row == column, one(eltype(matrix)), zero(eltype(matrix)))
    end
    return matrix
end

@inline function _sun_copy_matrix!(destination, source, ::Val{NC}) where NC
    @inbounds for column in 1:NC, row in 1:NC
        destination[row, column] = source[row, column]
    end
    return destination
end

@inline function _sun_max_retr_su2_rotation!(
    rotation, matrix, first_index, second_index, ::Val{NC},
) where NC
    _sun_identity_matrix!(rotation, Val(NC))
    a11 = matrix[first_index, first_index]
    a12 = matrix[first_index, second_index]
    a21 = matrix[second_index, first_index]
    a22 = matrix[second_index, second_index]
    normalization_squared =
        real(a11)^2 + imag(a11)^2 +
        2 * real(a11) * real(a22) +
        real(a12)^2 + imag(a12)^2 -
        2 * imag(a11) * imag(a22) +
        real(a21)^2 + imag(a21)^2 -
        2 * real(a12) * real(a21) +
        real(a22)^2 + imag(a22)^2 +
        2 * imag(a12) * imag(a21)
    inverse_normalization = inv(sqrt(normalization_squared))
    rotation[first_index, first_index] = complex(
        real(a11) + real(a22), -imag(a11) + imag(a22)) *
        inverse_normalization
    rotation[first_index, second_index] = complex(
        real(a21) - real(a12), -imag(a21) - imag(a12)) *
        inverse_normalization
    rotation[second_index, first_index] = complex(
        real(a12) - real(a21), -imag(a12) - imag(a21)) *
        inverse_normalization
    rotation[second_index, second_index] = complex(
        real(a11) + real(a22), imag(a11) - imag(a22)) *
        inverse_normalization
    return rotation
end

@inline function _kernel_sun_max_retr_initialize!(
    site_index, output, ::Val{NC}, ::Val{nw}, indexer,
) where {NC,nw}
    site = delinearize(indexer, site_index, nw)
    @inbounds for column in 1:NC, row in 1:NC
        output[row, column, site...] = ifelse(
            row == column, one(eltype(output)), zero(eltype(output)))
    end
    return nothing
end

@inline function _kernel_sun_max_retr_iteration!(
    site_index, maximizer_field, matrix_field, trace_field,
    ::Val{NC}, ::Val{nw}, indexer,
) where {NC,nw}
    site = delinearize(indexer, site_index, nw)
    element_type = eltype(matrix_field)
    matrix = MMatrix{NC,NC,element_type}(undef)
    maximizer = MMatrix{NC,NC,element_type}(undef)
    iteration_delta = MMatrix{NC,NC,element_type}(undef)
    rotation = MMatrix{NC,NC,element_type}(undef)
    temporary = MMatrix{NC,NC,element_type}(undef)
    @inbounds for column in 1:NC, row in 1:NC
        matrix[row, column] = matrix_field[row, column, site...]
        maximizer[row, column] = maximizer_field[row, column, site...]
    end
    _sun_identity_matrix!(iteration_delta, Val(NC))
    for first_index in 1:NC
        second_index = first_index == NC ? 1 : first_index + 1
        _sun_max_retr_su2_rotation!(
            rotation, matrix, first_index, second_index, Val(NC))
        gemm!(temporary, matrix, rotation)
        _sun_copy_matrix!(matrix, temporary, Val(NC))
        gemm!(temporary, maximizer, rotation)
        _sun_copy_matrix!(maximizer, temporary, Val(NC))
        gemm!(temporary, iteration_delta, rotation)
        _sun_copy_matrix!(iteration_delta, temporary, Val(NC))
    end

    @inbounds for column in 1:NC, row in 1:NC
        matrix_field[row, column, site...] = matrix[row, column]
        maximizer_field[row, column, site...] = maximizer[row, column]
    end
    real_trace = zero(real(zero(element_type)))
    @inbounds for color in 1:NC
        real_trace += real(iteration_delta[color, color])
    end
    trace_field[1, 1, site...] = real_trace
    return nothing
end

@inline function _kernel_sun_max_retr_trace(
    site_index, trace_field, ::Val{nw}, indexer,
) where nw
    site = delinearize(indexer, site_index, nw)
    return real(trace_field[1, 1, site...])
end

@inline function _kernel_sun_max_retr_finish!(
    site_index, output, ::Val{NC}, ::Val{nw}, indexer,
) where {NC,nw}
    site = delinearize(indexer, site_index, nw)
    matrix = MMatrix{NC,NC,eltype(output)}(undef)
    @inbounds for column in 1:NC, row in 1:NC
        matrix[row, column] = output[row, column, site...]
    end
    @inbounds for column in 1:NC, row in 1:NC
        output[row, column, site...] = conj(matrix[column, row])
    end
    return nothing
end

function _sun_max_retr_project_field!(
    output, input, scratch, maximum_iterations, convergence_tolerance,
)
    NC = input.NC1
    NC in (2, 3) || throw(ArgumentError(
        "MaxReTr SU(N) projection supports only N=2 or N=3"))
    _parallel_for_mutating!(
        output, prod(output.PN), _kernel_sun_max_retr_initialize!,
        output.A, Val(NC), Val(output.nw), output.indexer)
    mark_halo_dirty!(scratch)
    converged = false
    real_type = typeof(real(zero(eltype(input.A))))
    for _ in 1:maximum_iterations
        _parallel_for_mutating!(
            input, prod(input.PN), _kernel_sun_max_retr_iteration!,
            output.A, input.A, scratch.A,
            Val(NC), Val(input.nw), input.indexer)
        local_trace = JACC.parallel_reduce(
            prod(input.PN), _kernel_sun_max_retr_trace,
            scratch.A, Val(scratch.nw), scratch.indexer;
            init=zero(real_type), op=+)
        global_trace = _allreduce_sum(local_trace, input.comm)
        delta = one(real_type) - global_trace / (NC * prod(input.gsize))
        if delta < convergence_tolerance
            converged = true
            break
        end
    end
    converged || throw(ErrorException(
        "MaxReTr SU(N) projection did not converge in " *
        "$maximum_iterations iterations"))
    _parallel_for_mutating!(
        output, prod(output.PN), _kernel_sun_max_retr_finish!,
        output.A,
        Val(NC), Val(output.nw), output.indexer)
    return output
end

function _sun_project_field!(output, input, scratch, parameters)
    if parameters.projection === :polar
        return _sun_polar_project_field!(output, input)
    end
    return _sun_max_retr_project_field!(
        output, input, scratch, parameters.max_retr_iterations,
        parameters.max_retr_tolerance)
end

@inline function _sun_determinant(matrix, ::Val{2})
    return matrix[1, 1] * matrix[2, 2] -
        matrix[1, 2] * matrix[2, 1]
end

@inline function _sun_determinant(matrix, ::Val{3})
    return matrix[1, 1] * (matrix[2, 2] * matrix[3, 3] -
                           matrix[2, 3] * matrix[3, 2]) -
        matrix[1, 2] * (matrix[2, 1] * matrix[3, 3] -
                        matrix[2, 3] * matrix[3, 1]) +
        matrix[1, 3] * (matrix[2, 1] * matrix[3, 2] -
                        matrix[2, 2] * matrix[3, 1])
end

# Pull back V = s Q, where Q is unitary and
# s = exp(-i arg(det(Q)) / NC).  With the real Frobenius pairing,
# dQ_bar = conj(s) dV_bar + i Q Im(tr(dV_bar' V)) / NC.
@inline function _sun_phase_pullback_matrix!(
    output_cotangent, projected, ::Val{NC},
) where NC
    determinant = _sun_determinant(projected, Val(NC))
    RT = typeof(real(determinant))
    angle = atan(imag(determinant), real(determinant))
    sine, cosine = sincos(-angle / RT(NC))
    phase = complex(cosine, sine)
    overlap = zero(eltype(output_cotangent))
    @inbounds for column in 1:NC, row in 1:NC
        overlap += conj(output_cotangent[row, column]) *
            (phase * projected[row, column])
    end
    phase_coefficient = complex(zero(RT), imag(overlap) / RT(NC))
    @inbounds for column in 1:NC, row in 1:NC
        output_cotangent[row, column] =
            conj(phase) * output_cotangent[row, column] +
            phase_coefficient * projected[row, column]
    end
    return output_cotangent
end

@inline function _kernel_sun_project_u3_pullback_core!(
    site_index, dinput, doutput, input, ::Val{nw}, indexer,
) where nw
    site = delinearize(indexer, site_index, nw)
    element_type = eltype(input)
    V = MMatrix{3,3,element_type}(undef)
    Q = MMatrix{3,3,element_type}(undef)
    Q2 = MMatrix{3,3,element_type}(undef)
    inverse_sqrt = MMatrix{3,3,element_type}(undef)
    hermitian = MMatrix{3,3,element_type}(undef)
    projected = MMatrix{3,3,element_type}(undef)
    output_cotangent = MMatrix{3,3,element_type}(undef)
    skew_rhs = MMatrix{3,3,element_type}(undef)
    sylvester_solution = MMatrix{3,3,element_type}(undef)
    gradient = MMatrix{3,3,element_type}(undef)

    @inbounds for column in 1:3, row in 1:3
        V[row, column] = input[row, column, site...]
        output_cotangent[row, column] = doutput[row, column, site...]
    end
    _hisq_u3_project_matrix!(
        projected, V, Q, Q2, inverse_sqrt, hermitian)
    _sun_phase_pullback_matrix!(output_cotangent, projected, Val(3))

    @inbounds for column in 1:3, row in 1:3
        value = zero(element_type)
        adjoint_value = zero(element_type)
        for contracted in 1:3
            value += conj(projected[contracted, row]) *
                output_cotangent[contracted, column]
            adjoint_value += conj(output_cotangent[contracted, row]) *
                projected[contracted, column]
        end
        skew_rhs[row, column] = value - adjoint_value
    end
    _hisq_pullback_solve_sylvester_3x3!(
        sylvester_solution, hermitian, skew_rhs, Q, Q2, inverse_sqrt)
    gemm!(gradient, projected, sylvester_solution)
    @inbounds for column in 1:3, row in 1:3
        dinput[row, column, site...] += gradient[row, column]
    end
    return nothing
end

@inline function _kernel_sun_project_un_pullback_core!(
    site_index, dinput, doutput, input,
    ::Val{NC}, ::Val{nw}, indexer,
) where {NC,nw}
    site = delinearize(indexer, site_index, nw)
    element_type = eltype(input)
    V = MMatrix{NC,NC,element_type}(undef)
    projected = MMatrix{NC,NC,element_type}(undef)
    hermitian = MMatrix{NC,NC,element_type}(undef)
    polar_work = MMatrix{NC,NC,element_type}(undef)
    inverse = MMatrix{NC,NC,element_type}(undef)
    next = MMatrix{NC,NC,element_type}(undef)
    output_cotangent = MMatrix{NC,NC,element_type}(undef)
    skew_rhs = MMatrix{NC,NC,element_type}(undef)
    sylvester_solution = MMatrix{NC,NC,element_type}(undef)
    gradient = MMatrix{NC,NC,element_type}(undef)
    system = MMatrix{NC * NC,NC * NC,element_type}(undef)
    vector = MVector{NC * NC,element_type}(undef)

    @inbounds for column in 1:NC, row in 1:NC
        V[row, column] = input[row, column, site...]
        output_cotangent[row, column] = doutput[row, column, site...]
    end
    _hisq_un_project_matrix!(
        projected, hermitian, V, polar_work, inverse, next, Val(NC))
    _sun_phase_pullback_matrix!(output_cotangent, projected, Val(NC))

    @inbounds for column in 1:NC, row in 1:NC
        value = zero(element_type)
        adjoint_value = zero(element_type)
        for contracted in 1:NC
            value += conj(projected[contracted, row]) *
                output_cotangent[contracted, column]
            adjoint_value += conj(output_cotangent[contracted, row]) *
                projected[contracted, column]
        end
        skew_rhs[row, column] = value - adjoint_value
    end
    _hisq_pullback_solve_sylvester!(
        sylvester_solution, hermitian, skew_rhs,
        system, vector, Val(NC))
    gemm!(gradient, projected, sylvester_solution)
    @inbounds for column in 1:NC, row in 1:NC
        dinput[row, column, site...] += gradient[row, column]
    end
    return nothing
end

function _sun_project_pullback_accumulate!(dinput, doutput, input)
    NC = input.NC1
    if NC == 3
        _parallel_for_mutating!(
            dinput, prod(input.PN),
            _kernel_sun_project_u3_pullback_core!,
            dinput.A, doutput.A, input.A, Val(input.nw), input.indexer)
    else
        _parallel_for_mutating!(
            dinput, prod(input.PN),
            _kernel_sun_project_un_pullback_core!,
            dinput.A, doutput.A, input.A,
            Val(NC), Val(input.nw), input.indexer)
    end
    return dinput
end

"""
    APEParameters(alpha=0.6; projection=:max_retr,
                  max_retr_iterations=1000, max_retr_tolerance=1e-14)

Parameters for one four-dimensional APE step. `:max_retr` is the standard
iterative projection used for interoperability. Select `:polar` when an
analytic pullback is required.
"""
struct APEParameters{T<:AbstractFloat}
    alpha::T
    projection::Symbol
    max_retr_iterations::Int
    max_retr_tolerance::T

    function APEParameters{T}(
        alpha::T,
        projection::Symbol,
        max_retr_iterations::Int,
        max_retr_tolerance::T,
    ) where {T<:AbstractFloat}
        isfinite(alpha) || throw(ArgumentError("the APE coefficient must be finite"))
        projection in (:max_retr, :polar) || throw(ArgumentError(
            "APE projection must be :max_retr or :polar; got $projection"))
        max_retr_iterations >= 1 || throw(ArgumentError(
            "MaxReTr iterations must be positive; got $max_retr_iterations"))
        isfinite(max_retr_tolerance) && max_retr_tolerance > 0 ||
            throw(ArgumentError(
                "the MaxReTr convergence tolerance must be finite and positive"))
        return new{T}(
            alpha, projection, max_retr_iterations, max_retr_tolerance)
    end
end

function APEParameters(
    alpha::Real;
    projection::Symbol=:max_retr,
    max_retr_iterations::Integer=1000,
    max_retr_tolerance::Real=1e-14,
)
    promoted = promote(float(alpha), float(max_retr_tolerance))
    return APEParameters{typeof(promoted[1])}(
        promoted[1], projection, Int(max_retr_iterations), promoted[2])
end

APEParameters(;
    alpha=0.6,
    projection::Symbol=:max_retr,
    max_retr_iterations::Integer=1000,
    max_retr_tolerance::Real=1e-14,
) = APEParameters(
    alpha;
    projection,
    max_retr_iterations,
    max_retr_tolerance,
)

export APEParameters

struct APESmearingCache4D{T,P,S}
    unprojected::NTuple{4,T}
    cotangent::Union{Nothing,NTuple{4,T}}
    projection_scratch::Union{Nothing,T}
    parameters::P
    state::S
end

function APESmearingCache4D(
    thin_links::Union{Vector{T},NTuple{4,T}},
    parameters::APEParameters=APEParameters(),
) where {T<:LatticeMatrix{4}}
    _validate_staggered_gauge_links(thin_links)
    thin_links[1].nw >= 1 || throw(ArgumentError(
        "APE smearing requires a halo width nw >= 1"))
    thin_links[1].NC1 in (2, 3) || throw(ArgumentError(
        "APE smearing currently supports only SU(2) and SU(3) links"))
    real_type = typeof(real(zero(eltype(thin_links[1].A))))
    typed_parameters = APEParameters(
        convert(real_type, parameters.alpha);
        projection=parameters.projection,
        max_retr_iterations=parameters.max_retr_iterations,
        max_retr_tolerance=convert(
            real_type, parameters.max_retr_tolerance),
    )
    field_count = parameters.projection === :polar ? 8 : 5
    fields = ntuple(_ ->
        _nhyp_workspace_field(thin_links[1])
    , field_count)
    cotangent = parameters.projection === :polar ?
        ntuple(i -> fields[i + 4], Val(4)) : nothing
    projection_scratch = parameters.projection === :max_retr ? fields[5] : nothing
    source_links = ntuple(mu -> thin_links[mu], Val(4))
    state = _NHYPCacheState(
        source_links, ntuple(_ -> UInt64(0), Val(4)), false)
    return APESmearingCache4D(
        ntuple(i -> fields[i], Val(4)),
        cotangent,
        projection_scratch,
        typed_parameters,
        state,
    )
end

export APESmearingCache4D

function _validate_projected_forward(label, output, input, reference)
    _validate_staggered_gauge_links(input)
    _validate_nhyp_links(output, input[1], "smeared_links")
    input[1].nw >= 1 || throw(ArgumentError(
        "$label smearing requires a halo width nw >= 1"))
    input[1].NC1 in (2, 3) || throw(ArgumentError(
        "$label smearing currently supports only SU(2) and SU(3) links"))
    for destination in output, source in input
        _nhyp_aliases(destination, source) && throw(ArgumentError(
            "$label output and thin links must not alias"))
    end
    _nhyp_same_layout(reference, input[1]) || throw(ArgumentError(
        "the $label cache has a different lattice layout"))
    return nothing
end

"""
    ape_smear!(smeared_links, thin_links, cache)

Apply one four-dimensional APE step. `projection=:max_retr` follows the
standard iterative SU(N) projection; `projection=:polar` selects the
differentiable principal-polar variant.
"""
function ape_smear!(
    smeared_links::Union{Vector{TO},NTuple{4,TO}},
    thin_links::Union{Vector{TI},NTuple{4,TI}},
    cache::APESmearingCache4D,
) where {TO<:LatticeMatrix{4},TI<:LatticeMatrix{4}}
    _validate_projected_forward(
        "APE", smeared_links, thin_links, cache.unprojected[1])
    cache.state.valid = false
    alpha = cache.parameters.alpha
    ensure_halo!.(thin_links)
    for mu in 1:4
        side_axis1, side_axis2, side_axis3 = _stout_side_axes(mu)
        _nhyp_build_three_staples!(
            cache.unprojected[mu], thin_links[mu],
            thin_links[side_axis1], thin_links[mu],
            thin_links[side_axis2], thin_links[mu],
            thin_links[side_axis3], thin_links[mu],
            1 - alpha, alpha / 6,
            side_axis1, side_axis2, side_axis3, mu)
        _sun_project_field!(
            smeared_links[mu], cache.unprojected[mu],
            cache.projection_scratch, cache.parameters)
    end
    JACC.synchronize()
    cache.state.source_links = ntuple(mu -> thin_links[mu], Val(4))
    cache.state.core_epochs = ntuple(
        mu -> thin_links[mu].halo_epoch.core, Val(4))
    cache.state.valid = true
    return smeared_links
end

function ape_smear(
    thin_links::Union{Vector{T},NTuple{4,T}},
    parameters::APEParameters=APEParameters(),
) where {T<:LatticeMatrix{4}}
    output = [similar(link) for link in thin_links]
    cache = APESmearingCache4D(thin_links, parameters)
    ape_smear!(output, thin_links, cache)
    return output, cache
end

export ape_smear!, ape_smear

function _validate_projected_cache_current(label, cache, thin_links)
    cache.state.valid || throw(ArgumentError(
        "$(lowercase(label))_smear! must populate the cache before " *
        "$(lowercase(label))_pullback!"))
    for mu in 1:4
        cache.state.source_links[mu] === thin_links[mu] ||
            throw(ArgumentError(
                "the $label cache belongs to different thin links"))
        cache.state.core_epochs[mu] == thin_links[mu].halo_epoch.core ||
            throw(ArgumentError(
                "thin link U[$mu] changed after the cached $label forward pass"))
    end
    return nothing
end

function _validate_projected_pullback(
    label, dthin_links, dsmeared_links, thin_links, cache, reference,
)
    _validate_projected_forward(label, dsmeared_links, thin_links, reference)
    _validate_nhyp_links(dthin_links, thin_links[1], "dthin_links")
    for destination in dthin_links
        for input in thin_links
            _nhyp_aliases(destination, input) && throw(ArgumentError(
                "$label thin-link cotangents must not alias thin links"))
        end
        for source in dsmeared_links
            _nhyp_aliases(destination, source) && throw(ArgumentError(
                "$label input and output cotangents must not alias"))
        end
    end
    for first_mu in 1:4, second_mu in (first_mu + 1):4
        _nhyp_aliases(dthin_links[first_mu], dthin_links[second_mu]) &&
            throw(ArgumentError(
                "$label thin-link cotangents must not alias each other"))
    end
    _validate_projected_cache_current(label, cache, thin_links)
    return nothing
end

"""
    ape_pullback!(dthin_links, dsmeared_links, thin_links, cache)

Apply the analytic reverse pass for a cached [`ape_smear!`](@ref) call.
The derivative of the principal determinant phase is used away from its
negative-real branch cut.
"""
function ape_pullback!(
    dthin_links::Union{Vector{TD},NTuple{4,TD}},
    dsmeared_links::Union{Vector{TC},NTuple{4,TC}},
    thin_links::Union{Vector{TI},NTuple{4,TI}},
    cache::APESmearingCache4D,
) where {
    TD<:LatticeMatrix{4},TC<:LatticeMatrix{4},TI<:LatticeMatrix{4},
}
    cache.parameters.projection === :polar || throw(ArgumentError(
        "APE pullback is unavailable for projection=:max_retr; " *
        "use APEParameters(...; projection=:polar) to enable the " *
        "analytic pullback"))
    _validate_projected_pullback(
        "APE", dthin_links, dsmeared_links, thin_links, cache,
        cache.unprojected[1])
    clear_matrix!.(dthin_links)
    clear_matrix!.(cache.cotangent)

    alpha = cache.parameters.alpha
    staple_coefficient = alpha / 6
    for mu in 1:4
        _sun_project_pullback_accumulate!(
            cache.cotangent[mu], dsmeared_links[mu],
            cache.unprojected[mu])
    end
    for mu in 1:4
        chain = cache.cotangent[mu]
        add_matrix!(dthin_links[mu], chain, 1 - alpha)
        for nu in 1:4
            nu == mu && continue
            _nhyp_staple_pullback!(
                dthin_links[nu], dthin_links[mu], chain,
                thin_links[nu], thin_links[mu], nu, mu,
                staple_coefficient)
        end
    end
    return dthin_links
end

export ape_pullback!

"""
    HYPParameters(alpha_outer=0.75, alpha_middle=0.6, alpha_inner=0.3;
                  projection=:max_retr, max_retr_iterations=1000,
                  max_retr_tolerance=1e-14)

Parameters for one four-dimensional HYP step. `:max_retr` is the standard
iterative projection used for interoperability. Select `:polar` when an
analytic pullback is required.
"""
struct HYPParameters{T<:AbstractFloat}
    alpha_outer::T
    alpha_middle::T
    alpha_inner::T
    projection::Symbol
    max_retr_iterations::Int
    max_retr_tolerance::T

    function HYPParameters{T}(
        alpha_outer::T, alpha_middle::T, alpha_inner::T,
        projection::Symbol, max_retr_iterations::Int,
        max_retr_tolerance::T,
    ) where {T<:AbstractFloat}
        all(isfinite, (alpha_outer, alpha_middle, alpha_inner)) ||
            throw(ArgumentError("HYP coefficients must be finite"))
        projection in (:max_retr, :polar) || throw(ArgumentError(
            "HYP projection must be :max_retr or :polar; got $projection"))
        max_retr_iterations >= 1 || throw(ArgumentError(
            "MaxReTr iterations must be positive; got $max_retr_iterations"))
        isfinite(max_retr_tolerance) && max_retr_tolerance > 0 ||
            throw(ArgumentError(
                "the MaxReTr convergence tolerance must be finite and positive"))
        return new{T}(
            alpha_outer, alpha_middle, alpha_inner, projection,
            max_retr_iterations, max_retr_tolerance)
    end
end

function HYPParameters(
    alpha_outer::Real, alpha_middle::Real, alpha_inner::Real,
    ;
    projection::Symbol=:max_retr,
    max_retr_iterations::Integer=1000,
    max_retr_tolerance::Real=1e-14,
)
    promoted = promote(
        float(alpha_outer), float(alpha_middle), float(alpha_inner),
        float(max_retr_tolerance))
    return HYPParameters{typeof(promoted[1])}(
        promoted[1], promoted[2], promoted[3], projection,
        Int(max_retr_iterations), promoted[4])
end

HYPParameters(;
    alpha_outer=0.75,
    alpha_middle=0.6,
    alpha_inner=0.3,
    projection::Symbol=:max_retr,
    max_retr_iterations::Integer=1000,
    max_retr_tolerance::Real=1e-14,
) = HYPParameters(
    alpha_outer, alpha_middle, alpha_inner;
    projection,
    max_retr_iterations,
    max_retr_tolerance,
)

export HYPParameters

struct HYPSmearingCache4D{T,P,S}
    inner_unprojected::NTuple{12,T}
    inner_links::NTuple{12,T}
    middle_unprojected::NTuple{12,T}
    middle_links::NTuple{12,T}
    outer_unprojected::NTuple{4,T}
    inner_cotangent::Union{Nothing,NTuple{12,T}}
    middle_cotangent::Union{Nothing,NTuple{12,T}}
    outer_cotangent::Union{Nothing,NTuple{4,T}}
    projection_scratch::Union{Nothing,T}
    parameters::P
    state::S
end

function HYPSmearingCache4D(
    thin_links::Union{Vector{T},NTuple{4,T}},
    parameters::HYPParameters=HYPParameters(),
) where {T<:LatticeMatrix{4}}
    _validate_staggered_gauge_links(thin_links)
    thin_links[1].nw >= 1 || throw(ArgumentError(
        "HYP smearing requires a halo width nw >= 1"))
    thin_links[1].NC1 in (2, 3) || throw(ArgumentError(
        "HYP smearing currently supports only SU(2) and SU(3) links"))
    real_type = typeof(real(zero(eltype(thin_links[1].A))))
    typed_parameters = HYPParameters(
        convert(real_type, parameters.alpha_outer),
        convert(real_type, parameters.alpha_middle),
        convert(real_type, parameters.alpha_inner),
        projection=parameters.projection,
        max_retr_iterations=parameters.max_retr_iterations,
        max_retr_tolerance=convert(
            real_type, parameters.max_retr_tolerance),
    )
    field_count = parameters.projection === :polar ? 80 : 53
    fields = ntuple(_ ->
        _nhyp_workspace_field(thin_links[1])
    , field_count)
    inner_cotangent = parameters.projection === :polar ?
        ntuple(i -> fields[i + 52], Val(12)) : nothing
    middle_cotangent = parameters.projection === :polar ?
        ntuple(i -> fields[i + 64], Val(12)) : nothing
    outer_cotangent = parameters.projection === :polar ?
        ntuple(i -> fields[i + 76], Val(4)) : nothing
    projection_scratch = parameters.projection === :max_retr ? fields[53] : nothing
    source_links = ntuple(mu -> thin_links[mu], Val(4))
    state = _NHYPCacheState(
        source_links, ntuple(_ -> UInt64(0), Val(4)), false)
    return HYPSmearingCache4D(
        ntuple(i -> fields[i], Val(12)),
        ntuple(i -> fields[i + 12], Val(12)),
        ntuple(i -> fields[i + 24], Val(12)),
        ntuple(i -> fields[i + 36], Val(12)),
        ntuple(i -> fields[i + 48], Val(4)),
        inner_cotangent,
        middle_cotangent,
        outer_cotangent,
        projection_scratch,
        typed_parameters,
        state,
    )
end

export HYPSmearingCache4D

"""
    hyp_smear!(smeared_links, thin_links, cache)

Apply one four-dimensional HYP step. `projection=:max_retr` follows the
standard iterative SU(N) projection; `projection=:polar` selects the
differentiable principal-polar variant.
"""
function hyp_smear!(
    smeared_links::Union{Vector{TO},NTuple{4,TO}},
    thin_links::Union{Vector{TI},NTuple{4,TI}},
    cache::HYPSmearingCache4D,
) where {TO<:LatticeMatrix{4},TI<:LatticeMatrix{4}}
    _validate_projected_forward(
        "HYP", smeared_links, thin_links, cache.inner_links[1])
    cache.state.valid = false
    parameters = cache.parameters

    ensure_halo!.(thin_links)
    for (mu, nu) in _nhyp_direction_pairs
        candidate = _nhyp_pair_field(cache.inner_unprojected, mu, nu)
        projected = _nhyp_pair_field(cache.inner_links, mu, nu)
        _nhyp_build_one_staple!(
            candidate, thin_links[mu], thin_links[nu], thin_links[mu],
            1 - parameters.alpha_inner, parameters.alpha_inner / 2,
            nu, mu)
        _sun_project_field!(
            projected, candidate, cache.projection_scratch, parameters)
    end
    JACC.synchronize()
    ensure_halo!.(cache.inner_links)

    for (mu, nu) in _nhyp_direction_pairs
        candidate = _nhyp_pair_field(cache.middle_unprojected, mu, nu)
        projected = _nhyp_pair_field(cache.middle_links, mu, nu)
        side_axis1, side_axis2 = _nhyp_other_two_axes(mu, nu)
        excluded_axis1 = 10 - mu - nu - side_axis1
        excluded_axis2 = 10 - mu - nu - side_axis2
        _nhyp_build_two_staples!(
            candidate, thin_links[mu],
            _nhyp_pair_field(cache.inner_links, side_axis1, excluded_axis1),
            _nhyp_pair_field(cache.inner_links, mu, excluded_axis1),
            _nhyp_pair_field(cache.inner_links, side_axis2, excluded_axis2),
            _nhyp_pair_field(cache.inner_links, mu, excluded_axis2),
            1 - parameters.alpha_middle, parameters.alpha_middle / 4,
            side_axis1, side_axis2, mu)
        _sun_project_field!(
            projected, candidate, cache.projection_scratch, parameters)
    end
    JACC.synchronize()
    ensure_halo!.(cache.middle_links)

    for mu in 1:4
        candidate = cache.outer_unprojected[mu]
        side_axis1, side_axis2, side_axis3 = _stout_side_axes(mu)
        _nhyp_build_three_staples!(
            candidate, thin_links[mu],
            _nhyp_pair_field(cache.middle_links, side_axis1, mu),
            _nhyp_pair_field(cache.middle_links, mu, side_axis1),
            _nhyp_pair_field(cache.middle_links, side_axis2, mu),
            _nhyp_pair_field(cache.middle_links, mu, side_axis2),
            _nhyp_pair_field(cache.middle_links, side_axis3, mu),
            _nhyp_pair_field(cache.middle_links, mu, side_axis3),
            1 - parameters.alpha_outer, parameters.alpha_outer / 6,
            side_axis1, side_axis2, side_axis3, mu)
        _sun_project_field!(
            smeared_links[mu], candidate,
            cache.projection_scratch, parameters)
    end
    JACC.synchronize()
    cache.state.source_links = ntuple(mu -> thin_links[mu], Val(4))
    cache.state.core_epochs = ntuple(
        mu -> thin_links[mu].halo_epoch.core, Val(4))
    cache.state.valid = true
    return smeared_links
end

function hyp_smear(
    thin_links::Union{Vector{T},NTuple{4,T}},
    parameters::HYPParameters=HYPParameters(),
) where {T<:LatticeMatrix{4}}
    output = [similar(link) for link in thin_links]
    cache = HYPSmearingCache4D(thin_links, parameters)
    hyp_smear!(output, thin_links, cache)
    return output, cache
end

export hyp_smear!, hyp_smear

"""
    hyp_pullback!(dthin_links, dsmeared_links, thin_links, cache)

Apply the analytic reverse pass for a cached [`hyp_smear!`](@ref) call.
The derivative of the principal determinant phase is used away from its
negative-real branch cut.
"""
function hyp_pullback!(
    dthin_links::Union{Vector{TD},NTuple{4,TD}},
    dsmeared_links::Union{Vector{TC},NTuple{4,TC}},
    thin_links::Union{Vector{TI},NTuple{4,TI}},
    cache::HYPSmearingCache4D,
) where {
    TD<:LatticeMatrix{4},TC<:LatticeMatrix{4},TI<:LatticeMatrix{4},
}
    cache.parameters.projection === :polar || throw(ArgumentError(
        "HYP pullback is unavailable for projection=:max_retr; " *
        "use HYPParameters(...; projection=:polar) to enable the " *
        "analytic pullback"))
    _validate_projected_pullback(
        "HYP", dthin_links, dsmeared_links, thin_links, cache,
        cache.inner_links[1])
    clear_matrix!.(dthin_links)
    clear_matrix!.(cache.inner_cotangent)
    clear_matrix!.(cache.middle_cotangent)
    clear_matrix!.(cache.outer_cotangent)

    parameters = cache.parameters
    inner_staple_coefficient = parameters.alpha_inner / 2
    middle_staple_coefficient = parameters.alpha_middle / 4
    outer_staple_coefficient = parameters.alpha_outer / 6

    for mu in 1:4
        _sun_project_pullback_accumulate!(
            cache.outer_cotangent[mu], dsmeared_links[mu],
            cache.outer_unprojected[mu])
    end
    for mu in 1:4
        chain = cache.outer_cotangent[mu]
        add_matrix!(dthin_links[mu], chain, 1 - parameters.alpha_outer)
        for nu in 1:4
            nu == mu && continue
            dside = _nhyp_pair_field(cache.middle_cotangent, nu, mu)
            dmiddle = _nhyp_pair_field(cache.middle_cotangent, mu, nu)
            side = _nhyp_pair_field(cache.middle_links, nu, mu)
            middle = _nhyp_pair_field(cache.middle_links, mu, nu)
            _nhyp_staple_pullback!(
                dside, dmiddle, chain, side, middle, nu, mu,
                outer_staple_coefficient)
        end
    end

    projection_temporary = cache.outer_cotangent[1]
    for (mu, nu) in _nhyp_direction_pairs
        clear_matrix!(projection_temporary)
        _sun_project_pullback_accumulate!(
            projection_temporary,
            _nhyp_pair_field(cache.middle_cotangent, mu, nu),
            _nhyp_pair_field(cache.middle_unprojected, mu, nu))
        add_matrix!(
            dthin_links[mu], projection_temporary,
            1 - parameters.alpha_middle)
        for side_axis in 1:4
            (side_axis == mu || side_axis == nu) && continue
            excluded_axis = 10 - mu - nu - side_axis
            dside = _nhyp_pair_field(
                cache.inner_cotangent, side_axis, excluded_axis)
            dmiddle = _nhyp_pair_field(
                cache.inner_cotangent, mu, excluded_axis)
            side = _nhyp_pair_field(
                cache.inner_links, side_axis, excluded_axis)
            middle = _nhyp_pair_field(
                cache.inner_links, mu, excluded_axis)
            _nhyp_staple_pullback!(
                dside, dmiddle, projection_temporary, side, middle,
                side_axis, mu, middle_staple_coefficient)
        end
    end

    for (mu, nu) in _nhyp_direction_pairs
        clear_matrix!(projection_temporary)
        _sun_project_pullback_accumulate!(
            projection_temporary,
            _nhyp_pair_field(cache.inner_cotangent, mu, nu),
            _nhyp_pair_field(cache.inner_unprojected, mu, nu))
        add_matrix!(
            dthin_links[mu], projection_temporary,
            1 - parameters.alpha_inner)
        _nhyp_staple_pullback!(
            dthin_links[nu], dthin_links[mu], projection_temporary,
            thin_links[nu], thin_links[mu], nu, mu,
            inner_staple_coefficient)
    end
    return dthin_links
end

export hyp_pullback!
