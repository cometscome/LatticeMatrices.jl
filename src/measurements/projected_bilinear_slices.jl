@inline function _insert_axis(
    transverse::NTuple{Dminus1,<:Integer},
    axis_position::Integer,
    ::Val{axis},
) where {Dminus1,axis}
    D = Dminus1 + 1
    return ntuple(d -> begin
        if d == axis
            Int(axis_position)
        elseif d < axis
            Int(transverse[d])
        else
            Int(transverse[d - 1])
        end
    end, D)
end

@inline function _site_weight(
    local_position,
    mpi_coordinates,
    local_size,
    global_size,
    origin,
    momentum,
    parity_mask,
    ::Type{T},
) where T
    parity = 0
    angle = zero(typeof(real(zero(T))))
    two_pi = convert(typeof(angle), 2pi)
    @inbounds for d in eachindex(local_position)
        global_zero_based =
            mpi_coordinates[d] * local_size[d] + local_position[d] - 1
        relative = global_zero_based - (origin[d] - 1)
        parity += parity_mask[d] * relative
        angle -= two_pi * momentum[d] * relative / global_size[d]
    end
    staggered_sign = iseven(parity) ? one(typeof(angle)) : -one(typeof(angle))
    return convert(T, staggered_sign * cis(angle))
end

@inline function _site_spin_color_bilinear(
    propagators1,
    propagators2,
    left,
    right,
    indices,
    ::Val{NS},
    ::Val{NC},
) where {NS,NC}
    T = promote_type(eltype(propagators1[1]), eltype(propagators2[1]))
    value = zero(T)
    @inbounds for color in 1:NC
        P = SMatrix{NS,NS,T}(ntuple(Val(NS * NS)) do linear_index
            sink_spin = (linear_index - 1) % NS + 1
            source_spin = (linear_index - 1) ÷ NS + 1
            propagators1[source_spin][color, sink_spin, indices...]
        end)
        Q = SMatrix{NS,NS,T}(ntuple(Val(NS * NS)) do linear_index
            sink_spin = (linear_index - 1) % NS + 1
            source_spin = (linear_index - 1) ÷ NS + 1
            propagators2[source_spin][color, sink_spin, indices...]
        end)
        value += sum((left * P * right) .* conj.(Q))
    end
    return value
end

@inline function _kernel_projected_bilinear_slice(
    transverse_site,
    propagators1,
    propagators2,
    left,
    right,
    transverse_indexer,
    axis_position,
    ::Val{axis},
    ::Val{NS},
    ::Val{NC},
    halo_width,
    mpi_coordinates,
    local_size,
    global_size,
    origin,
    momentum,
    parity_mask,
    coefficient,
) where {axis,NS,NC}
    transverse = delinearize(transverse_indexer, transverse_site, 0)
    local_position = _insert_axis(transverse, axis_position, Val(axis))
    indices = ntuple(d -> local_position[d] + halo_width, length(local_position))
    contraction = _site_spin_color_bilinear(
        propagators1, propagators2, left, right, indices, Val(NS), Val(NC))
    weight = _site_weight(
        local_position,
        mpi_coordinates,
        local_size,
        global_size,
        origin,
        momentum,
        parity_mask,
        typeof(contraction),
    )
    return coefficient * weight * contraction
end

function _validate_projected_bilinear_inputs(
    propagators1,
    propagators2,
    left,
    right,
    axis,
    origin,
    momentum,
    parity_mask,
)
    NS = length(propagators1)
    NS > 0 || throw(ArgumentError("a propagator block cannot be empty"))
    length(propagators2) == NS || throw(DimensionMismatch(
        "the two propagator blocks must contain the same number of source spins"))
    reference = propagators1[1]
    D = length(reference.gsize)
    1 <= axis <= D || throw(ArgumentError("axis must be between 1 and $D"))
    size(left) == (NS, NS) || throw(DimensionMismatch(
        "left spin matrix must have size ($NS, $NS), got $(size(left))"))
    size(right) == (NS, NS) || throw(DimensionMismatch(
        "right spin matrix must have size ($NS, $NS), got $(size(right))"))
    length(origin) == D || throw(DimensionMismatch("origin must have $D entries"))
    length(momentum) == D || throw(DimensionMismatch("momentum must have $D entries"))
    length(parity_mask) == D ||
        throw(DimensionMismatch("parity_mask must have $D entries"))
    iszero(momentum[axis]) || throw(ArgumentError(
        "the momentum component along the correlator axis must be zero"))
    for d in 1:D
        1 <= origin[d] <= reference.gsize[d] ||
            throw(BoundsError(1:reference.gsize[d], origin[d]))
    end
    for (block_name, block) in (("first", propagators1), ("second", propagators2))
        for (source_spin, field) in pairs(block)
            field.NC1 == reference.NC1 || throw(DimensionMismatch(
                "$block_name block source spin $source_spin has a different color size"))
            field.NC2 == NS || throw(DimensionMismatch(
                "$block_name block source spin $source_spin has $(field.NC2) sink spins; expected $NS"))
            field.gsize == reference.gsize && field.PN == reference.PN &&
                field.dims == reference.dims && field.coords == reference.coords &&
                field.nw == reference.nw || throw(DimensionMismatch(
                "$block_name propagator block uses incompatible lattice geometry"))
        end
    end
    return nothing
end

"""
    projected_bilinear_slices(propagators1, propagators2, left, right; kwargs...)

Contract two tuples of spin-color propagators on every lattice site and sum
over the hyperplanes normal to `axis`. Optional integer `momentum` and
`parity_mask` tuples apply Fourier and staggered-sign projections relative to
`origin`. The returned vector is globally reduced across MPI ranks.
"""
function projected_bilinear_slices(
    propagators1::NTuple{NS,P1},
    propagators2::NTuple{NS,P2},
    left,
    right;
    axis::Integer=4,
    origin=ntuple(_ -> 1, length(propagators1[1].gsize)),
    momentum=ntuple(_ -> 0, length(propagators1[1].gsize)),
    parity_mask=ntuple(_ -> 0, length(propagators1[1].gsize)),
    coefficient=-1,
) where {NS,P1<:LatticeMatrix,P2<:LatticeMatrix}
    _validate_projected_bilinear_inputs(
        propagators1, propagators2, left, right, axis, origin, momentum, parity_mask)
    reference = propagators1[1]
    D = length(reference.gsize)
    NC = reference.NC1
    T = promote_type(eltype(propagators1[1].A), eltype(propagators2[1].A))
    left_static = SMatrix{NS,NS,T}(left)
    right_static = SMatrix{NS,NS,T}(right)
    origin_tuple = ntuple(d -> Int(origin[d]), D)
    momentum_tuple = ntuple(d -> Int(momentum[d]), D)
    parity_tuple = ntuple(d -> Int(parity_mask[d]), D)
    transverse_size = ntuple(d -> reference.PN[d < axis ? d : d + 1], D - 1)
    transverse_indexer = DIndexer(transverse_size)
    transverse_volume = prod(transverse_size)
    local_result = zeros(T, reference.gsize[axis])
    arrays1 = ntuple(source_spin -> propagators1[source_spin].A, NS)
    arrays2 = ntuple(source_spin -> propagators2[source_spin].A, NS)
    typed_coefficient = convert(T, coefficient)

    for local_axis_position in 1:reference.PN[axis]
        value = JACC.parallel_reduce(
            transverse_volume,
            _kernel_projected_bilinear_slice,
            arrays1,
            arrays2,
            left_static,
            right_static,
            transverse_indexer,
            local_axis_position,
            Val(Int(axis)),
            Val(NS),
            Val(NC),
            reference.nw,
            reference.coords,
            reference.PN,
            reference.gsize,
            origin_tuple,
            momentum_tuple,
            parity_tuple,
            typed_coefficient;
            init=zero(T),
            op=+,
        )
        global_axis_position =
            reference.coords[axis] * reference.PN[axis] + local_axis_position
        separation = mod(global_axis_position - origin_tuple[axis], reference.gsize[axis]) + 1
        local_result[separation] += value
    end
    return _allreduce_sum(local_result, reference.comm)
end

export projected_bilinear_slices
