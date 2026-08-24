@inline function _domainwall_surface_s(::Val{:physical}, spin, L5)
    return ifelse(spin <= 2, L5, 1)
end

@inline function _domainwall_surface_s(::Val{:midpoint}, spin, L5)
    midpoint = L5 ÷ 2
    return ifelse(spin <= 2, midpoint, midpoint + 1)
end

function _validate_domainwall_projection_geometry(field4, field5)
    field4.NC1 == field5.NC1 || throw(DimensionMismatch(
        "the four- and five-dimensional fields have different color sizes"))
    field4.NC2 == 4 && field5.NC2 == 4 || throw(DimensionMismatch(
        "domain-wall surface projection requires four spin components"))
    field5.gsize[1:4] == field4.gsize || throw(DimensionMismatch(
        "the physical dimensions of the four- and five-dimensional fields differ"))
    field5.PN[1:4] == field4.PN && field5.dims[1:4] == field4.dims &&
        field5.coords[1:4] == field4.coords || throw(DimensionMismatch(
        "the four- and five-dimensional fields use incompatible MPI decompositions"))
    field5.dims[5] == 1 || throw(ArgumentError(
        "the fifth domain-wall dimension must not be MPI-distributed"))
    return nothing
end

@inline function _kernel_domainwall_import_physical!(
    site, destination, source, indexer, source_nw, destination_nw, L5,
    ::Val{NC},
) where NC
    local_position = delinearize(indexer, site, 0)
    source_position = ntuple(d -> local_position[d] + source_nw, 4)
    @inbounds for spin in 1:4
        s = ifelse(spin <= 2, 1, L5)
        destination_position = (
            local_position[1] + destination_nw,
            local_position[2] + destination_nw,
            local_position[3] + destination_nw,
            local_position[4] + destination_nw,
            s + destination_nw,
        )
        for color in 1:NC
            destination[color, spin, destination_position...] =
                source[color, spin, source_position...]
        end
    end
    return nothing
end

"""
    domainwall_import_physical_source!(destination5, source4)

Embed a four-dimensional physical source into a Shamir domain-wall field. LM's
fifth-coordinate hopping is oriented oppositely to Grid's, so Grid's
`P_+ source4` at its first wall and `P_- source4` at its last wall map to
`P_- source4` at LM `s=1` and `P_+ source4` at LM `s=L5`. The fifth dimension
must be local to each MPI rank. This is the complete source import for Shamir
fermions, for which `D_- = 1`.
"""
function domainwall_import_physical_source!(
    destination::LatticeMatrix{5}, source::LatticeMatrix{4})
    _validate_domainwall_projection_geometry(source, destination)
    clear_matrix!(destination)
    L5 = destination.gsize[5]
    JACC.parallel_for(
        prod(source.PN),
        _kernel_domainwall_import_physical!,
        destination.A,
        source.A,
        source.indexer,
        source.nw,
        destination.nw,
        L5,
        Val(source.NC1),
    )
    mark_halo_dirty!(destination)
    return destination
end

@inline function _kernel_domainwall_export_surface!(
    site, destination, source, indexer, destination_nw, source_nw, L5,
    projection, ::Val{NC},
) where NC
    local_position = delinearize(indexer, site, 0)
    destination_position = ntuple(d -> local_position[d] + destination_nw, 4)
    @inbounds for spin in 1:4
        s = _domainwall_surface_s(projection, spin, L5)
        source_position = (
            local_position[1] + source_nw,
            local_position[2] + source_nw,
            local_position[3] + source_nw,
            local_position[4] + source_nw,
            s + source_nw,
        )
        for color in 1:NC
            destination[color, spin, destination_position...] =
                source[color, spin, source_position...]
        end
    end
    return nothing
end

function _domainwall_export_surface!(destination, source, projection)
    _validate_domainwall_projection_geometry(destination, source)
    L5 = source.gsize[5]
    projection === :physical || projection === :midpoint || throw(ArgumentError(
        "projection must be :physical or :midpoint, got $projection"))
    projection === :midpoint && isodd(L5) && throw(ArgumentError(
        "the midpoint projection requires an even fifth-dimensional extent"))
    JACC.parallel_for(
        prod(destination.PN),
        _kernel_domainwall_export_surface!,
        destination.A,
        source.A,
        destination.indexer,
        destination.nw,
        source.nw,
        L5,
        Val(projection),
        Val(destination.NC1),
    )
    mark_halo_dirty!(destination)
    return destination
end

"""
    domainwall_export_physical_solution!(destination4, source5)

Project a five-dimensional domain-wall solution to
`q = P_- source5(s=L5) + P_+ source5(s=1)`. This accounts for LM's reversed
fifth-coordinate orientation and is equivalent to Grid's
`ExportPhysicalFermionSolution`.
"""
domainwall_export_physical_solution!(destination::LatticeMatrix{4}, source::LatticeMatrix{5}) =
    _domainwall_export_surface!(destination, source, :physical)

"""
    domainwall_export_midpoint!(destination4, source5)

Project an even-`L5` domain-wall field to the central planes,
`P_- source5(s=L5/2) + P_+ source5(s=L5/2+1)`. This is Grid's central-plane
projection after accounting for LM's reversed fifth-coordinate orientation.
Its local norm gives the standard `J_5q` pseudoscalar-density contraction used
in residual-mass ratios.
"""
domainwall_export_midpoint!(destination::LatticeMatrix{4}, source::LatticeMatrix{5}) =
    _domainwall_export_surface!(destination, source, :midpoint)

function _validate_domainwall_bilinear_inputs(
    propagators1, propagators2, left, right, axis, origin, momentum, parity_mask,
    projection,
)
    NS = length(propagators1)
    NS == 4 || throw(ArgumentError(
        "a domain-wall propagator block must contain four source-spin fields"))
    length(propagators2) == NS || throw(DimensionMismatch(
        "the two propagator blocks must contain the same number of source spins"))
    projection === :physical || projection === :midpoint || throw(ArgumentError(
        "projection must be :physical or :midpoint, got $projection"))
    reference = propagators1[1]
    reference.gsize[5] == reference.PN[5] || throw(ArgumentError(
        "the fifth domain-wall dimension must not be MPI-distributed"))
    projection === :midpoint && isodd(reference.gsize[5]) && throw(ArgumentError(
        "the midpoint projection requires an even fifth-dimensional extent"))
    1 <= axis <= 4 || throw(ArgumentError("axis must be between 1 and 4"))
    size(left) == (4, 4) && size(right) == (4, 4) || throw(DimensionMismatch(
        "the spin matrices must both have size (4, 4)"))
    length(origin) == 4 && length(momentum) == 4 && length(parity_mask) == 4 ||
        throw(DimensionMismatch("origin, momentum, and parity_mask must have four entries"))
    iszero(momentum[axis]) || throw(ArgumentError(
        "the momentum component along the correlator axis must be zero"))
    for d in 1:4
        1 <= origin[d] <= reference.gsize[d] ||
            throw(BoundsError(1:reference.gsize[d], origin[d]))
    end
    for block in (propagators1, propagators2), field in block
        field.NC1 == reference.NC1 && field.NC2 == 4 &&
            field.gsize == reference.gsize && field.PN == reference.PN &&
            field.dims == reference.dims && field.coords == reference.coords &&
            field.nw == reference.nw || throw(DimensionMismatch(
            "the domain-wall propagator blocks use incompatible lattice geometry"))
    end
    return nothing
end

@inline function _domainwall_site_spin_color_bilinear(
    propagators1, propagators2, left, right, position4, source_nw, L5,
    projection, ::Val{NC},
) where NC
    T = promote_type(eltype(propagators1[1]), eltype(propagators2[1]))
    value = zero(T)
    @inbounds for color in 1:NC
        P = SMatrix{4,4,T}(ntuple(Val(16)) do linear_index
            sink_spin = (linear_index - 1) % 4 + 1
            source_spin = (linear_index - 1) ÷ 4 + 1
            s = _domainwall_surface_s(projection, sink_spin, L5)
            propagators1[source_spin][color, sink_spin, position4..., s + source_nw]
        end)
        Q = SMatrix{4,4,T}(ntuple(Val(16)) do linear_index
            sink_spin = (linear_index - 1) % 4 + 1
            source_spin = (linear_index - 1) ÷ 4 + 1
            s = _domainwall_surface_s(projection, sink_spin, L5)
            propagators2[source_spin][color, sink_spin, position4..., s + source_nw]
        end)
        value += sum((left * P * right) .* conj.(Q))
    end
    return value
end

@inline function _kernel_domainwall_bilinear_slice(
    transverse_site, propagators1, propagators2, left, right,
    transverse_indexer, axis_position, axis, projection, halo_width, L5,
    mpi_coordinates, local_size, global_size, origin, momentum, parity_mask,
    coefficient, ::Val{NC},
) where NC
    transverse = delinearize(transverse_indexer, transverse_site, 0)
    local_position = _insert_axis(transverse, axis_position, axis)
    position4 = ntuple(d -> local_position[d] + halo_width, 4)
    contraction = _domainwall_site_spin_color_bilinear(
        propagators1, propagators2, left, right, position4, halo_width, L5,
        projection, Val(NC))
    weight = _site_weight(
        local_position, mpi_coordinates, local_size, global_size, origin,
        momentum, parity_mask, typeof(contraction))
    return coefficient * weight * contraction
end

"""
    domainwall_projected_bilinear_slices(propagators1, propagators2, left, right;
                                         projection=:physical, kwargs...)

Contract two four-source-spin blocks of five-dimensional propagators after a
Grid-compatible `:physical` or `:midpoint` surface projection. Hyperplane,
momentum, parity and MPI reductions follow `projected_bilinear_slices`.
"""
function domainwall_projected_bilinear_slices(
    propagators1::NTuple{4,P1}, propagators2::NTuple{4,P2}, left, right;
    projection::Symbol=:physical,
    axis::Integer=4,
    origin=ntuple(_ -> 1, 4),
    momentum=ntuple(_ -> 0, 4),
    parity_mask=ntuple(_ -> 0, 4),
    coefficient=1,
) where {P1<:LatticeMatrix{5},P2<:LatticeMatrix{5}}
    _validate_domainwall_bilinear_inputs(
        propagators1, propagators2, left, right, axis, origin, momentum,
        parity_mask, projection)
    reference = propagators1[1]
    physical_gsize = ntuple(d -> reference.gsize[d], 4)
    physical_local_size = ntuple(d -> reference.PN[d], 4)
    physical_coords = ntuple(d -> reference.coords[d], 4)
    T = promote_type(eltype(propagators1[1].A), eltype(propagators2[1].A))
    left_static = SMatrix{4,4,T}(left)
    right_static = SMatrix{4,4,T}(right)
    origin_tuple = ntuple(d -> Int(origin[d]), 4)
    momentum_tuple = ntuple(d -> Int(momentum[d]), 4)
    parity_tuple = ntuple(d -> Int(parity_mask[d]), 4)
    transverse_size = ntuple(d -> physical_local_size[d < axis ? d : d + 1], 3)
    transverse_indexer = DIndexer(transverse_size)
    local_result = zeros(T, physical_gsize[axis])
    arrays1 = ntuple(source_spin -> propagators1[source_spin].A, 4)
    arrays2 = ntuple(source_spin -> propagators2[source_spin].A, 4)

    for local_axis_position in 1:physical_local_size[axis]
        value = JACC.parallel_reduce(
            prod(transverse_size),
            _kernel_domainwall_bilinear_slice,
            arrays1,
            arrays2,
            left_static,
            right_static,
            transverse_indexer,
            local_axis_position,
            Val(Int(axis)),
            Val(projection),
            reference.nw,
            reference.gsize[5],
            physical_coords,
            physical_local_size,
            physical_gsize,
            origin_tuple,
            momentum_tuple,
            parity_tuple,
            convert(T, coefficient),
            Val(reference.NC1);
            init=zero(T),
            op=+,
        )
        global_axis_position =
            physical_coords[axis] * physical_local_size[axis] + local_axis_position
        separation = mod(global_axis_position - origin_tuple[axis], physical_gsize[axis]) + 1
        local_result[separation] += value
    end
    return _allreduce_sum(local_result, reference.comm)
end

export domainwall_import_physical_source!, domainwall_export_physical_solution!
export domainwall_export_midpoint!, domainwall_projected_bilinear_slices
