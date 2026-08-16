@inline function _kernel_set_global_component!(
    _, storage, value, row, column, local_position, halo_width)
    position = ntuple(d -> local_position[d] + halo_width, length(local_position))
    @inbounds storage[row, column, position...] = value
    return nothing
end

"""
    set_global_component!(field, value, row, column, global_position)

Set one matrix component at a global lattice position. Only the MPI rank that
owns `global_position` writes to its local storage. The operation is portable
across the JACC CPU and accelerator backends.
"""
function set_global_component!(
    field::LatticeMatrix{D},
    value,
    row::Integer,
    column::Integer,
    global_position::NTuple{D,<:Integer},
) where D
    1 <= row <= field.NC1 || throw(BoundsError(1:field.NC1, row))
    1 <= column <= field.NC2 || throw(BoundsError(1:field.NC2, column))
    for d in 1:D
        1 <= global_position[d] <= field.gsize[d] ||
            throw(BoundsError(1:field.gsize[d], global_position[d]))
    end

    local_position = ntuple(
        d -> Int(global_position[d]) - field.coords[d] * field.PN[d], D)
    owns_position = all(d -> 1 <= local_position[d] <= field.PN[d], 1:D)
    if owns_position
        JACC.parallel_for(
            1,
            _kernel_set_global_component!,
            field.A,
            convert(eltype(field.A), value),
            Int(row),
            Int(column),
            local_position,
            field.nw,
        )
        mark_halo_dirty!(field)
    end
    return field
end

export set_global_component!
