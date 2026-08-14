using StaticArrays: SVector

const _hisq_transverse_signs = (-1, 1)

@inline function _hisq_parallel_for(args...)
    threads = JACC.backend == "cuda" ? 128 : 0
    specification = JACC.launch_spec(threads=threads, shmem_size=0)
    return JACC.parallel_for(specification, args...)
end

@inline function _hisq_link_element(
    U1, U2, U3, U4, direction, row, column, site,
)
    if direction == 1
        return U1[row, column, site...]
    elseif direction == 2
        return U2[row, column, site...]
    elseif direction == 3
        return U3[row, column, site...]
    end
    return U4[row, column, site...]
end

@inline function _hisq_shift_site(site, direction)
    axis = abs(direction)
    amount = ifelse(direction > 0, 1, -1)
    return ntuple(d -> site[d] + ifelse(d == axis, amount, 0), 4)
end

@inline function _hisq_oriented_row(
    U1, U2, U3, U4, site, direction, row, ::Val{NC},
) where NC
    if direction > 0
        values = SVector{NC}(ntuple(Val(NC)) do column
            _hisq_link_element(
                U1, U2, U3, U4, direction, row, column, site)
        end)
        return values, _hisq_shift_site(site, direction)
    end

    previous_site = _hisq_shift_site(site, direction)
    axis = -direction
    values = SVector{NC}(ntuple(Val(NC)) do column
        conj(_hisq_link_element(
            U1, U2, U3, U4, axis, column, row, previous_site))
    end)
    return values, previous_site
end

@inline function _hisq_row_times_oriented_link(
    row_values, U1, U2, U3, U4, site, direction, ::Val{NC},
) where NC
    matrix_site = ifelse(
        direction > 0, site, _hisq_shift_site(site, direction))
    axis = abs(direction)
    values = SVector{NC}(ntuple(Val(NC)) do column
        value = zero(eltype(row_values))
        @inbounds for contracted in 1:NC
            link_element = if direction > 0
                _hisq_link_element(
                    U1, U2, U3, U4, axis,
                    contracted, column, matrix_site)
            else
                conj(_hisq_link_element(
                    U1, U2, U3, U4, axis,
                    column, contracted, matrix_site))
            end
            value += row_values[contracted] * link_element
        end
        value
    end)
    next_site = ifelse(
        direction > 0, _hisq_shift_site(site, direction), matrix_site)
    return values, next_site
end

@inline function _hisq_path_row(
    U1, U2, U3, U4, origin, path, row, ::Val{NC},
) where NC
    product, site = _hisq_oriented_row(
        U1, U2, U3, U4, origin, path[1], row, Val(NC))
    @inbounds for path_index in 2:length(path)
        product, site = _hisq_row_times_oriented_link(
            product, U1, U2, U3, U4, site,
            path[path_index], Val(NC))
    end
    return product
end

@inline _hisq_five_signs(::Val{1}) = (1, 1)
@inline _hisq_five_signs(::Val{2}) = (-1, -1)
@inline _hisq_five_signs(::Val{3}) = (1, -1)
@inline _hisq_five_signs(::Val{4}) = (-1, 1)

@inline _hisq_seven_signs(::Val{1}) = (1, 1, 1)
@inline _hisq_seven_signs(::Val{2}) = (-1, -1, -1)
@inline _hisq_seven_signs(::Val{3}) = (-1, 1, 1)
@inline _hisq_seven_signs(::Val{4}) = (1, -1, 1)
@inline _hisq_seven_signs(::Val{5}) = (1, 1, -1)
@inline _hisq_seven_signs(::Val{6}) = (-1, -1, 1)
@inline _hisq_seven_signs(::Val{7}) = (-1, 1, -1)
@inline _hisq_seven_signs(::Val{8}) = (1, -1, -1)

# CUDA's register allocator spills the generic SVector/path-loop implementation
# heavily for the physical NC=3 case.  These helpers keep the same row-per-thread
# decomposition, but make the three color components and each path length
# explicit.  Flattening the existing NC-first arrays also lets a link address be
# formed once per matrix instead of once per scalar access.  The generic path
# below remains the fallback for other color counts and non-CUDA backends.
@inline function _hisq_flat_site_index3(site, padded_size)
    return site[1] + padded_size[1] * ((site[2] - 1) +
        padded_size[2] * ((site[3] - 1) +
        padded_size[3] * (site[4] - 1)))
end

@inline function _hisq_flat_link_element3(
    U1, U2, U3, U4, direction, row, column, site_index,
)
    element_index = row + 3 * (column - 1) + 9 * (site_index - 1)
    @inbounds if direction == 1
        return U1[element_index]
    elseif direction == 2
        return U2[element_index]
    elseif direction == 3
        return U3[element_index]
    end
    return U4[element_index]
end

@inline function _hisq_oriented_row3_flat(
    U1, U2, U3, U4, site, direction, row, padded_size,
)
    if direction > 0
        site_index = _hisq_flat_site_index3(site, padded_size)
        values = (
            _hisq_flat_link_element3(
                U1, U2, U3, U4, direction, row, 1, site_index),
            _hisq_flat_link_element3(
                U1, U2, U3, U4, direction, row, 2, site_index),
            _hisq_flat_link_element3(
                U1, U2, U3, U4, direction, row, 3, site_index),
        )
        return values, _hisq_shift_site(site, direction)
    end

    previous_site = _hisq_shift_site(site, direction)
    site_index = _hisq_flat_site_index3(previous_site, padded_size)
    axis = -direction
    values = (
        conj(_hisq_flat_link_element3(
            U1, U2, U3, U4, axis, 1, row, site_index)),
        conj(_hisq_flat_link_element3(
            U1, U2, U3, U4, axis, 2, row, site_index)),
        conj(_hisq_flat_link_element3(
            U1, U2, U3, U4, axis, 3, row, site_index)),
    )
    return values, previous_site
end

@inline function _hisq_row_times_oriented3_flat(
    row_values, U1, U2, U3, U4, site, direction, padded_size,
)
    matrix_site = ifelse(
        direction > 0, site, _hisq_shift_site(site, direction))
    site_index = _hisq_flat_site_index3(matrix_site, padded_size)
    axis = abs(direction)
    r1, r2, r3 = row_values

    if direction > 0
        values = (
            muladd(r1, _hisq_flat_link_element3(
                U1, U2, U3, U4, axis, 1, 1, site_index),
                muladd(r2, _hisq_flat_link_element3(
                    U1, U2, U3, U4, axis, 2, 1, site_index),
                    r3 * _hisq_flat_link_element3(
                        U1, U2, U3, U4, axis, 3, 1, site_index))),
            muladd(r1, _hisq_flat_link_element3(
                U1, U2, U3, U4, axis, 1, 2, site_index),
                muladd(r2, _hisq_flat_link_element3(
                    U1, U2, U3, U4, axis, 2, 2, site_index),
                    r3 * _hisq_flat_link_element3(
                        U1, U2, U3, U4, axis, 3, 2, site_index))),
            muladd(r1, _hisq_flat_link_element3(
                U1, U2, U3, U4, axis, 1, 3, site_index),
                muladd(r2, _hisq_flat_link_element3(
                    U1, U2, U3, U4, axis, 2, 3, site_index),
                    r3 * _hisq_flat_link_element3(
                        U1, U2, U3, U4, axis, 3, 3, site_index))),
        )
        return values, _hisq_shift_site(site, direction)
    end

    values = (
        muladd(r1, conj(_hisq_flat_link_element3(
            U1, U2, U3, U4, axis, 1, 1, site_index)),
            muladd(r2, conj(_hisq_flat_link_element3(
                U1, U2, U3, U4, axis, 1, 2, site_index)),
                r3 * conj(_hisq_flat_link_element3(
                    U1, U2, U3, U4, axis, 1, 3, site_index)))),
        muladd(r1, conj(_hisq_flat_link_element3(
            U1, U2, U3, U4, axis, 2, 1, site_index)),
            muladd(r2, conj(_hisq_flat_link_element3(
                U1, U2, U3, U4, axis, 2, 2, site_index)),
                r3 * conj(_hisq_flat_link_element3(
                    U1, U2, U3, U4, axis, 2, 3, site_index)))),
        muladd(r1, conj(_hisq_flat_link_element3(
            U1, U2, U3, U4, axis, 3, 1, site_index)),
            muladd(r2, conj(_hisq_flat_link_element3(
                U1, U2, U3, U4, axis, 3, 2, site_index)),
                r3 * conj(_hisq_flat_link_element3(
                    U1, U2, U3, U4, axis, 3, 3, site_index)))),
    )
    return values, matrix_site
end

@inline function _hisq_path3_row3_flat(
    U1, U2, U3, U4, origin, d1, d2, d3, row, padded_size,
)
    product, site = _hisq_oriented_row3_flat(
        U1, U2, U3, U4, origin, d1, row, padded_size)
    product, site = _hisq_row_times_oriented3_flat(
        product, U1, U2, U3, U4, site, d2, padded_size)
    product, _ = _hisq_row_times_oriented3_flat(
        product, U1, U2, U3, U4, site, d3, padded_size)
    return product
end

@inline function _hisq_path5_row3_flat(
    U1, U2, U3, U4, origin, d1, d2, d3, d4, d5, row, padded_size,
)
    product, site = _hisq_oriented_row3_flat(
        U1, U2, U3, U4, origin, d1, row, padded_size)
    product, site = _hisq_row_times_oriented3_flat(
        product, U1, U2, U3, U4, site, d2, padded_size)
    product, site = _hisq_row_times_oriented3_flat(
        product, U1, U2, U3, U4, site, d3, padded_size)
    product, site = _hisq_row_times_oriented3_flat(
        product, U1, U2, U3, U4, site, d4, padded_size)
    product, _ = _hisq_row_times_oriented3_flat(
        product, U1, U2, U3, U4, site, d5, padded_size)
    return product
end

@inline function _hisq_path7_row3_flat(
    U1, U2, U3, U4, origin, d1, d2, d3, d4, d5, d6, d7, row,
    padded_size,
)
    product, site = _hisq_oriented_row3_flat(
        U1, U2, U3, U4, origin, d1, row, padded_size)
    product, site = _hisq_row_times_oriented3_flat(
        product, U1, U2, U3, U4, site, d2, padded_size)
    product, site = _hisq_row_times_oriented3_flat(
        product, U1, U2, U3, U4, site, d3, padded_size)
    product, site = _hisq_row_times_oriented3_flat(
        product, U1, U2, U3, U4, site, d4, padded_size)
    product, site = _hisq_row_times_oriented3_flat(
        product, U1, U2, U3, U4, site, d5, padded_size)
    product, site = _hisq_row_times_oriented3_flat(
        product, U1, U2, U3, U4, site, d6, padded_size)
    product, _ = _hisq_row_times_oriented3_flat(
        product, U1, U2, U3, U4, site, d7, padded_size)
    return product
end

@inline function _hisq_add_tuple3(accumulator, value)
    return (
        accumulator[1] + value[1],
        accumulator[2] + value[2],
        accumulator[3] + value[3],
    )
end

@inline function _hisq_staple3_row3_flat(
    U1, U2, U3, U4, origin, mu, row, padded_size,
)
    z = zero(eltype(U1))
    accumulator = (z, z, z)
    @inbounds for nu in 1:4
        if nu != mu
            accumulator = _hisq_add_tuple3(accumulator,
                _hisq_path3_row3_flat(
                    U1, U2, U3, U4, origin,
                    nu, mu, -nu, row, padded_size))
            accumulator = _hisq_add_tuple3(accumulator,
                _hisq_path3_row3_flat(
                    U1, U2, U3, U4, origin,
                    -nu, mu, nu, row, padded_size))
        end
    end
    return accumulator
end

@inline function _hisq_staple5_row3_flat(
    U1, U2, U3, U4, origin, mu, row, part::Val{P}, padded_size,
) where P
    z = zero(eltype(U1))
    accumulator = (z, z, z)
    sign_nu, sign_rho = _hisq_five_signs(part)
    @inbounds for nu in 1:4
        if nu != mu
            for rho in 1:4
                if rho != mu && rho != nu
                    signed_nu = sign_nu * nu
                    signed_rho = sign_rho * rho
                    accumulator = _hisq_add_tuple3(accumulator,
                        _hisq_path5_row3_flat(
                            U1, U2, U3, U4, origin,
                            signed_nu, signed_rho, mu,
                            -signed_rho, -signed_nu, row, padded_size))
                end
            end
        end
    end
    return accumulator
end

@inline function _hisq_staple7_row3_flat(
    U1, U2, U3, U4, origin, mu, row, part::Val{P}, padded_size,
) where P
    z = zero(eltype(U1))
    accumulator = (z, z, z)
    sign_nu, sign_rho, sign_sigma = _hisq_seven_signs(part)
    @inbounds for nu in 1:4
        if nu != mu
            for rho in 1:4
                if rho != mu && rho != nu
                    sigma = 10 - mu - nu - rho
                    signed_nu = sign_nu * nu
                    signed_rho = sign_rho * rho
                    signed_sigma = sign_sigma * sigma
                    accumulator = _hisq_add_tuple3(accumulator,
                        _hisq_path7_row3_flat(
                            U1, U2, U3, U4, origin,
                            signed_nu, signed_rho, signed_sigma, mu,
                            -signed_sigma, -signed_rho, -signed_nu,
                            row, padded_size))
                end
            end
        end
    end
    return accumulator
end

@inline function _hisq_lepage_row3_flat(
    U1, U2, U3, U4, origin, mu, row, padded_size,
)
    z = zero(eltype(U1))
    accumulator = (z, z, z)
    @inbounds for nu in 1:4
        if nu != mu
            accumulator = _hisq_add_tuple3(accumulator,
                _hisq_path5_row3_flat(
                    U1, U2, U3, U4, origin,
                    nu, nu, mu, -nu, -nu, row, padded_size))
            accumulator = _hisq_add_tuple3(accumulator,
                _hisq_path5_row3_flat(
                    U1, U2, U3, U4, origin,
                    -nu, -nu, mu, nu, nu, row, padded_size))
        end
    end
    return accumulator
end

@inline function _hisq_store_row3_flat!(
    V1, V2, V3, V4, mu, site, row, values, coefficient, padded_size,
)
    site_index = _hisq_flat_site_index3(site, padded_size)
    index1 = row + 9 * (site_index - 1)
    index2 = index1 + 3
    index3 = index2 + 3
    @inbounds if mu == 1
        V1[index1] = coefficient * values[1]
        V1[index2] = coefficient * values[2]
        V1[index3] = coefficient * values[3]
    elseif mu == 2
        V2[index1] = coefficient * values[1]
        V2[index2] = coefficient * values[2]
        V2[index3] = coefficient * values[3]
    elseif mu == 3
        V3[index1] = coefficient * values[1]
        V3[index2] = coefficient * values[2]
        V3[index3] = coefficient * values[3]
    else
        V4[index1] = coefficient * values[1]
        V4[index2] = coefficient * values[2]
        V4[index3] = coefficient * values[3]
    end
    return nothing
end

@inline function _hisq_add_row3_flat!(
    V1, V2, V3, V4, mu, site, row, values, coefficient, padded_size,
)
    site_index = _hisq_flat_site_index3(site, padded_size)
    index1 = row + 9 * (site_index - 1)
    index2 = index1 + 3
    index3 = index2 + 3
    @inbounds if mu == 1
        V1[index1] += coefficient * values[1]
        V1[index2] += coefficient * values[2]
        V1[index3] += coefficient * values[3]
    elseif mu == 2
        V2[index1] += coefficient * values[1]
        V2[index2] += coefficient * values[2]
        V2[index3] += coefficient * values[3]
    elseif mu == 3
        V3[index1] += coefficient * values[1]
        V3[index2] += coefficient * values[2]
        V3[index3] += coefficient * values[3]
    else
        V4[index1] += coefficient * values[1]
        V4[index2] += coefficient * values[2]
        V4[index3] += coefficient * values[3]
    end
    return nothing
end

@inline function _hisq_staple3_row(
    U1, U2, U3, U4, origin, mu, row, ::Val{NC},
) where NC
    direct, _ = _hisq_oriented_row(
        U1, U2, U3, U4, origin, mu, row, Val(NC))
    accumulator = zero(direct)
    @inbounds for nu in 1:4
        nu == mu && continue
        accumulator += _hisq_path_row(
            U1, U2, U3, U4, origin, (nu, mu, -nu), row, Val(NC))
        accumulator += _hisq_path_row(
            U1, U2, U3, U4, origin, (-nu, mu, nu), row, Val(NC))
    end
    return accumulator
end

@inline function _hisq_staple5_row(
    U1, U2, U3, U4, origin, mu, row, part::Val{P}, ::Val{NC},
) where {P,NC}
    direct, _ = _hisq_oriented_row(
        U1, U2, U3, U4, origin, mu, row, Val(NC))
    accumulator = zero(direct)
    sign_nu, sign_rho = _hisq_five_signs(part)
    @inbounds for nu in 1:4
        nu == mu && continue
        for rho in 1:4
            (rho == mu || rho == nu) && continue
            signed_nu = sign_nu * nu
            signed_rho = sign_rho * rho
            accumulator += _hisq_path_row(
                U1, U2, U3, U4, origin,
                (signed_nu, signed_rho, mu, -signed_rho, -signed_nu),
                row, Val(NC))
        end
    end
    return accumulator
end

@inline function _hisq_staple7_row(
    U1, U2, U3, U4, origin, mu, row, part::Val{P}, ::Val{NC},
) where {P,NC}
    direct, _ = _hisq_oriented_row(
        U1, U2, U3, U4, origin, mu, row, Val(NC))
    accumulator = zero(direct)
    sign_nu, sign_rho, sign_sigma = _hisq_seven_signs(part)
    @inbounds for nu in 1:4
        nu == mu && continue
        for rho in 1:4
            (rho == mu || rho == nu) && continue
            sigma = 10 - mu - nu - rho
            signed_nu = sign_nu * nu
            signed_rho = sign_rho * rho
            signed_sigma = sign_sigma * sigma
            accumulator += _hisq_path_row(
                U1, U2, U3, U4, origin,
                (signed_nu, signed_rho, signed_sigma, mu,
                 -signed_sigma, -signed_rho, -signed_nu),
                row, Val(NC))
        end
    end
    return accumulator
end

@inline function _hisq_lepage_row(
    U1, U2, U3, U4, origin, mu, row, ::Val{NC},
) where NC
    direct, _ = _hisq_oriented_row(
        U1, U2, U3, U4, origin, mu, row, Val(NC))
    accumulator = zero(direct)
    @inbounds for nu in 1:4
        nu == mu && continue
        accumulator += _hisq_path_row(
            U1, U2, U3, U4, origin,
            (nu, nu, mu, -nu, -nu), row, Val(NC))
        accumulator += _hisq_path_row(
            U1, U2, U3, U4, origin,
            (-nu, -nu, mu, nu, nu), row, Val(NC))
    end
    return accumulator
end

@inline function _hisq_store_row!(
    V1, V2, V3, V4, mu, site, row, values, coefficient, ::Val{NC},
) where NC
    if mu == 1
        @inbounds for column in 1:NC
            V1[row, column, site...] = coefficient * values[column]
        end
    elseif mu == 2
        @inbounds for column in 1:NC
            V2[row, column, site...] = coefficient * values[column]
        end
    elseif mu == 3
        @inbounds for column in 1:NC
            V3[row, column, site...] = coefficient * values[column]
        end
    else
        @inbounds for column in 1:NC
            V4[row, column, site...] = coefficient * values[column]
        end
    end
    return nothing
end

@inline function _hisq_add_row!(
    V1, V2, V3, V4, mu, site, row, values, coefficient, ::Val{NC},
) where NC
    if mu == 1
        @inbounds for column in 1:NC
            V1[row, column, site...] += coefficient * values[column]
        end
    elseif mu == 2
        @inbounds for column in 1:NC
            V2[row, column, site...] += coefficient * values[column]
        end
    elseif mu == 3
        @inbounds for column in 1:NC
            V3[row, column, site...] += coefficient * values[column]
        end
    else
        @inbounds for column in 1:NC
            V4[row, column, site...] += coefficient * values[column]
        end
    end
    return nothing
end

@inline function _hisq_combined_row(combined_index, volume, ::Val{NC}) where NC
    zero_based = combined_index - 1
    site_index = mod(zero_based, volume) + 1
    row_and_direction = div(zero_based, volume)
    row = mod(row_and_direction, NC) + 1
    mu = div(row_and_direction, NC) + 1
    return site_index, row, mu
end

@inline function kernel_hisq_fat7_initialize!(
    combined_index, V1, V2, V3, V4, U1, U2, U3, U4,
    coefficient, volume, ::Val{NC}, ::Val{nw}, indexer,
) where {NC,nw}
    site_index, row, mu = _hisq_combined_row(
        combined_index, volume, Val(NC))
    origin = delinearize(indexer, site_index, nw)
    direct, _ = _hisq_oriented_row(
        U1, U2, U3, U4, origin, mu, row, Val(NC))
    _hisq_store_row!(
        V1, V2, V3, V4, mu, origin, row, direct, coefficient, Val(NC))
    return nothing
end

@inline function kernel_hisq_fat7_staple3!(
    combined_index, V1, V2, V3, V4, U1, U2, U3, U4,
    coefficient, volume, ::Val{NC}, ::Val{nw}, indexer,
) where {NC,nw}
    site_index, row, mu = _hisq_combined_row(
        combined_index, volume, Val(NC))
    origin = delinearize(indexer, site_index, nw)
    staple = _hisq_staple3_row(
        U1, U2, U3, U4, origin, mu, row, Val(NC))
    _hisq_add_row!(
        V1, V2, V3, V4, mu, origin, row, staple, coefficient, Val(NC))
    return nothing
end

@inline function kernel_hisq_fat7_staple5!(
    combined_index, V1, V2, V3, V4, U1, U2, U3, U4,
    coefficient, volume, part::Val{P}, ::Val{NC}, ::Val{nw}, indexer,
) where {P,NC,nw}
    site_index, row, mu = _hisq_combined_row(
        combined_index, volume, Val(NC))
    origin = delinearize(indexer, site_index, nw)
    staple = _hisq_staple5_row(
        U1, U2, U3, U4, origin, mu, row, part, Val(NC))
    _hisq_add_row!(
        V1, V2, V3, V4, mu, origin, row, staple, coefficient, Val(NC))
    return nothing
end

@inline function kernel_hisq_fat7_staple7!(
    combined_index, V1, V2, V3, V4, U1, U2, U3, U4,
    coefficient, volume, part::Val{P}, ::Val{NC}, ::Val{nw}, indexer,
) where {P,NC,nw}
    site_index, row, mu = _hisq_combined_row(
        combined_index, volume, Val(NC))
    origin = delinearize(indexer, site_index, nw)
    staple = _hisq_staple7_row(
        U1, U2, U3, U4, origin, mu, row, part, Val(NC))
    _hisq_add_row!(
        V1, V2, V3, V4, mu, origin, row, staple, coefficient, Val(NC))
    return nothing
end

@inline function kernel_hisq_fat7_lepage!(
    combined_index, V1, V2, V3, V4, U1, U2, U3, U4,
    coefficient, volume, ::Val{NC}, ::Val{nw}, indexer,
) where {NC,nw}
    site_index, row, mu = _hisq_combined_row(
        combined_index, volume, Val(NC))
    origin = delinearize(indexer, site_index, nw)
    staple = _hisq_lepage_row(
        U1, U2, U3, U4, origin, mu, row, Val(NC))
    _hisq_add_row!(
        V1, V2, V3, V4, mu, origin, row, staple, coefficient, Val(NC))
    return nothing
end

@inline function kernel_hisq_fat7_initialize_nc3_cuda!(
    combined_index, V1, V2, V3, V4, U1, U2, U3, U4,
    coefficient, volume, padded_size, ::Val{nw}, indexer,
) where nw
    site_index, row, mu = _hisq_combined_row(
        combined_index, volume, Val(3))
    origin = delinearize(indexer, site_index, nw)
    direct, _ = _hisq_oriented_row3_flat(
        U1, U2, U3, U4, origin, mu, row, padded_size)
    _hisq_store_row3_flat!(
        V1, V2, V3, V4, mu, origin, row, direct, coefficient, padded_size)
    return nothing
end

@inline function kernel_hisq_fat7_staple3_nc3_cuda!(
    combined_index, V1, V2, V3, V4, U1, U2, U3, U4,
    coefficient, volume, padded_size, ::Val{nw}, indexer,
) where nw
    site_index, row, mu = _hisq_combined_row(
        combined_index, volume, Val(3))
    origin = delinearize(indexer, site_index, nw)
    staple = _hisq_staple3_row3_flat(
        U1, U2, U3, U4, origin, mu, row, padded_size)
    _hisq_add_row3_flat!(
        V1, V2, V3, V4, mu, origin, row, staple, coefficient, padded_size)
    return nothing
end

@inline function kernel_hisq_fat7_staple5_nc3_cuda!(
    combined_index, V1, V2, V3, V4, U1, U2, U3, U4,
    coefficient, volume, part::Val{P}, padded_size, ::Val{nw}, indexer,
) where {P,nw}
    site_index, row, mu = _hisq_combined_row(
        combined_index, volume, Val(3))
    origin = delinearize(indexer, site_index, nw)
    staple = _hisq_staple5_row3_flat(
        U1, U2, U3, U4, origin, mu, row, part, padded_size)
    _hisq_add_row3_flat!(
        V1, V2, V3, V4, mu, origin, row, staple, coefficient, padded_size)
    return nothing
end

@inline function kernel_hisq_fat7_staple7_nc3_cuda!(
    combined_index, V1, V2, V3, V4, U1, U2, U3, U4,
    coefficient, volume, part::Val{P}, padded_size, ::Val{nw}, indexer,
) where {P,nw}
    site_index, row, mu = _hisq_combined_row(
        combined_index, volume, Val(3))
    origin = delinearize(indexer, site_index, nw)
    staple = _hisq_staple7_row3_flat(
        U1, U2, U3, U4, origin, mu, row, part, padded_size)
    _hisq_add_row3_flat!(
        V1, V2, V3, V4, mu, origin, row, staple, coefficient, padded_size)
    return nothing
end

@inline function kernel_hisq_fat7_lepage_nc3_cuda!(
    combined_index, V1, V2, V3, V4, U1, U2, U3, U4,
    coefficient, volume, padded_size, ::Val{nw}, indexer,
) where nw
    site_index, row, mu = _hisq_combined_row(
        combined_index, volume, Val(3))
    origin = delinearize(indexer, site_index, nw)
    staple = _hisq_lepage_row3_flat(
        U1, U2, U3, U4, origin, mu, row, padded_size)
    _hisq_add_row3_flat!(
        V1, V2, V3, V4, mu, origin, row, staple, coefficient, padded_size)
    return nothing
end

function _validate_hisq_smearing_output(fat_links, thin_links)
    _validate_staggered_gauge_links(thin_links)
    _validate_staggered_gauge_links(fat_links)
    reference = thin_links[1]

    for (mu, thin_link) in enumerate(thin_links)
        eltype(thin_link.A) == eltype(reference.A) || throw(ArgumentError(
            "thin link U[$mu] has a different element type"))
    end

    nw = reference.nw
    if !iszero(nw) && any(local_extent -> local_extent < nw, reference.PN)
        throw(ArgumentError(
            "each local lattice extent must be at least the halo width nw=$nw"))
    end

    for (mu, fat_link) in enumerate(fat_links)
        fat_link.NC1 == reference.NC1 && fat_link.NC2 == reference.NC2 ||
            throw(ArgumentError(
                "Fat7 output V[$mu] and thin links have different matrix sizes"))
        fat_link.gsize == reference.gsize && fat_link.PN == reference.PN &&
            fat_link.dims == reference.dims && fat_link.nw == reference.nw ||
            throw(ArgumentError(
                "Fat7 output V[$mu] and thin links use different lattice geometry"))
        eltype(fat_link.A) == eltype(reference.A) || throw(ArgumentError(
            "Fat7 output V[$mu] and thin links have different element types"))
    end

    for fat_link in fat_links, thin_link in thin_links
        (fat_link === thin_link || fat_link.A === thin_link.A) &&
            throw(ArgumentError(
            "HISQ Fat7 smearing does not support aliased input and output links"))
    end
    for mu in 1:4, nu in (mu+1):4
        (fat_links[mu] === fat_links[nu] ||
         fat_links[mu].A === fat_links[nu].A) &&
            throw(ArgumentError(
                "each HISQ Fat7 output direction must use distinct storage"))
    end
    return nothing
end

function _hisq_oriented_link_nowing(thin_links, direction, offset)
    axis = abs(direction)
    if direction < 0
        offset = ntuple(
            d -> offset[d] + ifelse(d == axis, -1, 0), 4)
    end
    shifted = _materialize_periodic_shift(thin_links[axis], offset)
    oriented = direction > 0 ? shifted : adjoint(shifted)
    if direction > 0
        offset = ntuple(
            d -> offset[d] + ifelse(d == axis, 1, 0), 4)
    end
    return oriented, offset
end

function _hisq_accumulate_path_nowing!(
    fat_link, thin_links, path, coefficient, product, scratch,
)
    offset = (0, 0, 0, 0)
    oriented, offset = _hisq_oriented_link_nowing(
        thin_links, path[1], offset)
    substitute!(product, oriented)

    for path_index in 2:length(path)
        oriented, offset = _hisq_oriented_link_nowing(
            thin_links, path[path_index], offset)
        mul!(scratch, product, oriented)
        product, scratch = scratch, product
    end
    add_matrix!(fat_link, product, coefficient)
    return nothing
end

function _hisq_fat7_nowing!(fat_links, thin_links, coefficients)
    coefficient_1, coefficient_3, coefficient_5, coefficient_7,
        coefficient_lepage = coefficients
    for mu in 1:4
        fat_link = fat_links[mu]
        clear_matrix!(fat_link)
        add_matrix!(fat_link, thin_links[mu], coefficient_1)
        product = similar(fat_link)
        scratch = similar(fat_link)

        for nu in 1:4
            nu == mu && continue
            for sign_nu in _hisq_transverse_signs
                signed_nu = sign_nu * nu
                _hisq_accumulate_path_nowing!(
                    fat_link, thin_links,
                    (signed_nu, mu, -signed_nu), coefficient_3,
                    product, scratch)
                if !iszero(coefficient_lepage)
                    _hisq_accumulate_path_nowing!(
                        fat_link, thin_links,
                        (signed_nu, signed_nu, mu,
                         -signed_nu, -signed_nu), coefficient_lepage,
                        product, scratch)
                end
            end

            for rho in 1:4
                (rho == mu || rho == nu) && continue
                for sign_nu in _hisq_transverse_signs,
                    sign_rho in _hisq_transverse_signs
                    signed_nu = sign_nu * nu
                    signed_rho = sign_rho * rho
                    _hisq_accumulate_path_nowing!(
                        fat_link, thin_links,
                        (signed_nu, signed_rho, mu,
                         -signed_rho, -signed_nu), coefficient_5,
                        product, scratch)
                end

                for sigma in 1:4
                    (sigma == mu || sigma == nu || sigma == rho) && continue
                    for sign_nu in _hisq_transverse_signs,
                        sign_rho in _hisq_transverse_signs,
                        sign_sigma in _hisq_transverse_signs
                        signed_nu = sign_nu * nu
                        signed_rho = sign_rho * rho
                        signed_sigma = sign_sigma * sigma
                        _hisq_accumulate_path_nowing!(
                            fat_link, thin_links,
                            (signed_nu, signed_rho, signed_sigma, mu,
                             -signed_sigma, -signed_rho, -signed_nu),
                            coefficient_7, product, scratch)
                    end
                end
            end
        end
    end
    return fat_links
end

"""
    hisq_fat7_level1!(fat_links, thin_links)
    hisq_fat7_level1(thin_links)

Construct the unprojected level-1 Fat7 links used by HISQ. The coefficients
and path multiplicities follow SIMULATeQCD:

```
V_mu = (1/8) U_mu + (1/16) sum(3-link paths)
     + (1/64) sum(5-link paths) + (1/384) sum(7-link paths).
```

The four input links must be square, periodic, and share one four-dimensional
lattice geometry. The halo path requires `nw >= 1`; `nw=0` is supported by a
slower shift-materializing fallback. Staggered and fermion boundary phases
are not included in the output.
"""
function hisq_fat7_level1!(
    fat_links::Vector{TO}, thin_links::Vector{TI},
) where {TO<:LatticeMatrix{4},TI<:LatticeMatrix{4}}
    _validate_staggered_gauge_links(thin_links)
    real_type = typeof(real(zero(eltype(thin_links[1].A))))
    coefficients = (
        one(real_type) / 8,
        one(real_type) / 16,
        one(real_type) / 64,
        one(real_type) / 384,
        zero(real_type),
    )

    return _hisq_fat7!(fat_links, thin_links, coefficients)
end

function _hisq_fat7_nc3_cuda!(fat_links, thin_links, coefficients)
    U1, U2, U3, U4 = thin_links
    V1, V2, V3, V4 = fat_links
    coefficient_1, coefficient_3, coefficient_5, coefficient_7,
        coefficient_lepage = coefficients
    volume = prod(V1.PN)
    combined_volume = 12 * volume
    padded_size = ntuple(d -> size(U1.A, d + 2), 4)
    common_arguments = (
        reshape(V1.A, :), reshape(V2.A, :),
        reshape(V3.A, :), reshape(V4.A, :),
        reshape(U1.A, :), reshape(U2.A, :),
        reshape(U3.A, :), reshape(U4.A, :),
    )
    geometry_arguments = (volume, padded_size, Val(V1.nw), V1.indexer)

    _hisq_parallel_for(
        combined_volume, kernel_hisq_fat7_initialize_nc3_cuda!,
        common_arguments..., coefficient_1, geometry_arguments...)
    _hisq_parallel_for(
        combined_volume, kernel_hisq_fat7_staple3_nc3_cuda!,
        common_arguments..., coefficient_3, geometry_arguments...)
    for part in 1:4
        _hisq_parallel_for(
            combined_volume, kernel_hisq_fat7_staple5_nc3_cuda!,
            common_arguments..., coefficient_5, volume, Val(part),
            padded_size, Val(V1.nw), V1.indexer)
    end
    for part in 1:8
        _hisq_parallel_for(
            combined_volume, kernel_hisq_fat7_staple7_nc3_cuda!,
            common_arguments..., coefficient_7, volume, Val(part),
            padded_size, Val(V1.nw), V1.indexer)
    end
    if !iszero(coefficient_lepage)
        _hisq_parallel_for(
            combined_volume, kernel_hisq_fat7_lepage_nc3_cuda!,
            common_arguments..., coefficient_lepage, geometry_arguments...)
    end
    mark_halo_dirty!.(fat_links)
    return fat_links
end

function _hisq_fat7!(fat_links, thin_links, coefficients)
    _validate_hisq_smearing_output(fat_links, thin_links)

    if iszero(thin_links[1].nw)
        return _hisq_fat7_nowing!(
            fat_links, thin_links, coefficients)
    end

    for link in thin_links
        ensure_halo!(link)
    end
    U1, U2, U3, U4 = thin_links
    V1, V2, V3, V4 = fat_links
    coefficient_1, coefficient_3, coefficient_5, coefficient_7,
        coefficient_lepage = coefficients
    if V1.NC1 == 3 && JACC.backend == "cuda"
        return _hisq_fat7_nc3_cuda!(fat_links, thin_links, coefficients)
    end
    volume = prod(V1.PN)
    combined_volume = 4 * V1.NC1 * volume
    common_arguments = (
        V1.A, V2.A, V3.A, V4.A, U1.A, U2.A, U3.A, U4.A)
    geometry_arguments = (volume, Val(V1.NC1), Val(V1.nw), V1.indexer)

    _hisq_parallel_for(
        combined_volume, kernel_hisq_fat7_initialize!, common_arguments...,
        coefficient_1, geometry_arguments...)
    _hisq_parallel_for(
        combined_volume, kernel_hisq_fat7_staple3!, common_arguments...,
        coefficient_3, geometry_arguments...)
    for part in 1:4
        _hisq_parallel_for(
            combined_volume, kernel_hisq_fat7_staple5!, common_arguments...,
            coefficient_5, volume, Val(part),
            Val(V1.NC1), Val(V1.nw), V1.indexer)
    end
    for part in 1:8
        _hisq_parallel_for(
            combined_volume, kernel_hisq_fat7_staple7!, common_arguments...,
            coefficient_7, volume, Val(part),
            Val(V1.NC1), Val(V1.nw), V1.indexer)
    end
    if !iszero(coefficient_lepage)
        _hisq_parallel_for(
            combined_volume, kernel_hisq_fat7_lepage!, common_arguments...,
            coefficient_lepage, geometry_arguments...)
    end
    mark_halo_dirty!.(fat_links)
    return fat_links
end

function hisq_fat7_level1(thin_links::Vector{T}) where {T<:LatticeMatrix{4}}
    _validate_staggered_gauge_links(thin_links)
    fat_links = [similar(link) for link in thin_links]
    return hisq_fat7_level1!(fat_links, thin_links)
end

export hisq_fat7_level1!, hisq_fat7_level1

"""
    hisq_fat7_level2!(fat_links, reunitarized_links, naik_epsilon=0)
    hisq_fat7_level2(reunitarized_links, naik_epsilon=0)

Construct the level-2 HISQ Fat7 links, including the five-link Lepage
correction.  The SIMULATeQCD coefficients are
`(1 + naik_epsilon/8, 1/16, 1/64, 1/384, -1/8)` for the one-link,
three-link, five-link, seven-link, and Lepage paths respectively.

The halo kernel requires `nw >= 2`; `nw=0` uses the periodic
shift-materializing fallback.
"""
function hisq_fat7_level2!(
    fat_links::Vector{TO}, reunitarized_links::Vector{TI}, naik_epsilon,
) where {TO<:LatticeMatrix{4},TI<:LatticeMatrix{4}}
    nw = reunitarized_links[1].nw
    !iszero(nw) && nw < 2 && throw(ArgumentError(
        "HISQ level-2 Fat7 smearing requires nw >= 2 or nw == 0"))
    real_type = typeof(real(zero(eltype(reunitarized_links[1].A))))
    epsilon = convert(real_type, naik_epsilon)
    coefficients = (
        one(real_type) + epsilon / 8,
        one(real_type) / 16,
        one(real_type) / 64,
        one(real_type) / 384,
        -one(real_type) / 8,
    )
    return _hisq_fat7!(fat_links, reunitarized_links, coefficients)
end

hisq_fat7_level2!(fat_links, reunitarized_links; naik_epsilon=0) =
    hisq_fat7_level2!(fat_links, reunitarized_links, naik_epsilon)

function hisq_fat7_level2(
    reunitarized_links::Vector{T}, naik_epsilon,
) where {T<:LatticeMatrix{4}}
    fat_links = [similar(link) for link in reunitarized_links]
    return hisq_fat7_level2!(
        fat_links, reunitarized_links, naik_epsilon)
end

hisq_fat7_level2(reunitarized_links; naik_epsilon=0) =
    hisq_fat7_level2(reunitarized_links, naik_epsilon)

export hisq_fat7_level2!, hisq_fat7_level2
