const _hisq_transverse_signs = (-1, 1)

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

@inline function _hisq_load_oriented_link!(
    link, U1, U2, U3, U4, site, direction, ::Val{NC},
) where NC
    if direction > 0
        @inbounds for column in 1:NC, row in 1:NC
            link[row, column] = _hisq_link_element(
                U1, U2, U3, U4, direction, row, column, site)
        end
        return _hisq_shift_site(site, direction)
    end

    previous_site = _hisq_shift_site(site, direction)
    axis = -direction
    @inbounds for column in 1:NC, row in 1:NC
        link[row, column] = conj(_hisq_link_element(
            U1, U2, U3, U4, axis, column, row, previous_site))
    end
    return previous_site
end

@inline function _hisq_accumulate_path!(
    accumulator, product, link, scratch,
    U1, U2, U3, U4, origin, path, coefficient, ::Val{NC},
) where NC
    eye!(product)
    site = origin
    @inbounds for step in path
        site = _hisq_load_oriented_link!(
            link, U1, U2, U3, U4, site, step, Val(NC))
        gemm!(scratch, product, link)
        product, scratch = scratch, product
    end
    axpy!(coefficient, product, accumulator)
    return nothing
end

@inline function kernel_hisq_fat7!(
    site_index, fat_link, U1, U2, U3, U4,
    coefficient_1, coefficient_3, coefficient_5, coefficient_7,
    ::Val{mu}, ::Val{NC}, ::Val{nw}, indexer,
) where {mu,NC,nw}
    origin = delinearize(indexer, site_index, nw)
    accumulator = MMatrix{NC,NC,eltype(fat_link)}(undef)
    product = MMatrix{NC,NC,eltype(fat_link)}(undef)
    link = MMatrix{NC,NC,eltype(fat_link)}(undef)
    scratch = MMatrix{NC,NC,eltype(fat_link)}(undef)

    @inbounds for column in 1:NC, row in 1:NC
        accumulator[row, column] = coefficient_1 * _hisq_link_element(
            U1, U2, U3, U4, mu, row, column, origin)
    end

    # The path sets are exactly the Fat7 3-, 5-, and 7-link staples used by
    # SIMULATeQCD. Ordered transverse axes account for every path ordering;
    # the signs account for forward and backward transverse links.
    for nu in 1:4
        nu == mu && continue
        for sign_nu in _hisq_transverse_signs
            signed_nu = sign_nu * nu
            path3 = (signed_nu, mu, -signed_nu)
            _hisq_accumulate_path!(
                accumulator, product, link, scratch,
                U1, U2, U3, U4, origin, path3, coefficient_3, Val(NC))
        end

        for rho in 1:4
            (rho == mu || rho == nu) && continue
            for sign_nu in _hisq_transverse_signs
                signed_nu = sign_nu * nu
                for sign_rho in _hisq_transverse_signs
                    signed_rho = sign_rho * rho
                    path5 = (
                        signed_nu, signed_rho, mu,
                        -signed_rho, -signed_nu,
                    )
                    _hisq_accumulate_path!(
                        accumulator, product, link, scratch,
                        U1, U2, U3, U4, origin, path5,
                        coefficient_5, Val(NC))
                end
            end

            for sigma in 1:4
                (sigma == mu || sigma == nu || sigma == rho) && continue
                for sign_nu in _hisq_transverse_signs
                    signed_nu = sign_nu * nu
                    for sign_rho in _hisq_transverse_signs
                        signed_rho = sign_rho * rho
                        for sign_sigma in _hisq_transverse_signs
                            signed_sigma = sign_sigma * sigma
                            path7 = (
                                signed_nu, signed_rho, signed_sigma, mu,
                                -signed_sigma, -signed_rho, -signed_nu,
                            )
                            _hisq_accumulate_path!(
                                accumulator, product, link, scratch,
                                U1, U2, U3, U4, origin, path7,
                                coefficient_7, Val(NC))
                        end
                    end
                end
            end
        end
    end

    @inbounds for column in 1:NC, row in 1:NC
        fat_link[row, column, origin...] = accumulator[row, column]
    end
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

function _hisq_fat7_level1_nowing!(fat_links, thin_links, coefficients)
    coefficient_1, coefficient_3, coefficient_5, coefficient_7 = coefficients
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
    )

    return _hisq_fat7!(fat_links, thin_links, coefficients)
end

function _hisq_fat7!(fat_links, thin_links, coefficients)
    _validate_hisq_smearing_output(fat_links, thin_links)

    if iszero(thin_links[1].nw)
        return _hisq_fat7_level1_nowing!(
            fat_links, thin_links, coefficients)
    end

    for link in thin_links
        ensure_halo!(link)
    end
    U1, U2, U3, U4 = thin_links
    coefficient_1, coefficient_3, coefficient_5, coefficient_7 = coefficients
    for mu in 1:4
        fat_link = fat_links[mu]
        _parallel_for_mutating!(fat_link,
            prod(fat_link.PN), kernel_hisq_fat7!, fat_link.A,
            U1.A, U2.A, U3.A, U4.A,
            coefficient_1, coefficient_3, coefficient_5, coefficient_7,
            Val(mu), Val(fat_link.NC1), Val(fat_link.nw), fat_link.indexer)
    end
    return fat_links
end

function hisq_fat7_level1(thin_links::Vector{T}) where {T<:LatticeMatrix{4}}
    _validate_staggered_gauge_links(thin_links)
    fat_links = [similar(link) for link in thin_links]
    return hisq_fat7_level1!(fat_links, thin_links)
end

export hisq_fat7_level1!, hisq_fat7_level1
