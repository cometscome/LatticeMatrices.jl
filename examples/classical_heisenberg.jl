#!/usr/bin/env julia

import JACC
JACC.@init_backend

using LatticeMatrices
using MPI
using Printf

module ClassicalHeisenbergExample

using LatticeMatrices
using MPI
using Printf
import JACC

const LITERATURE_KC = 0.693002
const LITERATURE_KC_ERROR = 0.000002
const LITERATURE_BINDER = 0.6217
const LITERATURE_BINDER_ERROR = 0.0008

struct SimulationResult
    L::Int
    coupling::Float64
    measurements::Int
    energy::Float64
    energy_error::Float64
    magnetization::Float64
    magnetization_error::Float64
    binder::Float64
    binder_error::Float64
    maximum_spin_norm_error::Float64
end

@inline function _global_parity(local_indices, coordinates, local_size)
    coordinate_sum = 0
    @inbounds for direction in eachindex(local_indices)
        coordinate_sum +=
            coordinates[direction] * local_size[direction] +
            local_indices[direction]
    end
    return iszero(coordinate_sum & 1)
end

@inline function _neighbor_field(spins, ix, iy, iz)
    h1 = spins[1, 1, ix + 1, iy, iz] + spins[1, 1, ix - 1, iy, iz] +
         spins[1, 1, ix, iy + 1, iz] + spins[1, 1, ix, iy - 1, iz] +
         spins[1, 1, ix, iy, iz + 1] + spins[1, 1, ix, iy, iz - 1]
    h2 = spins[2, 1, ix + 1, iy, iz] + spins[2, 1, ix - 1, iy, iz] +
         spins[2, 1, ix, iy + 1, iz] + spins[2, 1, ix, iy - 1, iz] +
         spins[2, 1, ix, iy, iz + 1] + spins[2, 1, ix, iy, iz - 1]
    h3 = spins[3, 1, ix + 1, iy, iz] + spins[3, 1, ix - 1, iy, iz] +
         spins[3, 1, ix, iy + 1, iz] + spins[3, 1, ix, iy - 1, iz] +
         spins[3, 1, ix, iy, iz + 1] + spins[3, 1, ix, iy, iz - 1]
    return h1, h2, h3
end

# Exact single-site heat-bath update for p(s) proportional to exp(K h dot s).
# A checkerboard color can be updated in parallel because all its neighbors
# belong to the other color.
@inline function _heatbath_color_kernel!(
    site,
    spins,
    coupling,
    key,
    target_even,
    coordinates,
    local_size,
    global_size,
    indexer,
    ::Val{nw},
    algorithm,
) where {nw}
    local_indices = delinearize(indexer, site, 0)
    _global_parity(local_indices, coordinates, local_size) == target_even ||
        return nothing

    ix = local_indices[1] + nw
    iy = local_indices[2] + nw
    iz = local_indices[3] + nw
    h1, h2, h3 = _neighbor_field(spins, ix, iy, iz)
    hnorm2 = h1 * h1 + h2 * h2 + h3 * h3

    global_indices = ntuple(
        direction -> coordinates[direction] * local_size[direction] +
                     local_indices[direction],
        3,
    )
    rng = site_rng(key, global_site_id(global_indices, global_size), algorithm)
    rng, uniform1 = rand_uniform_open(rng, eltype(spins))
    rng, uniform2 = rand_uniform(rng, eltype(spins))

    sine_phi = sinpi(2 * uniform2)
    cosine_phi = cospi(2 * uniform2)
    one_value = one(eltype(spins))
    two_value = one_value + one_value

    if hnorm2 <= eps(eltype(spins))
        cosine_theta = two_value * uniform1 - one_value
        sine_theta = sqrt(max(zero(cosine_theta),
                              one_value - cosine_theta * cosine_theta))
        spins[1, 1, ix, iy, iz] = sine_theta * cosine_phi
        spins[2, 1, ix, iy, iz] = sine_theta * sine_phi
        spins[3, 1, ix, iy, iz] = cosine_theta
        return nothing
    end

    hnorm = sqrt(hnorm2)
    inverse_hnorm = inv(hnorm)
    n1 = h1 * inverse_hnorm
    n2 = h2 * inverse_hnorm
    n3 = h3 * inverse_hnorm
    field_coupling = coupling * hnorm

    cosine_theta = if abs(field_coupling) <= sqrt(eps(eltype(spins)))
        two_value * uniform1 - one_value
    else
        # Stable inverse CDF for exp(field_coupling * cos(theta)).
        one_value + log(
            uniform1 + (one_value - uniform1) * exp(-two_value * field_coupling),
        ) / field_coupling
    end
    cosine_theta = min(one_value, max(-one_value, cosine_theta))
    sine_theta = sqrt(max(zero(cosine_theta),
                          one_value - cosine_theta * cosine_theta))

    radial2 = n1 * n1 + n2 * n2
    if radial2 > eps(eltype(spins))
        inverse_radial = inv(sqrt(radial2))
        e11 = -n2 * inverse_radial
        e12 = n1 * inverse_radial
        e13 = zero(n1)
        e21 = -n3 * n1 * inverse_radial
        e22 = -n3 * n2 * inverse_radial
        e23 = radial2 * inverse_radial

        spins[1, 1, ix, iy, iz] = cosine_theta * n1 +
            sine_theta * (cosine_phi * e11 + sine_phi * e21)
        spins[2, 1, ix, iy, iz] = cosine_theta * n2 +
            sine_theta * (cosine_phi * e12 + sine_phi * e22)
        spins[3, 1, ix, iy, iz] = cosine_theta * n3 +
            sine_theta * (cosine_phi * e13 + sine_phi * e23)
    else
        orientation = ifelse(n3 < zero(n3), -one_value, one_value)
        spins[1, 1, ix, iy, iz] = sine_theta * cosine_phi
        spins[2, 1, ix, iy, iz] = orientation * sine_theta * sine_phi
        spins[3, 1, ix, iy, iz] = cosine_theta * n3
    end
    return nothing
end

@inline function _overrelaxation_color_kernel!(
    site,
    spins,
    target_even,
    coordinates,
    local_size,
    indexer,
    ::Val{nw},
) where {nw}
    local_indices = delinearize(indexer, site, 0)
    _global_parity(local_indices, coordinates, local_size) == target_even ||
        return nothing

    ix = local_indices[1] + nw
    iy = local_indices[2] + nw
    iz = local_indices[3] + nw
    h1, h2, h3 = _neighbor_field(spins, ix, iy, iz)
    hnorm2 = h1 * h1 + h2 * h2 + h3 * h3
    hnorm2 > eps(eltype(spins)) || return nothing

    s1 = spins[1, 1, ix, iy, iz]
    s2 = spins[2, 1, ix, iy, iz]
    s3 = spins[3, 1, ix, iy, iz]
    scale = 2 * (s1 * h1 + s2 * h2 + s3 * h3) / hnorm2
    r1 = scale * h1 - s1
    r2 = scale * h2 - s2
    r3 = scale * h3 - s3
    inverse_norm = inv(sqrt(r1 * r1 + r2 * r2 + r3 * r3))
    spins[1, 1, ix, iy, iz] = r1 * inverse_norm
    spins[2, 1, ix, iy, iz] = r2 * inverse_norm
    spins[3, 1, ix, iy, iz] = r3 * inverse_norm
    return nothing
end

function heatbath_color!(spins, coupling, sweep, color, seed)
    ensure_halo!(spins)
    key = RNGStreamKey(seed, sweep, 0, color, 0)
    JACC.parallel_for(
        prod(spins.PN),
        _heatbath_color_kernel!,
        spins.A,
        coupling,
        key,
        iszero(color),
        spins.coords,
        spins.PN,
        spins.gsize,
        spins.indexer,
        Val(spins.nw),
        Philox4x32(),
    )
    mark_halo_dirty!(spins)
    return nothing
end

function heatbath_sweep!(spins, coupling, sweep, seed)
    heatbath_color!(spins, coupling, sweep, 0, seed)
    heatbath_color!(spins, coupling, sweep, 1, seed)
    return nothing
end

function overrelaxation_color!(spins, target_even)
    ensure_halo!(spins)
    JACC.parallel_for(
        prod(spins.PN),
        _overrelaxation_color_kernel!,
        spins.A,
        target_even,
        spins.coords,
        spins.PN,
        spins.indexer,
        Val(spins.nw),
    )
    mark_halo_dirty!(spins)
    return nothing
end

function overrelaxation_sweep!(spins)
    overrelaxation_color!(spins, true)
    overrelaxation_color!(spins, false)
    return nothing
end

@inline function _energy_kernel(site, spins, indexer, ::Val{nw}) where {nw}
    local_indices = delinearize(indexer, site, 0)
    ix = local_indices[1] + nw
    iy = local_indices[2] + nw
    iz = local_indices[3] + nw
    bond_sum = zero(eltype(spins))
    @inbounds for component in 1:3
        value = spins[component, 1, ix, iy, iz]
        bond_sum += value * (
            spins[component, 1, ix + 1, iy, iz] +
            spins[component, 1, ix, iy + 1, iz] +
            spins[component, 1, ix, iy, iz + 1]
        )
    end
    return -bond_sum
end

@inline function _magnetization_kernel(
    site,
    spins,
    component,
    indexer,
    ::Val{nw},
) where {nw}
    indices = delinearize(indexer, site, nw)
    return spins[component, 1, indices...]
end

@inline function _spin_norm_error_kernel(site, spins, indexer, ::Val{nw}) where {nw}
    indices = delinearize(indexer, site, nw)
    norm2 = zero(eltype(spins))
    @inbounds for component in 1:3
        norm2 += abs2(spins[component, 1, indices...])
    end
    return abs(sqrt(norm2) - one(norm2))
end

function measure(spins)
    ensure_halo!(spins)
    sites = prod(spins.PN)
    local_energy = JACC.parallel_reduce(
        sites,
        _energy_kernel,
        spins.A,
        spins.indexer,
        Val(spins.nw);
        init=zero(eltype(spins.A)),
        op=+,
    )
    energy = MPI.Allreduce(local_energy, MPI.SUM, spins.comm)

    magnetization = ntuple(3) do component
        local_component = JACC.parallel_reduce(
            sites,
            _magnetization_kernel,
            spins.A,
            component,
            spins.indexer,
            Val(spins.nw);
            init=zero(eltype(spins.A)),
            op=+,
        )
        MPI.Allreduce(local_component, MPI.SUM, spins.comm)
    end

    volume = prod(spins.gsize)
    magnetization2 = sum(abs2, magnetization) / volume^2
    return energy / volume, sqrt(magnetization2), magnetization2
end

function maximum_spin_norm_error(spins)
    local_error = JACC.parallel_reduce(
        prod(spins.PN),
        _spin_norm_error_kernel,
        spins.A,
        spins.indexer,
        Val(spins.nw);
        init=zero(eltype(spins.A)),
        op=max,
    )
    return MPI.Allreduce(local_error, MPI.MAX, spins.comm)
end

function _block_ranges(sample_count, requested_blocks)
    block_count = min(requested_blocks, sample_count)
    block_count >= 2 || return UnitRange{Int}[]
    return [
        (fld((block - 1) * sample_count, block_count) + 1):fld(
            block * sample_count,
            block_count,
        )
        for block in 1:block_count
    ]
end

function _blocking_error(values, ranges)
    isempty(ranges) && return NaN
    block_means = [sum(@view values[range]) / length(range) for range in ranges]
    mean_of_blocks = sum(block_means) / length(block_means)
    return sqrt(
        sum(abs2(value - mean_of_blocks) for value in block_means) /
        (length(block_means) * (length(block_means) - 1)),
    )
end

function _binder_and_error(magnetization2, ranges)
    sample_count = length(magnetization2)
    magnetization4 = abs2.(magnetization2)
    sum_m2 = sum(magnetization2)
    sum_m4 = sum(magnetization4)
    mean_m2 = sum_m2 / sample_count
    mean_m4 = sum_m4 / sample_count
    binder = 1 - mean_m4 / (3 * mean_m2^2)
    isempty(ranges) && return binder, NaN

    jackknife = similar(collect(ranges), Float64)
    for (block, range) in pairs(ranges)
        retained = sample_count - length(range)
        block_m2 = sum(@view magnetization2[range])
        block_m4 = sum(@view magnetization4[range])
        retained_m2 = (sum_m2 - block_m2) / retained
        retained_m4 = (sum_m4 - block_m4) / retained
        jackknife[block] = 1 - retained_m4 / (3 * retained_m2^2)
    end
    jackknife_mean = sum(jackknife) / length(jackknife)
    error = sqrt(
        (length(jackknife) - 1) / length(jackknife) *
        sum(abs2(value - jackknife_mean) for value in jackknife),
    )
    return binder, error
end

function _default_process_grid(process_count, L)
    best = nothing
    best_score = typemax(Int)
    for px in 1:process_count
        process_count % px == 0 || continue
        L % px == 0 || continue
        for py in 1:(process_count ÷ px)
            process_count % (px * py) == 0 || continue
            pz = process_count ÷ (px * py)
            L % py == 0 && L % pz == 0 || continue
            local_size = (L ÷ px, L ÷ py, L ÷ pz)
            score = maximum(local_size) - minimum(local_size)
            if score < best_score
                best = (px, py, pz)
                best_score = score
            end
        end
    end
    best === nothing && throw(ArgumentError(
        "cannot decompose an L=$L cube over $process_count MPI ranks",
    ))
    return best
end

function simulate(;
    L=12,
    coupling=LITERATURE_KC,
    thermalization=2_000,
    sweeps=10_000,
    measure_every=5,
    overrelaxation_sweeps=2,
    seed=0x48e15eeb,
    blocks=20,
    process_grid=nothing,
)
    iseven(L) || throw(ArgumentError(
        "checkerboard updates with periodic boundaries require an even L",
    ))
    coupling >= 0 || throw(ArgumentError("coupling must be nonnegative"))
    thermalization >= 0 || throw(ArgumentError("thermalization must be nonnegative"))
    sweeps > 0 || throw(ArgumentError("sweeps must be positive"))
    measure_every > 0 || throw(ArgumentError("measure_every must be positive"))
    overrelaxation_sweeps >= 0 || throw(ArgumentError(
        "overrelaxation_sweeps must be nonnegative",
    ))

    MPI.Initialized() || MPI.Init()
    process_count = MPI.Comm_size(MPI.COMM_WORLD)
    process_grid === nothing &&
        (process_grid = _default_process_grid(process_count, L))
    prod(process_grid) == process_count || throw(ArgumentError(
        "process_grid=$process_grid does not contain $process_count ranks",
    ))

    spins = LatticeMatrix(
        3,
        1,
        3,
        (L, L, L),
        process_grid;
        nw=1,
        elementtype=Float64,
    )

    # At K=0 the exact heat bath is a decomposition-independent hot start.
    heatbath_sweep!(spins, 0.0, 0, seed)
    for sweep in 1:thermalization
        heatbath_sweep!(spins, coupling, sweep, seed)
        for _ in 1:overrelaxation_sweeps
            overrelaxation_sweep!(spins)
        end
    end

    energies = Float64[]
    magnetizations = Float64[]
    magnetization2 = Float64[]
    for sweep in 1:sweeps
        global_sweep = thermalization + sweep
        heatbath_sweep!(spins, coupling, global_sweep, seed)
        for _ in 1:overrelaxation_sweeps
            overrelaxation_sweep!(spins)
        end
        if sweep % measure_every == 0
            energy, magnetization, m2 = measure(spins)
            push!(energies, energy)
            push!(magnetizations, magnetization)
            push!(magnetization2, m2)
        end
    end

    isempty(energies) && throw(ArgumentError(
        "sweeps must be at least measure_every to produce a measurement",
    ))
    ranges = _block_ranges(length(energies), blocks)
    binder, binder_error = _binder_and_error(magnetization2, ranges)
    return SimulationResult(
        L,
        coupling,
        length(energies),
        sum(energies) / length(energies),
        _blocking_error(energies, ranges),
        sum(magnetizations) / length(magnetizations),
        _blocking_error(magnetizations, ranges),
        binder,
        binder_error,
        maximum_spin_norm_error(spins),
    )
end

function print_result(result, process_grid)
    @printf("3D classical Heisenberg model, periodic L=%d cube\n", result.L)
    @printf("MPI process grid: (%d, %d, %d)\n", process_grid...)
    @printf("K = beta*J: %.9f\n", result.coupling)
    @printf("measurements: %d\n", result.measurements)
    @printf("E/(J V): %.8f +/- %.8f\n", result.energy, result.energy_error)
    @printf("<|m|>: %.8f +/- %.8f\n",
            result.magnetization, result.magnetization_error)
    @printf("Binder U_L: %.8f +/- %.8f\n", result.binder, result.binder_error)
    @printf("max ||s|-1|: %.3e\n", result.maximum_spin_norm_error)
    @printf("literature K_c: %.6f +/- %.6f\n",
            LITERATURE_KC, LITERATURE_KC_ERROR)
    @printf("literature U*: %.4f +/- %.4f (thermodynamic FSS limit)\n",
            LITERATURE_BINDER, LITERATURE_BINDER_ERROR)
    @printf("U_L - U*: %+.6f (finite-L corrections are expected)\n",
            result.binder - LITERATURE_BINDER)
    return nothing
end

function _usage()
    println("""
    Usage:
      julia --project=. examples/classical_heisenberg.jl [options]

    Options:
      --L=12                 Cubic lattice extent; must be even
      --coupling=0.693002    K = beta*J
      --thermalization=2000 Heat-bath sweeps discarded before measurement
      --sweeps=10000         Heat-bath sweeps used for measurement
      --measure-every=5      Measurement interval in sweeps
      --overrelaxation=2     Microcanonical sweeps after each heat-bath sweep
      --seed=1222729451      Global RNG seed
      --blocks=20            Blocks used for error estimates
      --pes=2,2,1            Optional MPI Cartesian process grid
    """)
end

function _parse_options(arguments)
    options = Dict{String,String}()
    for argument in arguments
        argument in ("-h", "--help") && return nothing
        startswith(argument, "--") || throw(ArgumentError(
            "unknown positional argument: $argument",
        ))
        pair = split(argument[3:end], "="; limit=2)
        length(pair) == 2 || throw(ArgumentError(
            "options must use --name=value syntax: $argument",
        ))
        options[pair[1]] = pair[2]
    end
    return options
end

function main(arguments=ARGS)
    options = _parse_options(arguments)
    if options === nothing
        _usage()
        return nothing
    end

    L = parse(Int, get(options, "L", "12"))
    coupling = parse(Float64, get(options, "coupling", string(LITERATURE_KC)))
    thermalization = parse(Int, get(options, "thermalization", "2000"))
    sweeps = parse(Int, get(options, "sweeps", "10000"))
    measure_every = parse(Int, get(options, "measure-every", "5"))
    overrelaxation = parse(Int, get(options, "overrelaxation", "2"))
    seed = parse(UInt64, get(options, "seed", string(UInt64(0x48e15eeb))))
    blocks = parse(Int, get(options, "blocks", "20"))

    MPI.Initialized() || MPI.Init()
    process_grid = if haskey(options, "pes")
        values = split(options["pes"], ',')
        length(values) == 3 || throw(ArgumentError("--pes requires three integers"))
        Tuple(parse.(Int, values))
    else
        _default_process_grid(MPI.Comm_size(MPI.COMM_WORLD), L)
    end

    result = simulate(;
        L,
        coupling,
        thermalization,
        sweeps,
        measure_every,
        overrelaxation_sweeps=overrelaxation,
        seed,
        blocks,
        process_grid,
    )
    MPI.Comm_rank(MPI.COMM_WORLD) == 0 && print_result(result, process_grid)
    return result
end

export SimulationResult,
    heatbath_sweep!,
    maximum_spin_norm_error,
    measure,
    overrelaxation_sweep!,
    simulate

end

if abspath(PROGRAM_FILE) == @__FILE__
    ClassicalHeisenbergExample.main()
end
