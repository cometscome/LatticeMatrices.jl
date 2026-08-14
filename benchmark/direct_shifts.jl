using LatticeMatrices
using MPI
using Printf
import JACC

JACC.@init_backend

function _parse_tuple(name, default)
    values = split(get(ENV, name, join(default, ',')), ',')
    length(values) == length(default) ||
        error("$name must contain $(length(default)) comma-separated integers")
    return Tuple(parse.(Int, values))
end

function _default_process_grid(nprocs)
    grids = Dict(
        1 => (1, 1, 1, 1),
        2 => (2, 1, 1, 1),
        4 => (2, 2, 1, 1),
        8 => (2, 2, 2, 1),
        16 => (2, 2, 2, 2),
    )
    haskey(grids, nprocs) || error(
        "set LM_BENCH_PROCESS_GRID for $nprocs MPI ranks",
    )
    return grids[nprocs]
end

function _synchronized_median_ms(f, evaluations, samples, comm)
    f()
    JACC.synchronize()
    measurements = Vector{Float64}(undef, samples)
    for sample in eachindex(measurements)
        # Keep unrelated GC/finalizer pauses outside the measured interval.
        GC.gc(false)
        MPI.Barrier(comm)
        start = time_ns()
        for _ in 1:evaluations
            f()
        end
        JACC.synchronize()
        elapsed = time_ns() - start
        measurements[sample] =
            MPI.Allreduce(elapsed, MPI.MAX, comm) / evaluations / 1e6
    end
    sort!(measurements)
    return measurements[(length(measurements) + 1) ÷ 2]
end

function main()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    global_size = _parse_tuple("LM_BENCH_GLOBAL_SIZE", (12, 12, 12, 24))
    default_grid = _default_process_grid(nprocs)
    process_grid =
        _parse_tuple("LM_BENCH_PROCESS_GRID", default_grid)
    prod(process_grid) == nprocs || error(
        "process grid $process_grid does not contain $nprocs ranks",
    )
    all(global_size .% process_grid .== 0) || error(
        "global size $global_size is not divisible by $process_grid",
    )

    nw = parse(Int, get(ENV, "LM_BENCH_NW", "1"))
    evaluations = parse(Int, get(ENV, "LM_BENCH_ITERS", "3"))
    samples = parse(Int, get(ENV, "LM_BENCH_SAMPLES", "7"))
    values = reshape(
        ComplexF64.(1:(2 * prod(global_size))), 2, 1, global_size...,
    )
    lattice = LatticeMatrix(
        values, 4, process_grid; nw, numtemps=2,
    )
    result = similar(lattice)
    initial_pool_size = length(lattice.temps)
    shifts = (
        ("within halo", (nw, 0, 0, 0)),
        ("just beyond halo", (nw + 1, 0, 0, 0)),
        ("mixed long", (5, -4, 7, -9)),
        ("multi-period", (37, -26, 31, -74)),
    )

    timings = map(shifts) do (_, shift)
        elapsed = _synchronized_median_ms(
            evaluations, samples, comm,
        ) do
            with_shifted_lattice(lattice, shift) do shifted
                substitute!(result, shifted)
            end
        end
        all_released = MPI.Allreduce(
            count(lattice.temps._flagusing) == 0 ? 1 : 0, MPI.MIN, comm,
        ) == 1
        pool_stable = MPI.Allreduce(
            length(lattice.temps) == initial_pool_size ? 1 : 0,
            MPI.MIN,
            comm,
        ) == 1
        all_released || error("shift $shift left a temporary buffer leased")
        pool_stable || error("shift $shift grew the temporary-buffer pool")
        elapsed
    end

    if rank == 0
        baseline = timings[1]
        println("Direct-shift benchmark")
        println("  MPI ranks:       $nprocs")
        println("  global size:     $global_size")
        println("  process grid:    $process_grid")
        println("  halo width:      $nw")
        println("  evaluations:     $evaluations")
        println("  samples:         $samples")
        println("  pool slots:      $initial_pool_size (stable)")
        println()
        println("| shift class | shift | median ms/call | vs. within halo |")
        println("|---|---:|---:|---:|")
        for ((label, shift), elapsed) in zip(shifts, timings)
            @printf(
                "| %s | `%s` | %.3f | %.2fx |\n",
                label,
                shift,
                elapsed,
                elapsed / baseline,
            )
        end
    end
end

initialized_here = !MPI.Initialized()
initialized_here && MPI.Init()
try
    main()
finally
    initialized_here && MPI.Finalize()
end
