using LatticeMatrices
using MPI
using Printf
import JACC

JACC.@init_backend

function _median(values)
    sorted = sort(values)
    return sorted[(length(sorted) + 1) ÷ 2]
end

# Prevent a tight benchmark loop from being algebraically reduced to one epoch
# increment. Production mutators can still inline `mark_halo_dirty!` normally,
# so this measures a conservative standalone-call upper bound.
Base.@noinline _standalone_mark!(lattice) = mark_halo_dirty!(lattice)

function _benchmark_ns(f, evaluations, comm; samples)
    f()
    measurements = Vector{Float64}(undef, samples)
    for sample in 1:samples
        MPI.Barrier(comm)
        start = time_ns()
        for _ in 1:evaluations
            f()
        end
        elapsed = time_ns() - start
        slowest_elapsed = MPI.Allreduce(elapsed, MPI.MAX, comm)
        measurements[sample] = slowest_elapsed / evaluations
    end
    return _median(measurements)
end

function main()
    comm = MPI.COMM_WORLD
    nprocs = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    local_x = parse(Int, get(ENV, "LM_BENCH_LOCAL_X", "16"))
    fast_evaluations = parse(
        Int, get(ENV, "LM_BENCH_FAST_ITERS", "2000000")
    )
    sync_evaluations = parse(
        Int, get(ENV, "LM_BENCH_SYNC_ITERS", "200")
    )
    samples = parse(Int, get(ENV, "LM_BENCH_SAMPLES", "9"))

    global_size = (local_x * nprocs, 8, 8, 8)
    process_grid = (nprocs, 1, 1, 1)
    lattice = LatticeMatrix(
        3, 3, 4, global_size, process_grid;
        nw=1,
        elementtype=ComplexF64,
        comm0=comm,
    )
    set_halo!(lattice)

    mark_ns = _benchmark_ns(
        () -> _standalone_mark!(lattice), fast_evaluations, comm; samples
    )
    ensure_halo!(lattice)
    clean_ensure_ns = _benchmark_ns(
        () -> ensure_halo!(lattice), fast_evaluations, comm; samples
    )
    dirty_ensure_ns = _benchmark_ns(
        () -> begin
            _standalone_mark!(lattice)
            ensure_halo!(lattice)
        end,
        sync_evaluations,
        comm;
        samples,
    )
    forced_sync_ns = _benchmark_ns(
        () -> set_halo!(lattice), sync_evaluations, comm; samples
    )

    if rank == 0
        println("Halo epoch benchmark")
        println("  MPI ranks:        $nprocs")
        println("  global size:      $global_size")
        println("  process grid:     $process_grid")
        println("  fast iterations:  $fast_evaluations")
        println("  sync iterations:  $sync_evaluations")
        println("  samples:          $samples")
        println()
        println("| operation | median ns/call (slowest rank) |")
        println("|---|---:|")
        @printf("| standalone `mark_halo_dirty!` upper bound | %.2f |\n", mark_ns)
        @printf("| clean `ensure_halo!` | %.2f |\n", clean_ensure_ns)
        @printf("| dirty mark + `ensure_halo!` | %.2f |\n", dirty_ensure_ns)
        @printf("| forced `set_halo!` | %.2f |\n", forced_sync_ns)
        println()
        @printf(
            "Forced synchronization / clean ensure ratio: %.1fx\n",
            forced_sync_ns / clean_ensure_ns,
        )
    end
end

initialized_here = !MPI.Initialized()
initialized_here && MPI.Init()

try
    main()
finally
    initialized_here && MPI.Finalize()
end
