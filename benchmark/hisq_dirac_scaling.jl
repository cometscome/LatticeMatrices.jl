using MPI

MPI.Initialized() || MPI.Init()

import JACC
JACC.@init_backend

using LatticeMatrices
using LinearAlgebra
using Printf
using Statistics

const _HISQ_DIRAC_PROCESS_GRIDS = Dict(
    1 => (1, 1, 1, 1),
    2 => (2, 1, 1, 1),
    4 => (2, 2, 1, 1),
    8 => (2, 2, 2, 1),
    16 => (2, 2, 2, 2),
)

function _hisq_dirac_parse_dimensions(name, default)
    values = Tuple(parse.(Int, split(get(ENV, name, default), ',')))
    length(values) == 4 || error(
        "$name must contain four comma-separated integers")
    return values
end

function _hisq_dirac_process_grid(nranks)
    if haskey(ENV, "HISQ_BENCH_GRID")
        grid = _hisq_dirac_parse_dimensions(
            "HISQ_BENCH_GRID", "1,1,1,1")
        prod(grid) == nranks || error(
            "HISQ_BENCH_GRID=$grid does not match $nranks MPI ranks")
        return grid
    end
    return get(_HISQ_DIRAC_PROCESS_GRIDS, nranks) do
        error("set HISQ_BENCH_GRID explicitly for $nranks MPI ranks")
    end
end

function _hisq_dirac_thin_links(global_size, process_grid)
    arrays = [zeros(ComplexF64, 3, 3, global_size...) for _ in 1:4]
    for mu in 1:4, site in CartesianIndices(global_size)
        coordinates = Tuple(site)
        coordinate = coordinates[1] + 3coordinates[2] +
            5coordinates[3] + 7coordinates[4]
        @inbounds for column in 1:3, row in 1:3
            deterministic_re = 0.013 * (
                2row - column + coordinate + 3mu)
            deterministic_im = 0.017 * (
                row + 2column - coordinate + mu)
            arrays[mu][row, column, coordinates...] =
                0.05deterministic_re + (row == column) +
                0.05deterministic_im * im
        end
    end
    return [LatticeMatrix(array, 4, process_grid; nw=3) for array in arrays]
end

function _hisq_dirac_apply_ready!(output, operator, input)
    mul!(output, operator, input)
    # Make the output immediately usable as the next Krylov vector.  This
    # includes the same output-halo readiness requested from SIMULATeQCD.
    ensure_halo!(output)
    return nothing
end

function _hisq_dirac_rank_maximum_seconds!(
    input_ref, output_ref, operator, repetitions, comm,
)
    JACC.synchronize()
    MPI.Barrier(comm)
    start = time_ns()
    for _ in 1:repetitions
        _hisq_dirac_apply_ready!(output_ref[], operator, input_ref[])
        input_ref[], output_ref[] = output_ref[], input_ref[]
    end
    JACC.synchronize()
    local_seconds = (time_ns() - start) / 1e9 / repetitions
    return MPI.Allreduce(local_seconds, max, comm)
end

function main()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    global_size = _hisq_dirac_parse_dimensions(
        "HISQ_BENCH_GLOBAL", "16,16,16,16")
    process_grid = _hisq_dirac_process_grid(nranks)
    all(global_size[d] % process_grid[d] == 0 for d in 1:4) || error(
        "global size $global_size is not divisible by process grid $process_grid")
    all(global_size[d] ÷ process_grid[d] >= 4 for d in 1:4) || error(
        "every local extent must be at least four")

    repetitions = parse(Int, get(ENV, "HISQ_BENCH_ITERS", "20"))
    samples = parse(Int, get(ENV, "HISQ_BENCH_SAMPLES", "7"))
    warmups = parse(Int, get(ENV, "HISQ_BENCH_WARMUPS", "4"))
    repetitions > 0 || error("HISQ_BENCH_ITERS must be positive")
    samples > 0 || error("HISQ_BENCH_SAMPLES must be positive")
    warmups >= 0 || error("HISQ_BENCH_WARMUPS must be nonnegative")

    thin = _hisq_dirac_thin_links(global_size, process_grid)
    workspace = HISQFat7Workspace(thin[1])
    fat = [similar(link) for link in thin]
    long = [similar(link) for link in thin]
    level1 = [similar(link) for link in thin]
    reunitarized = [similar(link) for link in thin]
    links = hisq_links_from_thin!(
        fat, long, level1, reunitarized, thin, -0.083, workspace)
    operator = HISQDiracOperator4D(links, 0.0; naik_epsilon=-0.083)

    phases = (1, 1, 1, -1)
    fermion_array = ones(ComplexF64, 3, 1, global_size...)
    input_ref = Ref(LatticeMatrix(
        fermion_array, 4, process_grid; nw=3, phases))
    output_ref = Ref(similar(input_ref[]))
    ensure_halo!(input_ref[])

    for _ in 1:warmups
        _hisq_dirac_apply_ready!(output_ref[], operator, input_ref[])
        input_ref[], output_ref[] = output_ref[], input_ref[]
    end
    JACC.synchronize()

    # Measure the steady-state operator rather than an unrelated Julia GC
    # cycle triggered by the (large) one-time smearing construction.
    GC.gc()
    gc_was_enabled = GC.enable(false)
    timings = try
        [_hisq_dirac_rank_maximum_seconds!(
            input_ref, output_ref, operator, repetitions, comm) * 1e3
            for _ in 1:samples]
    finally
        GC.enable(gc_was_enabled)
    end
    if rank == 0
        sorted = sort(timings)
        @printf(
            "RESULT operation=HISQDirac code=LatticeMatrices backend=%s ranks=%d threads=%d global=%dx%dx%dx%d grid=%dx%dx%dx%d iterations=%d samples=%d min_ms=%.9f median_ms=%.9f max_ms=%.9f all_ms=%s\n",
            JACC.backend, nranks, Threads.nthreads(), global_size...,
            process_grid..., repetitions, samples,
            first(sorted), median(sorted), last(sorted), repr(timings))
    end
    return nothing
end

main()
