using LatticeMatrices
using MPI
using Printf
import JACC

JACC.@init_backend

function _hisq_benchmark_links(extent)
    lattice_size = ntuple(_ -> extent, 4)
    arrays = [zeros(ComplexF64, 3, 3, lattice_size...) for _ in 1:4]
    for mu in 1:4, site in CartesianIndices(lattice_size)
        coordinates = Tuple(site)
        phase = 0.003 * (sum(coordinates) + mu)
        @inbounds for color in 1:3
            arrays[mu][color, color, coordinates...] = cis(color * phase)
        end
        arrays[mu][1, 2, coordinates...] = 0.002cis(2phase)
        arrays[mu][2, 1, coordinates...] = -0.002cis(-2phase)
    end
    return [LatticeMatrix(array, 4, (1, 1, 1, 1); nw=3)
            for array in arrays]
end

function _hisq_timing_samples(operation, evaluations, samples)
    operation()
    JACC.synchronize()
    measurements = Vector{Float64}(undef, samples)
    for sample in eachindex(measurements)
        start = time_ns()
        for _ in 1:evaluations
            operation()
        end
        JACC.synchronize()
        measurements[sample] =
            (time_ns() - start) / evaluations / 1e6
    end
    return sort!(measurements)
end

function _hisq_report(label, direct, factorized, evaluations, samples)
    direct_samples = _hisq_timing_samples(direct, evaluations, samples)
    factorized_samples =
        _hisq_timing_samples(factorized, evaluations, samples)
    middle = (samples + 1) ÷ 2
    @printf(
        "| %s | %.3f | %.3f | %.2fx | %.3f | %.3f |\n",
        label, first(direct_samples), first(factorized_samples),
        first(direct_samples) / first(factorized_samples),
        direct_samples[middle], factorized_samples[middle])
end

function _hisq_benchmark_extent(extent, evaluations, samples)
    thin = _hisq_benchmark_links(extent)
    workspace = HISQFat7Workspace(thin[1])
    epsilon = -0.083

    direct_level1 = [similar(link) for link in thin]
    factorized_level1 = [similar(link) for link in thin]
    direct_level2 = [similar(link) for link in thin]
    factorized_level2 = [similar(link) for link in thin]
    reunitarized = hisq_project_u3(hisq_fat7_level1(thin))

    direct_fat = [similar(link) for link in thin]
    direct_long = [similar(link) for link in thin]
    direct_work1 = [similar(link) for link in thin]
    direct_reunitarized = [similar(link) for link in thin]
    factorized_fat = [similar(link) for link in thin]
    factorized_long = [similar(link) for link in thin]
    factorized_work1 = [similar(link) for link in thin]
    factorized_reunitarized = [similar(link) for link in thin]

    println("\n`$(extent)^4`, evaluations=$evaluations, samples=$samples")
    println("| stage | direct min (ms) | factorized min (ms) | min speedup | direct median (ms) | factorized median (ms) |")
    println("|---|---:|---:|---:|---:|---:|")
    _hisq_report(
        "level 1",
        () -> hisq_fat7_level1!(direct_level1, thin),
        () -> hisq_fat7_level1!(factorized_level1, thin, workspace),
        evaluations, samples)
    _hisq_report(
        "level 2",
        () -> hisq_fat7_level2!(direct_level2, reunitarized, epsilon),
        () -> hisq_fat7_level2!(
            factorized_level2, reunitarized, epsilon, workspace),
        evaluations, samples)
    _hisq_report(
        "complete builder",
        () -> hisq_links_from_thin!(
            direct_fat, direct_long, direct_work1,
            direct_reunitarized, thin, epsilon),
        () -> hisq_links_from_thin!(
            factorized_fat, factorized_long, factorized_work1,
            factorized_reunitarized, thin, epsilon, workspace),
        evaluations, samples)
end

function main()
    MPI.Comm_size(MPI.COMM_WORLD) == 1 || error(
        "this direct SIMULATeQCD-comparison benchmark requires one MPI rank")
    extents = parse.(Int, split(get(ENV, "LM_BENCH_EXTENTS", "8"), ','))
    evaluations = parse(Int, get(ENV, "LM_BENCH_ITERS", "3"))
    samples = parse(Int, get(ENV, "LM_BENCH_SAMPLES", "7"))
    isodd(samples) || error("LM_BENCH_SAMPLES must be odd")
    println("Factorized HISQ Fat7 benchmark ($(JACC.backend) backend)")
    for extent in extents
        _hisq_benchmark_extent(extent, evaluations, samples)
    end
end

initialized_here = !MPI.Initialized()
initialized_here && MPI.Init()
try
    main()
finally
    initialized_here && MPI.Finalize()
end
