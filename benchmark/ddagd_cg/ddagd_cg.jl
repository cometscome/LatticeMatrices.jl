#!/usr/bin/env julia

# Portable D†D conjugate-gradient benchmark for the LatticeMatrices-backed
# LatticeDiracOperators v1 path.  Parse backend/device options before loading
# JACC, because its backend is selected through Julia Preferences.

import Pkg

using TOML

const BENCHMARK_VERSION = "0.4.0"
const SUPPORTED_BACKENDS = ("threads", "cuda", "amdgpu", "oneapi", "metal")
const BACKEND_PACKAGES = Dict(
    "cuda" => "CUDA",
    "amdgpu" => "AMDGPU",
    "oneapi" => "oneAPI",
    "metal" => "Metal",
)
const OFFICIAL_PACKAGES = (
    "LatticeMatrices",
    "Gaugefields",
    "LatticeDiracOperators",
    "JACC",
    "MPI",
)
const OFFICIAL_STACK_ID = "general-latest-compatible"
const SUPPORTED_OPTIONS = Set([
    "help", "setup", "list-devices", "backend", "devices", "operator",
    "lattice", "grid", "ranks", "threads", "precision", "gauge", "seed", "halo",
    "mass", "kappa", "csw", "naik-epsilon", "l5", "domain-wall-height",
    "mobius-b", "mobius-c", "a5", "b5", "c5", "rtol", "atol",
    "verify-rtol", "maxiter", "warmup-iterations", "repeats", "verbose",
    "log-every", "output",
])

function parse_cli(args)
    options = Dict{String,String}()
    positional = String[]
    i = 1
    while i <= length(args)
        arg = args[i]
        if !startswith(arg, "--")
            push!(positional, arg)
            i += 1
            continue
        end
        item = arg[3:end]
        if occursin('=', item)
            key, value = split(item, '='; limit=2)
            options[key] = value
        elseif item in ("help", "setup", "list-devices")
            options[item] = "true"
        else
            i == length(args) && throw(ArgumentError("--$item requires a value"))
            startswith(args[i + 1], "--") &&
                throw(ArgumentError("--$item requires a value"))
            options[item] = args[i + 1]
            i += 1
        end
        i += 1
    end
    isempty(positional) || throw(ArgumentError(
        "unexpected positional arguments: $(join(positional, ' '))"))
    unknown = sort!(collect(setdiff(keys(options), SUPPORTED_OPTIONS)))
    unknown_display = join(map(name -> "--" * name, unknown), ", ")
    isempty(unknown) || throw(ArgumentError(
        "unknown option(s): " * unknown_display))
    return options
end

option(options, name, default) = get(options, name, string(default))

function parse_bool(value, name)
    normalized = lowercase(strip(value))
    normalized in ("1", "true", "yes", "on") && return true
    normalized in ("0", "false", "no", "off") && return false
    throw(ArgumentError("--$name must be true or false, got '$value'"))
end

function parse_tuple(value, length_expected, name; element_type=Int)
    values = Tuple(parse.(element_type, split(value, ',')))
    length(values) == length_expected || throw(ArgumentError(
        "--$name must contain $length_expected comma-separated values"))
    return values
end

function launcher_rank()
    for name in (
        "OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK", "PMI_ID",
        "SLURM_PROCID", "MV2_COMM_WORLD_RANK",
    )
        haskey(ENV, name) && return parse(Int, ENV[name])
    end
    return 0
end

function inside_mpi_launcher()
    return any(haskey(ENV, name) for name in (
        "OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK", "PMI_ID",
        "SLURM_PROCID", "MV2_COMM_WORLD_RANK",
    ))
end

function preference_backend(path)
    isfile(path) || return nothing
    preferences = try
        TOML.parsefile(path)
    catch
        return nothing
    end
    jacc = get(preferences, "JACC", Dict{String,Any}())
    return lowercase(string(get(jacc, "default_backend", get(jacc, "backend", ""))))
end

function write_backend_preference!(path, backend)
    preferences = isfile(path) ? TOML.parsefile(path) : Dict{String,Any}()
    jacc = get!(preferences, "JACC", Dict{String,Any}())
    # JACC 0.6 used `backend`; JACC 1.x uses `default_backend`/`backends`.
    # Keeping all three makes this benchmark environment portable across both.
    jacc["backend"] = backend
    jacc["default_backend"] = backend
    jacc["backends"] = [backend]
    temporary = path * ".tmp." * string(getpid())
    open(temporary, "w") do io
        TOML.print(io, preferences; sorted=true)
    end
    mv(temporary, path; force=true)
    return nothing
end

function ensure_backend_preference!(backend, environment_directory)
    backend in SUPPORTED_BACKENDS || throw(ArgumentError(
        "--backend must be one of $(join(SUPPORTED_BACKENDS, ", ")), got '$backend'"))
    path = joinpath(environment_directory, "LocalPreferences.toml")
    preference_backend(path) == backend && return
    if launcher_rank() == 0
        write_backend_preference!(path, backend)
    else
        deadline = time() + 30
        while preference_backend(path) != backend
            time() < deadline || error(
                "timed out waiting for MPI rank 0 to set the JACC backend")
            sleep(0.05)
        end
    end
    return nothing
end

function installed_package_version(name)
    for (_, package) in Pkg.dependencies()
        package.name == name && return string(package.version)
    end
    return "not-installed"
end

function environment_package_names(backend)
    package_names = collect(OFFICIAL_PACKAGES)
    if backend != "threads"
        push!(package_names, BACKEND_PACKAGES[backend])
    end
    return package_names
end

function resolved_package_versions(backend)
    return join((
        "$name=$(installed_package_version(name))"
        for name in environment_package_names(backend)
    ), ", ")
end

function setup_environment!(backend)
    # No PackageSpec contains a version: every setup resolves the newest set of
    # mutually compatible General-registry releases supported by this Julia.
    package_specs = Pkg.PackageSpec[
        Pkg.PackageSpec(name=name) for name in environment_package_names(backend)
    ]
    Pkg.add(package_specs)
    Pkg.update()
    Pkg.instantiate()
    Pkg.precompile()
    return nothing
end

function print_help()
    println("""
    D†D CG benchmark (LatticeMatrices/LatticeDiracOperators v1)

    All packages are official Julia General-registry releases. Each --setup
    resolves and updates to the latest mutually compatible versions available.

    Setup/update (run without mpiexec):
      julia ddagd_cg.jl --setup --backend=threads
      julia ddagd_cg.jl --setup --backend=cuda
      julia ddagd_cg.jl --setup --backend=amdgpu
      julia ddagd_cg.jl --setup --backend=oneapi
      julia ddagd_cg.jl --setup --backend=metal

    Main options:
      --operator=NAME       staggered, wilson, wilson-clover, hisq,
                            mobius-domain-wall, domain-wall, or
                            general-domain-wall (default: wilson)
      --lattice=NX,NY,NZ,NT global 4D lattice size (default: 8,8,8,8)
      --grid=PX,PY,PZ,PT    MPI process grid (default: nranks,1,1,1)
      --ranks=N             launch N ranks with MPI.jl's matching mpiexec
      --backend=NAME        threads, cuda, amdgpu, oneapi, or metal
                            (default: threads)
      --devices=LIST        zero-based ordinals within the visible devices,
                            e.g. 0,1 (default: auto)
      --threads=N           Julia threads per rank; relaunch automatically if needed
      --precision=NAME      double/float64 or single/float32
                            (default: single on Metal, double otherwise)
      --gauge=NAME          cold or hot (default: hot)
      --seed=N              decomposition-independent source/gauge seed
      --halo=N              halo width; auto uses 3 for HISQ and 1 otherwise
      --rtol=X              relative L2 CG tolerance (default: precision-dependent)
      --atol=X              absolute L2 CG tolerance (default: 0)
      --verify-rtol=X       acceptance threshold for ||D†Dx-φ||/||φ||
      --maxiter=N           maximum CG iterations (default: 10000)
      --repeats=N           timed solves (default: 1)
      --warmup-iterations=N untimed CG iterations used for JIT warm-up (default: 2)
      --verbose=N           0 summary, 1 configuration, 2 CG progress
      --log-every=N         progress interval for verbose=2 (default: 50)
      --output=PATH         append one CSV result row on MPI rank 0

    Operator parameters:
      --mass=X --kappa=X --csw=X --naik-epsilon=X
      --l5=N --domain-wall-height=X --mobius-b=X --mobius-c=X
      --a5=X[,X...] --b5=X[,X...] --c5=X[,X...]

    Utilities:
      --list-devices        list devices visible to the selected backend
      --help                show this help

    Examples:
      julia -t 16 ddagd_cg.jl --backend=threads --operator=wilson \\
        --lattice=16,16,16,32 --threads=16

      julia -t 4 ddagd_cg.jl --ranks=2 --backend=amdgpu \\
        --operator=hisq --lattice=32,32,32,64 --grid=2,1,1,1

    With --devices=auto, one MPI rank is mapped to one accelerator by node-local
    rank. Existing scheduler/container visibility settings are respected. Use
    --devices only for an explicit subset of the devices already visible to the
    process. Metal currently supports one MPI rank per node in this benchmark.
    """)
end

const OPTIONS = try
    parse_cli(ARGS)
catch exception
    Base.display_error(stderr, exception, catch_backtrace())
    println(stderr, "Run with --help for usage.")
    exit(2)
end

if parse_bool(option(OPTIONS, "help", false), "help")
    print_help()
    exit()
end

const REQUESTED_BACKEND = lowercase(option(OPTIONS, "backend", "threads"))
REQUESTED_BACKEND in SUPPORTED_BACKENDS || throw(ArgumentError(
    "--backend must be one of $(join(SUPPORTED_BACKENDS, ", "))"))
const ENVIRONMENT_DIRECTORY = joinpath(
    @__DIR__, ".environments", OFFICIAL_STACK_ID, REQUESTED_BACKEND)
const SETUP_REQUESTED = parse_bool(option(OPTIONS, "setup", false), "setup")

if !SETUP_REQUESTED && !isfile(joinpath(ENVIRONMENT_DIRECTORY, "Project.toml"))
    error("benchmark environment for backend '$REQUESTED_BACKEND' is missing; " *
          "run: julia $(PROGRAM_FILE) --setup --backend=$REQUESTED_BACKEND")
end

SETUP_REQUESTED && mkpath(ENVIRONMENT_DIRECTORY)
Pkg.activate(ENVIRONMENT_DIRECTORY; io=devnull)
ensure_backend_preference!(REQUESTED_BACKEND, ENVIRONMENT_DIRECTORY)

const REQUESTED_DEVICES = strip(option(OPTIONS, "devices", "auto"))
isempty(REQUESTED_DEVICES) && throw(ArgumentError("--devices must not be empty"))
REQUESTED_BACKEND == "threads" && lowercase(REQUESTED_DEVICES) != "auto" &&
    throw(ArgumentError("--devices is not meaningful with --backend=threads"))

if SETUP_REQUESTED
    launcher_rank() == 0 || error("run --setup without mpiexec")
    setup_environment!(REQUESTED_BACKEND)
    println("Official benchmark environment is ready " *
            "(policy=$OFFICIAL_STACK_ID, backend=$REQUESTED_BACKEND).")
    println("Resolved packages: " *
            resolved_package_versions(REQUESTED_BACKEND))
    exit()
end

# Julia's thread count is fixed at process startup. For a direct invocation,
# relaunch this same script with the requested count before MPI is initialized.
# An external MPI launcher must set the count itself for every rank.
if haskey(OPTIONS, "threads")
    requested_threads = parse(Int, OPTIONS["threads"])
    requested_threads > 0 || throw(ArgumentError("--threads must be positive"))
    if requested_threads != Threads.nthreads()
        inside_mpi_launcher() && throw(ArgumentError(
            "--threads=$requested_threads requested under an external MPI launcher, " *
            "but Julia started with $(Threads.nthreads()) thread(s); pass " *
            "--threads=$requested_threads to the Julia command used by the launcher"))
        child_command = `$(Base.julia_cmd()) --threads=$requested_threads $(abspath(PROGRAM_FILE)) $ARGS`
        run(child_command)
        exit()
    end
end

# A self-launch option avoids accidentally combining MPI.jl with an
# ABI-incompatible system `mpiexec`. The child processes receive the current
# Julia thread count and all options except --ranks.
if haskey(OPTIONS, "ranks")
    requested_ranks = parse(Int, OPTIONS["ranks"])
    requested_ranks > 0 || throw(ArgumentError("--ranks must be positive"))
    if requested_ranks > 1
        inside_mpi_launcher() && throw(ArgumentError(
            "do not combine --ranks with mpiexec/srun; choose one launcher"))
        try
            @eval import MPI
        catch exception
            println(stderr, "The benchmark environment is not ready.")
            println(stderr, "Run: julia $(PROGRAM_FILE) --setup --backend=$REQUESTED_BACKEND")
            rethrow(exception)
        end
        child_arguments = String[
            "--$name=$value" for (name, value) in OPTIONS if name != "ranks"
        ]
        child_command = `$(MPI.mpiexec()) -n $requested_ranks $(Base.julia_cmd()) --threads=$(Threads.nthreads()) $(abspath(PROGRAM_FILE)) $child_arguments`
        run(child_command)
        exit()
    end
end

try
    @eval import MPI
    @eval import JACC
    @eval using Gaugefields
    @eval using LatticeDiracOperators
    @eval using LatticeMatrices
catch exception
    println(stderr, "The benchmark environment is not ready.")
    println(stderr, "Run: julia $(PROGRAM_FILE) --setup --backend=$REQUESTED_BACKEND")
    rethrow(exception)
end

JACC.@init_backend

using Dates
using LinearAlgebra
using Printf
using Statistics

MPI.Initialized() || MPI.Init()

const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NRANKS = MPI.Comm_size(COMM)

function requested_device_ordinals(value)
    lowercase(value) == "auto" && return nothing
    ordinals = parse.(Int, split(value, ','))
    isempty(ordinals) && throw(ArgumentError("--devices must not be empty"))
    all(>=(0), ordinals) || throw(ArgumentError(
        "--devices ordinals must be nonnegative"))
    allunique(ordinals) || throw(ArgumentError(
        "--devices must not contain duplicate ordinals"))
    return ordinals
end

function node_local_rank_and_size(comm)
    rank = MPI.Comm_rank(comm)
    local_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
    try
        return MPI.Comm_rank(local_comm), MPI.Comm_size(local_comm)
    finally
        MPI.free(local_comm)
    end
end

function select_requested_device!(comm, requested)
    if requested === nothing
        selection = LatticeMatrices.select_device_by_mpi_rank!(comm)
        cli_ordinal = selection.device_ordinal === nothing ? nothing :
            selection.device_ordinal - 1
        return merge(selection, (cli_device_ordinal=cli_ordinal,))
    end

    backend = Symbol(JACC.backend)
    backend === :threads && throw(ArgumentError(
        "--devices is not meaningful with --backend=threads"))
    local_rank, local_size = node_local_rank_and_size(comm)

    if backend === :metal
        local_size == 1 || throw(ArgumentError(
            "Metal currently supports only one MPI rank per node"))
        requested == [0] || throw(ArgumentError(
            "Metal currently exposes one device; use --devices=0 or auto"))
        devices = collect(Metal.MTL.devices())
        isempty(devices) && error("Metal reports no visible device")
        Metal.device!(first(devices))
        return (
            backend,
            local_rank,
            local_size,
            visible_devices=length(devices),
            device_ordinal=1,
            cli_device_ordinal=0,
        )
    end

    backend_value = Val(backend)
    visible_devices = LatticeMatrices._backend_device_count(backend_value)
    visible_devices > 0 || error("backend '$backend' reports no visible devices")
    selected_cli_ordinal = if visible_devices == 1 && requested == [0]
        # Schedulers commonly expose one different physical GPU to each rank.
        0
    else
        local_size <= length(requested) || throw(ArgumentError(
            "$local_size MPI ranks share this node, but --devices contains " *
            "only $(length(requested)) ordinal(s)"))
        requested[local_rank + 1]
    end
    selected_cli_ordinal < visible_devices || throw(ArgumentError(
        "--devices ordinal $selected_cli_ordinal is outside the $visible_devices " *
        "device(s) visible to backend '$backend'"))
    LatticeMatrices._select_backend_device!(
        backend_value, selected_cli_ordinal + 1)
    return (
        backend,
        local_rank,
        local_size,
        visible_devices,
        device_ordinal=selected_cli_ordinal + 1,
        cli_device_ordinal=selected_cli_ordinal,
    )
end

const DEVICE_SELECTION = select_requested_device!(
    COMM, requested_device_ordinals(REQUESTED_DEVICES))

function accelerator_device_description()
    JACC.backend == "threads" && return "CPU"
    JACC.backend == "cuda" && return string(CUDA.name(CUDA.device()))
    JACC.backend == "amdgpu" &&
        return string(AMDGPU.HIP.name(AMDGPU.device()))
    JACC.backend == "oneapi" && return string(oneAPI.device())
    JACC.backend == "metal" && return string(Metal.device())
    return JACC.backend
end

function visible_device_descriptions()
    JACC.backend == "threads" && return ["CPU"]
    JACC.backend == "cuda" && return [
        string(CUDA.name(device)) for device in CUDA.devices()
    ]
    JACC.backend == "amdgpu" && return [
        "$(AMDGPU.HIP.name(device)) arch=$(AMDGPU.HIP.gcn_arch(device))"
        for device in AMDGPU.devices()
    ]
    JACC.backend == "oneapi" && return [
        string(device) for device in oneAPI.devices()
    ]
    JACC.backend == "metal" && return [
        string(device) for device in Metal.MTL.devices()
    ]
    return String[]
end

if parse_bool(option(OPTIONS, "list-devices", false), "list-devices")
    MPI.Barrier(COMM)
    for rank_to_print in 0:(NRANKS - 1)
        if RANK == rank_to_print
            println("rank=$RANK backend=$(JACC.backend) visible devices:")
            for (ordinal, description) in enumerate(visible_device_descriptions())
                println("  ordinal=$(ordinal - 1) name=$description")
            end
            println("  selected_ordinal=$(something(
                DEVICE_SELECTION.cli_device_ordinal, "none"))")
            println("  selected=$(accelerator_device_description())")
        end
        MPI.Barrier(COMM)
    end
    exit()
end

const OPERATOR_ALIASES = Dict(
    "staggered" => (label="staggered", ldo="staggered", family="staggered"),
    "wilson" => (label="wilson", ldo="Wilson", family="Wilson"),
    "wilson-clover" => (label="wilson-clover", ldo="WilsonClover", family="Wilson"),
    "wilsonclover" => (label="wilson-clover", ldo="WilsonClover", family="Wilson"),
    "hisq" => (label="hisq", ldo="HISQ", family="staggered"),
    "domain-wall" => (label="domain-wall", ldo="Domainwall", family="Domainwall"),
    "domainwall" => (label="domain-wall", ldo="Domainwall", family="Domainwall"),
    "mobius-domain-wall" => (
        label="mobius-domain-wall", ldo="MobiusDomainwall", family="MobiusDomainwall"),
    "mobius" => (
        label="mobius-domain-wall", ldo="MobiusDomainwall", family="MobiusDomainwall"),
    "general-domain-wall" => (
        label="general-domain-wall", ldo="GeneralizedDomainwall",
        family="GeneralizedDomainwall"),
    "generalized-domain-wall" => (
        label="general-domain-wall", ldo="GeneralizedDomainwall",
        family="GeneralizedDomainwall"),
)

function positive_integer(value, name; allow_zero=false)
    parsed = parse(Int, value)
    minimum = allow_zero ? 0 : 1
    parsed >= minimum || throw(ArgumentError("--$name must be at least $minimum"))
    return parsed
end

function coefficient_vector(value, length_expected, name)
    coefficients = parse.(Float64, split(value, ','))
    length(coefficients) == 1 && return fill(only(coefficients), length_expected)
    length(coefficients) == length_expected || throw(ArgumentError(
        "--$name must contain either one value or L5=$length_expected values"))
    return coefficients
end

function parse_precision(value)
    normalized = lowercase(value)
    normalized in ("double", "float64", "f64") && return ComplexF64, "float64"
    normalized in ("single", "float32", "f32") && return ComplexF32, "float32"
    throw(ArgumentError("--precision must be double/float64 or single/float32"))
end

function make_parameters(operator, options, boundary, l5, maxiter)
    parameters = Dict{String,Any}(
        "Dirac_operator" => operator.ldo,
        "boundarycondition" => collect(boundary),
        # The script owns the relative stopping rule; these values only size
        # and configure the operator's internal workspaces.
        "eps_CG" => 0.0,
        "MaxCGstep" => maxiter,
        "verbose_level" => 0,
    )
    mass = parse(Float64, option(options, "mass", 0.1))
    if operator.label in ("wilson", "wilson-clover")
        parameters["kappa"] = parse(Float64, option(options, "kappa", 0.12))
        parameters["κ"] = parameters["kappa"]
    else
        parameters["mass"] = mass
    end
    operator.label == "wilson-clover" &&
        (parameters["cSW"] = parse(Float64, option(options, "csw", 1.0)))
    operator.label == "hisq" &&
        (parameters["naik_epsilon"] = parse(
            Float64, option(options, "naik-epsilon", -0.083)))
    if occursin("domain-wall", operator.label)
        parameters["L5"] = l5
        parameters["M"] = parse(
            Float64, option(options, "domain-wall-height", -1.0))
    end
    if operator.label == "mobius-domain-wall"
        parameters["b"] = parse(Float64, option(options, "mobius-b", 2.0))
        parameters["c"] = parse(Float64, option(options, "mobius-c", 1.0))
    elseif operator.label == "general-domain-wall"
        parameters["as"] = coefficient_vector(option(options, "a5", 1.0), l5, "a5")
        parameters["bs"] = coefficient_vector(option(options, "b5", 1.5), l5, "b5")
        parameters["cs"] = coefficient_vector(option(options, "c5", 0.5), l5, "c5")
    end
    return parameters
end

struct CGWorkspace{F}
    residual::F
    direction::F
    image::F
end

CGWorkspace(template) = CGWorkspace(
    similar(template), similar(template), similar(template))

"""Explicit `D†(Dx)` application with one reusable intermediate field."""
struct ExplicitDdagD{D,F}
    dirac::D
    intermediate::F
end

function LinearAlgebra.mul!(result, operator::ExplicitDdagD, source)
    mul!(operator.intermediate, operator.dirac, source)
    mul!(result, adjoint(operator.dirac), operator.intermediate)
    return result
end

function cg_zero_start!(
    solution, normal_operator, source, workspace;
    rtol, atol, maxiter, verbose=0, log_every=50,
)
    clear_fermion!(solution)
    substitute_fermion!(workspace.residual, source)
    substitute_fermion!(workspace.direction, source)

    source_norm_squared = real(dot(source, source))
    source_norm_squared > 0 || throw(ArgumentError("the source field has zero norm"))
    residual_norm_squared = source_norm_squared
    target_squared = max(atol^2, rtol^2 * source_norm_squared)
    residual_norm_squared <= target_squared && return (
        converged=true, iterations=0,
        recursive_relative_residual=sqrt(residual_norm_squared / source_norm_squared),
        source_norm_squared, target_squared,
    )

    for iteration in 1:maxiter
        mul!(workspace.image, normal_operator, workspace.direction)
        denominator = real(dot(workspace.direction, workspace.image))
        isfinite(denominator) || error(
            "CG breakdown at iteration $iteration: non-finite p†(D†D)p")
        denominator > 0 || error(
            "CG breakdown at iteration $iteration: p†(D†D)p=$denominator is not positive")
        alpha = residual_norm_squared / denominator
        axpby!(alpha, workspace.direction, 1, solution)
        axpby!(-alpha, workspace.image, 1, workspace.residual)

        next_residual_norm_squared = real(dot(workspace.residual, workspace.residual))
        relative = sqrt(max(next_residual_norm_squared, 0) / source_norm_squared)
        if verbose >= 2 && RANK == 0 &&
                (iteration == 1 || iteration % log_every == 0 ||
                 next_residual_norm_squared <= target_squared)
            @printf("  CG iteration=%d recursive_relative_residual=%.6e\n",
                iteration, relative)
        end
        if next_residual_norm_squared <= target_squared
            return (
                converged=true, iterations=iteration,
                recursive_relative_residual=relative,
                source_norm_squared, target_squared,
            )
        end
        isfinite(next_residual_norm_squared) || error(
            "CG breakdown at iteration $iteration: non-finite residual")
        beta = next_residual_norm_squared / residual_norm_squared
        axpby!(1, workspace.residual, beta, workspace.direction)
        residual_norm_squared = next_residual_norm_squared
    end

    return (
        converged=false, iterations=maxiter,
        recursive_relative_residual=sqrt(
            residual_norm_squared / source_norm_squared),
        source_norm_squared, target_squared,
    )
end

function true_relative_residual!(reconstructed, normal_operator, solution, source)
    mul!(reconstructed, normal_operator, solution)
    axpby!(-1, source, 1, reconstructed)
    return sqrt(real(dot(reconstructed, reconstructed)) / real(dot(source, source)))
end

function synchronized_solve!(solution, normal_operator, source, workspace; kwargs...)
    JACC.synchronize()
    MPI.Barrier(COMM)
    start = time_ns()
    diagnostics = cg_zero_start!(
        solution, normal_operator, source, workspace; kwargs...)
    JACC.synchronize()
    local_seconds = (time_ns() - start) / 1e9
    elapsed_seconds = MPI.Allreduce(local_seconds, max, COMM)
    return diagnostics, elapsed_seconds
end

csv_escape(value) = "\"" * replace(string(value), '"' => "\"\"") * "\""

function append_csv(path, row)
    fields = collect(keys(row))
    exists = isfile(path) && filesize(path) > 0
    open(path, "a") do io
        exists || println(io, join(fields, ','))
        println(io, join((csv_escape(row[field]) for field in fields), ','))
    end
end

function main(options)
    operator_name = lowercase(option(options, "operator", "wilson"))
    haskey(OPERATOR_ALIASES, operator_name) || throw(ArgumentError(
        "unsupported --operator '$operator_name'; run with --help for choices"))
    operator = OPERATOR_ALIASES[operator_name]

    global_size = parse_tuple(option(options, "lattice", "8,8,8,8"), 4, "lattice")
    all(>(0), global_size) || throw(ArgumentError("all lattice extents must be positive"))
    process_grid = haskey(options, "grid") ?
        parse_tuple(options["grid"], 4, "grid") : (NRANKS, 1, 1, 1)
    all(>(0), process_grid) || throw(ArgumentError("all grid extents must be positive"))
    prod(process_grid) == NRANKS || throw(ArgumentError(
        "prod(grid)=$(prod(process_grid)) must equal MPI ranks=$NRANKS"))
    all(global_size[d] % process_grid[d] == 0 for d in 1:4) ||
        throw(ArgumentError(
            "lattice $global_size must be divisible by process grid $process_grid"))
    local_size = global_size .÷ process_grid

    expected_threads = haskey(options, "threads") ?
        positive_integer(options["threads"], "threads") : nothing
    if expected_threads !== nothing && expected_threads != Threads.nthreads()
        throw(ArgumentError(
            "--threads=$expected_threads requested, but Julia started with " *
            "$(Threads.nthreads()) thread(s); launch with `julia -t $expected_threads`"))
    end

    default_precision = JACC.backend == "metal" ? "single" : "double"
    element_type, precision = parse_precision(
        option(options, "precision", default_precision))
    JACC.backend == "metal" && precision == "float64" && throw(ArgumentError(
        "Metal does not support Float64 kernels; use --precision=single"))
    default_rtol = precision == "float32" ? 1e-5 : 1e-10
    rtol = parse(Float64, option(options, "rtol", default_rtol))
    atol = parse(Float64, option(options, "atol", 0.0))
    rtol > 0 || throw(ArgumentError("--rtol must be positive"))
    atol >= 0 || throw(ArgumentError("--atol must be nonnegative"))
    verify_rtol = parse(Float64, option(options, "verify-rtol", 10rtol))
    verify_rtol > 0 || throw(ArgumentError("--verify-rtol must be positive"))

    maxiter = positive_integer(option(options, "maxiter", 10_000), "maxiter")
    repeats = positive_integer(option(options, "repeats", 1), "repeats")
    warmup_iterations = positive_integer(
        option(options, "warmup-iterations", 2), "warmup-iterations";
        allow_zero=true)
    verbose = positive_integer(option(options, "verbose", 1), "verbose"; allow_zero=true)
    verbose <= 2 || throw(ArgumentError("--verbose must be 0, 1, or 2"))
    log_every = positive_integer(option(options, "log-every", 50), "log-every")
    l5 = positive_integer(option(options, "l5", 12), "l5")
    seed = parse(Int, option(options, "seed", 1234))

    halo = if lowercase(option(options, "halo", "auto")) == "auto"
        operator.label == "hisq" ? 3 : 1
    else
        positive_integer(options["halo"], "halo"; allow_zero=true)
    end
    operator.label == "hisq" && halo < 3 && throw(ArgumentError(
        "HISQ requires --halo=3 or larger"))
    operator.label != "hisq" && halo < 1 && throw(ArgumentError(
        "this LatticeDiracOperators path requires --halo=1 or larger"))

    gauge_start = Symbol(lowercase(option(options, "gauge", "hot")))
    gauge_start in (:cold, :hot) || throw(ArgumentError(
        "--gauge must be cold or hot"))
    boundary = (1, 1, 1, -1)
    parameters = make_parameters(operator, options, boundary, l5, maxiter)

    device_name = accelerator_device_description()
    if verbose >= 1 && RANK == 0
        println("D†D CG benchmark v$BENCHMARK_VERSION")
        println("  package source: General registry releases")
        println("  version policy: $OFFICIAL_STACK_ID")
        println("  packages:       $(resolved_package_versions(JACC.backend))")
        println("  operator:       $(operator.label)")
        println("  lattice:        $global_size")
        println("  MPI:            ranks=$NRANKS grid=$process_grid local=$local_size")
        println("  backend:        $(JACC.backend)")
        println("  threads/rank:   $(Threads.nthreads())")
        println("  device:         $device_name")
        println("  device ordinal: $(something(
            DEVICE_SELECTION.cli_device_ordinal, "none"))")
        println("  precision:      $precision")
        println("  halo:           $halo")
        println("  CG:             rtol=$rtol atol=$atol maxiter=$maxiter repeats=$repeats")
    end

    setup_start = time_ns()
    gauge = gauge_configuration(
        global_size;
        colors=3,
        halo,
        start=gauge_start,
        seed=(gauge_start == :hot ? seed : nothing),
        process_grid,
        eltype=element_type,
        verbose=0,
    )
    source = Initialize_pseudofermion_fields(
        gauge[1], operator.family; L5=l5)
    sigma = sqrt(typeof(real(zero(element_type)))(0.5))
    LatticeMatrices.randomize_gaussian_matrix!(source.f; sigma, seed=seed + 1)
    solution = similar(source)
    reconstructed = similar(source)
    workspace = CGWorkspace(source)
    dirac_operator = Dirac_operator(gauge, source, parameters)
    # The high-level domain-wall wrapper represents D(m)D(1)^-1 and therefore
    # invokes an inner Pauli–Villars solve on every multiplication.  This
    # benchmark deliberately measures the selected raw 5D Dirac kernel, just
    # as the 4D choices measure their raw D kernels.
    raw_dirac_operator = occursin("domain-wall", operator.label) ?
        dirac_operator.D5DW : dirac_operator
    normal_operator = ExplicitDdagD(raw_dirac_operator, similar(source))
    JACC.synchronize()
    setup_seconds = MPI.Allreduce((time_ns() - setup_start) / 1e9, max, COMM)

    if warmup_iterations > 0
        cg_zero_start!(
            solution, normal_operator, source, workspace;
            rtol=0.0,
            atol=0.0,
            maxiter=warmup_iterations,
            verbose=0,
            log_every,
        )
        JACC.synchronize()
    else
        # Compile one D†D application even when iterative warm-up is disabled.
        mul!(reconstructed, normal_operator, source)
        JACC.synchronize()
    end

    elapsed = Float64[]
    diagnostics = nothing
    for repeat_index in 1:repeats
        diagnostics, elapsed_seconds = synchronized_solve!(
            solution, normal_operator, source, workspace;
            rtol,
            atol,
            maxiter,
            verbose,
            log_every,
        )
        push!(elapsed, elapsed_seconds)
        verbose >= 1 && RANK == 0 && @printf(
            "  solve %d/%d: iterations=%d time=%.6f s recursive_residual=%.6e\n",
            repeat_index, repeats, diagnostics.iterations, elapsed_seconds,
            diagnostics.recursive_relative_residual)
    end

    JACC.synchronize()
    MPI.Barrier(COMM)
    true_residual = true_relative_residual!(
        reconstructed, normal_operator, solution, source)
    JACC.synchronize()
    passed = diagnostics.converged && isfinite(true_residual) &&
        true_residual <= verify_rtol
    iterations = diagnostics.iterations
    median_seconds = median(elapsed)
    milliseconds_per_iteration = iterations == 0 ? 0.0 :
        1e3 * median_seconds / iterations

    row = (
        timestamp=Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sssZ"),
        benchmark_version=BENCHMARK_VERSION,
        latticematrices_version=string(pkgversion(LatticeMatrices)),
        latticediracoperators_version=string(pkgversion(LatticeDiracOperators)),
        gaugefields_version=string(pkgversion(Gaugefields)),
        julia_version=string(VERSION),
        package_source="General registry",
        official_stack=OFFICIAL_STACK_ID,
        jacc_version=installed_package_version("JACC"),
        mpi_version=installed_package_version("MPI"),
        backend_package=get(BACKEND_PACKAGES, JACC.backend, "threads"),
        backend_package_version=JACC.backend == "threads" ? "stdlib" :
            installed_package_version(BACKEND_PACKAGES[JACC.backend]),
        operator=operator.label,
        backend=JACC.backend,
        device=device_name,
        requested_devices=REQUESTED_DEVICES,
        selected_device_ordinal=something(
            DEVICE_SELECTION.cli_device_ordinal, "none"),
        backend_visible_device_count=DEVICE_SELECTION.visible_devices,
        precision,
        mpi_ranks=NRANKS,
        threads_per_rank=Threads.nthreads(),
        lattice=join(global_size, 'x'),
        process_grid=join(process_grid, 'x'),
        local_lattice=join(local_size, 'x'),
        halo,
        l5=occursin("domain-wall", operator.label) ? l5 : 0,
        gauge=String(gauge_start),
        seed,
        rtol,
        atol,
        verify_rtol,
        maxiter,
        repeats,
        iterations,
        converged=diagnostics.converged,
        recursive_relative_residual=diagnostics.recursive_relative_residual,
        true_relative_residual=true_residual,
        passed,
        setup_seconds,
        min_solve_seconds=minimum(elapsed),
        median_solve_seconds=median_seconds,
        max_solve_seconds=maximum(elapsed),
        milliseconds_per_iteration,
    )

    if RANK == 0
        @printf(
            "VERIFY passed=%s converged=%s true_relative_residual=%.9e threshold=%.9e\n",
            string(passed), string(diagnostics.converged), true_residual, verify_rtol)
        @printf(
            "RESULT operator=%s backend=%s device=%s device_ordinal=%s precision=%s ranks=%d threads=%d lattice=%s grid=%s iterations=%d median_seconds=%.9f ms_per_iteration=%.9f true_residual=%.9e passed=%s\n",
            operator.label, JACC.backend, replace(device_name, ' ' => '_'),
            string(something(DEVICE_SELECTION.cli_device_ordinal, "none")), precision,
            NRANKS, Threads.nthreads(), join(global_size, 'x'),
            join(process_grid, 'x'), iterations, median_seconds,
            milliseconds_per_iteration, true_residual, string(passed))
        if haskey(options, "output")
            append_csv(abspath(options["output"]), row)
            println("CSV appended: $(abspath(options["output"]))")
        end
    end
    MPI.Barrier(COMM)
    return passed ? 0 : 1
end

exit_code = try
    main(OPTIONS)
catch exception
    if RANK == 0
        Base.display_error(stderr, exception, catch_backtrace())
    end
    2
end

exit(exit_code)
