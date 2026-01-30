using LatticeMatrices
using MPI
using LinearAlgebra
using Statistics
import JACC
JACC.@init_backend

function _default_PEs(dim)
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    if length(ARGS) == 0
        n1 = nprocs ÷ 2
        if n1 == 0
            n1 = 1
        end
        PEs = ntuple(i -> i == 1 ? n1 : (i == 2 ? nprocs ÷ n1 : 1), dim)
    else
        PEs = Tuple(parse.(Int64, ARGS))
    end
    return PEs[1:dim]
end

function _time_mul!(M1, M2, M3; warmup=3, nrepeat=10)
    for _ in 1:warmup
        mul!(M1, M2, M3)
    end
    MPI.Barrier(MPI.COMM_WORLD)
    times = Vector{Float64}(undef, nrepeat)
    for i in 1:nrepeat
        times[i] = 1_000 * @elapsed mul!(M1, M2, M3)
    end
    return times
end

function jacc_mul_bench(; NC=3, dim=4, NX=16, nw=1, warmup=3, nrepeat=10)
    if !MPI.Initialized()
        MPI.Init()
    end
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    myrank = MPI.Comm_rank(MPI.COMM_WORLD)
    gsize = ntuple(_ -> NX, dim)
    PEs = _default_PEs(dim)

    M1 = LatticeMatrix(NC, NC, dim, gsize, PEs; nw)
    A2 = rand(ComplexF64, NC, NC, gsize...)
    A3 = rand(ComplexF64, NC, NC, gsize...)
    M2 = LatticeMatrix(A2, dim, PEs; nw)
    M3 = LatticeMatrix(A3, dim, PEs; nw)

    times = _time_mul!(M1, M2, M3; warmup, nrepeat)
    if myrank == 0
        println("== JACC mul! benchmark ==")
        println("Threads.nthreads(): ", Threads.nthreads())
        println("MPI nprocs: ", nprocs)
        println("NC=", NC, " dim=", dim, " gsize=", gsize, " nw=", nw)
        println("warmup=", warmup, " nrepeat=", nrepeat)
        println("mul!(M1, M2, M3) msec: min=", minimum(times),
            " median=", median(times), " mean=", mean(times))
    end
    return times
end

function run_all_threads()
    for nthreads in (1, 2, 4)
        cmd = Base.julia_cmd()
        project = "--project=@."
        args = [project, @__FILE__, "--child"]
        withenv("JULIA_NUM_THREADS" => string(nthreads)) do
            run(`$cmd $(args...)`)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--child" in ARGS
        jacc_mul_bench()
    else
        run_all_threads()
    end
end
