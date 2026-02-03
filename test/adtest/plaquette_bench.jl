using LatticeMatrices
using MPI
using LinearAlgebra
using Statistics
using Enzyme
import JACC
using Gaugefields
JACC.@init_backend
import LatticeMatrices: mul_simple!
using InteractiveUtils

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

function _calc_action_step_add!(Uout, C, D, Uμ, Uν, shift_μ, shift_ν)
    Uμ_pν = shift_L(Uμ, shift_ν)
    Uν_pμ = shift_L(Uν, shift_μ)

    mul!(C, Uμ, Uν_pμ)
    #mul!(D, C, Uμ_pν', 1, 0)
    mul!(D, C, Uμ_pν')
    #display(which(mul!, typeof.((D, C, Uμ_pν'))))
    #error("d")
    mul!(C, D, Uν')
    #mul!(C, D, Uν', 1, 0)
    add_matrix!(Uout, C)
    #S = realtrace(E)
    #=

    mul!(C, Uν, Uμ_pν)
    mul!(D, C, Uν_pμ')
    mul!(C, D, Uμ')
    add_matrix!(Uout, C)
    #S += realtrace(E)
    =#

    #return S
end

function _calc_action_step_add_simple!(Uout, C, D, Uμ, Uν, shift_μ, shift_ν)
    Uμ_pν = shift_L(Uμ, shift_ν)
    Uν_pμ = shift_L(Uν, shift_μ)

    mul_simple!(C, Uμ, Uν_pμ)
    mul_simple!(D, C, Uμ_pν', 1, 0)
    mul_simple!(C, D, Uν', 1, 0)
    add_matrix!(Uout, C)
    #S = realtrace(E)

    #=
    mul_simple!(C, Uν, Uμ_pν)
    mul_simple!(D, C, Uν_pμ')
    mul_simple!(C, D, Uμ')
    add_matrix!(Uout, C)
    #S += realtrace(E)
    =#

    #return S
end



function plaquette_mul!(U1, U2, U3, U4, temp)
    U = (U1, U2, U3, U4)
    ndir = length(U)
    dim = length(U1.PN)
    Uout = temp[1]
    C = temp[2]
    D = temp[3]

    S = 0.0
    clear_matrix!(Uout)

    for μ = 1:ndir
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        for ν = μ:ndir
            if ν == μ
                continue
            end
            shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)
            _calc_action_step_add!(Uout, C, D, U[μ], U[ν], shift_μ, shift_ν)

        end
    end
    S += realtrace(Uout)

    return S
end

function plaquette_mul_simple!(U1, U2, U3, U4, temp)
    U = (U1, U2, U3, U4)
    ndir = length(U)
    dim = length(U1.PN)
    Uout = temp[1]
    C = temp[2]
    D = temp[3]

    S = 0.0
    clear_matrix!(Uout)

    for μ = 1:ndir
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        for ν = μ:ndir
            if ν == μ
                continue
            end
            shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)
            _calc_action_step_add_simple!(Uout, C, D, U[μ], U[ν], shift_μ, shift_ν)
        end
    end
    S += realtrace(Uout)

    return S
end
#=

function plaquette_mul!(U1, U2, U3, U4, temp)
    _ = U3
    _ = U4
    C = temp[1]
    D = temp[2]
    E = temp[3]


    shift1 = (1, 0, 0, 0)
    shift2 = (0, 1, 0, 0)

    U2_p1 = shift_L(U2, shift1)
    U1_p2 = shift_L(U1, shift2)

    mul!(C, U1, U2_p1)
    mul!(D, C, U1_p2')
    mul!(E, D, U2')
    return realtrace(E)
end

=#

#=
function plaquette_mul_simple!(U1, U2, U3, U4, temp)
    _ = U3
    _ = U4
    C = temp[1]
    D = temp[2]
    E = temp[3]

    shift1 = (1, 0, 0, 0)
    shift2 = (0, 1, 0, 0)

    U2_p1 = shift_L(U2, shift1)
    U1_p2 = shift_L(U1, shift2)

    mul_simple!(C, U1, U2_p1)
    mul_simple!(D, C, U1_p2')
    mul_simple!(E, D, U2')
    return realtrace(E)
end
=#

function _time_plaquette(f, U1, U2, U3, U4, temp; warmup=3, nrepeat=10)
    for _ in 1:warmup
        f(U1, U2, U3, U4, temp)
    end
    JACC.synchronize()
    MPI.Barrier(MPI.COMM_WORLD)
    times = Vector{Float64}(undef, nrepeat)
    for i in 1:nrepeat
        times[i] = 1_000 * @elapsed begin
            f(U1, U2, U3, U4, temp)
            JACC.synchronize()
        end
    end
    return times
end

function plaquette_bench(; NC=3, dim=4, NX=20, nw=1, warmup=3, nrepeat=10)
    if !MPI.Initialized()
        MPI.Init()
    end
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    myrank = MPI.Comm_rank(MPI.COMM_WORLD)
    gsize = ntuple(_ -> NX, dim)
    PEs = _default_PEs(dim)

    U1 = LatticeMatrix(rand(ComplexF64, NC, NC, gsize...), dim, PEs; nw)
    U2 = LatticeMatrix(rand(ComplexF64, NC, NC, gsize...), dim, PEs; nw)
    U3 = LatticeMatrix(rand(ComplexF64, NC, NC, gsize...), dim, PEs; nw)
    U4 = LatticeMatrix(rand(ComplexF64, NC, NC, gsize...), dim, PEs; nw)
    set_halo!(U1)
    set_halo!(U2)
    set_halo!(U3)
    set_halo!(U4)
    temp = [similar(U1) for _ in 1:3]

    U = Initialize_Gaugefields(NC, 1, gsize...,
        condition="cold";
        isMPILattice=true)

    for i = 1:10

        Ucpu = Initialize_Gaugefields(NC, 0, gsize...,
            condition="hot")
        substitute_U!(U, Ucpu)
        U1.A .= U[1].U.A
        U2.A .= U[2].U.A
        U3.A .= U[3].U.A
        U4.A .= U[4].U.A
        set_halo!(U1)
        set_halo!(U2)
        set_halo!(U3)
        set_halo!(U4)


        set_wing_U!(U)
        Dim = 4

        if Dim == 4
            comb = 6 #4*3/2
        elseif Dim == 3
            comb = 3
        elseif Dim == 2
            comb = 1
        else
            error("dimension $Dim is not supported")
        end
        factor = 1 / (comb * U[1].NV * U[1].NC)


        temp1 = similar(U[1])
        temp2 = similar(U[1])

        temp1cpu = similar(Ucpu[1])
        temp2cpu = similar(Ucpu[1])


        println("JACC")
        @time plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
        println(plaq_t)
        println("CPU ")
        @time plaq_t = calculate_Plaquette(Ucpu, temp1cpu, temp2cpu) * factor
        println(plaq_t)

        println("LatticeMatrices")
        @time s_mul = plaquette_mul!(U1, U2, U3, U4, temp) * factor
        println("LatticeMatrices no JACC")
        @time s_simple = plaquette_mul_simple!(U1, U2, U3, U4, temp) * factor
        println(s_mul)
        println(s_simple)
    end

    #@code_warntype plaquette_mul_simple!(U1, U2, U3, U4, temp)

    # return


    s_mul = plaquette_mul!(U1, U2, U3, U4, temp)
    s_simple = plaquette_mul_simple!(U1, U2, U3, U4, temp)
    diff = abs(s_mul - s_simple)

    times_mul = _time_plaquette(plaquette_mul!, U1, U2, U3, U4, temp; warmup, nrepeat)
    times_simple = _time_plaquette(plaquette_mul_simple!, U1, U2, U3, U4, temp; warmup, nrepeat)

    if myrank == 0
        println("== plaquette mul! vs mul_simple! ==")
        println("Threads.nthreads(): ", Threads.nthreads())
        println("MPI nprocs: ", nprocs)
        println("NC=", NC, " dim=", dim, " gsize=", gsize, " nw=", nw)
        println("warmup=", warmup, " nrepeat=", nrepeat)
        println("realtrace diff: ", diff)
        println("mul! msec: min=", minimum(times_mul),
            " median=", median(times_mul), " mean=", mean(times_mul))
        println("mul_simple! msec: min=", minimum(times_simple),
            " median=", median(times_simple), " mean=", mean(times_simple))
        println("speedup (mul!/mul_simple!): ", median(times_mul) / median(times_simple))
    end
    return times_mul, times_simple
end

if abspath(PROGRAM_FILE) == @__FILE__
    plaquette_bench()
end
