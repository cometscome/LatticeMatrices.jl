
using LatticeMatrices
using MPI
using LinearAlgebra
using Enzyme
import JACC
JACC.@init_backend
import LatticeMatrices: mul_AshiftB!
import Enzyme: autodiff, Duplicated, Const
import Enzyme.EnzymeRules: forward, augmented_primal, reverse, FwdConfig, RevConfigWidth,
    Active, needs_primal, needs_shadow, AugmentedReturn,
    overwritten

import Enzyme: Duplicated, Const, Active
import Enzyme.EnzymeRules: augmented_primal, reverse, RevConfig, AugmentedReturn
using InteractiveUtils

function _report_diff(label, dU, dUn, indices; tol=1e-6)
    diffs = dU.A[:, :, indices...] .- dUn
    maxerr = maximum(abs, diffs)
    if maxerr > tol
        println(label, " max |diff| = ", maxerr)
        println("AD:")
        display(dU.A[:, :, indices...])
        println("Numerical:")
        display(dUn)
        println("Diff:")
        display(diffs)
    else
        println(label, " max |diff| = ", maxerr)
    end
    return nothing
end


function _halo_to_core_indices(indices, PN, nw)
    return ntuple(i -> begin
            idx = indices[i]
            if idx <= nw
                idx + PN[i]
            elseif idx > PN[i] + nw
                idx - PN[i]
            else
                idx
            end
        end, length(indices))
end

function run_case_all(label, f, f_num, U1, U2, U3, U4, dU1, dU2, dU3, dU4, temp, dtemp, indices_mid, indices_halo; tol=1e-4)
    println("=== ", label, " ===")

    dU = [dU1, dU2, dU3, dU4]
    U = [U1, U2, U3, U4]

    clear_matrix!.(dU)
    clear_matrix!.(temp)
    clear_matrix!.(dtemp)

    Enzyme_derivative!(
        f,
        U1,
        U2,
        U3,
        U4,
        dU1,
        dU2,
        dU3,
        dU4; temp=temp, dtemp=dtemp)

    clear_matrix!.(temp)
    f_num_1(Uvec) = f_num(Uvec[1], Uvec[2], Uvec[3], Uvec[4], temp)
    dUn_mid = Numerical_derivative_Enzyme(f_num_1, indices_mid, U)
    for k in 1:length(U)
        _report_diff("U$k mid", dU[k], dUn_mid[k], indices_mid; tol)
    end

    indices_halo_core = _halo_to_core_indices(indices_halo, U[1].PN, U[1].nw)
    clear_matrix!.(temp)
    dUn_halo = Numerical_derivative_Enzyme(f_num_1, indices_halo_core, U)
    for k in 1:length(U)
        _report_diff("U$k halo", dU[k], dUn_halo[k], indices_halo_core; tol)
    end
end

function run_case_all_vector(label, f, f_num, U, dU, temp, dtemp, indices_mid, indices_halo; tol=1e-4)
    println("=== ", label, " ===")



    clear_matrix!.(dU)
    clear_matrix!.(temp)
    clear_matrix!.(dtemp)

    Enzyme_derivative!(
        f,
        U,
        dU; temp=temp, dtemp=dtemp)

    clear_matrix!.(temp)
    f_num_1(Uvec) = f_num(Uvec, temp)
    dUn_mid = Numerical_derivative_Enzyme(f_num_1, indices_mid, U)
    for k in 1:length(U)
        _report_diff("U$k mid", dU[k], dUn_mid[k], indices_mid; tol)
    end

    indices_halo_core = _halo_to_core_indices(indices_halo, U[1].PN, U[1].nw)
    clear_matrix!.(temp)
    dUn_halo = Numerical_derivative_Enzyme(f_num_1, indices_halo_core, U)
    for k in 1:length(U)
        _report_diff("U$k halo", dU[k], dUn_halo[k], indices_halo_core; tol)
    end
end


function loss_mulABtest(U1, U2, U3, U4, temp)
    C = temp[1]
    mul!(C, U1, U2)
    return realtrace(C)
end

function loss_add_matrix_test(U1, U2, U3, U4, temp)
    C = temp[1]
    mul!(C, U1, U2)
    add_matrix!(C, U1, 0.7)
    add_matrix!(C, U2, -0.2)
    return realtrace(C)
end

function loss_add_matrix_adj_test(U1, U2, U3, U4, temp)
    C = temp[1]
    mul!(C, U1, U2)
    add_matrix!(C, U1', 0.3)
    return realtrace(C)
end

function loss_add_matrix_shiftedA_test(U1, U2, U3, U4, shift1, temp)
    C = temp[1]
    mul!(C, U1, U2)
    U1_p1 = shift_L(U1, shift1)
    add_matrix!(C, U1_p1)
    #LatticeMatrices.add_matrix_shiftedA!(C, U1, shift1, 0.7)
    return realtrace(C)
end

function loss_add_matrix_shifted_adj_test(U1, U2, U3, U4, shift1, temp)
    C = temp[1]
    mul!(C, U1, U2)
    Ashift = shift_L(U1, shift1)
    add_matrix!(C, Ashift', 0.4)
    return realtrace(C)
end

function loss_mulABtest_loop(U1, U2, U3, U4, temp)
    C = temp[1]
    D = temp[2]
    mul!(C, U1, U2)
    mul!(D, C, U3)
    mul!(C, D, U1)
    mul!(D, C, U4)
    return realtrace(D)
end

function loss_mulAshiftedBtest(U1, U2, U3, U4, shift1, temp)
    C = temp[1]
    # U2p = shift_L(U2, shift1)
    # mul!(C, U1, U2p)
    mul_AshiftB!(C, U1, U2, shift1)
    return realtrace(C)
end

function _calc_action_step_add_exp_traceless_antihermitian!(Uout, t, C, D, Uμ, Uν, shift_μ, shift_ν)
    Uμ_pν = shift_L(Uμ, shift_ν)
    Uν_pμ = shift_L(Uν, shift_μ)

    mul!(C, Uμ, Uν_pμ)
    mul!(D, C, Uμ_pν')
    mul!(Uout, D, Uν')
    #add_matrix!(Uout, C)
    #S = realtrace(E)

    mul!(C, Uν, Uμ_pν)
    mul!(D, C, Uν_pμ')
    mul!(C, D, Uμ')
    add_matrix!(Uout, C)

    traceless_antihermitian!(D, Uout)
    expt!(C, D, t)
    mul!(Uout, C, Uμ)
    #S += realtrace(E)

    #return S
end


function loss_exp_traceless_antihermitian(U1, U2, U3, U4, t, temp)
    TA = temp[1]
    C = temp[2]
    traceless_antihermitian!(TA, U1)
    expt!(C, TA, t)
    return realtrace(C)
end

function calc_action_loss_exp_traceless_antihermitian(U1, U2, U3, U4, β, NC, t, temp)
    U = (U1, U2, U3, U4)
    ndir = length(U)
    dim = length(U1.PN)
    Uout = temp[1]
    C = temp[2]
    D = temp[3]
    S = 0.0

    for μ = 1:ndir
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        for ν = μ:ndir
            if ν == μ
                continue
            end
            shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)
            _calc_action_step_add_exp_traceless_antihermitian!(Uout, t, C, D, Uμ, Uν, shift_μ, shift_ν)
            S += realtrace(Uout)
        end
    end

    return -S * β / NC
end



function loss_mulAshiftedBtest_loop(U1, U2, U3, U4, shift1, shift2, temp)
    C = temp[1]
    D = temp[2]
    # U2p = shift_L(U2, shift1)
    # mul!(C, U1, U2p)
    mul_AshiftB!(C, U1, U2, shift1)
    mul!(D, C, U3)
    mul!(C, D, U1)
    # U4p = shift_L(U4, shift2)
    # mul!(D, C, U4p)
    mul_AshiftB!(D, C, U4, shift2)
    return realtrace(D)
end

function loss_mulAshiftedBtest_munuloop(U1, U2, U3, U4, temp)
    C = temp[1]
    D = temp[2]
    S = 0.0
    dim = 4
    Uvec = (U1, U2, U3, U4)
    for μ = 1:3
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        for ν = μ+1:3
            shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)

            # Uμ_pν = shift_L(Uvec[μ], shift_ν)
            # Uν_pμ = shift_L(Uvec[ν], shift_μ)
            # mul!(C, Uvec[μ], Uν_pμ)
            # mul!(D, C, Uμ_pν)
            mul_AshiftB!(C, Uvec[μ], Uvec[ν], shift_μ)
            mul_AshiftB!(D, C, Uvec[μ], shift_ν)
            S += realtrace(D)
        end
    end
    return S
end

function loss_mulAshiftedBtest_munuloop_shiftarray(U1, U2, U3, U4, temp)
    C = temp[1]
    D = temp[2]
    S = 0.0
    Uvec = (U1, U2, U3, U4)
    shifts = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    for μ = 1:3
        shift_μ = shifts[μ]
        for ν = μ+1:3
            shift_ν = shifts[ν]
            mul_AshiftB!(C, Uvec[μ], Uvec[ν], shift_μ)
            mul_AshiftB!(D, C, Uvec[μ], shift_ν)
            S += realtrace(D)
        end
    end
    return S
end

function loss_mulAshiftedBtest_munuloop_pairs(U1, U2, U3, U4, temp)
    C = temp[1]
    D = temp[2]
    S = 0.0
    Uvec = (U1, U2, U3, U4)
    shifts = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    pairs = ((1, 2), (1, 3), (2, 3))
    for (μ, ν) in pairs
        shift_μ = shifts[μ]
        shift_ν = shifts[ν]
        mul_AshiftB!(C, Uvec[μ], Uvec[ν], shift_μ)
        mul_AshiftB!(D, C, Uvec[μ], shift_ν)
        S += realtrace(D)
    end
    return S
end

function _loss_mulAshiftedBtest_munuloop_val_step!(C, D, Uvec, ::Val{μ}, ::Val{ν}) where {μ,ν}
    shifts = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    shift_μ = shifts[μ]
    shift_ν = shifts[ν]
    mul_AshiftB!(C, Uvec[μ], Uvec[ν], shift_μ)
    mul_AshiftB!(D, C, Uvec[μ], shift_ν)
    return realtrace(D)
end

function loss_mulAshiftedBtest_munuloop_val(U1, U2, U3, U4, temp)
    C = temp[1]
    D = temp[2]
    S = 0.0
    Uvec = (U1, U2, U3, U4)
    pairs = ((Val(1), Val(2)), (Val(1), Val(3)), (Val(2), Val(3)))
    for (μ, ν) in pairs
        S += _loss_mulAshiftedBtest_munuloop_val_step!(C, D, Uvec, μ, ν)
    end
    return S
end

function loss_mulAshiftedBtest_munuloop_unrolled(U1, U2, U3, U4, temp)
    C = temp[1]
    D = temp[2]
    S = 0.0

    shift_1 = (1, 0, 0, 0)
    shift_2 = (0, 1, 0, 0)
    shift_3 = (0, 0, 1, 0)

    mul_AshiftB!(C, U1, U2, shift_1)
    mul_AshiftB!(D, C, U1, shift_2)
    S += realtrace(D)

    mul_AshiftB!(C, U1, U3, shift_1)
    mul_AshiftB!(D, C, U1, shift_3)
    S += realtrace(D)

    mul_AshiftB!(C, U2, U3, shift_2)
    mul_AshiftB!(D, C, U2, shift_3)
    S += realtrace(D)

    return S
end

function loss_mulAshiftedBtest_munuloop_unrolled_shiftmul(U1, U2, U3, U4, temp)
    C = temp[1]
    D = temp[2]
    S = 0.0

    shift_1 = (1, 0, 0, 0)
    shift_2 = (0, 1, 0, 0)
    shift_3 = (0, 0, 1, 0)

    U2_p1 = shift_L(U2, shift_1)
    mul!(C, U1, U2_p1)
    U1_p2 = shift_L(U1, shift_2)
    mul!(D, C, U1_p2)
    S += realtrace(D)

    U3_p1 = shift_L(U3, shift_1)
    mul!(C, U1, U3_p1)
    U1_p3 = shift_L(U1, shift_3)
    mul!(D, C, U1_p3)
    S += realtrace(D)

    U3_p2 = shift_L(U3, shift_2)
    mul!(C, U2, U3_p2)
    U2_p3 = shift_L(U2, shift_3)
    mul!(D, C, U2_p3)
    S += realtrace(D)

    return S
end



function loss_mulABdagtest(U1, U2, U3, U4, temp)
    C = temp[1]
    mul!(C, U1, U2')
    return realtrace(C)
end

function loss_mulABdagtest_loop(U1, U2, U3, U4, temp)
    C = temp[1]
    D = temp[2]
    mul!(C, U1, U2')
    mul!(D, C, U3)
    mul!(C, D, U1')
    mul!(D, C, U4)
    return realtrace(D)
end

function loss_plaquette(U1, U2, U3, U4, shift1, shift2, temp)
    C = temp[1]
    D = temp[2]
    E = temp[3]

    U2_p1 = shift_L(U2, shift1)  # U_nu(x+mu)
    U1_p2 = shift_L(U1, shift2)  # U_mu(x+nu)

    mul!(C, U1, U2_p1)    # U_mu(x) * U_nu(x+mu)
    mul!(D, C, U1_p2')      # ... * U_mu(x+nu)†
    mul!(E, D, U2')       # ... * U_nu(x)†
    return realtrace(E)
    #mul!(C, D, U[2]')       # ... * U_nu(x)†
    #return realtrace(C)
end

function calc_action(U1, U2, U3, U4, β, NC, temp)
    U = (U1, U2, U3, U4)
    ndir = length(U)
    dim = length(U1.PN)
    C = temp[1]
    D = temp[2]
    E = temp[3]
    S = 0.0

    for μ = 1:ndir
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        for ν = μ:ndir
            if ν == μ
                continue
            end
            shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)
            Uμ_pν = shift_L(U[μ], shift_ν)
            Uν_pμ = shift_L(U[ν], shift_μ)
            mul!(C, U[μ], Uν_pμ)
            mul!(D, C, Uμ_pν')
            mul!(E, D, U[ν]')
            S += realtrace(E)

            mul!(C, U[ν], Uμ_pν)
            mul!(D, C, Uν_pμ')
            mul!(E, D, U[μ]')
            S += realtrace(E)
        end
    end

    return -S * β / NC
end

function calc_action_unrolled(U1, U2, U3, U4, β, NC, temp)
    C = temp[1]
    D = temp[2]
    E = temp[3]
    S = 0.0

    shift_1 = (1, 0, 0, 0)
    shift_2 = (0, 1, 0, 0)
    shift_3 = (0, 0, 1, 0)
    shift_4 = (0, 0, 0, 1)

    U1_p2 = shift_L(U1, shift_2)
    U2_p1 = shift_L(U2, shift_1)
    mul!(C, U1, U2_p1)
    mul!(D, C, U1_p2')
    mul!(E, D, U2')
    S += realtrace(E)
    mul!(C, U2, U1_p2)
    mul!(D, C, U2_p1')
    mul!(E, D, U1')
    S += realtrace(E)

    U1_p3 = shift_L(U1, shift_3)
    U3_p1 = shift_L(U3, shift_1)
    mul!(C, U1, U3_p1)
    mul!(D, C, U1_p3')
    mul!(E, D, U3')
    S += realtrace(E)
    mul!(C, U3, U1_p3)
    mul!(D, C, U3_p1')
    mul!(E, D, U1')
    S += realtrace(E)

    U1_p4 = shift_L(U1, shift_4)
    U4_p1 = shift_L(U4, shift_1)
    mul!(C, U1, U4_p1)
    mul!(D, C, U1_p4')
    mul!(E, D, U4')
    S += realtrace(E)
    mul!(C, U4, U1_p4)
    mul!(D, C, U4_p1')
    mul!(E, D, U1')
    S += realtrace(E)

    U2_p3 = shift_L(U2, shift_3)
    U3_p2 = shift_L(U3, shift_2)
    mul!(C, U2, U3_p2)
    mul!(D, C, U2_p3')
    mul!(E, D, U3')
    S += realtrace(E)
    mul!(C, U3, U2_p3)
    mul!(D, C, U3_p2')
    mul!(E, D, U2')
    S += realtrace(E)

    U2_p4 = shift_L(U2, shift_4)
    U4_p2 = shift_L(U4, shift_2)
    mul!(C, U2, U4_p2)
    mul!(D, C, U2_p4')
    mul!(E, D, U4')
    S += realtrace(E)
    mul!(C, U4, U2_p4)
    mul!(D, C, U4_p2')
    mul!(E, D, U2')
    S += realtrace(E)

    U3_p4 = shift_L(U3, shift_4)
    U4_p3 = shift_L(U4, shift_3)
    mul!(C, U3, U4_p3)
    mul!(D, C, U3_p4')
    mul!(E, D, U4')
    S += realtrace(E)
    mul!(C, U4, U3_p4)
    mul!(D, C, U4_p3')
    mul!(E, D, U3')
    S += realtrace(E)

    return -S * β / NC
end

function _calc_action_step!(C, D, E, Uμ, Uν, shift_μ, shift_ν)
    Uμ_pν = shift_L(Uμ, shift_ν)
    Uν_pμ = shift_L(Uν, shift_μ)

    mul!(C, Uμ, Uν_pμ)
    mul!(D, C, Uμ_pν')
    mul!(E, D, Uν')
    S = realtrace(E)

    mul!(C, Uν, Uμ_pν)
    mul!(D, C, Uν_pμ')
    mul!(E, D, Uμ')
    S += realtrace(E)

    return S
end

function calc_action_loopfn(U1, U2, U3, U4, β, NC, temp)
    U = (U1, U2, U3, U4)
    ndir = length(U)
    dim = length(U1.PN)
    C = temp[1]
    D = temp[2]
    E = temp[3]
    S = 0.0

    for μ = 1:ndir
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        for ν = μ:ndir
            if ν == μ
                continue
            end
            shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)
            S += _calc_action_step!(C, D, E, U[μ], U[ν], shift_μ, shift_ν)
        end
    end

    return -S * β / NC
end

function _calc_action_step_add!(Uout, C, D, Uμ, Uν, shift_μ, shift_ν)
    Uμ_pν = shift_L(Uμ, shift_ν)
    Uν_pμ = shift_L(Uν, shift_μ)

    mul!(C, Uμ, Uν_pμ)
    mul!(D, C, Uμ_pν')
    mul!(Uout, D, Uν')
    #add_matrix!(Uout, C)
    #S = realtrace(E)

    mul!(C, Uν, Uμ_pν)
    mul!(D, C, Uν_pμ')
    mul!(C, D, Uμ')
    add_matrix!(Uout, C)
    #S += realtrace(E)

    #return S
end


function calc_action_loopfn_add(U1, U2, U3, U4, β, NC, temp)
    U = (U1, U2, U3, U4)
    ndir = length(U)
    dim = length(U1.PN)
    Uout = temp[1]
    C = temp[2]
    D = temp[3]
    S = 0.0

    for μ = 1:ndir
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        for ν = μ:ndir
            if ν == μ
                continue
            end
            shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)
            _calc_action_step_add!(Uout, C, D, U[μ], U[ν], shift_μ, shift_ν)
            S += realtrace(Uout)
        end
    end

    return -S * β / NC
end


function calc_action_loopfn(U, β, NC, temp)
    ndir = length(U)
    dim = length(U[1].PN)
    C = temp[1]
    D = temp[2]
    E = temp[3]
    S = 0.0

    for μ = 1:ndir
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        for ν = μ:ndir
            if ν == μ
                continue
            end
            shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)
            S += _calc_action_step!(C, D, E, U[μ], U[ν], shift_μ, shift_ν)
        end
    end

    return -S * β / NC
end

function main()
    MPI.Init()

    NC = 2
    dim = 4
    gsize = (4, 4, 4, 4)
    nw = 1
    PEs = (1, 1, 1, 1)

    UA = randn(ComplexF64, NC, NC, gsize...)
    U1 = LatticeMatrix(UA, dim, PEs; nw, numtemps=4)
    set_halo!(U1)
    U2 = deepcopy(U1)
    U3 = deepcopy(U1)
    U4 = deepcopy(U1)
    U = [U1, U2, U3, U4]


    shift1 = (1, 0, 0, 0)
    dU = [similar(U1), similar(U1), similar(U1), similar(U1)]
    clear_matrix!.(dU)
    temp = [similar(U1), similar(U1), similar(U1), similar(U1)]
    dtemp = [similar(U1), similar(U1), similar(U1), similar(U1)]

    indices_mid = (3, 3, 3, 3)
    indices_halo = (2, 3, 3, 3)

    β = 3.0
    texp = 0.3

    S = realtrace(U[1])
    println(S)

    run_case_all("add_matrix", loss_add_matrix_test, loss_add_matrix_test,
        U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    run_case_all("add_matrix_adj", loss_add_matrix_adj_test, loss_add_matrix_adj_test,
        U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    fs_add_shift(U1, U2, U3, U4, temp) = loss_add_matrix_shiftedA_test(U1, U2, U3, U4, shift1, temp)
    run_case_all("add_matrix_shiftedA", fs_add_shift, fs_add_shift,
        U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    fs_add_shiftdag(U1, U2, U3, U4, temp) = loss_add_matrix_shifted_adj_test(U1, U2, U3, U4, shift1, temp)
    run_case_all("add_matrix_shifted_adj", fs_add_shiftdag, fs_add_shiftdag,
        U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)


    fs_exp_ta(U1, U2, U3, U4, temp) = calc_action_loss_exp_traceless_antihermitian(U1, U2, U3, U4, β, NC, texp, temp)
    run_case_all("calc_action_loss_exp_traceless_antihermitian", fs_exp_ta, fs_exp_ta,
        U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)


    fs_exp_ta(U1, U2, U3, U4, temp) = loss_exp_traceless_antihermitian(U1, U2, U3, U4, texp, temp)
    run_case_all("exp_traceless_antihermitian", fs_exp_ta, fs_exp_ta,
        U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    fs4l_add(U1, U2, U3, U4, temp) = calc_action_loopfn_add(U1, U2, U3, U4, β, NC, temp)
    run_case_all("calc_action_loopfn_add", fs4l_add, fs4l_add, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)




    run_case_all("mulAshiftedBtest_munuloop", loss_mulAshiftedBtest_munuloop,
        loss_mulAshiftedBtest_munuloop, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)


    fs3(U1, U2, U3, U4, temp) = loss_plaquette(U1, U2, U3, U4, (1, 0, 0, 0), (0, 1, 0, 0), temp)
    run_case_all("loss_plaquette", fs3, fs3, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)



    #fs4l_1(U, temp) = calc_action_loopfn(U, β, NC, temp)
    #run_case_all_vector("calc_action_loopfn_1", fs4l_1, fs4l_1, U, dU, temp, dtemp, indices_mid, indices_halo)


    fs4l(U1, U2, U3, U4, temp) = calc_action_loopfn(U1, U2, U3, U4, β, NC, temp)
    run_case_all("calc_action_loopfn", fs4l, fs4l, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)


    fs4u(U1, U2, U3, U4, temp) = calc_action_unrolled(U1, U2, U3, U4, β, NC, temp)
    run_case_all("calc_action_unrolled", fs4u, fs4u, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)


    #
    #fs4(U1, U2, U3, U4, temp) = calc_action(U1, U2, U3, U4, β, NC, temp)
    #run_case_all("calc_action", fs4, fs4, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)



    run_case_all("mulAshiftedBtest_munuloop_unrolled_shiftmul", loss_mulAshiftedBtest_munuloop_unrolled_shiftmul,
        loss_mulAshiftedBtest_munuloop_unrolled_shiftmul, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)


    run_case_all("mulAshiftedBtest_munuloop_unrolled", loss_mulAshiftedBtest_munuloop_unrolled,
        loss_mulAshiftedBtest_munuloop_unrolled, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)


    run_case_all("mulAshiftedBtest_munuloop_shiftarray", loss_mulAshiftedBtest_munuloop_shiftarray,
        loss_mulAshiftedBtest_munuloop_shiftarray, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    run_case_all("mulAshiftedBtest_munuloop_pairs", loss_mulAshiftedBtest_munuloop_pairs,
        loss_mulAshiftedBtest_munuloop_pairs, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    run_case_all("mulAshiftedBtest_munuloop_val", loss_mulAshiftedBtest_munuloop_val,
        loss_mulAshiftedBtest_munuloop_val, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)


    fs(U1, U2, U3, U4, temp) = loss_mulAshiftedBtest(U1, U2, U3, U4, (1, 0, 0, 0), temp)
    run_case_all("mulAshiftedBtest", fs, fs, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)
    fs2(U1, U2, U3, U4, temp) = loss_mulAshiftedBtest_loop(U1, U2, U3, U4, (1, 0, 0, 0), (0, 1, 0, 0), temp)
    run_case_all("mulAshiftedBtest_loop", fs2, fs2, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)


    run_case_all("mulABdagtest", loss_mulABdagtest, loss_mulABdagtest, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)
    run_case_all("mulABdagtest_loop", loss_mulABdagtest_loop, loss_mulABdagtest_loop, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)





    run_case_all("mulABtest_loop", loss_mulABtest_loop, loss_mulABtest_loop, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)
    f_num(U1, U2, U3, U4, t) = realtrace(U1)
    f(U1, U2, U3, U4, t) = realtrace(U1)
    run_case_all("realtrace", f, f_num, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)
    run_case_all("mulABtest", loss_mulABtest, loss_mulABtest, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)








end
main()
