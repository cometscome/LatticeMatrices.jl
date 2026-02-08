using LatticeMatrices
using MPI
using LinearAlgebra
using Enzyme
import JACC
JACC.@init_backend
import LatticeMatrices: mul_AshiftB!, expt_TA
import Enzyme: autodiff, Duplicated, Const
import Enzyme.EnzymeRules: forward, augmented_primal, reverse, FwdConfig, RevConfigWidth,
    Active, needs_primal, needs_shadow, AugmentedReturn,
    overwritten

import Enzyme: Duplicated, Const, Active
import Enzyme.EnzymeRules: augmented_primal, reverse, RevConfig, AugmentedReturn
using InteractiveUtils

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

function loss_mul_stAB_test(U1, U2, U3, U4, temp)
    C = temp[1]
    μ = 1
    sU1 = Staggered_Lattice(U1, μ)
    mul!(C, sU1, U2)
    return realtrace(C)
end

function loss_mul_stAB_mu2_test(U1, U2, U3, U4, temp)
    C = temp[1]
    μ = 2
    sU1 = Staggered_Lattice(U1, μ)
    mul!(C, sU1, U2)
    return realtrace(C)
end

function loss_mul_AstB_test(U1, U2, U3, U4, temp)
    C = temp[1]
    μ = 2
    sU2 = Staggered_Lattice(U2, μ)
    mul!(C, U1, sU2)
    return realtrace(C)
end

function loss_mul_stAstB_test(U1, U2, U3, U4, temp)
    C = temp[1]
    μA = 1
    μB = 2
    sU1 = Staggered_Lattice(U1, μA)
    sU2 = Staggered_Lattice(U2, μB)
    mul!(C, sU1, sU2)
    return realtrace(C)
end

function loss_mul_stAdagB_test(U1, U2, U3, U4, temp)
    C = temp[1]
    μA = 1
    sU1 = Staggered_Lattice(U1, μA)
    mul!(C, sU1', U2)
    return realtrace(C)
end

function loss_mul_AdagstB_test(U1, U2, U3, U4, temp)
    C = temp[1]
    μB = 2
    sU2 = Staggered_Lattice(U2, μB)
    mul!(C, U1', sU2)
    return realtrace(C)
end

function loss_mul_stAdagstB_test(U1, U2, U3, U4, temp)
    C = temp[1]
    μA = 1
    μB = 2
    sU1 = Staggered_Lattice(U1, μA)
    sU2 = Staggered_Lattice(U2, μB)
    mul!(C, sU1', sU2)
    return realtrace(C)
end

function loss_mul_stABdag_test(U1, U2, U3, U4, temp)
    C = temp[1]
    μA = 1
    sU1 = Staggered_Lattice(U1, μA)
    mul!(C, sU1, U2')
    return realtrace(C)
end

function loss_mul_AstBdag_test(U1, U2, U3, U4, temp)
    C = temp[1]
    μB = 2
    sU2 = Staggered_Lattice(U2, μB)
    mul!(C, U1, sU2')
    return realtrace(C)
end

function loss_mul_stAstBdag_test(U1, U2, U3, U4, temp)
    C = temp[1]
    μA = 1
    μB = 2
    sU1 = Staggered_Lattice(U1, μA)
    sU2 = Staggered_Lattice(U2, μB)
    mul!(C, sU1, sU2')
    return realtrace(C)
end

function loss_mul_stAdagBdag_test(U1, U2, U3, U4, temp)
    C = temp[1]
    μA = 1
    sU1 = Staggered_Lattice(U1, μA)
    mul!(C, sU1', U2')
    return realtrace(C)
end

function loss_mul_AdagstBdag_test(U1, U2, U3, U4, temp)
    C = temp[1]
    μB = 2
    sU2 = Staggered_Lattice(U2, μB)
    mul!(C, U1', sU2')
    return realtrace(C)
end

function loss_mul_stAdagstBdag_test(U1, U2, U3, U4, temp)
    C = temp[1]
    μA = 1
    μB = 2
    sU1 = Staggered_Lattice(U1, μA)
    sU2 = Staggered_Lattice(U2, μB)
    mul!(C, sU1', sU2')
    return realtrace(C)
end

function main()
    MPI.Init()
    ENV["LM_DEBUG_SET_HALO"] = "1"
    ENV["LM_DEBUG_ADD_SHIFT"] = "1"
    ENV["LM_DEBUG_SHIFT_L"] = "1"
    ENV["LM_DEBUG_SHIFT_MUL"] = "1"
    ENV["LM_DEBUG_SHIFT_L"] = "1"
    ENV["LM_DEBUG_SHIFT_MUL"] = "1"
    ENV["LM_DEBUG_SUBSTITUTE"] = "1"
    NC = 2
    dim = 4
    gsize = (4, 4, 4, 4)
    nw = 1
    PEs = (1, 1, 1, 1)
    NG = 4

    UA = randn(ComplexF64, NC, NC, gsize...)
    U1 = LatticeMatrix(UA, dim, PEs; nw, numtemps=10)
    set_halo!(U1)
    U2 = deepcopy(U1)
    U3 = deepcopy(U1)
    U4 = deepcopy(U1)
    U = [U1, U2, U3, U4]


    phiA = randn(ComplexF64, NC, NG, gsize...)
    phi = LatticeMatrix(phiA, dim, PEs; nw, numtemps=10)
    set_halo!(phi)
    println(dot(phi, phi))


    shift1 = (1, 0, 0, 0)
    dU = [similar(U1), similar(U1), similar(U1), similar(U1)]
    clear_matrix!.(dU)
    temp = typeof(U1)[]
    dtemp = typeof(U1)[]
    phitemp = typeof(phi)[]
    dphitemp = typeof(phi)[]
    for i = 1:9
        push!(temp, similar(U1))
        push!(dtemp, similar(U1))
        push!(phitemp, similar(phi))
        push!(dphitemp, similar(phi))
    end

    #temp = [similar(U1), similar(U1), similar(U1), similar(U1)]
    #dtemp = [similar(U1), similar(U1), similar(U1), similar(U1)]

    indices_mid = (3, 3, 3, 3)
    indices_halo = (2, 3, 3, 3)

    β = 3.0
    texp = 0.3

    indices_halo_set = (1, 3, 3, 3)

    t = 0.3

    floss_mul_stAB_test(U1, U2, U3, U4, temp) = loss_mul_stAB_test(U1, U2, U3, U4, temp)
    run_case_all("loss_mul_stAB_test", floss_mul_stAB_test, floss_mul_stAB_test, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    floss_mul_stAB_mu2_test(U1, U2, U3, U4, temp) = loss_mul_stAB_mu2_test(U1, U2, U3, U4, temp)
    run_case_all("loss_mul_stAB_mu2_test", floss_mul_stAB_mu2_test, floss_mul_stAB_mu2_test, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    floss_mul_AstB_test(U1, U2, U3, U4, temp) = loss_mul_AstB_test(U1, U2, U3, U4, temp)
    run_case_all("loss_mul_AstB_test", floss_mul_AstB_test, floss_mul_AstB_test, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    floss_mul_stAstB_test(U1, U2, U3, U4, temp) = loss_mul_stAstB_test(U1, U2, U3, U4, temp)
    run_case_all("loss_mul_stAstB_test", floss_mul_stAstB_test, floss_mul_stAstB_test, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    floss_mul_stAdagB_test(U1, U2, U3, U4, temp) = loss_mul_stAdagB_test(U1, U2, U3, U4, temp)
    run_case_all("loss_mul_stAdagB_test", floss_mul_stAdagB_test, floss_mul_stAdagB_test, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    floss_mul_AdagstB_test(U1, U2, U3, U4, temp) = loss_mul_AdagstB_test(U1, U2, U3, U4, temp)
    run_case_all("loss_mul_AdagstB_test", floss_mul_AdagstB_test, floss_mul_AdagstB_test, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    floss_mul_stAdagstB_test(U1, U2, U3, U4, temp) = loss_mul_stAdagstB_test(U1, U2, U3, U4, temp)
    run_case_all("loss_mul_stAdagstB_test", floss_mul_stAdagstB_test, floss_mul_stAdagstB_test, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    floss_mul_stABdag_test(U1, U2, U3, U4, temp) = loss_mul_stABdag_test(U1, U2, U3, U4, temp)
    run_case_all("loss_mul_stABdag_test", floss_mul_stABdag_test, floss_mul_stABdag_test, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    floss_mul_AstBdag_test(U1, U2, U3, U4, temp) = loss_mul_AstBdag_test(U1, U2, U3, U4, temp)
    run_case_all("loss_mul_AstBdag_test", floss_mul_AstBdag_test, floss_mul_AstBdag_test, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    floss_mul_stAstBdag_test(U1, U2, U3, U4, temp) = loss_mul_stAstBdag_test(U1, U2, U3, U4, temp)
    run_case_all("loss_mul_stAstBdag_test", floss_mul_stAstBdag_test, floss_mul_stAstBdag_test, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    floss_mul_stAdagBdag_test(U1, U2, U3, U4, temp) = loss_mul_stAdagBdag_test(U1, U2, U3, U4, temp)
    run_case_all("loss_mul_stAdagBdag_test", floss_mul_stAdagBdag_test, floss_mul_stAdagBdag_test, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    floss_mul_AdagstBdag_test(U1, U2, U3, U4, temp) = loss_mul_AdagstBdag_test(U1, U2, U3, U4, temp)
    run_case_all("loss_mul_AdagstBdag_test", floss_mul_AdagstBdag_test, floss_mul_AdagstBdag_test, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    floss_mul_stAdagstBdag_test(U1, U2, U3, U4, temp) = loss_mul_stAdagstBdag_test(U1, U2, U3, U4, temp)
    run_case_all("loss_mul_stAdagstBdag_test", floss_mul_stAdagstBdag_test, floss_mul_stAdagstBdag_test, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)



end

main()
