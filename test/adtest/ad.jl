
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

function _dot_all(A::LatticeMatrix, B::LatticeMatrix)
    a = Array(A.A)
    b = Array(B.A)
    return sum(conj.(a) .* b)
end

function _report_max_grad(label, dU::LatticeMatrix)
    absvals = abs.(dU.A)
    maxval = maximum(absvals)
    idx = argmax(absvals)
    inds = Tuple(CartesianIndices(size(dU.A))[idx])
    println(label, " max |grad| = ", maxval, " idx = ", inds)
    return nothing
end

function _max_grad_indices(dU::LatticeMatrix)
    absvals = abs.(dU.A)
    idx = argmax(absvals)
    return Tuple(CartesianIndices(size(dU.A))[idx])
end

function _report_grad_at(label, dU::LatticeMatrix, indices)
    block = dU.A[:, :, indices...]
    println(label, " |grad| max = ", maximum(abs, block), " idx = ", indices)
    display(block)
    return nothing
end

function _shift_delta_probe!(label, U::LatticeMatrix, shift, idx)
    B = similar(U)
    clear_matrix!(B)
    B.A[1, 1, idx...] = 1.0
    set_halo!(B)
    Bs = shift_L(B, shift)
    C = similar(U)
    substitute!(C, Bs)
    maxval = maximum(abs, C.A)
    pos = Tuple(CartesianIndices(size(C.A))[argmax(abs.(C.A))])
    println(label, " idx=", idx, " shift=", shift, " max=", maxval, " pos=", pos)
    return nothing
end

function check_set_halo_adjoint!(X::LatticeMatrix, Y::LatticeMatrix; tol=1e-10)
    X.A .= randn(eltype(X.A), size(X.A))
    Y.A .= randn(eltype(Y.A), size(Y.A))

    Xh = deepcopy(X)
    set_halo!(Xh)
    lhs = _dot_all(Xh, Y)

    Yh = deepcopy(Y)
    for d in length(Yh.PN):-1:1
        LatticeMatrices.fold_halo_dim_to_core_grad!(Yh, d)
        LatticeMatrices.zero_halo_dim!(Yh, d)
    end
    rhs = _dot_all(X, Yh)

    diff = abs(lhs - rhs)
    rel = diff / (abs(lhs) + abs(rhs) + eps(Float64))
    println("set_halo adjoint check: |lhs-rhs|=", diff, " rel=", rel)
    if rel > tol
        println("set_halo adjoint check failed: rel=", rel, " tol=", tol)
    end
    return rel
end

function check_mul_AshiftB_adjoint!(Ain::LatticeMatrix, Bin::LatticeMatrix; shift=(1, 0, 0, 0), tol=1e-10)
    A = deepcopy(Ain)
    B = deepcopy(Bin)
    A.A .= randn(eltype(A.A), size(A.A))
    B.A .= randn(eltype(B.A), size(B.A))
    set_halo!(A)
    set_halo!(B)

    Cref = similar(A)
    Y = similar(A)
    Y.A .= randn(eltype(Y.A), size(Y.A))
    set_halo!(Y)

    temp = [similar(A)]
    dtemp = [similar(A)]
    clear_matrix!.(temp)
    clear_matrix!.(dtemp)

    dA = similar(A)
    dB = similar(B)
    dU3 = similar(A)
    dU4 = similar(A)
    clear_matrix!.((dA, dB, dU3, dU4))

    function f_mul_shift(U1, U2, U3, U4, temp)
        C = temp[1]
        mul_AshiftB!(C, U1, U2, shift)
        return real(_dot_all(C, Y))
    end

    fval = f_mul_shift(A, B, A, A, temp)
    clear_matrix!.(temp)
    clear_matrix!.(dtemp)

    Enzyme_derivative!(
        f_mul_shift,
        A,
        B,
        A,
        A,
        dA,
        dB,
        dU3,
        dU4; temp=temp, dtemp=dtemp)

    rhsA = real(_dot_all(dA, A))
    rhsB = real(_dot_all(dB, B))
    relA = abs(fval - rhsA) / (abs(fval) + abs(rhsA) + eps(Float64))
    relB = abs(fval - rhsB) / (abs(fval) + abs(rhsB) + eps(Float64))
    println("mul_AshiftB adjoint check (A): |lhs-rhs|=", abs(fval - rhsA), " rel=", relA)
    println("mul_AshiftB adjoint check (B): |lhs-rhs|=", abs(fval - rhsB), " rel=", relB)
    if relA > tol || relB > tol
        println("mul_AshiftB adjoint check failed: relA=", relA, " relB=", relB, " tol=", tol)
    end
    return max(relA, relB)
end

function check_fold_halo_dim_to_core!(X::LatticeMatrix; dim=1)
    C = deepcopy(X)
    clear_matrix!(C)
    nw = C.nw
    pn = C.PN
    D = length(pn)
    @assert nw == 1 "check_fold_halo_dim_to_core! assumes nw=1"
    @assert pn[dim] >= 2 "check_fold_halo_dim_to_core! needs pn[dim] >= 2"

    idx_face = ntuple(i -> (i == dim ? 2 : 2), D)
    idx_ghost = ntuple(i -> (i == dim ? pn[i] + 2 * nw : 2), D)

    C.A[:, :, idx_ghost...] .= 1 + 0im
    LatticeMatrices.fold_halo_dim_to_core_grad!(C, dim)
    LatticeMatrices.zero_halo_dim!(C, dim)

    face_val = C.A[1, 1, idx_face...]
    ghost_val = C.A[1, 1, idx_ghost...]
    println("fold_halo_dim_to_core check dim=$dim face=", face_val, " ghost=", ghost_val)
    return face_val, ghost_val
end

function check_add_matrix_shiftedA_adjoint!(Ain::LatticeMatrix; shift=(1, 0, 0, 0), tol=1e-10)
    A = deepcopy(Ain)
    A.A .= randn(eltype(A.A), size(A.A))
    set_halo!(A)

    Y = similar(A)
    Y.A .= randn(eltype(Y.A), size(Y.A))
    set_halo!(Y)

    temp = [similar(A)]
    dtemp = [similar(A)]
    clear_matrix!.(temp)
    clear_matrix!.(dtemp)

    dA = similar(A)
    dU2 = similar(A)
    dU3 = similar(A)
    dU4 = similar(A)
    clear_matrix!.((dA, dU2, dU3, dU4))

    function f_add_shift(U1, U2, U3, U4, temp)
        C = temp[1]
        clear_matrix!(C)
        LatticeMatrices.add_matrix_shiftedA!(C, U1, shift)
        return real(_dot_all(C, Y))
    end

    fval = f_add_shift(A, A, A, A, temp)
    clear_matrix!.(temp)
    clear_matrix!.(dtemp)

    Enzyme_derivative!(
        f_add_shift,
        A,
        A,
        A,
        A,
        dA,
        dU2,
        dU3,
        dU4; temp=temp, dtemp=dtemp)

    rhs = real(_dot_all(dA, A))
    rel = abs(fval - rhs) / (abs(fval) + abs(rhs) + eps(Float64))
    println("add_matrix_shiftedA adjoint check: |lhs-rhs|=", abs(fval - rhs), " rel=", rel)
    if rel > tol
        println("add_matrix_shiftedA adjoint check failed: rel=", rel, " tol=", tol)
    end
    return rel
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

function run_case_all_phi(label, f, f_num, U1, U2, U3, U4, dU1, dU2, dU3, dU4,
    phi, phitemp, dphitemp, temp, dtemp, indices_mid, indices_halo; tol=1e-4)
    println("=== ", label, " ===")

    dU = [dU1, dU2, dU3, dU4]
    U = [U1, U2, U3, U4]

    clear_matrix!.(dU)
    clear_matrix!.(temp)
    clear_matrix!.(dtemp)
    clear_matrix!.(phitemp)
    clear_matrix!.(dphitemp)

    Enzyme_derivative!(
        f,
        U1,
        U2,
        U3,
        U4,
        dU1,
        dU2,
        dU3,
        dU4,
        nodiff(phi); temp=temp, dtemp=dtemp, phitemp=phitemp, dphitemp=dphitemp)

    clear_matrix!.(temp)
    clear_matrix!.(phitemp)
    f_num_1(Uvec) = f_num(Uvec[1], Uvec[2], Uvec[3], Uvec[4], phi, phitemp, temp)
    dUn_mid = Numerical_derivative_Enzyme(f_num_1, indices_mid, U)
    for k in 1:length(U)
        _report_diff("U$k mid", dU[k], dUn_mid[k], indices_mid; tol)
    end

    indices_halo_core = _halo_to_core_indices(indices_halo, U[1].PN, U[1].nw)
    clear_matrix!.(temp)
    clear_matrix!.(phitemp)
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

function run_case_set_halo(label, f, f_num, U1, U2, U3, U4, dU1, dU2, dU3, dU4, temp, dtemp, indices_halo; tol=1e-4)
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
    indices_halo_core = _halo_to_core_indices(indices_halo, U1.PN, U1.nw)
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

function loss_substitute_test(U1, U2, U3, U4, temp)
    C = temp[1]
    substitute!(C, U1)
    return realtrace(C)
end

function loss_set_halo_test(U1, U2, U3, U4, temp)
    C = temp[1]
    mul!(C, U1, U2)
    set_halo!(C)
    #set_halo!(U1)
    idx = (1, 3, 3, 3)
    s = zero(eltype(U1.A))
    s = realtrace(C)
    #@inbounds for ic = 1:U1.NC1
    #    s += C.A[ic, ic, idx...]
    #end
    return real(s)
end

function loss_shift_mul_test(U1, U2, U3, U4, temp)
    C = temp[1]
    shift1 = (1, 0, 0, 0)
    mul!(C, U1, shift_L(U2, shift1))
    return realtrace(C)
end

function loss_shift_muladj_test(U1, U2, U3, U4, temp)
    C = temp[1]
    shift1 = (1, 0, 0, 0)
    mul!(C, U1, shift_L(U2, shift1)')
    return realtrace(C)
end

function loss_calc_action_step_matrixadd_test(U1, U2, U3, U4, temp)
    C = temp[1]
    D = temp[2]
    E = temp[3]
    clear_matrix!(E)
    shift_μ = (1, 0, 0, 0)
    shift_ν = (0, 1, 0, 0)
    _calc_action_step_matrixadd!(C, D, E, U1, U2, shift_μ, shift_ν)
    return realtrace(E)
end

function loss_calc_action_step_matrixadd_sethalo_test(U1, U2, U3, U4, temp)
    Ufat1 = temp[4]
    Ufat2 = temp[5]
    C = temp[1]
    D = temp[2]
    E = temp[3]
    clear_matrix!(E)
    mul!(Ufat1, U1, U2)
    set_halo!(Ufat1)
    mul!(Ufat2, U3, U4)
    set_halo!(Ufat2)
    shift_μ = (1, 0, 0, 0)
    shift_ν = (0, 1, 0, 0)
    _calc_action_step_matrixadd!(C, D, E, Ufat1, Ufat2, shift_μ, shift_ν)
    return realtrace(E)
end

function loss_calc_action_step_matrixadd_sethalo_alloc_test(U1, U2, U3, U4, temp)
    Ufat1 = similar(U1)
    Ufat2 = similar(U1)
    C = temp[1]
    D = temp[2]
    E = temp[3]
    clear_matrix!(E)
    mul!(Ufat1, U1, U2)
    set_halo!(Ufat1)
    mul!(Ufat2, U3, U4)
    set_halo!(Ufat2)
    shift_μ = (1, 0, 0, 0)
    shift_ν = (0, 1, 0, 0)
    _calc_action_step_matrixadd!(C, D, E, Ufat1, Ufat2, shift_μ, shift_ν)
    return realtrace(E)
end

function loss_calc_action_step_matrixadd_sethalo_copy_test(U1, U2, U3, U4, temp)
    Ufat1 = similar(U1)
    Ufat2 = similar(U1)
    C = temp[1]
    D = temp[2]
    E = temp[3]
    clear_matrix!(E)
    substitute!(Ufat1, U1)
    set_halo!(Ufat1)
    substitute!(Ufat2, U3)
    set_halo!(Ufat2)
    shift_μ = (1, 0, 0, 0)
    shift_ν = (0, 1, 0, 0)
    _calc_action_step_matrixadd!(C, D, E, Ufat1, Ufat2, shift_μ, shift_ν)
    return realtrace(E)
end

function loss_shift_sethalo_mul_test(U1, U2, U3, U4, temp)
    C = temp[1]
    B = temp[2]
    substitute!(B, U3)
    set_halo!(B)
    shift1 = (1, 0, 0, 0)
    mul!(C, U1, shift_L(B, shift1))
    return realtrace(C)
end

function loss_shift_sethalo_mul_identity_test(U1, U2, U3, U4, temp)
    C = temp[1]
    B = temp[2]
    substitute!(B, U3)
    set_halo!(B)
    shift1 = (1, 0, 0, 0)
    I = temp[3]
    clear_matrix!(I)
    idx = (U1.nw + 1, 3, 3, 3)
    for ic = 1:U1.NC1
        I.A[ic, ic, idx...] = 1.0
    end
    set_halo!(I)
    mul!(C, I, shift_L(B, shift1))
    return realtrace(C)
end

function loss_shift_sethalo_mul_point_test(U1, U2, U3, U4, temp)
    C = temp[1]
    B = temp[2]
    substitute!(B, U3)
    set_halo!(B)
    shift1 = (1, 0, 0, 0)
    mul!(C, U1, shift_L(B, shift1))
    idx = (2, 3, 3, 3)
    s = zero(eltype(C.A))
    @inbounds for ic = 1:U1.NC1
        s += C.A[ic, ic, idx...]
    end
    return real(s)
end

function loss_shift_sethalo_muladj_test(U1, U2, U3, U4, temp)
    C = temp[1]
    B = temp[2]
    substitute!(B, U3)
    set_halo!(B)
    shift1 = (1, 0, 0, 0)
    mul!(C, U1, shift_L(B, shift1)')
    return realtrace(C)
end

function loss_sethalo_adjmul_test(U1, U2, U3, U4, temp)
    C = temp[1]
    B = temp[2]
    substitute!(B, U3)
    set_halo!(B)
    mul!(C, U1, B')
    return realtrace(C)
end

function loss_calc_action_step_matrixadd_noshift_test(U1, U2, U3, U4, temp)
    C = temp[1]
    D = temp[2]
    E = temp[3]
    clear_matrix!(E)
    shift_μ = (1, 0, 0, 0)
    shift_ν = (0, 1, 0, 0)
    _calc_action_step_matrixadd_noshift!(C, D, E, U1, U2, shift_μ, shift_ν)
    return realtrace(E)
end
function loss_expt_TA_test(U1, U2, U3, U4, temp)
    C = temp[1]
    t = 0.3
    expt_TA!(C, U1, t)
    return realtrace(C)
end

function loss_expt_TA_wrapper_test(U1, U2, U3, U4, temp)
    C = temp[1]
    t = 0.3
    UTA = Traceless_AntiHermitian(U1)
    expt!(C, UTA, t)
    return realtrace(C)
end

function loss_sethalo_shift_test(U1, U2, U3, U4, temp)
    C = temp[1]
    Cshift = temp[2]
    shift1 = (-1, 0, 0, 0)
    mul!(C, U1, U2)
    set_halo!(C)
    clear_matrix!(Cshift)
    add_matrix_shiftedA!(Cshift, C, shift1)
    return realtrace(Cshift)
end

function loss_sethalo_shift_pos_test(U1, U2, U3, U4, temp)
    C = temp[1]
    Cshift = temp[2]
    shift1 = (1, 0, 0, 0)
    mul!(C, U1, U2)
    set_halo!(C)
    clear_matrix!(Cshift)
    LatticeMatrices.add_matrix_shiftedA!(Cshift, C, shift1)
    return realtrace(Cshift)
end

function loss_stout_core_test(U1, U2, U3, U4, temp)
    dim = 4
    U = (U1, U2, U3, U4)
    C = temp[1]
    D = temp[2]
    Uout = temp[3]
    Ufat1 = temp[4]
    Ufat2 = temp[5]
    Ufat3 = temp[6]
    Ufat4 = temp[7]
    E = temp[8]
    clear_matrix!(E)

    μ = 1
    ν = 2
    shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
    shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)

    make_μloop(Uout, C, D, E, μ, U, shift_μ, dim, 0.3)
    mul!(Ufat1, Uout, U1)
    set_halo!(Ufat1)

    clear_matrix!(E)
    _calc_action_step_matrixadd!(C, D, E, Ufat1, Ufat2, shift_μ, shift_ν)
    return realtrace(E)
end

function loss_stout_core_nohalo_test(U1, U2, U3, U4, temp)
    dim = 4
    U = (U1, U2, U3, U4)
    C = temp[1]
    D = temp[2]
    Uout = temp[3]
    Ufat1 = temp[4]
    Ufat2 = temp[5]
    E = temp[8]
    clear_matrix!(E)

    μ = 1
    ν = 2
    shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
    shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)

    make_μloop(Uout, C, D, E, μ, U, shift_μ, dim, 0.3)
    mul!(Ufat1, Uout, U1)
    # no set_halo! on Ufat1 here

    clear_matrix!(E)
    _calc_action_step_matrixadd!(C, D, E, Ufat1, Ufat2, shift_μ, shift_ν)
    return realtrace(E)
end

function loss_stout_core_onlyhalo_test(U1, U2, U3, U4, temp)
    dim = 4
    U = (U1, U2, U3, U4)
    C = temp[1]
    D = temp[2]
    Uout = temp[3]
    Ufat1 = temp[4]
    Ufat2 = temp[5]
    E = temp[8]
    clear_matrix!(E)

    μ = 1
    ν = 2
    shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
    shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)

    make_μloop(Uout, C, D, E, μ, U, shift_μ, dim, 0.3)
    mul!(Ufat1, Uout, U1)
    set_halo!(Ufat1)

    clear_matrix!(E)
    _calc_action_step_matrixadd!(C, D, E, Ufat1, Ufat2, shift_μ, shift_ν)
    return realtrace(E)
end
function loss_substitute_shifted_test(U1, U2, U3, U4, shift1, temp)
    C = temp[1]
    U2_p1 = shift_L(U2, shift1)
    substitute!(C, U2_p1)
    return realtrace(C)
end

function loss_real_of_dot_test(U1, U2, U3, U4, temp)
    return real(LinearAlgebra.dot(U1, U2))
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

function _calc_action_step_add_expt_traceless_antihermitian!(Uout, t, C, D, Uμ, Uν, shift_μ, shift_ν)
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

    UTA = Traceless_AntiHermitian(Uout)
    expt!(C, UTA, t)

    #traceless_antihermitian!(D, Uout)
    #expt!(C, D, t)
    mul!(Uout, C, Uμ)
    #S += realtrace(E)

    #return S
end

function loss_expt_TA(U1, U2, U3, U4, t, temp)
    C = temp[1]
    UTA = Traceless_AntiHermitian(U1)
    expt!(C, UTA, t)
    return realtrace(C)
end

function run_expt_ta_t_grad_test(U1, U2, U3, U4, dU, temp, dtemp, t; tol=1e-6, eps=1e-6)
    println("=== expt_TA t grad ===")

    clear_matrix!.(dU)
    clear_matrix!.(temp)
    clear_matrix!.(dtemp)

    f(U1, U2, U3, U4, tt, temp) = loss_expt_TA(U1, U2, U3, U4, tt, temp)
    result = Enzyme.autodiff(
        Reverse,
        Enzyme.Const(f),
        Active,
        Duplicated(U1, dU[1]),
        Duplicated(U2, dU[2]),
        Duplicated(U3, dU[3]),
        Duplicated(U4, dU[4]),
        Active(t),
        Duplicated(temp, dtemp)
    )
    if result isa Tuple && length(result) == 1
        result = result[1]
    end
    dt_ad = result isa Tuple ? result[5] : result

    f_num(tt) = begin
        clear_matrix!.(temp)
        f(U1, U2, U3, U4, tt, temp)
    end
    dt_num = (f_num(t + eps) - f_num(t - eps)) / (2 * eps)

    diff = abs(dt_ad - dt_num)
    println("t grad AD = ", dt_ad, " numeric = ", dt_num, " diff = ", diff)
    if diff > tol
        println("WARNING: expt_TA t grad diff > tol (", tol, ")")
    end
    return nothing
end

function calc_action_loss_expt_traceless_antihermitian(U1, U2, U3, U4, β, NC, t, temp)
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
            _calc_action_step_add_expt_traceless_antihermitian!(Uout, t, C, D, U[μ], U[ν], shift_μ, shift_ν)
            S += realtrace(Uout)
        end
    end

    return -S * β / NC
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
            _calc_action_step_add_exp_traceless_antihermitian!(Uout, t, C, D, U[μ], U[ν], shift_μ, shift_ν)
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

function _calc_action_step_addsum!(Uout, C, D, Uμ, Uν, shift_μ, shift_ν)
    Uμ_pν = shift_L(Uμ, shift_ν)
    Uν_pμ = shift_L(Uν, shift_μ)

    mul!(C, Uμ, Uν_pμ)
    mul!(D, C, Uμ_pν')
    mul!(C, D, Uν')
    add_matrix!(Uout, C)
    #add_matrix!(Uout, C)
    #S = realtrace(E)

    mul!(C, Uν, Uμ_pν)
    mul!(D, C, Uν_pμ')
    mul!(C, D, Uμ')
    add_matrix!(Uout, C)
    #S += realtrace(E)

    #return S
end

function _calc_action_step_matrixadd!(C, D, E, Uμ, Uν, shift_μ, shift_ν)
    #clear_U!(E)
    Uμ_pν = shift_L(Uμ, shift_ν)
    Uν_pμ = shift_L(Uν, shift_μ)

    mul!(C, Uμ, Uν_pμ)
    mul!(D, C, Uμ_pν')
    mul!(C, D, Uν')
    add_matrix!(E, C)
    #S = realtrace(E)

    mul!(C, Uν, Uμ_pν)
    mul!(D, C, Uν_pμ')
    mul!(C, D, Uμ')
    add_matrix!(E, C)
    #S += realtrace(E)
    return
end

function _calc_action_step_matrixadd_noshift!(C, D, E, Uμ, Uν, shift_μ, shift_ν)
    mul_AshiftB!(C, Uμ, Uν, shift_μ)
    mul_A_shiftBdag!(D, C, Uμ, shift_ν)
    mul!(C, D, Uν')
    add_matrix!(E, C)

    mul_AshiftB!(C, Uν, Uμ, shift_ν)
    mul_A_shiftBdag!(D, C, Uν, shift_μ)
    mul!(C, D, Uμ')
    add_matrix!(E, C)
    return
end


function calc_action_loopfn_addsum(U1, U2, U3, U4, β, NC, temp)
    U = (U1, U2, U3, U4)
    ndir = length(U)
    dim = length(U1.PN)
    Uout = temp[1]
    C = temp[2]
    D = temp[3]
    clear_matrix!(Uout)

    for μ = 1:ndir
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        for ν = μ:ndir
            if ν == μ
                continue
            end
            shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)
            _calc_action_step_addsum!(Uout, C, D, U[μ], U[ν], shift_μ, shift_ν)
            #S += realtrace(Uout)
        end
    end
    S = realtrace(Uout)

    return -S * β / NC
end

function make_fatU(Uout, C, D, E, μ, U, shift_μ, dim, t)
    make_μloop(Uout, C, D, E, μ, U, shift_μ, dim, t)
end

function make_μloop(Uout, C, D, E, μ, U, shift_μ, dim, t)
    clear_matrix!(E)
    for ν = μ:dim
        if ν == μ
            continue
        end
        shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)
        _calc_action_step_addsum!(E, C, D, U[μ], U[ν], shift_μ, shift_ν)
        #S += realtrace(Uout)
    end
    UTA = Traceless_AntiHermitian(E)
    expt!(Uout, UTA, t)
end

function stoutsmearing_test(U1, U2, U3, U4, β, NC, temp, t)
    U = (U1, U2, U3, U4)
    ndir = length(U)
    dim = length(U1.PN)
    Uout = temp[1]
    C = temp[2]
    D = temp[3]
    clear_matrix!(Uout)
    S = 0.0
    Ufat1 = temp[4]
    Ufat2 = temp[5]
    Ufat3 = temp[6]
    Ufat4 = temp[7]
    E = temp[8]
    Ufat = (Ufat1, Ufat2, Ufat3, Ufat4)

    for μ = 1:ndir
        clear_matrix!(E)
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        make_μloop(Uout, C, D, E, μ, U, shift_μ, dim, t)
        mul!(Ufat[μ], Uout, U[μ])
        S += realtrace(Ufat[μ])
    end

    return -S * β / NC
end

function stoutsmearing_action(U1, U2, U3, U4, β, NC, temp, t)
    U = (U1, U2, U3, U4)
    ndir = length(U)
    dim = length(U1.PN)
    Uout = temp[1]
    C = temp[2]
    D = temp[3]
    clear_matrix!(Uout)
    S = 0.0
    Ufat1 = temp[4]
    Ufat2 = temp[5]
    Ufat3 = temp[6]
    Ufat4 = temp[7]
    E = temp[8]
    Ufat = (Ufat1, Ufat2, Ufat3, Ufat4)

    for μ = 1:ndir
        clear_matrix!(E)
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        make_μloop(Uout, C, D, E, μ, U, shift_μ, dim, t)
        mul!(Ufat[μ], Uout, U[μ])        #S += realtrace(Ufat[μ])
        set_halo!(Ufat)
    end

    S = 0.0

    for μ = 1:ndir
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        for ν = μ:ndir
            if ν == μ
                continue
            end
            shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)
            _calc_action_step_add!(Uout, C, D, Ufat[μ], Ufat[ν], shift_μ, shift_ν)
            S += realtrace(Uout)
        end
    end


    return -S * β / NC
end


function calc_action_stout(U1, U2, U3, U4, β, NC, t, temp)
    dim = 4
    U = (U1, U2, U3, U4)
    for μ = 1:dim
        set_halo!(U[μ])
    end
    C = temp[1]
    D = temp[2]
    Uout = temp[3]
    S = 0.0

    Ufat1 = temp[4]
    Ufat2 = temp[5]
    Ufat3 = temp[6]
    Ufat4 = temp[7]
    E = temp[8]
    clear_matrix!(E)

    Ufat = (Ufat1, Ufat2, Ufat3, Ufat4)
    for μ = 1:dim
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        make_μloop(Uout, C, D, E, μ, U, shift_μ, dim, t)
        mul!(Ufat[μ], Uout, U[μ])

    end
    for μ = 1:dim
        set_halo!(Ufat[μ])
    end


    for μ = 1:dim
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        for ν = μ:dim
            if ν == μ
                continue
            end
            clear_matrix!(E)
            shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)
            _calc_action_step_matrixadd!(C, D, E, Ufat[μ], Ufat[ν], shift_μ, shift_ν)
            S += realtrace(E)
        end
    end


    return -S * β / NC
end

function calc_action_stout_tmpfix(U1, U2, U3, U4, β, NC, t, temp)
    dim = 4
    U = (U1, U2, U3, U4)
    for μ = 1:dim
        set_halo!(U[μ])
    end
    S = 0.0

    Ufat1 = temp[4]
    Ufat2 = temp[5]
    Ufat3 = temp[6]
    Ufat4 = temp[7]
    Ufat = (Ufat1, Ufat2, Ufat3, Ufat4)

    for μ = 1:dim
        Uout = similar(U1)
        C = similar(U1)
        D = similar(U1)
        E = similar(U1)
        clear_matrix!.((Uout, C, D, E))
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        make_μloop(Uout, C, D, E, μ, U, shift_μ, dim, t)
        mul!(Ufat[μ], Uout, U[μ])
    end
    for μ = 1:dim
        set_halo!(Ufat[μ])
    end

    for μ = 1:dim
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        for ν = μ:dim
            if ν == μ
                continue
            end
            C = similar(U1)
            D = similar(U1)
            E = similar(U1)
            clear_matrix!.((C, D, E))
            shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)
            _calc_action_step_matrixadd!(C, D, E, Ufat[μ], Ufat[ν], shift_μ, shift_ν)
            S += realtrace(E)
        end
    end

    return -S * β / NC
end



function fermiontest(U1, U2, U3, U4, phi, phitemp, temp)
    C = temp[1]
    U = (U1, U2, U3, U4)
    phiC = phitemp[1]
    shift = (1, 0, 0, 0)
    #phi_p1 = Shifted_Lattice(phi, shift)
    phi_p1 = shift_L(phi, shift)
    mul!(phiC, U[1], phi_p1)
    #mul_AshiftB!(phiC, U[1], phi, shift)
    return real(dot(phi, phiC))
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

    check_set_halo_adjoint!(similar(U1), similar(U1))
    check_mul_AshiftB_adjoint!(similar(U1), similar(U1))
    check_fold_halo_dim_to_core!(similar(U1))
    check_add_matrix_shiftedA_adjoint!(similar(U1))

    fstout_stout(U1, U2, U3, U4, temp) = calc_action_stout(U1, U2, U3, U4, β, NC, t, temp)
    run_case_all("calc_action_stout", fstout_stout, fstout_stout, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)




    f_shift_sethalo_mul_id(U1, U2, U3, U4, temp) = loss_shift_sethalo_mul_identity_test(U1, U2, U3, U4, temp)
    run_case_all("shift_sethalo_mul_id", f_shift_sethalo_mul_id, f_shift_sethalo_mul_id, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    f_shift_sethalo_mul_point(U1, U2, U3, U4, temp) = loss_shift_sethalo_mul_point_test(U1, U2, U3, U4, temp)
    run_case_all("shift_sethalo_mul_point", f_shift_sethalo_mul_point, f_shift_sethalo_mul_point, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    f_shift_sethalo_mul(U1, U2, U3, U4, temp) = loss_shift_sethalo_mul_test(U1, U2, U3, U4, temp)
    run_case_all("shift_sethalo_mul", f_shift_sethalo_mul, f_shift_sethalo_mul, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    _report_max_grad("shift_sethalo_mul U3", dU[3])
    idx_core = indices_halo
    idx_ghost_plus = (U1.PN[1] + 2 * U1.nw, indices_halo[2], indices_halo[3], indices_halo[4])
    _report_grad_at("shift_sethalo_mul U3 core", dU[3], idx_core)
    _report_grad_at("shift_sethalo_mul U3 ghost_plus", dU[3], idx_ghost_plus)
    _shift_delta_probe!("shift_delta_core", U1, shift1, idx_core)
    _shift_delta_probe!("shift_delta_core_neg", U1, ntuple(i -> -shift1[i], length(shift1)), idx_core)

    idx_max = _max_grad_indices(dU[3])
    indices_max = idx_max[3:end]
    clear_matrix!.(temp)
    f_num_1(Uvec) = f_shift_sethalo_mul(Uvec[1], Uvec[2], Uvec[3], Uvec[4], temp)
    dUn_max = Numerical_derivative_Enzyme(f_num_1, indices_max, [U1, U2, U3, U4])
    _report_diff("U3 maxpos", dU[3], dUn_max[3], indices_max; tol=1e-4)

    #return


    f_shift_sethalo_muladj(U1, U2, U3, U4, temp) = loss_shift_sethalo_muladj_test(U1, U2, U3, U4, temp)
    run_case_all("shift_sethalo_muladj", f_shift_sethalo_muladj, f_shift_sethalo_muladj, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)
    _report_max_grad("shift_sethalo_muladj U3", dU[3])
    _report_grad_at("shift_sethalo_muladj U3 core", dU[3], idx_core)
    _report_grad_at("shift_sethalo_muladj U3 ghost_plus", dU[3], idx_ghost_plus)

    f_matrixadd_sethalo(U1, U2, U3, U4, temp) = loss_calc_action_step_matrixadd_sethalo_test(U1, U2, U3, U4, temp)
    run_case_all("calc_action_step_matrixadd_sethalo", f_matrixadd_sethalo, f_matrixadd_sethalo, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    #return


    f_sethalo_adjmul(U1, U2, U3, U4, temp) = loss_sethalo_adjmul_test(U1, U2, U3, U4, temp)
    run_case_all("sethalo_adjmul", f_sethalo_adjmul, f_sethalo_adjmul, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)



    #return




    #return


    f_matrixadd_sethalo_alloc(U1, U2, U3, U4, temp) = loss_calc_action_step_matrixadd_sethalo_alloc_test(U1, U2, U3, U4, temp)
    run_case_all("calc_action_step_matrixadd_sethalo_alloc", f_matrixadd_sethalo_alloc, f_matrixadd_sethalo_alloc, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    f_matrixadd_sethalo_copy(U1, U2, U3, U4, temp) = loss_calc_action_step_matrixadd_sethalo_copy_test(U1, U2, U3, U4, temp)
    run_case_all("calc_action_step_matrixadd_sethalo_copy", f_matrixadd_sethalo_copy, f_matrixadd_sethalo_copy, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)




    fstout_stout_tmpfix(U1, U2, U3, U4, temp) = calc_action_stout_tmpfix(U1, U2, U3, U4, β, NC, t, temp)
    run_case_all("calc_action_stout_tmpfix", fstout_stout_tmpfix, fstout_stout_tmpfix, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)



    f_sethalo_shift(U1, U2, U3, U4, temp) = loss_sethalo_shift_test(U1, U2, U3, U4, temp)
    run_case_all("sethalo_shift_neg", f_sethalo_shift, f_sethalo_shift, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    indices_halo_pos = (U1.PN[1] + 1, 3, 3, 3)
    f_sethalo_shift_pos(U1, U2, U3, U4, temp) = loss_sethalo_shift_pos_test(U1, U2, U3, U4, temp)
    run_case_all("sethalo_shift_pos", f_sethalo_shift_pos, f_sethalo_shift_pos, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo_pos)
    # return


    ndices_halo_pos = indices_halo
    f_sethalo_shift_pos(U1, U2, U3, U4, temp) = loss_sethalo_shift_pos_test(U1, U2, U3, U4, temp)
    run_case_all("sethalo_shift_pos2", f_sethalo_shift_pos, f_sethalo_shift_pos, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo_pos)

    f_matrixadd_noshift(U1, U2, U3, U4, temp) = loss_calc_action_step_matrixadd_noshift_test(U1, U2, U3, U4, temp)
    run_case_all("calc_action_step_matrixadd_noshift", f_matrixadd_noshift, f_matrixadd_noshift, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    f_matrixadd(U1, U2, U3, U4, temp) = loss_calc_action_step_matrixadd_test(U1, U2, U3, U4, temp)
    run_case_all("calc_action_step_matrixadd", f_matrixadd, f_matrixadd, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)


    f_stout_core(U1, U2, U3, U4, temp) = loss_stout_core_test(U1, U2, U3, U4, temp)
    run_case_all("stout_core", f_stout_core, f_stout_core, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    f_stout_core_nohalo(U1, U2, U3, U4, temp) = loss_stout_core_nohalo_test(U1, U2, U3, U4, temp)
    run_case_all("stout_core_nohalo", f_stout_core_nohalo, f_stout_core_nohalo, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    f_stout_core_onlyhalo(U1, U2, U3, U4, temp) = loss_stout_core_onlyhalo_test(U1, U2, U3, U4, temp)
    run_case_all("stout_core_onlyhalo", f_stout_core_onlyhalo, f_stout_core_onlyhalo, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)








    f_shift_mul(U1, U2, U3, U4, temp) = loss_shift_mul_test(U1, U2, U3, U4, temp)
    run_case_all("shift_mul", f_shift_mul, f_shift_mul, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    f_shift_muladj(U1, U2, U3, U4, temp) = loss_shift_muladj_test(U1, U2, U3, U4, temp)
    run_case_all("shift_muladj", f_shift_muladj, f_shift_muladj, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)


    f_expt_ta(U1, U2, U3, U4, temp) = loss_expt_TA_test(U1, U2, U3, U4, temp)
    run_case_all("expt_TA", f_expt_ta, f_expt_ta, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    f_expt_wrapper(U1, U2, U3, U4, temp) = loss_expt_TA_wrapper_test(U1, U2, U3, U4, temp)
    run_case_all("expt_TA_wrapper", f_expt_wrapper, f_expt_wrapper, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)





    #fstout(U1, U2, U3, U4, temp) = stoutsmearing_action(U1, U2, U3, U4, β, NC, temp, t)
    #run_case_all("stoutsmearing_action", fstout, fstout, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    run_case_set_halo("set_halo", loss_set_halo_test, loss_set_halo_test,
        U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_halo_set)

    run_case_all("real_of_dot", loss_real_of_dot_test, loss_real_of_dot_test, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)



    return

    #fermitest(U1, U2, U3, U4, phi, phitemp, temp) = fermiontest(U1, U2, U3, U4, phi, phitemp, temp)
    # run_case_all_phi("fermitest", fermitest, fermitest, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], phi, phitemp, dphitemp, temp, dtemp, indices_mid, indices_halo)



    run_case_all("substitute", loss_substitute_test, loss_substitute_test,
        U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    fs_sub_shift(U1, U2, U3, U4, temp) = loss_substitute_shifted_test(U1, U2, U3, U4, shift1, temp)
    run_case_all("substitute_shifted", fs_sub_shift, fs_sub_shift,
        U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    t = 0.3
    fstout(U1, U2, U3, U4, temp) = stoutsmearing_test(U1, U2, U3, U4, β, NC, temp, t)
    run_case_all("stoutsmearing_test", fstout, fstout, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)


    run_expt_ta_t_grad_test(U1, U2, U3, U4, dU, temp, dtemp, texp)

    S = realtrace(U[1])
    println(S)

    t = 0.2
    traceless_antihermitian!(temp[1], U[1])
    expt!(temp[2], temp[1], t)
    display(temp[2].A[:, :, 2, 2, 2, 2])
    UTA = Traceless_AntiHermitian(U[1])
    expt!(temp[1], UTA, t)
    display(temp[1].A[:, :, 2, 2, 2, 2])



    fstout(U1, U2, U3, U4, temp) = stoutsmearing_test(U1, U2, U3, U4, β, NC, temp, t)
    run_case_all("stoutsmearing_test", fstout, fstout, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)


    return
    fs4l_addsum(U1, U2, U3, U4, temp) = calc_action_loopfn_addsum(U1, U2, U3, U4, β, NC, temp)
    run_case_all("calc_action_loopfn_addsum", fs4l_addsum, fs4l_addsum, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)



    fs_expt_ta(U1, U2, U3, U4, temp) = loss_expt_TA(U1, U2, U3, U4, texp, temp)
    run_case_all("expt_TA", fs_expt_ta, fs_expt_ta,
        U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    fs_expt_tan(U1, U2, U3, U4, temp) = calc_action_loss_expt_traceless_antihermitian(U1, U2, U3, U4, β, NC, texp, temp)
    run_case_all("calc_action_loss_expt_traceless_antihermitian", fs_expt_tan, fs_expt_tan,
        U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], temp, dtemp, indices_mid, indices_halo)

    #return

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


    fs_exp_tan(U1, U2, U3, U4, temp) = calc_action_loss_exp_traceless_antihermitian(U1, U2, U3, U4, β, NC, texp, temp)
    run_case_all("calc_action_loss_exp_traceless_antihermitian", fs_exp_tan, fs_exp_tan,
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
