using LatticeMatrices
using MPI
using LinearAlgebra
using Enzyme
import JACC
using InteractiveUtils

JACC.@init_backend

function loss_plain(U, temp)
    C = temp[1]
    mul!(C, U[1], U[2])
    return realtrace(C)
end

function loss_shifted(U, shift1, temp)
    C = temp[1]
    U2_p1 = Shifted_Lattice(U[2], shift1)  # U_nu(x+mu)
    mul!(C, U[1], U2_p1)   # U_mu(x) * U_nu(x+mu)
    return realtrace(C)
end

function loss_combined(U, shift1, temp)
    C = temp[1]
    D = temp[2]
    mul!(C, U[1], U[2])
    base = realtrace(C)
    U2_p1 = Shifted_Lattice(U[2], shift1)
    mul!(D, U[1], U2_p1)
    return base + realtrace(D)
end

function loss_shifted_adjoint(U, shift1, temp)
    C = temp[1]
    U2_p1 = Shifted_Lattice(U[2], shift1)
    mul!(C, U[1], U2_p1')   # U_mu(x) * U_nu(x+mu)†
    return realtrace(C)
end

function loss_plain_adjoint(U, temp)
    C = temp[1]
    mul!(C, U[1], U[2]')    # U_mu(x) * U_nu(x)†
    return realtrace(C)
end

function loss_plain_left_adjoint(U, temp)
    C = temp[1]
    mul!(C, U[1]', U[2])    # U_mu(x)† * U_nu(x)
    return realtrace(C)
end

function loss_plain_both_adjoint(U, temp)
    C = temp[1]
    mul!(C, U[1]', U[2]')   # U_mu(x)† * U_nu(x)†
    return realtrace(C)
end

function loss_plaquette(U, shift1, shift2, temp)
    C = temp[1]
    D = temp[2]
    E = temp[3]

    U2_p1 = Shifted_Lattice(U[2], shift1)  # U_nu(x+mu)
    U1_p2 = Shifted_Lattice(U[1], shift2)  # U_mu(x+nu)

    mul!(C, U[1], U2_p1)    # U_mu(x) * U_nu(x+mu)
    mul!(D, C, U1_p2')      # ... * U_mu(x+nu)†
    mul!(E, D, U[2]')       # ... * U_nu(x)†
    return realtrace(E)
end

function loss_shift_shift_adj(U, shift1, shift2, temp)
    C = temp[1]
    D = temp[2]
    E = temp[3]

    U2_p1 = Shifted_Lattice(U[2], shift1)  # U_nu(x+mu)
    U1_p2 = Shifted_Lattice(U[1], shift2)  # U_mu(x+nu)

    mul!(C, U[1], U2_p1)     # U_mu(x) * U_nu(x+mu)
    mul!(D, C, U1_p2')       # ... * U_mu(x+nu)†
    mul!(E, D, U[2]')        # ... * U_nu(x)†
    return realtrace(E)
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

function run_case(label, f, U, dU, temp, dtemp, indices_mid, indices_halo; tol=1e-5)
    println("=== ", label, " ===")

    clear_matrix!.(dU)
    clear_matrix!.(temp)
    clear_matrix!.(dtemp)
    f_num(Uin) = f(Uin, temp)
    Wiltinger_derivative!(f, U, dU; temp=temp, dtemp=dtemp)

    clear_matrix!.(temp)
    dUn_mid = Wiltinger_numerical_derivative(f_num, indices_mid, U)
    _report_diff("U1 mid", dU[1], dUn_mid[1], indices_mid; tol)
    _report_diff("U2 mid", dU[2], dUn_mid[2], indices_mid; tol)

    clear_matrix!.(temp)
    dUn_halo = Wiltinger_numerical_derivative(f_num, indices_halo, U)
    _report_diff("U1 halo", dU[1], dUn_halo[1], indices_halo; tol)
    _report_diff("U2 halo", dU[2], dUn_halo[2], indices_halo; tol)
end

function main()
    MPI.Init()

    NC = 2
    dim = 4
    gsize = (4, 4, 4, 4)
    nw = 1
    PEs = (1, 1, 1, 1)

    UA = randn(ComplexF64, NC, NC, gsize...)
    U1 = LatticeMatrix(UA, dim, PEs; nw)
    set_halo!(U1)
    U2 = deepcopy(U1)
    U = [U1, U2]

    shift1 = (1, 0, 0, 0)
    dU = [similar(U1), similar(U1)]
    clear_matrix!.(dU)
    temp = [similar(U1), similar(U1), similar(U1)]
    dtemp = [similar(U1), similar(U1), similar(U1)]

    indices_mid = (3, 3, 3, 3)
    indices_halo = (5, 2, 2, 2)
    indices_halo = (3, 3, 4, 2)

    loss_plain_f(Uin, t) = loss_plain(Uin, t)
    loss_shifted_f(Uin, t) = loss_shifted(Uin, shift1, t)

    loss_combined_f(Uin, t) = loss_combined(Uin, shift1, t)
    loss_shifted_adj_f(Uin, t) = loss_shifted_adjoint(Uin, shift1, t)
    loss_plain_adj_f(Uin, t) = loss_plain_adjoint(Uin, t)
    loss_plain_left_adj_f(Uin, t) = loss_plain_left_adjoint(Uin, t)
    loss_plain_both_adj_f(Uin, t) = loss_plain_both_adjoint(Uin, t)
    loss_plaquette_f(Uin, t) = loss_plaquette(Uin, shift1, (0, 1, 0, 0), t)
    loss_shift_shift_adj_f(Uin, t) = loss_shift_shift_adj(Uin, shift1, (0, 1, 0, 0), t)

    run_case("plain mul! (U1 * U2)", loss_plain_f, U, dU, temp, dtemp, indices_mid, indices_halo)
    run_case("shifted mul! (U1 * shifted(U2))", loss_shifted_f, U, dU, temp, dtemp, indices_mid, indices_halo)
    run_case("combined (plain + shifted)", loss_combined_f, U, dU, temp, dtemp, indices_mid, indices_halo)
    run_case("shifted adjoint mul! (U1 * shifted(U2)')", loss_shifted_adj_f, U, dU, temp, dtemp, indices_mid, indices_halo)
    run_case("plain adjoint mul! (U1 * U2')", loss_plain_adj_f, U, dU, temp, dtemp, indices_mid, indices_halo)
    run_case("plain adjoint mul! (U1' * U2)", loss_plain_left_adj_f, U, dU, temp, dtemp, indices_mid, indices_halo)
    run_case("plain adjoint mul! (U1' * U2')", loss_plain_both_adj_f, U, dU, temp, dtemp, indices_mid, indices_halo)
    run_case("plaquette (U1 * U2_p1 * U1_p2' * U2')", loss_plaquette_f, U, dU, temp, dtemp, indices_mid, indices_halo)
    run_case("shifted * shifted adjoint (U1 * U2_p1 * U1_p2' * U2')", loss_shift_shift_adj_f, U, dU, temp, dtemp, indices_mid, indices_halo)

    MPI.Finalize()
end

main()
