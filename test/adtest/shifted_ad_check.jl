using LatticeMatrices
using MPI
using LinearAlgebra
using Enzyme
import JACC
using InteractiveUtils

JACC.@init_backend

function loss_plain(U)
    C = similar(U[1])

    mul!(C, U[1], U[2])
    return realtrace(C)
end

function loss_shifted(U, shift1)
    C = similar(U[1])
    U2_p1 = Shifted_Lattice(U[2], shift1)  # U_nu(x+mu)
    mul!(C, U[1], U2_p1)   # U_mu(x) * U_nu(x+mu)
    return realtrace(C)
end

function loss_combined(U, shift1)
    C = similar(U[1])
    mul!(C, U[1], U[2])
    base = realtrace(C)
    U2_p1 = Shifted_Lattice(U[2], shift1)
    mul!(C, U[1], U2_p1)
    return base + realtrace(C)
end

function loss_shifted_adjoint(U, shift1)
    C = similar(U[1])
    U2_p1 = Shifted_Lattice(U[2], shift1)
    mul!(C, U[1], U2_p1')   # U_mu(x) * U_nu(x+mu)†
    return realtrace(C)
end

function loss_plain_adjoint(U)
    C = similar(U[1])
    mul!(C, U[1], U[2]')    # U_mu(x) * U_nu(x)†
    return realtrace(C)
end

function loss_plain_left_adjoint(U)
    C = similar(U[1])
    mul!(C, U[1]', U[2])    # U_mu(x)† * U_nu(x)
    return realtrace(C)
end

function loss_plain_both_adjoint(U)
    C = similar(U[1])
    mul!(C, U[1]', U[2]')   # U_mu(x)† * U_nu(x)†
    return realtrace(C)
end

function loss_plaquette(U, shift1, shift2)
    C = similar(U[1])
    D = similar(U[1])
    E = similar(U[1])

    U2_p1 = Shifted_Lattice(U[2], shift1)  # U_nu(x+mu)
    U1_p2 = Shifted_Lattice(U[1], shift2)  # U_mu(x+nu)

    mul!(C, U[1], U2_p1)    # U_mu(x) * U_nu(x+mu)
    mul!(D, C, U1_p2')      # ... * U_mu(x+nu)†
    mul!(E, D, U[2]')       # ... * U_nu(x)†
    return realtrace(E)
end

function run_case(label, f, U, dU, indices_mid, indices_halo)
    println("=== ", label, " ===")

    clear_matrix!.(dU)
    dUn_mid = Wiltinger_numerical_derivative(f, indices_mid, U)
    Wiltinger_derivative!(f, U, dU)

    println("AD grad (U1) at mid indices:")
    display(dU[1].A[:, :, indices_mid...])
    println("Numerical grad (U1) at mid indices:")
    display(dUn_mid[1])

    println("AD grad (U2) at mid indices:")
    display(dU[2].A[:, :, indices_mid...])
    println("Numerical grad (U2) at mid indices:")
    display(dUn_mid[2])

    clear_matrix!.(dU)
    dUn_halo = Wiltinger_numerical_derivative(f, indices_halo, U)
    Wiltinger_derivative!(f, U, dU)

    println("AD grad (U1) at halo indices:")
    display(dU[1].A[:, :, indices_halo...])
    println("Numerical grad (U1) at halo indices:")
    display(dUn_halo[1])

    println("AD grad (U2) at halo indices:")
    display(dU[2].A[:, :, indices_halo...])
    println("Numerical grad (U2) at halo indices:")
    display(dUn_halo[2])
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

    indices_mid = (3, 3, 3, 3)
    indices_halo = (5, 2, 2, 2)

    loss_plain_f(Uin) = loss_plain(Uin)
    loss_shifted_f(Uin) = loss_shifted(Uin, shift1)

    loss_combined_f(Uin) = loss_combined(Uin, shift1)
    loss_shifted_adj_f(Uin) = loss_shifted_adjoint(Uin, shift1)
    loss_plain_adj_f(Uin) = loss_plain_adjoint(Uin)
    loss_plain_left_adj_f(Uin) = loss_plain_left_adjoint(Uin)
    loss_plain_both_adj_f(Uin) = loss_plain_both_adjoint(Uin)
    loss_plaquette_f(Uin) = loss_plaquette(Uin, shift1, (0, 1, 0, 0))

    run_case("plain mul! (U1 * U2)", loss_plain_f, U, dU, indices_mid, indices_halo)
    run_case("shifted mul! (U1 * shifted(U2))", loss_shifted_f, U, dU, indices_mid, indices_halo)
    run_case("combined (plain + shifted)", loss_combined_f, U, dU, indices_mid, indices_halo)
    run_case("shifted adjoint mul! (U1 * shifted(U2)')", loss_shifted_adj_f, U, dU, indices_mid, indices_halo)
    run_case("plain adjoint mul! (U1 * U2')", loss_plain_adj_f, U, dU, indices_mid, indices_halo)
    run_case("plain adjoint mul! (U1' * U2)", loss_plain_left_adj_f, U, dU, indices_mid, indices_halo)
    run_case("plain adjoint mul! (U1' * U2')", loss_plain_both_adj_f, U, dU, indices_mid, indices_halo)
    run_case("plaquette (U1 * U2_p1 * U1_p2' * U2')", loss_plaquette_f, U, dU, indices_mid, indices_halo)

    MPI.Finalize()
end

main()
