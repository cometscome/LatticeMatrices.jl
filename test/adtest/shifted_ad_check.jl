using LatticeMatrices
using MPI
using LinearAlgebra
using Enzyme
import JACC
using InteractiveUtils
using Enzyme.EnzymeRules: shadow_type

JACC.@init_backend

const shift_L = LatticeMatrices.shift_L

function loss_plain(U, temp)
    C = temp[1]
    mul!(C, U[1], U[2])
    return realtrace(C)
end

function loss_shifted(U, shift1, temp)
    C = temp[1]
    U2_p1 = shift_L(U[2], shift1)  # U_nu(x+mu)
    mul!(C, U[1], U2_p1)   # U_mu(x) * U_nu(x+mu)
    return realtrace(C)
end

function loss_shifted_trace(U, shift1, temp)
    C = temp[1]
    U2_p1 = shift_L(U[2], shift1)
    substitute!(C, U2_p1)
    return realtrace(C)
end

function loss_shifted_U2U1(U, shift2, temp)
    C = temp[1]
    U1_p2 = shift_L(U[1], shift2)  # U_mu(x+nu)
    mul!(C, U[2], U1_p2)   # U_nu(x) * U_mu(x+nu)
    return realtrace(C)
end

function loss_shifted_sum(U, shift1, temp)
    C = temp[1]
    D = temp[2]
    U2_p1 = shift_L(U[2], shift1)
    U3_p1 = shift_L(U[3], shift1)
    mul!(C, U[1], U2_p1)
    mul!(D, U[1], U3_p1)
    add_matrix!(C, D)
    return realtrace(C)
end

function loss_shifted_two_terms_simple(U, shift1, shift2, temp)
    C = temp[1]
    D = temp[2]
    U2_p1 = shift_L(U[2], shift1)
    U3_p2 = shift_L(U[3], shift2)
    mul!(C, U[1], U2_p1)
    mul!(D, U[1], U3_p2)
    return realtrace(C) + realtrace(D)
end

const shift_μs = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]

function loss_shifted_two_terms_loop(U, temp)
    C = temp[1]
    S = 0.0
    dim = length(U[1].PN)
    for μ = 1:2
        #shift_μ = shift_μs[μ]
        #shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        #shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        #U2_pμ = shift_L(U[2], shift_μ)
        U2_pμ = shift_L(U[2], shift_μs[μ])
        mul!(C, U[1], U2_pμ)
        S += realtrace(C)
    end
    return S
end

function loss_shifted_two_terms_reuse(U, shift1, shift2, temp)
    C = temp[1]
    U2_p1 = shift_L(U[2], shift1)
    mul!(C, U[1], U2_p1)
    s = realtrace(C)
    U3_p2 = shift_L(U[3], shift2)
    mul!(C, U[1], U3_p2)
    s += realtrace(C)
    return s
end

function calc_action_shift_LA(U, temp)
    C = temp[1]
    shift2 = (0, 1, 0, 0)
    U1_p2 = shift_L(U[1], shift2)
    mul!(C, U[2], U1_p2)
    return realtrace(C)
end

function calc_action_three_mul_LA(U, temp)
    C = temp[1]
    D = temp[2]
    shift1 = (1, 0, 0, 0)
    shift2 = (0, 1, 0, 0)
    U2_p1 = shift_L(U[2], shift1)
    U1_p2 = shift_L(U[1], shift2)
    mul!(C, U[1], U2_p1)
    mul!(D, C, U1_p2')
    return realtrace(D)
end



function calc_action(U, β, NC, temp)
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

function loss_combined(U, shift1, temp)
    C = temp[1]
    D = temp[2]
    mul!(C, U[1], U[2])
    base = realtrace(C)
    U2_p1 = shift_L(U[2], shift1)
    mul!(D, U[1], U2_p1)
    return base + realtrace(D)
end

function loss_shifted_adjoint(U, shift1, temp)
    C = temp[1]
    U2_p1 = shift_L(U[2], shift1)
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

    U2_p1 = shift_L(U[2], shift1)  # U_nu(x+mu)
    U1_p2 = shift_L(U[1], shift2)  # U_mu(x+nu)

    mul!(C, U[1], U2_p1)    # U_mu(x) * U_nu(x+mu)
    mul!(D, C, U1_p2')      # ... * U_mu(x+nu)†
    mul!(E, D, U[2]')       # ... * U_nu(x)†
    return realtrace(E)
    #mul!(C, D, U[2]')       # ... * U_nu(x)†
    #return realtrace(C)
end

function loss_plaquette_sum_no_last(U, temp)
    C = temp[1]
    D = temp[2]
    E = temp[3]
    ndir = length(U)
    dim = length(U[1].PN)
    S = 0.0
    for μ = 1:ndir
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        for ν = μ+1:ndir
            shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)
            Uμ_pν = shift_L(U[μ], shift_ν)
            Uν_pμ = shift_L(U[ν], shift_μ)
            mul!(C, U[μ], Uν_pμ)
            mul!(D, C, Uμ_pν')
            #mul!(E, D, U[ν]')
            S += realtrace(D)
        end
    end
    return S
end

function loss_plaquette_mu1_nu2_no_last(U, temp)
    C = temp[1]
    D = temp[2]
    dim = length(U[1].PN)
    shift_μ = ntuple(i -> ifelse(i == 1, 1, 0), dim)
    shift_ν = ntuple(i -> ifelse(i == 2, 1, 0), dim)
    Uμ_pν = shift_L(U[1], shift_ν)
    Uν_pμ = shift_L(U[2], shift_μ)
    mul!(C, U[1], Uν_pμ)
    mul!(D, C, Uμ_pν')
    return realtrace(D)
end

function loss_plaquette_mu1_nu2_firstmul(U, temp)
    C = temp[1]
    dim = length(U[1].PN)
    shift_μ = ntuple(i -> ifelse(i == 1, 1, 0), dim)
    shift_ν = ntuple(i -> ifelse(i == 2, 1, 0), dim)
    Uν_pμ = shift_L(U[2], shift_μ)
    mul!(C, U[1], Uν_pμ)
    return realtrace(C)
end

function loss_plaquette_mu1to3_nu_no_last(U, temp)
    C = temp[1]
    D = temp[2]
    E = temp[3]
    clear_matrix!(E)
    dim = length(U[1].PN)
    S = 0.0
    for μ = 1:3
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        for ν = μ+1:3
            shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)
            Uμ_pν = shift_L(U[μ], shift_ν)
            Uν_pμ = shift_L(U[ν], shift_μ)
            mul!(C, U[μ], Uν_pμ)
            mul!(D, C, Uμ_pν')
            add_matrix!(E, D)
        end
    end
    S += realtrace(D)
    return S
end

function loss_plaquette_mu1to3_firstmul(U, temp)
    C = temp[1]
    dim = length(U[1].PN)
    S = 0.0
    for μ = 1:3
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        for ν = μ+1:3
            Uν_pμ = shift_L(U[ν], shift_μ)
            mul!(C, U[μ], Uν_pμ)
            S += realtrace(C)
        end
    end
    return S
end

function loss_shift_shift_adj(U, shift1, shift2, temp)
    C = temp[1]
    D = temp[2]
    E = temp[3]

    U2_p1 = shift_L(U[2], shift1)  # U_nu(x+mu)
    U1_p2 = shift_L(U[1], shift2)  # U_mu(x+nu)

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

function _report_analytic(label, dU, expected, indices; tol=1e-6)
    diffs = dU.A[:, :, indices...] .- expected
    maxerr = maximum(abs, diffs)
    if maxerr > tol
        println(label, " max |diff| = ", maxerr)
        println("AD:")
        display(dU.A[:, :, indices...])
        println("Analytical:")
        display(expected)
        println("Diff:")
        display(diffs)
    else
        println(label, " max |diff| = ", maxerr)
    end
    return nothing
end

function _expected_shifted_trace_site(U2, scale=0.5)
    NC = U2.NC1
    expected = zeros(eltype(U2.A), NC, NC)
    @inbounds for i in 1:NC
        expected[i, i] = scale
    end
    return expected
end

function _expected_shifted_mul_site(U1, U2, indices, shift, scale=0.5)
    indices_p = shiftindices(indices, shift)
    indices_m = shiftindices(indices, ntuple(i -> -shift[i], length(shift)))
    u2_site = U2.A[:, :, indices_p...]
    u1_site = U1.A[:, :, indices_m...]
    expected_u1 = scale .* permutedims(u2_site, (2, 1))
    expected_u2 = scale .* permutedims(u1_site, (2, 1))
    return expected_u1, expected_u2
end

function _expected_shifted_mul_sum_site(U1, U2, U3, indices, shift, scale=0.5)
    expected_u1_u2, expected_u2 = _expected_shifted_mul_site(U1, U2, indices, shift, scale)
    expected_u1_u3, expected_u3 = _expected_shifted_mul_site(U1, U3, indices, shift, scale)
    expected_u1 = expected_u1_u2 .+ expected_u1_u3
    return expected_u1, expected_u2, expected_u3
end

function _expected_plaquette_sum_site(U, indices, scale=0.5; mu_max=length(U), nu_max=length(U))
    ndir = length(U)
    dim = length(indices)
    expected = [zeros(eltype(U[1].A), U[1].NC1, U[1].NC2) for _ in 1:ndir]
    for μ = 1:min(mu_max, ndir)
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        shift_mμ = ntuple(i -> -shift_μ[i], dim)
        for ν = μ+1:min(nu_max, ndir)
            shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)
            shift_mν = ntuple(i -> -shift_ν[i], dim)

            # Uμ at x (A role).
            x = indices
            A = U[μ].A[:, :, x...]
            B = U[ν].A[:, :, shiftindices(x, shift_μ)...]
            C = U[μ].A[:, :, shiftindices(x, shift_ν)...]
            expected[μ] .+= scale .* transpose(B * adjoint(C))

            # Uμ at x+ν (C role).
            x_mν = shiftindices(indices, shift_mν)
            A = U[μ].A[:, :, x_mν...]
            B = U[ν].A[:, :, shiftindices(x_mν, shift_μ)...]
            expected[μ] .+= scale .* conj(A * B)

            # Uν at x+μ (B role).
            x_mμ = shiftindices(indices, shift_mμ)
            A = U[μ].A[:, :, x_mμ...]
            C = U[μ].A[:, :, shiftindices(x_mμ, shift_ν)...]
            expected[ν] .+= scale .* transpose(adjoint(C) * A)
        end
    end
    return expected
end

function run_case_all(label, f, U, dU, temp, dtemp, indices_mid, indices_halo; tol=1e-5)
    println("=== ", label, " ===")

    clear_matrix!.(dU)
    clear_matrix!.(temp)
    clear_matrix!.(dtemp)
    f_num(Uin) = f(Uin, temp)
    Wiltinger_derivative!(f, U, dU; temp=temp, dtemp=dtemp)

    clear_matrix!.(temp)
    dUn_mid = Wiltinger_numerical_derivative(f_num, indices_mid, U)
    for k in 1:length(U)
        _report_diff("U$k mid", dU[k], dUn_mid[k], indices_mid; tol)
    end

    clear_matrix!.(temp)
    dUn_halo = Wiltinger_numerical_derivative(f_num, indices_halo, U)
    for k in 1:length(U)
        _report_diff("U$k halo", dU[k], dUn_halo[k], indices_halo; tol)
    end
    return nothing
end

const run_case = run_case_all

function report_shifted_trace_analytic(U, dU, indices_mid, indices_halo; tol=1e-6)
    # Verify analytical gradients for realtrace(shifted(U2)).
    expected = _expected_shifted_trace_site(U[2])
    zeros_expected = zeros(eltype(U[1].A), size(expected))
    _report_analytic("U1 mid (analytic)", dU[1], zeros_expected, indices_mid; tol)
    _report_analytic("U2 mid (analytic)", dU[2], expected, indices_mid; tol)
    _report_analytic("U1 halo (analytic)", dU[1], zeros_expected, indices_halo; tol)
    _report_analytic("U2 halo (analytic)", dU[2], expected, indices_halo; tol)
    return nothing
end

function report_shifted_mul_analytic(U, dU, shift, indices_mid, indices_halo; tol=1e-6)
    # Verify analytical gradients for realtrace(U1 * shifted(U2)).
    expected_u1_mid, expected_u2_mid =
        _expected_shifted_mul_site(U[1], U[2], indices_mid, shift)
    expected_u1_halo, expected_u2_halo =
        _expected_shifted_mul_site(U[1], U[2], indices_halo, shift)
    _report_analytic("U1 mid (analytic)", dU[1], expected_u1_mid, indices_mid; tol)
    _report_analytic("U2 mid (analytic)", dU[2], expected_u2_mid, indices_mid; tol)
    _report_analytic("U1 halo (analytic)", dU[1], expected_u1_halo, indices_halo; tol)
    _report_analytic("U2 halo (analytic)", dU[2], expected_u2_halo, indices_halo; tol)
    return nothing
end

function report_shifted_mul_sum_analytic(U, dU, shift, indices_mid, indices_halo; tol=1e-6)
    # Verify analytical gradients for realtrace(U1 * shifted(U2) + U1 * shifted(U3)).
    expected_u1_mid, expected_u2_mid, expected_u3_mid =
        _expected_shifted_mul_sum_site(U[1], U[2], U[3], indices_mid, shift)
    expected_u1_halo, expected_u2_halo, expected_u3_halo =
        _expected_shifted_mul_sum_site(U[1], U[2], U[3], indices_halo, shift)
    zeros_expected = zeros(eltype(U[4].A), size(expected_u1_mid))
    _report_analytic("U1 mid (analytic)", dU[1], expected_u1_mid, indices_mid; tol)
    _report_analytic("U2 mid (analytic)", dU[2], expected_u2_mid, indices_mid; tol)
    _report_analytic("U3 mid (analytic)", dU[3], expected_u3_mid, indices_mid; tol)
    _report_analytic("U4 mid (analytic)", dU[4], zeros_expected, indices_mid; tol)
    _report_analytic("U1 halo (analytic)", dU[1], expected_u1_halo, indices_halo; tol)
    _report_analytic("U2 halo (analytic)", dU[2], expected_u2_halo, indices_halo; tol)
    _report_analytic("U3 halo (analytic)", dU[3], expected_u3_halo, indices_halo; tol)
    _report_analytic("U4 halo (analytic)", dU[4], zeros_expected, indices_halo; tol)
    return nothing
end

function report_plaquette_sum_analytic(U, dU, indices_mid, indices_halo; tol=1e-6)
    # Verify analytical gradients for plaquette sum over mu<nu without trailing Uν'.
    expected_mid = _expected_plaquette_sum_site(U, indices_mid)
    expected_halo = _expected_plaquette_sum_site(U, indices_halo)
    for k in 1:length(U)
        _report_analytic("U$k mid (analytic)", dU[k], expected_mid[k], indices_mid; tol)
        _report_analytic("U$k halo (analytic)", dU[k], expected_halo[k], indices_halo; tol)
    end
    return nothing
end

function report_plaquette_sum_analytic_range(U, dU, indices_mid, indices_halo; tol=1e-6, mu_max, nu_max)
    # Verify analytical gradients for plaquette sum over restricted mu/nu range.
    expected_mid = _expected_plaquette_sum_site(U, indices_mid; mu_max=mu_max, nu_max=nu_max)
    expected_halo = _expected_plaquette_sum_site(U, indices_halo; mu_max=mu_max, nu_max=nu_max)
    for k in 1:length(U)
        _report_analytic("U$k mid (analytic)", dU[k], expected_mid[k], indices_mid; tol)
        _report_analytic("U$k halo (analytic)", dU[k], expected_halo[k], indices_halo; tol)
    end
    return nothing
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
    U3 = deepcopy(U1)
    U4 = deepcopy(U1)
    U = [U1, U2, U3, U4]

    shift1 = (1, 0, 0, 0)
    dU = [similar(U1), similar(U1), similar(U1), similar(U1)]
    clear_matrix!.(dU)
    temp = [similar(U1), similar(U1), similar(U1)]
    dtemp = [similar(U1), similar(U1), similar(U1)]

    indices_mid = (3, 3, 3, 3)
    indices_halo = (5, 2, 2, 2)
    indices_halo = (3, 3, 4, 2)

    loss_plain_f(Uin, t) = loss_plain(Uin, t)
    loss_shifted_f(Uin, t) = loss_shifted(Uin, shift1, t)
    loss_shifted_trace_f(Uin, t) = loss_shifted_trace(Uin, shift1, t)
    loss_shifted_sum_f(Uin, t) = loss_shifted_sum(Uin, shift1, t)
    loss_shifted_two_terms_simple_f(Uin, t) =
        loss_shifted_two_terms_simple(Uin, shift1, (0, 1, 0, 0), t)
    loss_shifted_two_terms_reuse_f(Uin, t) =
        loss_shifted_two_terms_reuse(Uin, shift1, (0, 1, 0, 0), t)
    loss_shifted_two_terms_loop_f(Uin, t) = loss_shifted_two_terms_loop(Uin, t)
    loss_shifted_u2u1_f(Uin, t) = loss_shifted_U2U1(Uin, (0, 1, 0, 0), t)
    loss_plaquette_sum_f(Uin, t) = loss_plaquette_sum_no_last(Uin, t)
    loss_plaquette_mu1_nu2_f(Uin, t) = loss_plaquette_mu1_nu2_no_last(Uin, t)
    loss_plaquette_mu1_nu2_firstmul_f(Uin, t) = loss_plaquette_mu1_nu2_firstmul(Uin, t)
    loss_plaquette_mu1to3_f(Uin, t) = loss_plaquette_mu1to3_nu_no_last(Uin, t)
    loss_plaquette_mu1to3_firstmul_f(Uin, t) = loss_plaquette_mu1to3_firstmul(Uin, t)
    loss_action_shift_la_f(Uin, t) = calc_action_shift_LA(Uin, t)
    loss_action_three_mul_la_f(Uin, t) = calc_action_three_mul_LA(Uin, t)
    loss_action_f(Uin, t) = calc_action(Uin, 0.5, NC, t)

    loss_combined_f(Uin, t) = loss_combined(Uin, shift1, t)
    loss_shifted_adj_f(Uin, t) = loss_shifted_adjoint(Uin, shift1, t)
    loss_plain_adj_f(Uin, t) = loss_plain_adjoint(Uin, t)
    loss_plain_left_adj_f(Uin, t) = loss_plain_left_adjoint(Uin, t)
    loss_plain_both_adj_f(Uin, t) = loss_plain_both_adjoint(Uin, t)
    loss_plaquette_f(Uin, t) = loss_plaquette(Uin, shift1, (0, 1, 0, 0), t)
    loss_shift_shift_adj_f(Uin, t) = loss_shift_shift_adj(Uin, shift1, (0, 1, 0, 0), t)

    #debug_shifted_call(U, shift1, (0, 1, 0, 0), temp)
    #println("\n=== @code_lowered Shifted_Lattice ===")
    #@code_lowered Shifted_Lattice(U[2], shift1)

    #@code_warntype loss_shifted_two_terms_loop_f(U, temp)
    #return

    # Small sanity check: sum of two shifted terms with separate buffers.
    run_case_all("shifted mul sum (two terms, separate buffers)", loss_shifted_two_terms_simple_f, U, dU, temp, dtemp, indices_mid, indices_halo)
    # Small sanity check: sum of two shifted terms with reused buffer.
    run_case_all("shifted mul sum (two terms, reused buffer)", loss_shifted_two_terms_reuse_f, U, dU, temp, dtemp, indices_mid, indices_halo)
    # Minimal looped-shift case: sum of two terms with dynamic shift tuple.
    run_case_all("shifted mul sum (two terms, loop shift)", loss_shifted_two_terms_loop_f, U, dU, temp, dtemp, indices_mid, indices_halo)
    # Check AD vs numerical gradients for single term mu=1, nu=2 (first mul only).
    run_case_all("plaquette term (mu=1, nu=2, first mul only)", loss_plaquette_mu1_nu2_firstmul_f, U, dU, temp, dtemp, indices_mid, indices_halo)
    # Check AD vs numerical gradients for mu=1..3, nu=mu+1..3 (first mul only).
    run_case_all("plaquette sum (mu=1..3, nu=mu+1..3, first mul only)", loss_plaquette_mu1to3_firstmul_f, U, dU, temp, dtemp, indices_mid, indices_halo)
    # Check AD vs numerical gradients for single term mu=1, nu=2.
    return
    run_case_all("plaquette term (mu=1, nu=2)", loss_plaquette_mu1_nu2_f, U, dU, temp, dtemp, indices_mid, indices_halo)
    # Check AD vs analytical gradients for single term mu=1, nu=2.
    report_plaquette_sum_analytic_range(U, dU, indices_mid, indices_halo; mu_max=1, nu_max=2)
    # Check AD vs numerical gradients for mu=1..3, nu=mu+1..3.
    run_case_all("plaquette sum (mu=1..3, nu=mu+1..3)", loss_plaquette_mu1to3_f, U, dU, temp, dtemp, indices_mid, indices_halo)
    # Check AD vs analytical gradients for mu=1..3, nu=mu+1..3.
    report_plaquette_sum_analytic_range(U, dU, indices_mid, indices_halo; mu_max=3, nu_max=3)
    # Check AD vs numerical gradients for plaquette sum over mu<nu without trailing Uν'.
    run_case_all("plaquette sum (mu<nu, Uμ*Uν_pμ*Uμ_pν')", loss_plaquette_sum_f, U, dU, temp, dtemp, indices_mid, indices_halo)
    # Check AD vs analytical gradients for plaquette sum over mu<nu without trailing Uν'.
    report_plaquette_sum_analytic(U, dU, indices_mid, indices_halo)

    run_case("shifted trace (realtrace(shifted(U2)))", loss_shifted_trace_f, U, dU, temp, dtemp, indices_mid, indices_halo)
    # Check AD vs analytical gradients for realtrace(shifted(U2)).
    report_shifted_trace_analytic(U, dU, indices_mid, indices_halo)
    # Check AD vs numerical gradients for realtrace(U1 * shifted(U2)).
    run_case("shifted mul! (U1 * shifted(U2))", loss_shifted_f, U, dU, temp, dtemp, indices_mid, indices_halo)
    # Check AD vs analytical gradients for realtrace(U1 * shifted(U2)).
    report_shifted_mul_analytic(U, dU, shift1, indices_mid, indices_halo)
    # Check AD vs numerical gradients for sum of two shifted products.
    run_case("shifted mul sum (U1*shifted(U2) + U1*shifted(U3))", loss_shifted_sum_f, U, dU, temp, dtemp, indices_mid, indices_halo)
    # Check AD vs analytical gradients for sum of two shifted products.
    report_shifted_mul_sum_analytic(U, dU, shift1, indices_mid, indices_halo)


    MPI.Finalize()
end

function debug_shifted_call(U, shift1, shift2, temp)
    println("=== @which Shifted_Lattice ===")
    println(@which Shifted_Lattice(U[2], shift1))
    println(@which Shifted_Lattice(U[2], shift1, Val(4)))

    println("\n=== @which shift_U (if used) ===")
    try
        println(@which shift_U(U[2], shift1))
    catch err
        println("shift_U not found: ", err)
    end

    println("\n=== @code_lowered loss_plaquette ===")
    @code_lowered loss_plaquette(U, shift1, shift2, temp)
    return nothing
end


main()
