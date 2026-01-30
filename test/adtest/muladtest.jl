
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

function run_case_mul(label, f, f_num, U1, dU1, A, temp, dtemp, indices_mid, indices_halo; tol=1e-4)
    println("=== ", label, " ===")

    clear_matrix!(dU1)
    clear_matrix!.(temp)
    clear_matrix!.(dtemp)

    Enzyme_derivative!(
        f,
        U1,
        dU1,
        nodiff(A); temp=temp, dtemp=dtemp)

    clear_matrix!.(temp)
    f_num_1(Uvec) = f_num(Uvec[1], A, temp)
    dUn_mid = Numerical_derivative_Enzyme(f_num_1, indices_mid, [U1])
    _report_diff("U1 mid", dU1, dUn_mid[1], indices_mid; tol)

    indices_halo_core = _halo_to_core_indices(indices_halo, U1.PN, U1.nw)
    clear_matrix!.(temp)
    dUn_halo = Numerical_derivative_Enzyme(f_num_1, indices_halo_core, [U1])
    _report_diff("U1 halo", dU1, dUn_halo[1], indices_halo_core; tol)
end

function run_case_mulB(label, f, f_num, U1, dU1, B, temp, dtemp, indices_mid, indices_halo; tol=1e-4)
    println("=== ", label, " ===")

    clear_matrix!(dU1)
    clear_matrix!.(temp)
    clear_matrix!.(dtemp)

    Enzyme_derivative!(
        f,
        U1,
        dU1,
        nodiff(B); temp=temp, dtemp=dtemp)

    clear_matrix!.(temp)
    f_num_1(Uvec) = f_num(Uvec[1], B, temp)
    dUn_mid = Numerical_derivative_Enzyme(f_num_1, indices_mid, [U1])
    _report_diff("U1 mid", dU1, dUn_mid[1], indices_mid; tol)

    indices_halo_core = _halo_to_core_indices(indices_halo, U1.PN, U1.nw)
    clear_matrix!.(temp)
    dUn_halo = Numerical_derivative_Enzyme(f_num_1, indices_halo_core, [U1])
    _report_diff("U1 halo", dU1, dUn_halo[1], indices_halo_core; tol)
end

function loss_mulA_matrix_test(U1, A, temp)
    C = temp[1]
    mul!(C, A, U1)
    return realtrace(C)
end

function loss_mulB_matrix_test(U1, B, temp)
    C = temp[1]
    mul!(C, U1, B)
    return realtrace(C)
end

function shiftedadd(y, Uμ, x, γμ, shift_p, shift_m, phi1, phi2, κ)
    #U_n[ν](1 - γν) * ψ_{n + ν}
    mul_AshiftB!(phi1, Uμ, x, shift_p)
    mul!(phi2, phi1, transpose(I(4) - γμ))
    add_matrix!(y, phi2, -κ)

    # U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
    mul_shiftAshiftB!(phi1, Uμ', x, shift_m, shift_m)
    mul!(phi2, phi1, transpose(I(4) + γμ))
    add_matrix!(y, phi2, -κ)
end


#ψ_n - κ sum_ν U_n[ν](1 - γν)*ψ_{n+ν} + U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
function apply_wilson!(y, U1, U2, U3, U4, x, params, phitemps)
    U = (U1, U2, U3, U4)

    clear_matrix!(y)
    add_matrix!(y, x, 1)
    γs = (γ1, γ2, γ3, γ4)
    κ = params.κ

    phi1 = phitemps[3]
    phi2 = phitemps[4]
    dim = 4
    for μ = 1:dim
        shift_p = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        shift_m = ntuple(i -> ifelse(i == μ, -1, 0), dim)
        shiftedadd(y, U[μ], x, γs[μ], shift_p, shift_m, phi1, phi2, κ)
    end
end

function shifteddagadd(y, Uμ, x, γμ, shift_p, shift_m, phi1, phi2, κ)
    #U_n[ν](1 - γν) * ψ_{n + ν}
    mul_AshiftB!(phi1, Uμ, x, shift_p)
    mul!(phi2, phi1, transpose(I(4) + γμ))
    add_matrix!(y, phi2, -κ)

    # U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
    mul_shiftAshiftB!(phi1, Uμ', x, shift_m, shift_m)
    mul!(phi2, phi1, transpose(I(4) - γμ))
    add_matrix!(y, phi2, -κ)
end


#ψ_n - κ sum_ν U_n[ν](1 + γν)*ψ_{n+ν} + U_{n-ν}[-ν]^+ (1 - γν)*ψ_{n-ν}
function apply_wilson_dag!(y, U1, U2, U3, U4, x, params, phitemps)
    U = (U1, U2, U3, U4)

    clear_matrix!(y)
    add_matrix!(y, x, 1)
    γs = (γ1, γ2, γ3, γ4)
    κ = params.κ

    phi1 = phitemps[3]
    phi2 = phitemps[4]
    dim = 4
    for μ = 1:dim
        shift_p = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        shift_m = ntuple(i -> ifelse(i == μ, -1, 0), dim)
        shifteddagadd(y, U[μ], x, γs[μ], shift_p, shift_m, phi1, phi2, κ)
    end
end

function loss_wilson(U1, U2, U3, U4, phi, phitemp, params, temp)
    y = phitemp[3]
    x = phi
    #params = nothing
    apply_wilson!(y, U1, U2, U3, U4, x, params, phitemp)
    return real(dot(phi, y))
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

function main()
    MPI.Init()

    NC = 2
    dim = 4
    gsize = (4, 4, 4, 4)
    nw = 1
    PEs = (1, 1, 1, 1)
    NG = 4

    UA = randn(ComplexF64, NC, NC, gsize...)
    U1 = LatticeMatrix(UA, dim, PEs; nw, numtemps=4)
    set_halo!(U1)
    dU1 = similar(U1)
    clear_matrix!(dU1)
    U2 = deepcopy(U1)
    U3 = deepcopy(U1)
    U4 = deepcopy(U1)
    U = [U1, U2, U3, U4]

    phiA = randn(ComplexF64, NC, NG, gsize...)
    phi = LatticeMatrix(phiA, dim, PEs; nw, numtemps=4)
    phinorm = sqrt(dot(phi, phi))
    mul!(phi, 1 / phinorm, phi)
    set_halo!(phi)
    println(dot(phi, phi))

    phiB = randn(ComplexF64, NC, NG, gsize...)
    phi2 = LatticeMatrix(phiB, dim, PEs; nw, numtemps=4)
    set_halo!(phi2)

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


    y = phitemp[4]
    x = phi
    κ = 0.141139
    params = (κ=κ,)
    apply_wilson!(y, U1, U2, U3, U4, x, params, phitemp)
    println(dot(y, y))
    #return

    indices_mid = (3, 3, 3, 3)
    indices_halo = (2, 3, 3, 3)

    A = randn(ComplexF64, NC, NC)

    B = randn(ComplexF64, NG, NG)
    display(I(NG))
    display(γ1)
    B = (I(4) + γ1)

    mul!(phitemp[1], phi, B)
    println(dot(phitemp[1], phitemp[1]))

    set_halo!.(U)

    Dwilson = WilsonDiracOperator4D(U, κ)
    mul!(phitemp[1], Dwilson, phi)
    println("wilson test ", dot(phitemp[1], phitemp[1]))
    mul!(phitemp[1], Dwilson', phi)
    println("wilson dag test ", dot(phitemp[1], phitemp[1]))


    fermitest(U1, U2, U3, U4, phi, phitemp, temp) = loss_wilson(U1, U2, U3, U4, phi, phitemp, params, temp)
    println(fermitest(U1, U2, U3, U4, phi, phitemp, temp))

    apply_wilson!(phitemp[1], U1, U2, U3, U4, phi, params, phitemp)
    println("wilson ", dot(phitemp[1], phitemp[1]))

    apply_wilson_dag!(phitemp[1], U1, U2, U3, U4, phi, params, phitemp)
    println("wilson dag ", dot(phitemp[1], phitemp[1]))

    apply(y, U1, U2, U3, U4, x, params, phitemp, temp) = apply_wilson!(y, U1, U2, U3, U4, x, params, phitemp)
    apply_dag(y, U1, U2, U3, U4, x, params, phitemp, temp) = apply_wilson_dag!(y, U1, U2, U3, U4, x, params, phitemp)
    D = DiracOp(U, apply, apply_dag, params, phi; numtemp=4, numphitemp=4)

    S = pseudofermion_action(D, phi)
    function f(Up)
        #Up = (U1, U2, U3, U4)
        D = DiracOp(Up, apply, apply_dag, params, phi; numtemp=4, numphitemp=4)
        S = pseudofermion_action(D, phi)
        println("calculated")
        return S
    end
    dSFdUn = Numerical_derivative_Enzyme(f, indices_mid, U)
    for μ = 1:4
        display(dSFdUn[μ])
    end

    dSFdUn_halo = Numerical_derivative_Enzyme(f, indices_halo, U)
    for μ = 1:4
        display(dSFdUn_halo[μ])
    end


    mul!(phitemp[1], D, phi)
    println("wilson D test ", dot(phitemp[1], phitemp[1]))


    mul!(phitemp[1], D', phi)
    println("wilson D dag test ", dot(phitemp[1], phitemp[1]))

    clear_matrix!(phitemp[1])
    mul!(phitemp[1], D, phi) #D*phi
    println(dot(phi2, phitemp[1])) # phi2 D*phi

    clear_matrix!(phitemp[2])
    mul!(phitemp[2], D', phi2) #D^+ phi2
    println(dot(phitemp[2], phi))

    DdagD = DdagDOp(D)
    mul!(phitemp[1], DdagD, phi)
    println("wilson DdagD test ", dot(phitemp[1], phitemp[1]))

    clear_matrix!(phitemp[1])
    mul!(phitemp[1], DdagD, phi) #DdagD*phi
    println(dot(phi2, phitemp[1])) # phi2 DdagD*phi

    clear_matrix!(phitemp[2])
    mul!(phitemp[2], DdagD, phi2) #DdagD phi2
    println(dot(phitemp[2], phi))


    dSFdU(dU, D, phi)
    println("halo ", indices_halo)
    for μ = 1:4
        println("mu = $μ ")
        display(dU[μ].A[:, :, indices_halo...])
        println("AD ")
        display(dSFdUn_halo[μ])
    end
    println("mid ", indices_mid)
    for μ = 1:4
        println("mu = $μ ")
        display(dU[μ].A[:, :, indices_mid...])
        println("AD ")
        display(dSFdUn[μ])
    end
    return

    #return



    #return

    set_halo!(phi)
    println("dot ", dot(phi, phi))
    clear_matrix!(phitemp[1])
    solve!(phitemp[1], DdagD, phi; verboselevel=2)
    set_halo!(phitemp[1])
    mul!(phitemp[2], DdagD, phitemp[1])
    println("dot ", dot(phitemp[2], phitemp[2]))
    #return

    return


    run_case_all_phi("fermitest", fermitest, fermitest, U1, U2, U3, U4, dU[1], dU[2], dU[3], dU[4], phi, phitemp, dphitemp, temp, dtemp, indices_mid, indices_halo)

    #=
    Enzyme_derivative!(
        fermitest,
        U1,
        U2,
        U3,
        U4,
        dU[1],
        dU[2],
        dU[3],
        dU[4], nodiff(phi);
        temp,
        dtemp,
        phitemp,
        dphitemp
    )
    ]=#
    #f_num_1(Uvec) = f_num(Uvec[1], A, temp)
    #dUn_mid = Numerical_derivative_Enzyme(f_num_1, indices_mid, U)
    #_report_diff("U1 mid", dU[1], dUn_mid[1], indices_mid; tol)

    #display(dU[1].A[:, :, indices_halo...])


    return
end
main()
