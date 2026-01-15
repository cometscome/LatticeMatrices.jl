using LatticeMatrices
using MPI
using LinearAlgebra
using Enzyme
import JACC
JACC.@init_backend
using PreallocatedArrays

function calc_action(U, β, NC, temp)
    dim = 4
    C = temp[1]
    D = temp[2]
    S = 0.0

    for μ = 1:dim
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        for ν = μ:dim
            if ν == μ
                continue
            end
            shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)
            Uμ_pν = Shifted_Lattice(U[μ], shift_ν)
            Uν_pμ = Shifted_Lattice(U[ν], shift_μ)
            mul!(C, U[μ], Uν_pμ)
            mul!(D, C, Uμ_pν')
            mul!(C, D, U[ν]')
            S += realtrace(C)

            mul!(C, U[ν], Uμ_pν)
            mul!(D, C, Uν_pμ')
            mul!(C, D, U[μ]')
            S += realtrace(C)
        end
    end

    return -S * β / 2NC
end

function calc_staple!(U, X, μ, temp)
    dim = 4
    clear_matrix!(X)

    C = temp[1]
    D = temp[2]

    shift_μ = ntuple(i -> (i == μ ? 1 : 0), dim)

    for ν = 1:dim
        ν == μ && continue

        shift_ν = ntuple(i -> (i == ν ? 1 : 0), dim)
        shift_νm = ntuple(i -> (i == ν ? -1 : 0), dim)

        # -------- term 1: Uν(x+μ) Uμ†(x+ν) Uν†(x) --------
        Uν_pμ = Shifted_Lattice(U[ν], shift_μ)   # Uν(x+μ)
        Uμ_pν = Shifted_Lattice(U[μ], shift_ν)   # Uμ(x+ν)
        mul!(C, Uν_pμ, Uμ_pν')                   # Uν(x+μ) * Uμ†(x+ν)
        mul!(D, C, U[ν]')                        # ... * Uν†(x)
        add_matrix!(X, D)

        # -------- term 2: Uν†(x+μ-ν) Uμ†(x-ν) Uν(x-ν) --------
        shift_μmν = ntuple(i -> (i == μ ? 1 : (i == ν ? -1 : 0)), dim)

        Uν_pμmν = Shifted_Lattice(U[ν], shift_μmν)  # Uν(x+μ-ν)
        Uμ_mν = Shifted_Lattice(U[μ], shift_νm)   # Uμ(x-ν)
        Uν_mν = Shifted_Lattice(U[ν], shift_νm)   # Uν(x-ν)

        mul!(C, Uν_pμmν', Uμ_mν')                # Uν†(x+μ-ν) * Uμ†(x-ν)
        mul!(D, C, Uν_mν)                        # ... * Uν(x-ν)
        add_matrix!(X, D)
    end
end


function calc_staple2!(U, X, μ, temp)
    dim = 4
    clear_matrix!(X)

    C = temp[1]
    D = temp[2]
    shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
    shift_μm = ntuple(i -> ifelse(i == μ, -1, 0), dim)

    for ν = μ+1:dim
        shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)

        Uμ_pν = Shifted_Lattice(U[μ], shift_ν)
        Uν_pμ = Shifted_Lattice(U[ν], shift_μ)
        mul!(C, U[ν], Uμ_pν)
        mul!(D, C, Uν_pμ')
        add_matrix!(X, D)

        shift_νm = ntuple(i -> ifelse(i == ν, -1, 0), dim)
        Uμ_mν = Shifted_Lattice(U[μ], shift_νm)
        Uν_mν = Shifted_Lattice(U[ν], shift_νm)
        mul!(C, Uν_mν', Uμ_mν)
        shift_νmpμ = ntuple(i -> ifelse(i == ν, -1, ifelse(i == μ, 1, 0)), dim)
        Uν_mνpμ = Shifted_Lattice(U[ν], shift_νmpμ)
        mul!(D, C, Uν_mνpμ)
        add_matrix!(X, D)
    end
end


function calc_staple3!(U, X, μ, temp)
    dim = 4
    clear_matrix!(X)

    C = temp[1]
    D = temp[2]

    shift_μ = ntuple(i -> (i == μ ? 1 : 0), dim)

    for ν = 1:dim
        ν == μ && continue

        shift_ν = ntuple(i -> (i == ν ? 1 : 0), dim)
        shift_νm = ntuple(i -> (i == ν ? -1 : 0), dim)

        # -------------------------
        # forward:  Uν(x+μ) * Uμ†(x+ν) * Uν†(x)
        # -------------------------
        Uν_pμ = Shifted_Lattice(U[ν], shift_μ)   # Uν(x+μ)
        Uμ_pν = Shifted_Lattice(U[μ], shift_ν)   # Uμ(x+ν)

        mul!(C, Uν_pμ, Uμ_pν')   # C = Uν(x+μ) * Uμ†(x+ν)
        mul!(D, C, U[ν]')        # D = C * Uν†(x)
        add_matrix!(X, D)

        # -------------------------
        # backward: Uν†(x+μ-ν) * Uμ†(x-ν) * Uν(x-ν)
        # -------------------------
        shift_μmν = ntuple(i -> (i == μ ? 1 : (i == ν ? -1 : 0)), dim)  # +μ -ν

        Uν_pμmν = Shifted_Lattice(U[ν], shift_μmν)  # Uν(x+μ-ν)
        Uμ_mν = Shifted_Lattice(U[μ], shift_νm)   # Uμ(x-ν)
        Uν_mν = Shifted_Lattice(U[ν], shift_νm)   # Uν(x-ν)

        mul!(C, Uν_pμmν', Uμ_mν')  # C = Uν†(x+μ-ν) * Uμ†(x-ν)
        mul!(D, C, Uν_mν)          # D = C * Uν(x-ν)
        add_matrix!(X, D)
    end

    return nothing
end

function calc_grad!(U, dU, β, NC, temp)
    dim = 4
    X = temp[3]
    Y = temp[4]
    set_halo!.(U)

    for μ = 1:dim
        calc_staple!(U, X, μ, temp)

        # V = staple * U†   （← こっちを基準にする）
        mul!(Y, X, U[μ]')
        #mul!(Y, U[μ], X')

        clear_matrix!(dU[μ])
        # force = -(β/(2Nc)) * TA(V)  みたいな形にする
        traceless_antihermitian_add!(dU[μ], -β / (2NC), Y)
    end
end



function calc_grad_old!(U, dU, β, NC, temp)
    dim = 4
    X = temp[3]
    Y = temp[4]
    set_halo!.(U)
    for μ = 1:dim
        calc_staple!(U, X, μ, temp)
        mul!(Y, U[μ], X')
        #display(X.A[:, :, 2, 2, 2, 2])
        #mul!(dU[μ], -β / NC, X')
        #traceless_antihermitian_add!(dU[μ], -β / NC, Y)
        traceless_antihermitian_add!(dU[μ], β / NC, Y)
    end
end

function calc_grad_AD!(U, dU, β, NC, temp, dtemp)
    Wiltinger_derivative!(
        calc_action,
        U,
        dU, β, NC;
        temp,
        dtemp
    )
    dim = 4
    Y = temp[4]
    for μ = 1:dim
        mul!(Y, U[μ], dU[μ])
        traceless_antihermitian!(dU[μ], -2.0, Y)
        #display(dU[μ].A[:, :, 2, 2, 2, 2])
    end

end


function copy_U!(dest, src)
    for μ in eachindex(src)
        substitute!(dest[μ], src[μ])
    end
    return nothing
end

function init_momenta!(P)
    for μ in eachindex(P)
        P[μ].A .= randn(eltype(P[μ].A), size(P[μ].A)...)
        traceless_antihermitian!(P[μ])
    end
    return nothing
end

function kinetic_energy(P, temp)
    C = temp[1]
    K = 0.0
    for μ in eachindex(P)
        mul!(C, P[μ], P[μ]')
        K += realtrace(C)
    end
    return 0.5 * K
end

function hmc_step!(Uin, U, β, NC, temp, dtemp;
    nsteps::Int=10,
    ϵ::Float64=0.1)
    dim = length(U)

    copy_U!(Uin, U)

    P = [similar(U[μ]) for μ in 1:dim]
    init_momenta!(P)

    dU = [similar(U[μ]) for μ in 1:dim]

    expP = similar(U[1])
    tmpU = similar(U[1])
    Sgold = calc_action(U, β, NC, temp)
    Spold = kinetic_energy(P, temp)
    H0 = Sgold + Spold#calc_action(U, β, NC, temp) + kinetic_energy(P, temp)
    println("Sgold $Sgold Spold $Spold")

    clear_matrix!.(dU)
    calc_grad!(U, dU, β, NC, temp)
    #calc_grad_AD!(U, dU, nodiff(β), nodiff(NC), temp, dtemp)
    for μ in 1:dim
        add_matrix!(P[μ], dU[μ], -ϵ / 2)
        #add_matrix!(P[μ], dU[μ], ϵ / 2)
        traceless_antihermitian!(P[μ])
    end

    for step = 1:nsteps
        for μ in 1:dim
            #traceless_antihermitian!(P[μ])
            #=
            expt_TA!(expP, P[μ], ϵ)
            set_halo!(expP)
            p = P[μ].A[:, :, 2, 2, 2, 2]
            E = expP.A[:, :, 2, 2, 2, 2]
            println("diff")
            display(exp(p * ϵ) .- E)
            =#
            expt!(expP, P[μ], ϵ)
            #set_halo!(expP)
            #p = P[μ].A[:, :, 2, 2, 2, 2]
            #E = expP.A[:, :, 2, 2, 2, 2]
            #println("diff2")
            #display(exp(p * ϵ) .- E)
            #display(exp(p * ϵ) * exp(p * ϵ)')
            #println("unitarity err = ", norm(E' * E - I), "  det = ", det(E))
            # mul!(tmpU, expP, expP')
            #println("check")
            #display(tmpU.A[:, :, 2, 2, 2, 2])
            mul!(tmpU, expP, U[μ])
            substitute!(U[μ], tmpU)
        end
        set_halo!.(U)
        #display(U[1].A[:, :, 2, 2, 2, 2])

        clear_matrix!.(dU)
        #calc_grad_AD!(U, dU, nodiff(β), nodiff(NC), temp, dtemp)
        calc_grad!(U, dU, β, NC, temp)
        for μ in 1:dim
            add_matrix!(P[μ], dU[μ], step == nsteps ? -ϵ / 2 : -ϵ)
            #add_matrix!(P[μ], dU[μ], step == nsteps ? ϵ / 2 : ϵ)
            traceless_antihermitian!(P[μ])
        end

        #println("step $step")
        Sgnew = calc_action(U, β, NC, temp)
        Spnew = kinetic_energy(P, temp)
        #println("Sg $Sgnew Sp $Spnew")
    end

    Sgnew = calc_action(U, β, NC, temp)
    Spnew = kinetic_energy(P, temp)
    println("Sgnew $Sgnew Spnew $Spnew")
    H1 = Sgnew + Spnew
    #H1 = calc_action(U, β, NC, temp) + kinetic_energy(P, temp)
    println("H1 $H1 H0 $H0")
    accept = rand() < exp(-(H1 - H0))
    if !accept
        copy_U!(U, Uin)
        set_halo!.(U)
    end

    return accept, H0, H1
end

function TA_antiherm(M::AbstractMatrix)
    N = size(M)[1]
    Ah = (M - M') / 2             # anti-Hermitian part
    for i = 1:N
        Ah[i, i] -= tr(Ah) / N
    end
    #Ah .-= (tr(Ah) / N) * I       # traceless
    return Ah
end


# --- helper: site の 3x3 を取り出す（コピーして安全に扱う） ---
function _site3x3(M::LatticeMatrix, idx::NTuple{4,Int})
    return Matrix(@view M.A[:, :, idx...])
end

# --- helper: 3x3 の traceless anti-hermitian を作る（site 行列用） ---
function _TA_antiherm_site(M::AbstractMatrix{<:Complex})
    N = size(M, 1)
    Ah = (M - M') / 2
    trAh = tr(Ah) / N
    for i = 1:N
        Ah[i, i] -= trAh
    end
    return Ah
end

# --- helper: staple X と numerical dUn の対応を色々な候補で比較する ---
function _compare_X_vs_dUn!(U, β, NC, temps; indices::NTuple{4,Int}=(2, 2, 2, 2))
    dim = 4
    Xbuf = temps[3]

    # 数値 Wirtinger（あなたの関数をそのまま）
    dUn = Wiltinger_numerical_derivative(calc_action, indices, U; params=(β, NC, temps))

    println("=== Compare staple X vs numerical dUn at site ", indices, " ===")
    for μ = 1:dim
        clear_matrix!(Xbuf)
        calc_staple!(U, Xbuf, μ, temps)

        Xs = _site3x3(Xbuf, indices)   # staple at site
        dS = dUn[μ]                    # 3x3 (from your numerical routine)

        # staple と dUn の関係は規約で転置/共役/随伴が混ざり得るので全列挙で見る
        # 係数はとりあえず (2NC/β) を掛けた形を並べ、norm を比較
        scale = (2 * NC) / β

        cand = Dict{String,Matrix{ComplexF64}}()
        cand["dUn"] = dS
        cand["dUn'"] = dS'
        cand["transpose"] = transpose(dS)
        cand["conj"] = conj.(dS)
        cand["conjT"] = transpose(conj.(dS))
        cand["adjoint"] = adjoint(dS)

        # staple 側も X, X', transpose, conj など見ておく
        Xcand = Dict{String,Matrix{ComplexF64}}()
        Xcand["X"] = Xs
        Xcand["X'"] = Xs'
        Xcand["Xt"] = transpose(Xs)
        Xcand["conjX"] = conj.(Xs)
        Xcand["conjXt"] = transpose(conj.(Xs))
        Xcand["adjX"] = adjoint(Xs)

        # 最小の差を探す（符号 ± も見る）
        best = (Inf, "", "", +1)
        for (xn, Xm) in Xcand
            for (dn, Dm) in cand
                for sgn in (-1, +1)
                    diff = norm(Xm - sgn * (-scale) * Dm)  # ← -β/(2Nc) を仮定した形
                    if diff < best[1]
                        best = (diff, xn, dn, sgn)
                    end
                end
            end
        end

        println("μ = $μ")
        println("  best match:  Xform=$(best[2])  vs  dUnform=$(best[3])  sign=$(best[4])")
        println("  || Xform - sign*(-2Nc/β)*dUnform || = ", best[1])

        # 参考に、最良候補のときの scale を反映した行列を表示（大きすぎるならコメントアウト）
        # Xm = Xcand[best[2]]
        # Dm = cand[best[3]]
        # println("  check matrix (Xform):"); display(Xm)
        # println("  check matrix (mapped dUn):"); display(best[4] * (-scale) * Dm)
    end
    println("=== end compare ===")
end

# --- helper: staple force と numerical force を “同じ定義” に揃えて比較 ---
function _compare_force!(U, β, NC, temps; indices::NTuple{4,Int}=(2, 2, 2, 2))
    dim = 4

    # staple から作った force（あなたの calc_grad! の出力）
    dU_staple = [similar(U[μ]) for μ in 1:dim]
    clear_matrix!.(dU_staple)
    calc_grad!(U, dU_staple, β, NC, temps)

    # numerical
    dUn = Wiltinger_numerical_derivative(calc_action, indices, U; params=(β, NC, temps))

    println("=== Compare FORCE (su(N)) at site ", indices, " ===")
    for μ = 1:dim
        Usite = _site3x3(U[μ], indices)
        Fst = _site3x3(dU_staple[μ], indices)

        # ここが問題の核心：numerical dUn をどう force に落とすべきか規約が未確定
        # まず、あなたが今やっている変換をそのまま計算して比較する
        Fnum_user = _TA_antiherm_site(Usite * (2 * dUn[μ]'))

        println("μ = $μ")
        println("  ||Fst - Fnum_user|| = ", norm(Fst - Fnum_user))
        println("  ||Fst + Fnum_user|| = ", norm(Fst + Fnum_user))  # 符号反転も一応見る
    end
    println("=== end force compare ===")
end



function run_hmc(;
    ntraj::Int=5,
    nsteps::Int=10,
    ϵ::Float64=0.1,
    do_grad_check::Bool=false)
    NC = 3
    dim = 4
    NX = 4
    NY = NX
    NZ = NX
    NT = NX
    gsize = (NX, NY, NZ, NT)
    #gsize = (NX, NY)
    nw = 1
    NG = NC

    PEs = (1, 1, 1, 1)
    UA = zeros(ComplexF64, NC, NG, gsize...)

    for jc = 1:NC
        ic = jc
        UA[ic, jc, :, :, :, :] .= 1
    end
    UA = randn(ComplexF64, NC, NG, gsize...)

    U1 = LatticeMatrix(UA, dim, PEs; nw)
    set_halo!(U1)
    display(realtrace(U1))
    U2 = deepcopy(U1)
    clear_matrix!(U2)
    #traceless_antihermitian!(U1)
    display(U1.A[:, :, 2, 2, 2, 2])
    traceless_antihermitian_add!(U2, 1, U1)
    expt!(U1, U2, 1)



    U2 = deepcopy(U1)
    U3 = deepcopy(U1)
    U4 = deepcopy(U1)
    U = [U1, U2, U3, U4]
    set_halo!.(U)

    #clear_matrix!(U2)
    #traceless_antihermitian_add!(U2, 1, U1)
    #display(U2.A[:, :, 2, 2, 2, 2])
    #return

    dUA = zeros(ComplexF64, NC, NG, gsize...)
    dU1 = LatticeMatrix(dUA, dim, PEs; nw)
    dU2 = deepcopy(dU1)
    dU3 = deepcopy(dU1)
    dU4 = deepcopy(dU1)
    dU = [dU1, dU2, dU3, dU4]

    β = 6.0

    tempvec = PreallocatedArray(U1; num=20, haslabel=false)

    temps, indices_temps = get_block(tempvec, 4)
    dtemps, indices_dtemps = get_block(tempvec, 4)

    #=
    if do_grad_check
        S = calc_action(U, β, NC, temps)
        println("action: ", S)

        indices = (2, 2, 2, 2)
        dUn = Wiltinger_numerical_derivative(calc_action, indices, U; params=(β, NC, temps))
        # Convert Wirtinger ∂S/∂U to the staple-force form (U * V with V=staple).
        for μ = 1:dim
            Y = U[μ].A[:, :, indices...] * (2 * dUn[μ]')
            dUn[μ] = TA_antiherm(Y)
        end
        clear_matrix!.(dU)


        calc_grad!(U, dU, β, NC, temps)
        for μ = 1:dim
            println("μ = ", μ)
            println("grad from staple ")
            display(dU[μ].A[:, :, indices...])
            println("numerical ")
            display(dUn[μ])
        end
        unused!(tempvec, indices_temps)
        unused!(tempvec, indices_dtemps)
        return
    end
    =#
    # ---------------------------
    # run_hmc 内 do_grad_check 部分（ここから置換）
    # ---------------------------
    if do_grad_check
        S = calc_action(U, β, NC, temps)
        println("action: ", S)

        indices = (2, 2, 2, 2)

        # numerical Wirtinger
        dUn = Wiltinger_numerical_derivative(calc_action, indices, U; params=(β, NC, temps))

        # staple force
        clear_matrix!.(dU)
        calc_grad!(U, dU, β, NC, temps)

        # compare at one site
        for μ = 1:dim
            println("μ = ", μ)

            # --- numerical: reconstruct X† from dUn using the proven transpose convention ---
            # dUn ≈ -(β/(2Nc)) * X^T  =>  X† ≈ -(2Nc/β) * transpose(adjoint(dUn))
            #Xdag_num = -(2NC / β) * transpose(adjoint(dUn[μ]))
            Xdag_num = -(2NC / β) * dUn[μ]'

            # build su(N) force in the SAME way as your staple-side mapping does:
            # Y = U * X†  then TA_antiherm
            Usite = U[μ].A[:, :, indices...]
            Ynum = Usite * Xdag_num
            Fnum = TA_antiherm(Ynum)

            println("grad from staple (site)")
            display(dU[μ].A[:, :, indices...])

            println("numerical (fixed transpose convention)")
            display(Fnum)

            println("||diff|| = ", norm(dU[μ].A[:, :, indices...] - Fnum))
            println("||sum || = ", norm(dU[μ].A[:, :, indices...] + Fnum))  # sign flip check
        end

        unused!(tempvec, indices_temps)
        unused!(tempvec, indices_dtemps)
        return
    end



    Uin = [deepcopy(U[μ]) for μ in 1:dim]
    accept_count = 0
    for traj in 1:ntraj
        accepted, H0, H1 = hmc_step!(Uin, U, β, NC, temps, dtemps; nsteps=2, ϵ=ϵ)
        accept_count += accepted ? 1 : 0
        println("traj=", traj, " ΔH=", H1 - H0, " accepted=", accepted)
    end
    println("acceptance: ", accept_count, "/", ntraj)

    unused!(tempvec, indices_temps)
    unused!(tempvec, indices_dtemps)

    return

end

function main()
    MPI.Init()
    run_hmc(; do_grad_check=true)
    return

    NC = 3
    dim = 4
    NX = 16
    NY = NX
    NZ = NX
    NT = NX
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    myrank = MPI.Comm_rank(MPI.COMM_WORLD)
    gsize = (NX, NY, NZ, NT)
    #gsize = (NX, NY)
    nw = 1
    NG = NC

    PEs = (1, 1, 1, 1)
    UA = zeros(ComplexF64, NC, NG, gsize...)
    for jc = 1:NC
        ic = jc
        UA[ic, jc, :, :, :, :] .= 1
    end

    U1 = LatticeMatrix(UA, dim, PEs; nw)
    U2 = deepcopy(U1)
    U3 = deepcopy(U1)
    U4 = deepcopy(U1)
    U = [U1, U2, U3, U4]
    Uin = deepcopy(U)

    dUA = zeros(ComplexF64, NC, NG, gsize...)

    dU1 = LatticeMatrix(UA, dim, PEs; nw)
    dU2 = deepcopy(dU1)
    dU3 = deepcopy(dU1)
    dU4 = deepcopy(dU1)
    dU = [dU1, dU2, dU3, dU4]

    temp1 = deepcopy(U1)
    temp2 = deepcopy(U2)
    temp = [temp1, temp2]
    β = 6.0

    S = calc_action(U, β, temp)
end
main()
