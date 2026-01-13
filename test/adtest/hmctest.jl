using LatticeMatrices
using MPI
using LinearAlgebra
using Enzyme
import JACC
JACC.@init_backend

function calc_action(U, β, temp)
    U1 = U[1]
    U2 = U[2]
    U3 = U[3]
    U4 = U[4]
    C = temp[1]

    D = temp[2]

    dim = 4
    shift_1 = ntuple(i -> ifelse(i == 1, 1, 0), dim)
    shift_2 = ntuple(i -> ifelse(i == 2, 1, 0), dim)
    shift_3 = ntuple(i -> ifelse(i == 3, 1, 0), dim)
    shift_4 = ntuple(i -> ifelse(i == 4, 1, 0), dim)

    U2_p = Shifted_Lattice(U2, shift_1)
    U1_p = Shifted_Lattice(U1, shift_2)
    mul!(C, U1, U2_p)
    mul!(D, C, U1_p')
    mul!(C, D, U2')
    return realtrace(C) * β
end

function main()
    MPI.Init()
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