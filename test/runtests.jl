using LatticeMatrices
using Test
using MPI
import JACC
using LinearAlgebra
using InteractiveUtils
JACC.@init_backend
using MPI, JACC, StaticArrays

function normalizetest()
    Random.seed!(1234)
    for dim = 1:4
        for NC = 2:4
            @testset "NC = $NC" begin
                normalizetest(NC, dim)
            end
        end
    end
end

function normalizetest(NC, dim)
    NX = 8
    NY = 8
    NZ = 8
    NT = 8
    if dim == 1
        gsize = (NX,)
    elseif dim == 2
        gsize = (NX, NY)
    elseif dim == 3
        gsize = (NX, NY, NZ)
    elseif dim == 4
        gsize = (NX, NY, NZ, NT)
    else
        error("dim should be smaller than 5.")
    end
    #gsize = (NX, NY)
    nw = 1


    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    if length(ARGS) == 0
        n1 = nprocs ÷ 2
        if n1 == 0
            n1 = 1
        end
        PEs = (n1, nprocs ÷ n1, 1, 1)
    else
        PEs = Tuple(parse.(Int64, ARGS))
    end
    M1 = LatticeMatrix(NC, NC, dim, gsize, PEs; nw)
    comm = M1.cart

    A2 = rand(ComplexF64, NC, NC, gsize...)
    M2 = LatticeMatrix(A2, dim, PEs; nw)

    ix, iy, iz, it = 1, 8, 1, 1
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]
    ixp = ifelse(ixp < 1, ixp + NX, ixp)
    iyp = ifelse(iyp < 1, iyp + NY, iyp)
    izp = ifelse(izp < 1, izp + NZ, izp)
    itp = ifelse(itp < 1, itp + NT, itp)
    ixp = ifelse(ixp > NX, ixp - NX, ixp)
    iyp = ifelse(iyp > NY, iyp - NY, iyp)
    izp = ifelse(izp > NZ, izp - NZ, izp)
    itp = ifelse(itp > NT, itp - NT, itp)

    normalize_matrix!(M2)
    display(M2.A[:, :, nw+ix, nw+iy, nw+iz, nw+it] * M2.A[:, :, nw+ix, nw+iy, nw+iz, nw+it]')

end

@testset "MPILattice.jl" begin
    MPI.Init()
    # Write your tests here.
    #latticetest4D()
    normalizetest()
end
