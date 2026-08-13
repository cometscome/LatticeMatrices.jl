using LatticeMatrices
using LinearAlgebra
using MPI
using Test
import JACC

JACC.@init_backend

initialized_here = !MPI.Initialized()
initialized_here && MPI.Init()

try
    include("halo_epoch.jl")
    halo_epoch_tests()
    MPI.Barrier(MPI.COMM_WORLD)
finally
    initialized_here && MPI.Finalize()
end
