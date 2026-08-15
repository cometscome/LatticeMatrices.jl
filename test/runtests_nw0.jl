using LatticeMatrices
using Test
using MPI
import JACC
using LinearAlgebra

JACC.@init_backend
include("nw0.jl")

MPI.Init()
try
    nw0test()
finally
    MPI.Finalize()
end
