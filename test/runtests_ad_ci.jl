using Enzyme
using LatticeMatrices
using LinearAlgebra
using MPI
using Test
import JACC

JACC.@init_backend

@assert Base.get_extension(
    LatticeMatrices, :LatticeMatricesEnzymeExt) !== nothing

initialized_here = !MPI.Initialized()
initialized_here && MPI.Init()

try
    include("enzyme.jl")
    include("wilson_dirac_ad.jl")

    @testset "LatticeMatrices CI AD smoke" begin
        enzymetests()
        wilson_dirac_ad_tests()
    end
    MPI.Barrier(MPI.COMM_WORLD)
finally
    initialized_here && MPI.Finalize()
end
