using Enzyme
using LatticeMatrices
using LinearAlgebra
using Test
import JACC

JACC.@init_backend

const LATTICEMATRICES_TEST_MPI = lowercase(get(
    ENV, "LATTICEMATRICES_TEST_MPI", "true")) in
    ("1", "true", "yes", "on")

@assert Base.get_extension(
    LatticeMatrices, :LatticeMatricesEnzymeExt) !== nothing

initialized_here = false
if LATTICEMATRICES_TEST_MPI
    @eval using MPI
    initialized_here = !MPI.Initialized()
    initialized_here && MPI.Init()
    @assert Base.get_extension(
        LatticeMatrices, :LatticeMatricesMPIExt) !== nothing
else
    @assert Base.find_package("MPI") === nothing
    @assert Base.get_extension(
        LatticeMatrices, :LatticeMatricesMPIExt) === nothing
end

include("communication_helpers.jl")

try
    include("enzyme.jl")
    include("wilson_dirac_ad.jl")

    @testset "LatticeMatrices CI AD smoke" begin
        enzymetests()
        wilson_dirac_ad_tests()
    end
    LATTICEMATRICES_TEST_MPI && MPI.Barrier(MPI.COMM_WORLD)
finally
    LATTICEMATRICES_TEST_MPI && initialized_here && MPI.Finalize()
end
