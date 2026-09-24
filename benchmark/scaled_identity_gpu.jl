# Opt-in wrapper: prevents accidentally reporting CPU numbers as GPU timings.
using CUDA, LatticeMatrices
import JACC
JACC.@init_backend

CUDA.functional() || error("A functional CUDA device is required")
CUDA.allowscalar(false)
probe = LatticeMatrix(1, 1, 1, (2,), (1,); nw=1)
probe.A isa CUDA.CuArray || error("Enable the JACC cuda backend before benchmarking")
println((device=CUDA.name(CUDA.device()), julia=VERSION,
    cuda=Base.pkgversion(CUDA), jacc=Base.pkgversion(JACC),
    timing="synchronized host wall time per public operation"))

include("scaled_identity.jl")
CUDA.synchronize()
