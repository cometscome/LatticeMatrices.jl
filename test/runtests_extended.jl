# The full suite is intentionally opt-in. It includes expensive finite-
# difference AD checks, complete HISQ pipelines, and production-size legacy
# regression cases that are unsuitable for the normal GitHub Actions matrix.
ENV["LATTICEMATRICES_EXTENDED_TESTS"] = "true"
include("runtests.jl")
