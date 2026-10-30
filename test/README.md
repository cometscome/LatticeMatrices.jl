# Test suites

The tests are split by cost so that normal GitHub Actions runs remain useful
and predictable while the full validation suite remains available before a
release.

## Fast suite (run by GitHub Actions)

The normal package test does not load Enzyme and uses small lattices for the
legacy matrix-algebra and Wilson checks:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

This suite covers indexing, random streams, halo epochs, `nw=0`, direct
shifts, matrix exponentials, CG and its pre-v1 compatibility API, Wilson,
Wilson--clover, staggered, HISQ and domain-wall stencils, plus one complete
HISQ thin-link smoke test on the minimum `3^4` local lattice required by
`nw=3`.

GitHub Actions additionally runs two-rank halo and README tests. Julia 1.11
and 1.12 run the small Enzyme suite in a separate environment, and Julia 1.11
runs the two-rank Enzyme halo-gradient regression.

## Transport defaults and buffer allocation

`transport_defaults.jl` checks environment/keyword precedence and the absence
of packed MPI buffers in undecomposed directions. It also verifies all halo
faces and corners after exchange. It is included in the normal package test
and in the existing two-rank `mpi_halo.jl` CI entry point.

To run this regression alone in a prepared test environment:

```sh
julia --startup-file=no --project=. test/transport_defaults.jl
```

With MPI available in the active environment:

```sh
LATTICEMATRICES_TEST_MPI=true julia --startup-file=no --project=. -e 'using MPI; run(`$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --startup-file=no --project=. test/transport_defaults.jl`)'
```

These tests are contained in this repository and do not depend on the
LatticeDiracOperators checkout or a workspace validation runner.

## Extended suite (not run by GitHub Actions)

The extended suite adds all finite-difference AD checks, complete HISQ
smearing and end-to-end AD validation, domain-wall AD, every legacy
`NC`/dimension combination on `16^D` lattices, and the `32^4` Wilson
integration/performance regression.

Create an isolated environment once:

```bash
julia --startup-file=no -e '
using Pkg
Pkg.activate("test/extended-env")
Pkg.develop(path=pwd())
Pkg.add(["Enzyme", "JACC", "MPI", "StaticArrays"])
'
```

Then run:

```bash
julia --project=test/extended-env --startup-file=no test/runtests_extended.jl
```

The accelerator and multi-GPU suites are also opt-in; see
[`multigpu/README.md`](multigpu/README.md).
