# External staggered reference

`bridgepp_staggered_reference.cpp` generates the fixed numerical fingerprints
used by `test/staggered_dirac.jl`.  It is deliberately not linked into the
normal Julia test suite: build it against a separately downloaded, unmodified
Bridge++ 2.1.x release.

For a Bridge++ tree built in `/path/to/bridge/build` with its default GNU/MPI
configuration, run from the LatticeMatrices.jl directory:

```bash
BRIDGE_BUILD=/path/to/bridge/build
mpic++ -fopenmp -std=gnu++11 -O2 \
  -DNDEBUG -DPC_GNU -DUSE_MPI -DUSE_IMP -DUSE_GROUP_SU3 \
  -DUSE_OPENMP -DUSE_STD_COMPLEX -DLIB_CPP11 -DUSE_FACTORY \
  -I"$BRIDGE_BUILD/include/bridge" \
  -I"$BRIDGE_BUILD/include/bridge/lib" \
  test/reference/bridgepp_staggered_reference.cpp \
  -L"$BRIDGE_BUILD" -lbridge -lm \
  -o /tmp/bridgepp_staggered_reference
/tmp/bridgepp_staggered_reference
```

The program constructs deterministic `4x2x2x2`, `NC=3` gauge and one-spinor
fields, applies Bridge++ `Fopr_Staggered` in `D` and `Ddag` modes with mass
`0.17` and boundary condition `(1,1,1,-1)`, then prints sums, index-weighted
sums, and norms.  This keeps the Julia test independent of
LatticeDiracOperators.jl and makes normalization, matrix layout, staggered
phases, and temporal boundary signs part of the external comparison.

## External SIMULATeQCD HISQ reference

`simulateqcd/hisq_level1_reference.cpp` generates the fixed level-1 Fat7
fingerprints used by `test/hisq_smearing.jl`. It is built against a separately
checked-out, unmodified SIMULATeQCD tree through the small overlay CMake
project in the same directory.

For an H100 and the SIMULATeQCD source at the path shown below, configure in a
fresh build directory:

```bash
cmake -S test/reference/simulateqcd \
  -B /tmp/latticematrices-simulateqcd-reference \
  -DSIMULATEQCD_SOURCE_DIR=/path/to/SIMULATeQCD \
  -DARCHITECTURE=90 \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/12.4/bin/nvcc \
  -DCUDAToolkit_ROOT=/opt/cuda/12.4 \
  -DCMAKE_CXX_COMPILER=/opt/ompi-cuda/bin/mpicxx \
  -DMPI_INCLUDE_PATH=/opt/ompi-cuda/include
cmake --build /tmp/latticematrices-simulateqcd-reference \
  --target simulateqcd_hisq_level1_reference --parallel 4
CUDA_VISIBLE_DEVICES=<GPU-UUID> \
  /tmp/latticematrices-simulateqcd-reference/reference/\
simulateqcd_hisq_level1_reference
```

The program constructs deterministic `4x4x4x4`, `NC=3` thin links directly
in both codes. It calls SIMULATeQCD `HisqSmearing::SmearLvl1` and prints, for
each direction, sums, explicit Julia-layout-weighted sums, and norms. Neither
SIMULATeQCD's Git LFS test configurations nor its built-in reference values
are involved.
