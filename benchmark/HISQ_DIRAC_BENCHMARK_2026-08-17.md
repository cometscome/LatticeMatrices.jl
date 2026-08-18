# HISQ Dirac benchmark: LatticeMatrices v1.1.0 and SIMULATeQCD

Date: 2026-08-17

This benchmark times the **HISQ Dirac stencil**, not HISQ smearing.  Smearing
is performed once before the warm-up and is excluded from every number below.
One reported application means one full-lattice, massless HISQ Dslash together
with making its output halo ready for use as the next Krylov vector.

## Compared operation

Both implementations use double precision, `epsilon_N=-0.083`, mass zero,
temporal antiperiodic boundary conditions, and the same deterministic thin
links and unit spinor source.  The timed stencil is

```text
1/2 * [X_mu(x) psi(x+mu) - X_mu(x-mu)' psi(x-mu)]
- (1+epsilon_N)/48
    * [L_mu(x) psi(x+3mu) - L_mu(x-3mu)' psi(x-3mu)].
```

LatticeMatrices applies both parities in one `mul!`.  SIMULATeQCD's
`HisqDSlash` produces one parity, so one full application is the sum of its
Even-to-Odd and Odd-to-Even calls.  Both calls request output-halo updates.
The input and output fields are swapped after every application.

The two codes retain their native link formats.  LatticeMatrices stores full
`3 x 3` complex matrices.  SIMULATeQCD uses `R18` corrected-fat links and
`U3R14` Naik links.  Thus this is an application-level implementation
comparison, not an isolated arithmetic-kernel comparison.

Each result is the maximum wall time over MPI ranks, divided by the number of
applications.  The tables give the minimum and median of repeated samples.
JIT compilation, allocation, link construction, and the explicit pre-timing
GC are excluded.  GC is disabled only inside the LatticeMatrices steady-state
timing interval.  GPU runs use nine samples; CPU runs use five longer samples.

## Software and hardware

- LatticeMatrices v1.1.0 working tree and JACC backend selected through
  preferences; the HISQ source contains no CUDA/Threads dispatch.
- Unmodified SIMULATeQCD commit `767a1b1`, built by the external reference
  target with CUDA 12.4, `sm_90`, GPU-aware OpenMPI 5.0.7, P2P, and
  communication streams enabled.
- Two NVIDIA H100 NVL GPUs:
  `GPU-2644154d-7268-af42-6631-59e1f3c6e7f3` and
  `GPU-2378bb6e-0b0c-e257-61f3-ff7993a641d7`.
- The H100s are connected by NVLink.  UUIDs, rather than CUDA ordinals, were
  used because CUDA's default device order differed from `nvidia-smi` order.
- CPU: 32 physical cores in two 16-core NUMA nodes, one hardware thread per
  core.

The local OpenMPI links UCX 1.20.  CPU/JACC runs set
`UCX_ERROR_SIGNALS=ILL,BUS,FPE` so UCX does not claim Julia's task-stack
`SIGSEGV`.  This does not alter the kernels.

## H100 comparison

Times are milliseconds per complete full-lattice Dslash.  `LM/SQCD` is the
ratio of medians; smaller is better.

### One H100

| global lattice | SQCD min | SQCD median | LM min | LM median | LM/SQCD |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `8^4` | 0.01429 | 0.01431 | 0.11056 | 0.11104 | 7.76x |
| `16^4` | 0.04999 | 0.05011 | 0.18414 | 0.18496 | 3.69x |
| `24^4` | 0.21887 | 0.24036 | 0.62361 | 0.64086 | 2.67x |
| `32^4` | 0.68527 | 0.70998 | 1.98241 | 2.56497 | 3.61x |

The `32^4` LatticeMatrices samples had scheduler variation; its minimum ratio
was 2.89x.  The smaller rows were stable after excluding the pre-timing GC.

### Two H100s, two MPI ranks: final communication path

| global lattice | SQCD min | SQCD median | LM min | LM median | LM/SQCD |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `16^4` | 0.21245 | 0.43459 | 0.38746 | 0.39470 | 0.91x |
| `24^4` | 0.56208 | 0.84935 | 0.94615 | 0.95412 | 1.12x |
| `32^4` | 0.97413 | 1.05370 | 2.77835 | 2.91394 | 2.77x |

Both programs used distinct GPUs.  The measured LatticeMatrices mapping was
rank 0 to UUID `2644...` and rank 1 to UUID `2378...`.

The final one-to-two-GPU median speedups for LatticeMatrices are 0.47x, 0.67x,
and 0.88x at `16^4`, `24^4`, and `32^4`; the corresponding SQCD values are
0.12x, 0.28x, and 0.67x.  These fixed global volumes remain below the
strong-scaling crossover, but the two-rank penalty is now small at `32^4`.

Before the communication revision, the LatticeMatrices `2x1x1x1` medians
were `1.91479`, `6.00296`, and `12.64365` ms.  The final values are therefore
4.85x, 6.29x, and 4.34x faster, respectively.

#### Communication path A/B

GPU-aware support alone did not predict the fastest path.  A two-rank buffer
exchange on the same NVLink-connected H100 pair gave:

| payload | CUDA-aware device median | pageable host median | pinned host median |
| ---: | ---: | ---: | ---: |
| 32 KiB | 0.04228 | 0.03356 | 0.03000 |
| 128 KiB | 0.32702 | 0.16000 | 0.04290 |
| 512 KiB | 0.39607 | 0.15734 | 0.09532 |
| 2 MiB | 1.64983 | 0.89560 | 0.27662 |

Times are milliseconds per simultaneous rank-to-rank exchange and include
device/host copies for the staged paths.  Forcing OpenMPI `ob1/smcuda`
improved direct device communication relative to the default UCX choice, but
it still did not beat pinned staging in the application benchmark.

The implemented default therefore keeps accelerator communication staged but
registers the host Arrays for decomposed directions through the CUDA package
extension.  CPU Arrays continue to go directly to MPI.  The same pinning is
applied lazily to the `Alltoallv!` buffers used for long shifts.  This is
transparent to callers and introduces no CUDA/Threads branch in the lattice
or HISQ source.

The second change is a staircase face section.  At direction `d`, dimensions
already exchanged include their halos so corners propagate correctly, while
later dimensions include only their core.  For a `16^4`, `nw=3`, x-split
spinor, one face drops from `3 * 22^3` to `3 * 16^3` lattice positions, a
61.5% reduction.  Existing communication-buffer capacity and the public
lattice memory layout are unchanged; only the active packed prefix is sent.
The per-direction global `MPI.Barrier` was also removed and receives are
posted before sends.

#### Process-grid direction, `16^4`

| process grid | split direction | LM min | LM median |
| ---: | ---: | ---: | ---: |
| `2x1x1x1` | x | 0.38746 | 0.39470 |
| `1x2x1x1` | y | 0.43145 | 0.43760 |
| `1x1x2x1` | z | 0.50088 | 0.52793 |
| `1x1x1x2` | t | 0.62136 | 0.62332 |

The x split is preferred after staircase packing because it is exchanged
first and therefore has the smallest active cross-section.  For reference,
SIMULATeQCD with the t split gave medians `0.39949`, `0.85811`, and `0.87859`
ms at `16^4`, `24^4`, and `32^4`; its x-split values are the SQCD column in
the main table.

SIMULATeQCD requires a decomposed local extent greater than twice the spinor
halo.  With the minimal HISQ spinor halo of three, an `8^4` lattice split over
two ranks has local extent four and is therefore rejected; the two-GPU table
starts at `16^4`.

## CPU JACC Threads backend

The global lattice is `16^4`, with one MPI rank.  `JULIA_EXCLUSIVE=1` was used
for this pure-Threads sweep.

| Julia threads | min ms | median ms | median speedup |
| ---: | ---: | ---: | ---: |
| 1 | 37.485 | 39.957 | 1.00x |
| 2 | 11.018 | 15.517 | 2.58x |
| 4 | 7.158 | 16.491 | 2.42x |
| 8 | 17.196 | 18.970 | 2.11x |
| 16 | 19.130 | 20.149 | 1.98x |

The JACC/Polyester thread scaling is not monotone.  Two threads had the best
median and four threads the best isolated sample; worker scheduling overhead
dominates beyond that point for this stencil and lattice size.

## CPU MPI process-grid sweep (pre-communication-revision baseline)

The global lattice remains `16^4`, with one Julia thread per rank.  These
historical CPU MPI numbers were recorded before barrier removal and staircase
packing; they are retained to document the original sweep and should not be
read as post-revision results.

| ranks | process grid | local lattice | min ms | median ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `1x1x1x1` | `16x16x16x16` | 37.485 | 39.957 |
| 2 | `2x1x1x1` | `8x16x16x16` | 23.509 | 24.051 |
| 2 | `1x1x1x2` | `16x16x16x8` | 21.261 | 22.676 |
| 4 | `4x1x1x1` | `4x16x16x16` | 18.604 | 20.965 |
| 4 | `2x2x1x1` | `8x8x16x16` | 13.607 | 18.169 |
| 8 | `4x2x1x1` | `4x8x16x16` | 24.763 | 26.900 |
| 8 | `2x2x2x1` | `8x8x8x16` | 36.339 | 50.647 |
| 16 | `4x4x1x1` | `4x4x16x16` | 61.305 | 82.719 |
| 16 | `4x2x2x1` | `4x8x8x16` | 90.408 | 103.480 |
| 16 | `2x2x2x2` | `8x8x8x8` | 102.967 | 114.809 |

Four ranks with `2x2x1x1` was the best pure-MPI median.  At 8 and 16 ranks,
decomposing fewer dimensions was substantially faster than the most balanced
local lattice.  For example, `4x2x1x1` was 1.88x faster than `2x2x2x1` at
eight ranks.  Current message/synchronization count is therefore more
important than geometric balance for this implementation.

## CPU hybrid sweep (pre-communication-revision baseline)

These runs keep the total requested CPU cores at 16.  MPI `PE=` mapping binds
the appropriate number of cores to each process.

| MPI ranks x threads | process grid | min ms | median ms |
| ---: | ---: | ---: | ---: |
| `1x16` | `1x1x1x1` | 19.130 | 20.149 |
| `2x8` | `2x1x1x1` | 37.696 | 40.337 |
| `4x4` | `2x2x1x1` | 36.542 | 38.976 |
| `8x2` | `2x2x2x1` | 29.827 | 30.075 |
| `16x1` | `2x2x2x2` | 102.967 | 114.809 |

No hybrid point beat the pure two-thread or four-rank results on `16^4`.

## Correctness and reproducibility

After the benchmark, the HISQ dense SIMULATeQCD-convention oracle, adjoint and
epsilon-Hermiticity identities, gauge covariance, `nw=0` versus `nw=3`, free
`D' D` spectrum, validation errors, and generic-color path all passed with one
rank and again with two ranks.

After the communication revision, the two-H100 integration suite also passed
the full padded-halo reference (including corners), long shifts, distributed
matrix kernels, staggered Dslash, domain-wall operators, and CG.  A two-rank
CPU run passed halo-epoch, explicit-shift, and Wilson dirty-halo regressions.
The two-rank Enzyme suite passed halo-epoch gradients, pooled long-shift
gradients, fused SU(3) exponential gradients, and small/degenerate TA fields.

LatticeMatrices driver:

```bash
HISQ_BENCH_GLOBAL=16,16,16,16 \
HISQ_BENCH_GRID=1,1,1,1 \
HISQ_BENCH_ITERS=100 HISQ_BENCH_SAMPLES=9 \
julia --project=. benchmark/hisq_dirac_scaling.jl
```

Two-rank form:

```bash
HISQ_BENCH_GRID=2,1,1,1 \
mpirun --bind-to core -np 2 \
    julia --project=. benchmark/hisq_dirac_scaling.jl
```

The external SIMULATeQCD executable is built without editing its source tree:

```bash
cmake -S test/reference/simulateqcd -B build-sqcd \
    -DSIMULATEQCD_SOURCE_DIR=/path/to/SIMULATeQCD \
    -DARCHITECTURE=90 -DUSE_GPU_AWARE_MPI=ON -DUSE_GPU_P2P=ON \
    -DCOMMS_STREAMS=ON
cmake --build build-sqcd --target simulateqcd_hisq_dirac_benchmark
```

The same `HISQ_BENCH_*` variables configure the SQCD executable.

The communication microbenchmark is:

```bash
mpirun --bind-to core -np 2 \
    julia --project=. benchmark/mpi_gpu_buffer_exchange.jl
```
