# Scaled-identity H100 validation, 2026-09-24

## Environment and scope

- NVIDIA H100 NVL, driver 590.48.01, Julia 1.11.8, CUDA.jl 5.11.3,
  JACC.jl 1.3.1, one Julia CPU thread.
- Multiplication runs through JACC's CUDA backend (`JACC.parallel_for`),
  not direct `CUDA.@cuda` launches. CUDA.jl is also used by the runners for
  device/storage checks and to disable scalar host fallback.
- Single GPU, SU(3), ComplexF64, `nw=1`; no multi-GPU timing claim.
- GPU 0 was shared with a resident process using 80,552 MiB according to
  nvidia-smi (80,575 MiB total usage). GPU utilization
  immediately before/after the final benchmark was 0%/5%. This was not an exclusive
  GPU reservation. Our CUDA allocator had a 4 GiB hard memory limit.
- Explicit CuArray checks and `CUDA.allowscalar(false)` prevent silently
  benchmarking the CPU backend or scalar host fallback.

## Correctness

`test/scaled_identity_gpu.jl` passes 1,172 checks: four explicit CUDA storage
and backend-dispatch checks, the same 1,144 focused checks used on CPU,
and 24 partial-block checks on a 5×7×3×2 lattice. These cover SU(2)/SU(3)/SU(4),
Float32/Float64, scalar updates, zero/nonzero halos, long shifts, nontrivial
boundary phases, left/right products, adjoints, alpha/beta accumulation,
rectangular adjoints with unequal halos, and shifted-copy aliasing.
Comparisons with the initial scalar kernel include bitwise equality.
The partial-block cases compare shifted adjoints and accumulation with an
independent CPU expression, and reuse shifted views after scalar updates.

## Final v1.2.8 public-operation timings

All times below are median microseconds over 51 warmed, randomized,
interleaved samples per method in one process. They include public dispatch,
launch, and synchronization, not just device-event kernel time. The final
16^4 and 32^4 measurements were run sequentially after all correctness tests.

| Lattice | B shift | Full matrix | Initial scalar | CUDA component-wise | Full/current | Initial/current |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 16^4 | (0,0,0,0) | 44.636 | 44.213 | 22.609 | 1.974 | 1.956 |
| 16^4 | (1,0,0,0) | 44.817 | 44.150 | 22.622 | 1.981 | 1.952 |
| 32^4 | (0,0,0,0) | 571.481 | 858.802 | 159.504 | 3.583 | 5.384 |
| 32^4 | (1,0,0,0) | 562.275 | 852.394 | 158.590 | 3.545 | 5.375 |

Every row has maximum absolute difference `4.440892098500626e-16` from
full matrices, and `0.0` plus bitwise equality with the initial scalar
reference. The earlier slowdown is resolved in these measured cases.
An independent pilot run gave 22.63--22.66 microseconds at 16^4 and
155.55--160.22 microseconds at 32^4 for the component-wise kernel.

Host allocations per call for full/initial/current are 320/1744/1744 bytes
at 16^4 and 336/1760/1744 bytes at 32^4. The optimization does not remove all
public-API host allocations. B component storage is 9x smaller than a full
SU(3) matrix field; halo padding is included in that comparison.

## Public-operation timings before GPU-specific tuning

The benchmark warms each function, then measures 51 samples per method in
randomized order in the same process. Times are median microseconds of
host wall time including public dispatch, launch, and GPU synchronization;
they are not device-event-only kernel timings. Geometry/halos are already
prepared. `Initial scalar` is the frozen pre-linear-index reference.

| Lattice | B shift | Full matrix | Initial scalar | Site-wise scalar | Full/site-wise |
| --- | --- | ---: | ---: | ---: | ---: |
| 16^4 | (0,0,0,0) | 44.461 | 44.600 | 44.005 | 1.010 |
| 16^4 | (1,0,0,0) | 44.597 | 44.211 | 43.936 | 1.015 |
| 32^4 | (0,0,0,0) | 593.337 | 881.522 | 913.942 | 0.649 |
| 32^4 | (1,0,0,0) | 582.463 | 870.081 | 910.372 | 0.640 |

Repeating 16^4 after the 32^4 run gives 43.716/43.948 microseconds for the
site-wise scalar method, versus 44.889/44.004 for full matrices. The small
16^4 differences should be treated as near parity, not a robust speedup.
At 32^4, the site-wise scalar method is about 54--56% slower than full matrices
and 4--5% slower than the initial scalar reference. CPU speedups must not be
extrapolated to this GPU result.

All runs give maximum absolute difference `4.440892098500626e-16` against
full matrices, and `0.0` with bitwise equality against the initial scalar
reference. Full-matrix GPU arithmetic is therefore numerically equivalent,
but is not bitwise identical in these measurements.

## Diagnosis and production fix

The JACC default launch selects 768 threads/block and 80 registers/thread
for full matrices, but 1024 threads/block and 28 registers/thread for the
site-wise scalar kernel. Both reserve 49,152 bytes of dynamic shared memory
per block. Neither kernel reports local-memory spills.

At 32^4, direct launch diagnostics reduce the scalar kernel to about
503 microseconds using 128 threads/block with the same shared-memory
reservation. A separate fixed-SU(3) prototype assigns consecutive color
components to adjacent threads instead of assigning an entire site to one
thread. It takes about 105--110 microseconds with suitable launch settings
and agrees with the numerical reference within `1e-12`.

Those prototype timings omit the public dispatch and cover only a restricted
case. They isolate a GPU memory-access/launch bottleneck, not a package-level
speedup. The production fix uses a general component-wise kernel selected
only for CUDA arrays by the CUDA extension, and launched through JACC just
like the site-wise kernel. It preserves the site-wise
kernel's arithmetic order, rectangular/adjoint offsets, halo behavior, and
alias validation. CPU and other backends keep the site-wise kernel. No
device-specific block-size or shared-memory override is introduced.

CUDA correctness and timings here are single-device results; multi-GPU,
AMDGPU, and oneAPI performance have not been validated in this work.

## Reproduction

Use a separate Julia environment developing this checkout, with CUDA.jl 5.11.3
and JACC.jl 1.3.1 and the JACC `cuda` backend selected before Julia starts.
Do not change another active project's preferences merely to run this test.

```bash
CUDA_VISIBLE_DEVICES=0 JULIA_CUDA_HARD_MEMORY_LIMIT=4GiB \
  julia --startup-file=no --project=/path/to/cuda-env test/scaled_identity_gpu.jl
CUDA_VISIBLE_DEVICES=0 JULIA_CUDA_HARD_MEMORY_LIMIT=4GiB \
  julia --startup-file=no --project=/path/to/cuda-env benchmark/scaled_identity_gpu.jl 16
CUDA_VISIBLE_DEVICES=0 JULIA_CUDA_HARD_MEMORY_LIMIT=4GiB \
  julia --startup-file=no --project=/path/to/cuda-env benchmark/scaled_identity_gpu.jl 32
```
