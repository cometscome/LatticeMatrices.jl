# Classical Heisenberg validation

This records a small validation of
[`classical_heisenberg.jl`](classical_heisenberg.jl), performed with Julia
1.12.1, the JACC threads backend with eight threads, one MPI rank, and the
default seed `0x48e15eeb`. It is an implementation check, not a replacement
for a high-statistics finite-size-scaling analysis.

## Exact and parallel checks

- A fully aligned configuration gives `E/(J*V)=-3` and `|m|=1`, as required
  for a simple-cubic lattice with each bond counted once.
- A complete checkerboard over-relaxation sweep changed `E/(J*V)` by only
  `2.8e-17`.
- The maximum spin-length error was `2.2e-16`.
- The same short run on one rank with process grid `(1,1,1)` and two ranks
  with process grid `(2,1,1)` printed identical observables, including error
  estimates.

## Critical-point check

At the literature critical coupling `K=0.693002`, the measured Binder
parameters were

| L | Thermalization | Measurement sweeps | Samples | `U_L` |
|---:|---------------:|-------------------:|--------:|------:|
| 8  | 1,000 | 5,000  | 1,000 | 0.6228(23) |
| 12 | 3,000 | 20,000 | 4,000 | 0.6232(11) |
| 16 | 5,000 | 30,000 | 6,000 | 0.6226(16) |

The parenthesized uncertainties are block/jackknife statistical errors. All
three values are consistent, allowing for finite-size effects, with the
finite-size-scaling result `U*=0.6217(8)` of Holm and Janke.

As a second check, runs on both sides of the transition gave

| K | `U_8` | `U_12` |
|---:|------:|-------:|
| 0.688 | 0.6157(23) | 0.6097(24) |
| 0.698 | 0.6268(15) | 0.6335(12) |

Thus `U_8-U_12` changes sign. Linear interpolation of these two points gives

```text
K_cross(L=8,12) = 0.69276 +/- 0.00155,
```

where the uncertainty is propagated from the four block errors and does not
include finite-size corrections. This agrees with the high-precision
literature result `K_c=0.693002(2)`.

## References

1. Y. Deng, H. W. J. Blöte, and M. P. Nightingale, *Surface and bulk
   transitions in three-dimensional O(n) models*, Physical Review E **72**,
   016128 (2005),
   [doi:10.1103/PhysRevE.72.016128](https://doi.org/10.1103/PhysRevE.72.016128).
2. C. Holm and W. Janke, *Finite-size scaling study of the three-dimensional
   classical Heisenberg model*, Physics Letters A **173**, 8-12 (1993),
   [doi:10.1016/0375-9601(93)90077-D](https://doi.org/10.1016/0375-9601(93)90077-D).
