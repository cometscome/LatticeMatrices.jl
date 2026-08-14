# Examples

## 3D classical Heisenberg model

[`classical_heisenberg.jl`](classical_heisenberg.jl) simulates unit-length
three-component classical spins on a periodic simple-cubic lattice,

```math
H/J=-\sum_{x,\mu=1}^{3} \boldsymbol{s}(x)\cdot
\boldsymbol{s}(x+\hat\mu), \qquad |\boldsymbol{s}(x)|=1.
```

The spin field is a `3x1` `LatticeMatrix`. The example uses halo exchange for
nearest neighbors, an exact checkerboard heat-bath update, deterministic
global-site Philox streams, optional microcanonical over-relaxation, MPI
reductions, and the active JACC backend.

Run the default literature-point calculation with

```sh
julia --project=. examples/classical_heisenberg.jl
```

or, for example, use four MPI ranks and a larger sample:

```sh
mpiexec -n 4 julia --project=. examples/classical_heisenberg.jl \
    --L=16 --pes=2,2,1 --thermalization=10000 --sweeps=50000 \
    --measure-every=5 --overrelaxation=2
```

The coupling printed by the program is `K=beta*J`. The energy convention is
`E/(J*V)=-sum(s(x) dot s(x+mu))/V`, with every bond counted once. Holm and
Janke instead add one per bond, so their intensive energy is related by
`e_HJ = 3 + E/(J*V)`.

### Literature comparison

The default coupling is the high-precision bulk critical value
`K_c=0.693002(2)` reported by Y. Deng, H. W. J. Blöte, and M. P. Nightingale,
*Physical Review E* **72**, 016128 (2005),
[doi:10.1103/PhysRevE.72.016128](https://doi.org/10.1103/PhysRevE.72.016128).

The program measures the Binder parameter

```math
U_L=1-\frac{\langle m^4\rangle}{3\langle m^2\rangle^2}.
```

For the same model and convention, C. Holm and W. Janke found the
thermodynamic finite-size-scaling limit `U*=0.6217(8)` and
`K_c=0.6930(1)`, *Physics Letters A* **173**, 8-12 (1993),
[doi:10.1016/0375-9601(93)90077-D](https://doi.org/10.1016/0375-9601(93)90077-D).

`U_L` at a single finite `L` is not expected to equal `U*` within the quoted
literature uncertainty: it has finite-size corrections. Agreement should be
assessed by increasing `L` (the reference used `L=12,...,48`) or by locating
crossings of `U_L(K)` for multiple sizes. The example uses a local heat-bath
algorithm, whereas the high-precision work used cluster updates; substantially
more sweeps may therefore be needed close to the critical point.

Results from a small reproducibility and literature check are recorded in
[`heisenberg_validation.md`](heisenberg_validation.md).
