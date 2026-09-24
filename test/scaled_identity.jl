using Random: MersenneTwister
include("reference/scaled_identity_prelinear.jl")

function _center_reference(A, B, shift, phases, dagger)
    C = similar(A)
    lattice = size(A)[3:end]
    for site in CartesianIndices(lattice)
        x = Tuple(site)
        y = ntuple(d -> mod1(x[d] + shift[d], lattice[d]), length(x))
        phase = B[1, 1, y...]
        for d in eachindex(x)
            phase *= phases[d]^fld(x[d] + shift[d] - 1, lattice[d])
        end
        phase = dagger ? conj(phase) : phase
        for j in axes(A, 2), i in axes(A, 1)
            C[i, j, x...] = A[i, j, x...] * phase
        end
    end
    return C
end

@testset "scaled-identity lattice multiplication" begin
    nprocs = test_comm_size()
    lattice = (2nprocs, 3)
    grid = (nprocs, 1)
    phases = (1.0 + 0im, -1.0im)
    rng = MersenneTwister(91503)
    for nc in (2, 3, 4), T in (ComplexF32, ComplexF64), nw in (0, 1, 2)
        tol = T == ComplexF32 ? 3e-6 : 3e-13
        data = randn(rng, T, nc, nc, lattice...)
        bdata = zeros(T, 1, 1, lattice...)
        for site in CartesianIndices(lattice)
            bdata[1, 1, Tuple(site)...] = T(cis(2pi * sum(Tuple(site)) / nc))
        end
        A = LatticeMatrix(data, 2, grid; nw)
        s = LatticeMatrix(bdata, 2, grid; nw, phases)
        B = ScaledIdentityLattice(nc, s)
        dense_B = LatticeMatrix(zeros(T, nc, nc, lattice...), 2, grid; nw, phases)
        substitute!(dense_B, B)
        @test length(s.A) * nc^2 == length(dense_B.A)
        C, full, inplace = similar(A), similar(A), similar(A)
        for shift in ((0, 0), (1, -1), (2nprocs + 1, -4)), dagger in (false, true)
            with_shifted_lattice(B, shift) do sb
                operand = dagger ? sb' : sb
                @test mul!(C, A, operand) === C
                with_shifted_lattice(dense_B, shift) do dense_sb
                    mul!(full, A, dagger ? dense_sb' : dense_sb)
                end
                expected = _center_reference(data, bdata, shift, phases, dagger)
                @test gather_and_bcast_matrix(C) ≈ expected atol=tol rtol=tol
                @test gather_and_bcast_matrix(C) ≈ gather_and_bcast_matrix(full) atol=tol rtol=tol
                PrelinearScaledIdentity.mul_scalar!(full, A, operand.scalar)
                @test isequal(gather_and_bcast_matrix(C), gather_and_bcast_matrix(full))
                substitute!(inplace, A)
                mul!(inplace, inplace, operand)
                @test gather_and_bcast_matrix(inplace) ≈ expected atol=tol rtol=tol
                @test isopen(sb)
            end
        end
        with_shifted_lattice(A, (1, -1)) do sa
            for left in (sa, sa')
                mul!(C, left, B')
                mul!(full, left, dense_B')
                @test gather_and_bcast_matrix(C) ≈ gather_and_bcast_matrix(full) atol=tol rtol=tol
                PrelinearScaledIdentity.mul_scalar!(full, left, B.scalar')
                @test isequal(gather_and_bcast_matrix(C), gather_and_bcast_matrix(full))
                mul!(C, B', left)
                @test gather_and_bcast_matrix(C) ≈ gather_and_bcast_matrix(full) atol=tol rtol=tol
            end
        end
        with_shifted_lattice(B', (1, -1)) do sb
            mul!(C, A, sb)
            @test gather_and_bcast_matrix(C) ≈
                _center_reference(data, bdata, (1, -1), phases, true) atol=tol rtol=tol
        end
        alpha, beta = T(0.3 + 0.2im), T(-0.1)
        substitute!(C, A)
        mul!(C, A, B, alpha, beta)
        expected_scaled = alpha .* _center_reference(data, bdata, (0, 0), phases, false) .+ beta .* data
        @test gather_and_bcast_matrix(C) ≈ expected_scaled atol=tol rtol=tol
        substitute!(full, A)
        PrelinearScaledIdentity.mul_scalar!(full, A, B.scalar, alpha, beta)
        @test isequal(gather_and_bcast_matrix(C), gather_and_bcast_matrix(full))
        substitute!(C, A)
        mul!(C, B, A, alpha, beta)
        @test gather_and_bcast_matrix(C) ≈ expected_scaled atol=tol rtol=tol
        # Refresh a previously valid halo after changing B in-place.
        set_halo!(s)
        bdata .*= T(cis(2pi / nc))
        updated_B = ScaledIdentityLattice(nc, LatticeMatrix(bdata, 2, grid; nw, phases))
        substitute!(B, updated_B)
        with_shifted_lattice(B, (1, -1)) do sb
            mul!(C, A, sb)
            @test gather_and_bcast_matrix(C) ≈
                _center_reference(data, bdata, (1, -1), phases, false) atol=tol rtol=tol
        end
        # The new kernel must invalidate the output halo for downstream shifts.
        set_halo!(C)
        mul!(C, A, B)
        with_shifted_lattice(C, (1, -1)) do sc
            substitute!(full, sc)
        end
        plain = _center_reference(data, bdata, (0, 0), phases, false)
        expected_shift = circshift(plain, (0, 0, -1, 1))
        @test gather_and_bcast_matrix(full) ≈ expected_shift atol=tol rtol=tol
        @test_throws ArgumentError mul!(s, s, ScaledIdentityLattice(1, s))
        @test_throws ArgumentError mul!(A, A', B)
        if nw > 0
            with_shifted_lattice(A, (1, 0)) do sa
                @test_throws ArgumentError mul!(A, sa, B)
            end
        end
        long_shift = shift_L(B, (2nprocs + 1, -4))
        release!(long_shift)
        @test_throws ArgumentError mul!(C, A, long_shift)
        @test_throws DimensionMismatch ScaledIdentityLattice(nc, A)
        @test_throws DimensionMismatch mul!(C, A, ScaledIdentityLattice(nc+1, s))
        @test_throws ArgumentError ScaledIdentityLattice(0, s)
        copy_B = similar(B)
        substitute!(copy_B, B)
        @test gather_and_bcast_matrix(copy_B.scalar) == gather_and_bcast_matrix(s)
        substitute!(dense_B, B')
        expected_dense = zeros(T, nc, nc, lattice...)
        for site in CartesianIndices(lattice), i in 1:nc
            expected_dense[i, i, Tuple(site)...] = conj(bdata[1, 1, Tuple(site)...])
        end
        @test gather_and_bcast_matrix(dense_B) ≈ expected_dense atol=tol rtol=tol

        # Regression: copying a shifted view into its own parent needs a snapshot.
        for dagger in (false, true), shift in ((-1, 0), (1, 0))
            substitute!(inplace, A)
            with_shifted_lattice(inplace, shift) do sa
                substitute!(full, dagger ? sa' : sa)
                substitute!(inplace, dagger ? sa' : sa)
            end
            @test gather_and_bcast_matrix(inplace) == gather_and_bcast_matrix(full)
        end
    end
end

@testset "linear scalar indexing with unequal halos and rectangular adjoints" begin
    nprocs = test_comm_size()
    rng = MersenneTwister(92426)
    for D in (1, 2, 4), dagger in (false, true)
        lattice = ntuple(d -> d == 1 ? 2nprocs : 2, D)
        grid = ntuple(d -> d == 1 ? nprocs : 1, D)
        shift = ntuple(d -> isodd(d) ? 1 : -1, D)
        data = randn(rng, ComplexF64, 2, 3, lattice...)
        coeff = randn(rng, ComplexF64, 1, 1, lattice...)
        A = LatticeMatrix(data, D, grid; nw=1)
        s = LatticeMatrix(coeff, D, grid; nw=0)
        rows, cols = dagger ? (3, 2) : (2, 3)
        initial = randn(rng, ComplexF64, rows, cols, lattice...)
        C = LatticeMatrix(initial, D, grid; nw=2)
        old = similar(C)
        B = ScaledIdentityLattice(cols, s)
        with_shifted_lattice(A, shift) do shifted_A
            with_shifted_lattice(B, shift) do shifted_B
                left = dagger ? shifted_A' : shifted_A
                for (alpha, beta) in ((1.0 + 0im, 0.0 + 0im), (0.3 + 0.2im, -0.1 + 0im))
                    seed = LatticeMatrix(initial, D, grid; nw=2)
                    substitute!(C, seed); substitute!(old, seed)
                    mul!(C, left, shifted_B', alpha, beta)
                    PrelinearScaledIdentity.mul_scalar!(old, left, shifted_B.scalar', alpha, beta)
                    actual = gather_and_bcast_matrix(C)
                    @test isequal(actual, gather_and_bcast_matrix(old))
                    shifted_data = circshift(data, (0, 0, map(-, shift)...))
                    input = dagger ? permutedims(conj.(shifted_data), (2, 1, (3:(D+2))...)) : shifted_data
                    expected = _center_reference(input, coeff, shift, ntuple(_ -> 1, D), true)
                    @test actual ≈ alpha .* expected .+ beta .* initial atol=3e-13 rtol=3e-13
                end
                # beta=0 must not read an uninitialized/NaN destination.
                fill!(C.A, ComplexF64(NaN, NaN))
                mark_halo_dirty!(C)
                mul!(C, left, shifted_B', 1.0, 0.0)
                @test all(isfinite, gather_and_bcast_matrix(C))
            end
        end
    end
end

@testset "scaled identity with rectangular matrices and general coefficients" begin
    grid = (test_comm_size(),)
    lattice = (3test_comm_size(),)
    data = reshape(ComplexF64.(1:(6prod(lattice))), 2, 3, lattice...)
    coeff = reshape(ComplexF64[(-0.2 + 0.3im) * i for i in 1:prod(lattice)], 1, 1, lattice...)
    A = LatticeMatrix(data, 1, grid; nw=1)
    s = LatticeMatrix(coeff, 1, grid; nw=0)
    C = similar(A)
    right = ScaledIdentityLattice(3, s)
    left = ScaledIdentityLattice(2, s)
    mul!(C, A, right)
    expected = data .* coeff
    @test gather_and_bcast_matrix(C) ≈ expected
    mul!(C, left, A)
    @test gather_and_bcast_matrix(C) ≈ expected
    @test_throws DimensionMismatch mul!(C, A, left)
    @test_throws DimensionMismatch mul!(C, right, A)
end
