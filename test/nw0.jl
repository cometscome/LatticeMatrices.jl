function _reference_periodic_shift(A, shift, phases)
    global_size = size(A)[3:end]
    shifted = similar(A)
    for site in CartesianIndices(global_size)
        indices = Tuple(site)
        source_indices = ntuple(d -> mod(indices[d] + shift[d] - 1, global_size[d]) + 1,
            length(global_size))
        factor = one(eltype(A))
        for d in eachindex(global_size)
            wraps = fld(indices[d] + shift[d] - 1, global_size[d])
            factor *= convert(eltype(A), phases[d])^wraps
        end
        @views shifted[:, :, indices...] .= factor .* A[:, :, source_indices...]
    end
    return shifted
end

function _reference_site_mul(A, B)
    result = similar(A)
    for site in CartesianIndices(size(A)[3:end])
        indices = Tuple(site)
        @views result[:, :, indices...] .= A[:, :, indices...] * B[:, :, indices...]
    end
    return result
end

function _reference_site_adjoint(A)
    result = similar(A)
    for site in CartesianIndices(size(A)[3:end])
        indices = Tuple(site)
        @views result[:, :, indices...] .= A[:, :, indices...]'
    end
    return result
end

function nw0test()
    nprocs = test_comm_size()
    rank = test_comm_rank()

    @testset "nw=0 halo-free lattice" begin
        dim = 2
        global_size = (4 * nprocs, 3)
        process_grid = (nprocs, 1)
        phases = (-1 + 0im, -im)
        NC = 2

        A = reshape(ComplexF64.(1:(NC * NC * prod(global_size))),
            NC, NC, global_size...)
        B = reverse(A; dims=1)
        # Two materialized views of M coexist below.  MPI direct materialization
        # also needs one transient receive block, hence three pool blocks.
        M = LatticeMatrix(A, dim, process_grid; nw=0, phases, numtemps=3)
        M2 = LatticeMatrix(B, dim, process_grid; nw=0, phases, numtemps=2)

        @test size(M.A) == (NC, NC, M.PN...)
        @test isempty(M.buf)
        @test isempty(M.buf_host)

        before = copy(Array(M.A))
        @test set_halo!(M) === nothing
        @test Array(M.A) == before

        C = similar(M)
        mul!(C, M, M2)
        product = gather_matrix(C)
        if rank == 0
            @test product ≈ _reference_site_mul(A, B)
        end

        shifts = ((1, 0), (-1, 0),
            (global_size[1] + 1, -global_size[2] - 1))
        for shift in shifts
            with_shifted_lattice(M, shift) do shifted
                substitute!(C, shifted)
            end
            result = gather_matrix(C)
            if rank == 0
                @test result ≈ _reference_periodic_shift(A, shift, phases)
            end
            @test count(M.temps._flagusing) == 0
        end

        shift_A = (1, -1)
        shift_B = (-1, 1)
        shifted_A = Shifted_Lattice(M, shift_A)
        shifted_B = Shifted_Lattice(M2, shift_B)
        zero_shift = ntuple(_ -> 0, dim)
        @test shifted_A.data !== M
        @test shifted_B.data !== M2
        @test LatticeMatrices.get_shift(shifted_A) == zero_shift
        @test LatticeMatrices.get_shift(shifted_B) == zero_shift
        shifted_via_function = shift_L(M, shift_A)
        @test shifted_via_function.data !== M
        @test LatticeMatrices.get_shift(shifted_via_function) == zero_shift
        reference_A = _reference_periodic_shift(A, shift_A, phases)
        reference_B = _reference_periodic_shift(B, shift_B, phases)
        reference_Adag = _reference_site_adjoint(reference_A)
        reference_Bdag = _reference_site_adjoint(reference_B)

        products = (
            (shifted_A, shifted_B, _reference_site_mul(reference_A, reference_B)),
            (shifted_A', shifted_B, _reference_site_mul(reference_Adag, reference_B)),
            (shifted_A, shifted_B', _reference_site_mul(reference_A, reference_Bdag)),
            (shifted_A', shifted_B', _reference_site_mul(reference_Adag, reference_Bdag)),
            (M, shifted_B, _reference_site_mul(A, reference_B)),
            (M, shifted_B', _reference_site_mul(A, reference_Bdag)),
        )

        for (left, right, reference) in products
            mul!(C, left, right)
            result = gather_matrix(C)
            if rank == 0
                @test result ≈ reference
            end
        end

        substitute!(C, M)
        mul!(C, shifted_A, shifted_B, 2.0, -0.5)
        result = gather_matrix(C)
        if rank == 0
            reference = 2 .* _reference_site_mul(reference_A, reference_B) .- 0.5 .* A
            @test result ≈ reference
        end

        # Public shifts are materialized, so every existing shifted API remains
        # safe and observes a consistent snapshot on one or many MPI ranks.
        public_products = (
            (shifted_A, M2, _reference_site_mul(reference_A, B)),
            (shifted_A', M2, _reference_site_mul(reference_Adag, B)),
            (shifted_A, M2', _reference_site_mul(reference_A, _reference_site_adjoint(B))),
            (M', shifted_B, _reference_site_mul(_reference_site_adjoint(A), reference_B)),
        )
        for (left, right, reference) in public_products
            mul!(C, left, right)
            result = gather_matrix(C)
            if rank == 0
                @test result ≈ reference
            end
        end

        clear_matrix!(C)
        add_matrix!(C, shifted_A)
        result = gather_matrix(C)
        if rank == 0
            @test result ≈ reference_A
        end

        clear_matrix!(C)
        add_matrix!(C, shifted_A')
        result = gather_matrix(C)
        if rank == 0
            @test result ≈ reference_Adag
        end

        snapshot_source = LatticeMatrix(
            A, dim, process_grid; nw=0, phases, numtemps=2)
        snapshot = Shifted_Lattice(snapshot_source, shift_A)
        snapshot_source.A .= 0
        substitute!(C, snapshot)
        result = gather_matrix(C)
        if rank == 0
            @test result ≈ reference_A
        end

        if nprocs == 1
            lazy_A = LatticeMatrices._lazy_shift_nowing(M, shift_A)
            lazy_B = LatticeMatrices._lazy_shift_nowing(M2, shift_B)
            @test lazy_A.data === M
            @test lazy_B.data === M2
            @test LatticeMatrices.get_shift(lazy_A) == shift_A
            @test LatticeMatrices.get_shift(lazy_B) == shift_B

            lazy_products = (
                (lazy_A, lazy_B, _reference_site_mul(reference_A, reference_B)),
                (lazy_A', lazy_B, _reference_site_mul(reference_Adag, reference_B)),
                (lazy_A, lazy_B', _reference_site_mul(reference_A, reference_Bdag)),
                (lazy_A', lazy_B', _reference_site_mul(reference_Adag, reference_Bdag)),
                (M, lazy_B, _reference_site_mul(A, reference_B)),
                (M, lazy_B', _reference_site_mul(A, reference_Bdag)),
            )
            for (left, right, reference) in lazy_products
                mul!(C, left, right)
                @test Array(C.A) ≈ reference
            end


            substitute!(C, M)
            mul!(C, lazy_A, lazy_B, 2.0, -0.5)
            lazy_scaled_reference =
                2 .* _reference_site_mul(reference_A, reference_B) .- 0.5 .* A
            @test Array(C.A) ≈ lazy_scaled_reference

            substitute!(C, lazy_A)
            @test Array(C.A) ≈ reference_A
            substitute!(C, lazy_A')
            @test Array(C.A) ≈ reference_Adag

            @test_throws MethodError mul!(C, lazy_A, M2)
            @test_throws MethodError add_matrix!(C, lazy_A)
        else
            @test_throws ArgumentError LatticeMatrices._lazy_shift_nowing(M, shift_A)
        end

        release!(snapshot)
        release!(shifted_via_function)
        release!(shifted_A)
        release!(shifted_B)
        @test count(M.temps._flagusing) == 0
        @test count(M2.temps._flagusing) == 0
        @test count(snapshot_source.temps._flagusing) == 0
    end

    @testset "nw=0 Wilson operators" begin
        global_size = (4 * nprocs, 2, 2, 2)
        process_grid = (nprocs, 1, 1, 1)
        NC = 2
        fermion_phases = (1, 1, 1, -1)

        gauge_arrays = [begin
            U = zeros(ComplexF64, NC, NC, global_size...)
            for site in CartesianIndices(global_size), ic in 1:NC
                U[ic, ic, Tuple(site)...] = 1 + 0.1 * d + 0.01im * site[1]
            end
            U
        end for d in 1:4]
        ψarray = reshape(ComplexF64.(1:(NC * 4 * prod(global_size))),
            NC, 4, global_size...)

        U0 = [LatticeMatrix(U, 4, process_grid; nw=0) for U in gauge_arrays]
        U1 = [LatticeMatrix(U, 4, process_grid; nw=1) for U in gauge_arrays]
        ψ0 = LatticeMatrix(ψarray, 4, process_grid; nw=0, phases=fermion_phases)
        ψ1 = LatticeMatrix(ψarray, 4, process_grid; nw=1, phases=fermion_phases)
        C0 = similar(ψ0)
        C1 = similar(ψ1)

        operators = (
            (WilsonDiracOperator4D(U0, 0.12), WilsonDiracOperator4D(U1, 0.12)),
            (adjoint(WilsonDiracOperator4D(U0, 0.12)),
                adjoint(WilsonDiracOperator4D(U1, 0.12))),
            (WilsonDiracOperator4D_Donly(U0), WilsonDiracOperator4D_Donly(U1)),
            (adjoint(WilsonDiracOperator4D_Donly(U0)),
                adjoint(WilsonDiracOperator4D_Donly(U1))),
        )

        for (operator0, operator1) in operators
            mul!(C0, operator0, ψ0)
            mul!(C1, operator1, ψ1)
            result0 = gather_matrix(C0)
            result1 = gather_matrix(C1)
            if rank == 0
                @test result0 ≈ result1
            end
        end
    end
end
