function halo_epoch_tests()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)

    @testset "halo epochs synchronize before shifts" begin
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        global_size = (4 * nprocs,)
        process_grid = (nprocs,)
        zeros_global = zeros(ComplexF64, 1, 1, global_size...)
        ones_global = ones(ComplexF64, 1, 1, global_size...)
        M = LatticeMatrix(zeros_global, 1, process_grid; nw=1)
        source = LatticeMatrix(ones_global, 1, process_grid; nw=1)
        shifted_result = similar(M)

        @test !halo_is_dirty(M)
        initial_epochs = halo_epochs(M)

        add_matrix!(M, source)
        @test halo_is_dirty(M)
        @test halo_epochs(M).core == initial_epochs.core + 1
        @test halo_epochs(M).halo == initial_epochs.halo

        shifted_view = Shifted_Lattice(M, (1,))
        @test !halo_is_dirty(M)
        substitute!(shifted_result, shifted_view)
        result = gather_matrix(shifted_result)
        if rank == 0
            @test result == ones_global
        end

        add_matrix!(M, source)
        @test halo_is_dirty(M)
        substitute!(shifted_result, shifted_view)
        @test !halo_is_dirty(M)
        result = gather_matrix(shifted_result)
        if rank == 0
            @test result == 2 .* ones_global
        end

        core_range = (M.nw + 1):(M.nw + M.PN[1])
        @views M.A[:, :, core_range] .= 3 + 0im
        mark_halo_dirty!(M)
        @test halo_is_dirty(M)
        substitute!(shifted_result, Shifted_Lattice(M, (-1,)))
        @test !halo_is_dirty(M)
        result = gather_matrix(shifted_result)
        if rank == 0
            @test result == 3 .* ones_global
        end

        add_matrix!(M, source)
        @test halo_is_dirty(M)
        LatticeMatrices.mul_shiftA_B!(shifted_result, shifted_view, source, (1,))
        @test !halo_is_dirty(M)
        result = gather_matrix(shifted_result)
        if rank == 0
            @test result == 4 .* ones_global
        end

        clean_epochs = halo_epochs(M)
        @test ensure_halo!(M) === nothing
        @test halo_epochs(M) == clean_epochs

        no_halo = LatticeMatrix(zeros_global, 1, process_grid; nw=0)
        mark_halo_dirty!(no_halo)
        @test !halo_is_dirty(no_halo)
        @test halo_epochs(no_halo).core == halo_epochs(no_halo).halo
    end

    @testset "explicit shift kernels synchronize their inputs" begin
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        global_size = (4 * nprocs,)
        process_grid = (nprocs,)
        values = reshape(ComplexF64.(1:prod(global_size)), 1, 1, global_size...)
        ones_global = ones(ComplexF64, 1, 1, global_size...)
        shift_plus = (1,)
        shift_minus = (-1,)

        make_dirty_pair() = begin
            raw = LatticeMatrix(values, 1, process_grid; nw=1)
            reference = LatticeMatrix(values, 1, process_grid; nw=1)
            delta = LatticeMatrix(values, 1, process_grid; nw=1)
            add_matrix!(raw, delta)
            add_matrix!(reference, delta)
            @test halo_is_dirty(raw)
            @test halo_is_dirty(reference)
            return raw, reference
        end

        left = LatticeMatrix(ones_global, 1, process_grid; nw=1)

        raw, reference = make_dirty_pair()
        actual = similar(left)
        expected = similar(left)
        mul_AshiftB!(actual, left, raw, shift_plus)
        mul!(expected, left, Shifted_Lattice(reference, shift_plus))
        @test !halo_is_dirty(raw)
        actual_global = gather_matrix(actual)
        expected_global = gather_matrix(expected)
        if rank == 0
            @test actual_global == expected_global
        end

        raw_A, reference_A = make_dirty_pair()
        raw_B, reference_B = make_dirty_pair()
        clear_matrix!.((actual, expected))
        mul_shiftAshiftB!(actual, raw_A, raw_B, shift_plus, shift_minus)
        mul!(expected,
            Shifted_Lattice(reference_A, shift_plus),
            Shifted_Lattice(reference_B, shift_minus))
        @test !halo_is_dirty(raw_A)
        @test !halo_is_dirty(raw_B)
        actual_global = gather_matrix(actual)
        expected_global = gather_matrix(expected)
        if rank == 0
            @test actual_global == expected_global
        end

        raw, reference = make_dirty_pair()
        clear_matrix!.((actual, expected))
        mul_A_shiftBdag!(actual, left, raw, shift_plus)
        mul!(expected, left, Shifted_Lattice(reference, shift_plus)')
        @test !halo_is_dirty(raw)
        actual_global = gather_matrix(actual)
        expected_global = gather_matrix(expected)
        if rank == 0
            @test actual_global == expected_global
        end

        raw, reference = make_dirty_pair()
        clear_matrix!.((actual, expected))
        add_matrix_shiftedA!(actual, raw, shift_plus)
        add_matrix!(expected, Shifted_Lattice(reference, shift_plus))
        @test !halo_is_dirty(raw)
        actual_global = gather_matrix(actual)
        expected_global = gather_matrix(expected)
        if rank == 0
            @test actual_global == expected_global
        end

        raw, reference = make_dirty_pair()
        clear_matrix!.((actual, expected))
        LatticeMatrices.add_matrix_shiftedAdag!(actual, raw, shift_plus)
        add_matrix!(expected, Shifted_Lattice(reference, shift_plus)')
        @test !halo_is_dirty(raw)
        actual_global = gather_matrix(actual)
        expected_global = gather_matrix(expected)
        if rank == 0
            @test actual_global == expected_global
        end
    end
end
