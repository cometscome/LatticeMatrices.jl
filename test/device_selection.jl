function device_selection_tests()
    @testset "JACC device selection" begin
        @test LatticeMatrices._device_ordinal_for_local_rank(0, 2, 2) == 1
        @test LatticeMatrices._device_ordinal_for_local_rank(1, 2, 2) == 2
        @test LatticeMatrices._device_ordinal_for_local_rank(1, 2, 1) == 1
        # Two nodes with two ranks and two visible GPUs per node.  Node-local
        # ranks restart at zero, so each node selects its own device pair.
        local_ranks = (0, 1, 0, 1)
        selected_ordinals = map(local_ranks) do local_rank
            LatticeMatrices._device_ordinal_for_local_rank(local_rank, 2, 2)
        end
        @test selected_ordinals == [1, 2, 1, 2]
        @test_throws ArgumentError LatticeMatrices._device_ordinal_for_local_rank(2, 2, 2)
        @test_throws ArgumentError LatticeMatrices._device_ordinal_for_local_rank(0, 2, 0)
        @test_throws ArgumentError LatticeMatrices._device_ordinal_for_local_rank(0, 3, 2)

        if JACC.backend == "threads"
            selection = select_device_by_mpi_rank!()
            @test selection.backend === :threads
            @test selection.visible_devices == 0
            @test selection.device_ordinal === nothing
        end

        nprocs = MPI.Comm_size(MPI.COMM_WORLD)
        lattice = LatticeMatrix(
            1, 1, 1, (2 * nprocs,), (nprocs,);
            device_mapping=:current,
        )
        @test size(lattice.A, 3) == 4
        @test_throws ArgumentError LatticeMatrix(
            1, 1, 1, (2 * nprocs,), (nprocs,);
            device_mapping=:invalid,
        )
    end
end
