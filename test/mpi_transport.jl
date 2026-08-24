function mpi_transport_tests()
    @testset "MPI transport selection" begin
        serial_one_rank = LatticeMatrix(
            1, 1, 1, (4,), (1,);
            comm0=SerialCommunicator(), device_mapping=:current)
        mpi_one_rank = LatticeMatrix(
            1, 1, 1, (4,), (1,);
            comm0=MPI.COMM_SELF, device_mapping=:current)

        @test serial_one_rank.comm isa SerialCommunicator
        @test mpi_one_rank.comm isa MPI.Comm
        @test mpi_one_rank.comm === MPI.COMM_SELF
        @test isempty(serial_one_rank.buf)
        @test isempty(serial_one_rank.buf_host)
        @test isempty(mpi_one_rank.buf)
        @test isempty(mpi_one_rank.buf_host)
        @test @inferred(LatticeMatrices._cart_coords(
            mpi_one_rank.cart, 0, Val(1))) == (0,)
        @test mpi_transport_info(serial_one_rank).resolved === :local
        @test mpi_transport_info(mpi_one_rank).resolved === :host_direct

        number_of_processes = MPI.Comm_size(MPI.COMM_WORLD)
        global_size = (4 * number_of_processes,)
        process_grid = (number_of_processes,)

        automatic = LatticeMatrix(
            1, 1, 1, global_size, process_grid;
            nw=1, device_mapping=:current, mpi_transport=:auto)
        automatic_info = mpi_transport_info(automatic)
        @test automatic_info.requested === :auto
        @test automatic_info.resolved in
              (:host_direct, :host_staged, :device_direct)
        @test automatic_info.mpi_library isa AbstractString
        @test automatic_info.mpi_library_version isa VersionNumber

        staged = LatticeMatrix(
            1, 1, 1, global_size, process_grid;
            nw=1, device_mapping=:current, mpi_transport=:host_staged)
        staged_info = mpi_transport_info(staged)
        @test staged_info.requested === :host_staged
        expected_staged = staged.A isa Array ? :host_direct : :host_staged
        # CPU arrays already reside in MPI-accessible host memory, so forcing
        # host staging does not add a redundant host-to-host copy.
        @test staged_info.resolved === expected_staged

        staged_similar = similar(staged)
        @test mpi_transport_info(staged_similar).requested === :host_staged
        @test mpi_transport_info(staged_similar).resolved === expected_staged

        if automatic.A isa Array
            @test_throws ArgumentError LatticeMatrix(
                1, 1, 1, global_size, process_grid;
                nw=1, device_mapping=:current, mpi_transport=:device_direct)
        end
        @test_throws ArgumentError LatticeMatrix(
            1, 1, 1, global_size, process_grid;
            nw=1, device_mapping=:current, mpi_transport=:invalid)
    end
end
