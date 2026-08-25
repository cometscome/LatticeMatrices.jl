function _serial_mul_allocation_probe!(destination, left, right)
    mul!(destination, left, right)
    return @allocated mul!(destination, left, right)
end

@testset "serial communicator" begin
    comm = SerialCommunicator()
    global_size = (4, 3)
    process_grid = (1, 1)
    source = reshape(ComplexF64.(1:(2 * 2 * prod(global_size))),
        2, 2, global_size...)

    lattice = LatticeMatrix(
        source, 2, process_grid; nw=1, comm0=comm, device_mapping=:current)

    @test lattice.comm === comm
    @test lattice.cart === comm
    @test lattice.coords == (0, 0)
    @test lattice.dims == process_grid
    @test lattice.myrank == 0
    @test isempty(lattice.buf)
    @test isempty(lattice.buf_host)
    @test @inferred(LatticeMatrices._comm_size(comm)) == 1
    @test @inferred(LatticeMatrices._allreduce_sum(1.0, comm)) == 1.0

    @test gather_matrix(lattice) == source
    @test gather_and_bcast_matrix(lattice) == source
    @test allsum(lattice) == sum(source)
    @test tr(lattice) == sum(source[i, i, x, y]
        for i in 1:2, x in 1:global_size[1], y in 1:global_size[2])

    mark_halo_dirty!(lattice)
    @test halo_is_dirty(lattice)
    set_halo!(lattice)
    @test !halo_is_dirty(lattice)

    copied = similar(lattice)
    @test copied.comm isa SerialCommunicator
    @test copied.cart isa SerialCommunicator
    @test isempty(copied.buf)
    @test isempty(copied.buf_host)

    if lattice.A isa Array
        product = similar(lattice)
        allocated = _serial_mul_allocation_probe!(product, lattice, lattice)
        @test allocated <= 64
        for x in axes(source, 3), y in axes(source, 4)
            @test product.A[:, :, x + 1, y + 1] ≈
                  source[:, :, x, y] * source[:, :, x, y]
        end
    end

    @test LatticeMatrices.exchange_dim!(copied, 1) === nothing

    transport = mpi_transport_info(lattice)
    @test transport.resolved === :local
    @test transport.reason === :serial_communicator
    @test transport.mpi_library === nothing
    @test transport.mpi_library_version === nothing

    @test_throws ArgumentError LatticeMatrix(
        1, 1, 2, global_size, (2, 1); comm0=comm)
    @test_throws ArgumentError LatticeMatrix(
        1, 1, 2, global_size, process_grid;
        comm0=comm, mpi_transport=:device_direct)
end
