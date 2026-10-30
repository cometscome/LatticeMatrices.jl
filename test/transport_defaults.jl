module TransportDefaultTests
using Test, JACC, LatticeMatrices
JACC.@init_backend
const distributed = lowercase(get(ENV, "LATTICEMATRICES_TEST_MPI", "false")) == "true"
if distributed
    using MPI
    MPI.Initialized() || MPI.Init()
end
const comm = distributed ? MPI.COMM_WORLD : SerialCommunicator()
const ranks = distributed ? MPI.Comm_size(comm) : 1
select_device_by_mpi_rank!(comm)

@testset "Transport defaults and undecomposed halo buffers" begin
    withenv("LATTICEMATRICES_MPI_TRANSPORT" => nothing, "GENERALIZED_CUDAAWARE_MPI" => nothing) do
        @test LatticeMatrices._default_mpi_transport() == :auto
        for (setting, expected) in (("0", :host_staged), ("1", :device_direct), ("auto", :auto))
            withenv("GENERALIZED_CUDAAWARE_MPI" => setting) do
                @test LatticeMatrices._default_mpi_transport() == expected
            end
        end
        withenv("LATTICEMATRICES_MPI_TRANSPORT" => "host_staged", "GENERALIZED_CUDAAWARE_MPI" => "bad") do
            @test LatticeMatrices._default_mpi_transport() == :host_staged
            lattice = LatticeMatrix(1, 1, 3, (4, 4, 4), (ranks, 1, 1); comm0=comm)
            @test mpi_transport_info(lattice).requested == :host_staged
            for field in (lattice, similar(lattice))
                if ranks > 1
                    @test all(!isempty, field.buf[1:4])
                    @test all(isempty, field.buf[5:12])
                    @test all(isempty, field.buf_host[5:12])
                else
                    @test isempty(field.buf) && isempty(field.buf_host)
                end
                data = zeros(eltype(field.A), size(field.A))
                for pos in CartesianIndices(field.PN)
                    local_pos = Tuple(pos)
                    global_pos = ntuple(d -> field.coords[d]*field.PN[d]+local_pos[d], 3)
                    data[1, 1, (local_pos .+ field.nw)...] = sum((1, 10, 100) .* global_pos)
                end
                copyto!(field.A, data)
                mark_halo_dirty!(field)
                set_halo!(field)
                padded = Array(field.A)
                for pos in CartesianIndices(size(field.A)[3:end])
                    global_pos = ntuple(d -> mod1(field.coords[d]*field.PN[d]+pos[d]-field.nw, 4), 3)
                    @test padded[1, 1, Tuple(pos)...] == sum((1, 10, 100) .* global_pos)
                end
            end
        end
        withenv("LATTICEMATRICES_MPI_TRANSPORT" => "bad") do
            @test_throws ArgumentError LatticeMatrix(1, 1, 1, (4,), (1,); comm0=SerialCommunicator())
            explicit = LatticeMatrix(1, 1, 1, (4,), (1,); comm0=SerialCommunicator(), mpi_transport=:auto)
            @test mpi_transport_info(explicit).requested == :auto
        end
        withenv("GENERALIZED_CUDAAWARE_MPI" => "bad") do
            @test_throws ArgumentError LatticeMatrices._default_mpi_transport()
        end
    end
end
end
