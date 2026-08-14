using LatticeMatrices
using MPI
using Test
import JACC

JACC.@init_backend

function reference_direct_shift(A, shift, phases)
    global_size = size(A)[3:end]
    shifted = similar(A)
    for site in CartesianIndices(global_size)
        indices = Tuple(site)
        source_indices, factor =
            LatticeMatrices._shifted_global_indices_and_phase(
                indices, shift, global_size, phases, eltype(A))
        @views shifted[:, :, indices...] .= factor .* A[:, :, source_indices...]
    end
    return shifted
end

function direct_shift_16_tests()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    nprocs == 16 || error("this test requires 16 MPI ranks, got $nprocs")

    global_size = (12, 12, 12, 24)
    process_grids = (
        (2, 2, 2, 2),
        (4, 2, 2, 1),
        (1, 4, 1, 4),
        (1, 1, 2, 8),
    )
    phases = (cis(0.11), cis(-0.23), -1.0 + 0im, im)
    values = reshape(
        ComplexF64.(1:(2 * prod(global_size))), 2, 1, global_size...)
    shifts = (
        (2, 0, 0, 0),
        (0, -3, 0, 0),
        (0, 0, 5, 0),
        (0, 0, 0, -7),
        (5, -4, 7, -9),
        (12, -12, 25, -49),
        (37, -26, 31, -74),
    )

    function run_process_grid(process_grid)
        for nw in (0, 1, 2)
            lattice = LatticeMatrix(
                values, 4, process_grid; nw, phases, numtemps=2)
            result_lattice = similar(lattice)
            initial_pool_size = length(lattice.temps)

            for shift in shifts
                materialized =
                    LatticeMatrices._shift_requires_materialization(lattice, shift)
                lease_state_ok = Ref(true)
                with_shifted_lattice(lattice, shift) do shifted
                    expected_used = materialized ? 1 : 0
                    lease_state_ok[] &= isopen(shifted)
                    lease_state_ok[] &=
                        count(lattice.temps._flagusing) == expected_used
                    substitute!(result_lattice, shifted)
                end
                lease_state_ok[] &= count(lattice.temps._flagusing) == 0
                lease_state_ok[] &= length(lattice.temps) == initial_pool_size
                all_leases_ok = MPI.Allreduce(
                    lease_state_ok[] ? 1 : 0, MPI.MIN, comm) == 1

                result = gather_matrix(result_lattice)
                if rank == 0
                    @test all_leases_ok
                    expected = reference_direct_shift(values, shift, phases)
                    @test result ≈ expected
                end
            end
        end
        MPI.Barrier(comm)
        return nothing
    end

    if rank == 0
        @testset "16-rank direct shifts on 12x12x12x24" begin
            for process_grid in process_grids
                @testset "process grid $process_grid" begin
                    run_process_grid(process_grid)
                end
            end
        end
    else
        for process_grid in process_grids
            run_process_grid(process_grid)
        end
    end
end

initialized_here = !MPI.Initialized()
initialized_here && MPI.Init()
try
    direct_shift_16_tests()
    MPI.Barrier(MPI.COMM_WORLD)
finally
    initialized_here && MPI.Finalize()
end
