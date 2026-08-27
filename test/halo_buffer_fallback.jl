function _poison_halo_keep_core!(lattice)
    D = length(lattice.PN)
    core_indices = (
        Colon(),
        Colon(),
        ntuple(d -> (lattice.nw + 1):(lattice.nw + lattice.PN[d]), D)...,
    )
    core = copy(@view lattice.A[core_indices...])
    fill!(lattice.A, complex(NaN, NaN))
    @views lattice.A[core_indices...] .= core
    mark_halo_dirty!(lattice)
    return nothing
end

function halo_buffer_fallback_tests()
    @testset "full-buffer halo fallback" begin
        nprocs = test_comm_size()
        global_size = (4 * nprocs, 5, 6, 7)
        process_grid = (nprocs, 1, 1, 1)
        phases = (cis(0.17), cis(-0.31), -1.0 + 0im, im)
        values = reshape(
            complex.(Float64.(1:(2 * prod(global_size)))),
            2, 1, global_size...,
        )

        optimized = LatticeMatrix(
            values, 4, process_grid; nw=2, phases, numtemps=1)
        fallback = LatticeMatrix(
            values, 4, process_grid; nw=2, phases, numtemps=1)

        @test !LatticeMatrices._uses_full_halo_buffers(optimized.A)
        _poison_halo_keep_core!(optimized)
        _poison_halo_keep_core!(fallback)

        set_halo!(optimized)
        for d in 1:4
            rankM, rankP = fallback.nbr[d]
            me = fallback.myrank
            if rankM == me && rankP == me
                LatticeMatrices.exchange_dim_local!(fallback, d)
            else
                LatticeMatrices._exchange_dim_full_buffers!(
                    fallback, d, rankM, rankP, me)
            end
        end

        @test Array(fallback.A) ≈ Array(optimized.A)
    end
end
