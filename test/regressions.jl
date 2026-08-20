using Random: MersenneTwister

function _reduce_three_lattices_with_scale(i, C, A, B, scale,
    ::Val{NC1}, ::Val{NG}, ::Val{nw},
    ::Val{NC2}, ::Val{NG2}, ::Val{nw2},
    ::Val{NC3}, ::Val{NG3}, ::Val{nw3}, dindexer) where {
    NC1,NG,nw,NC2,NG2,nw2,NC3,NG3,nw3}
    indices = delinearize(dindexer, i, nw)
    return scale * (C[1, 1, indices...] + A[1, 1, indices...] + B[1, 1, indices...])
end

function _deterministic_checkerboard_map!(U, V)
    for jc in 1:size(U, 2)
        for ic in 1:size(U, 1)
            U[ic, jc] = 2 * U[ic, jc] - conj(V[jc, ic])
        end
    end
    return nothing
end

function _reference_direct_shift(A, shift, phases)
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

Base.@noinline function _abandon_materialized_shift(M, shift)
    shifted = Shifted_Lattice(M, shift)
    return WeakRef(getfield(shifted, :lease))
end

function regressiontests()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)

    @testset "NC3 normalization in Float32 and Float64" begin
        lattice_size = (2, 2, 2, 2)
        indexer = DIndexer(lattice_size)
        rng = MersenneTwister(0x4c4d)

        for (T, tolerance) in ((ComplexF32, 5f-5), (ComplexF64, 1e-12))
            for nw in (1, 3)
                storage_size = ntuple(d -> lattice_size[d] + 2nw, 4)
                data = rand(rng, T, 3, 3, storage_size...)
                for site in 1:prod(lattice_size)
                    LatticeMatrices.kernel_normalize_NC3!(
                        site, data, indexer, Val(nw))
                    indices = delinearize(indexer, site, nw)
                    matrix = @view data[:, :, indices...]
                    @test matrix * matrix' ≈ Matrix{T}(I, 3, 3) atol=tolerance rtol=tolerance
                    @test det(matrix) ≈ one(T) atol=tolerance rtol=tolerance
                end
            end
        end
    end

    @testset "generic SU(N)/SO(N) normalization in Float32 and Float64" begin
        lattice_size = (2, 2, 2, 2)
        indexer = DIndexer(lattice_size)
        rng = MersenneTwister(0x53554e)

        for NC in (4, 5)
            for (T, tolerance) in (
                (Float32, 2f-4),
                (Float64, 2e-12),
                (ComplexF32, 2f-4),
                (ComplexF64, 2e-12),
            )
                for nw in (1, 3)
                    storage_size = ntuple(d -> lattice_size[d] + 2nw, 4)
                    data = rand(rng, T, NC, NC, storage_size...)
                    for site in 1:prod(lattice_size)
                        LatticeMatrices.kernel_normalize_generic!(
                            site, data, indexer, Val(NC), Val(nw))
                        indices = delinearize(indexer, site, nw)
                        matrix = @view data[:, :, indices...]
                        @test matrix' * matrix ≈ Matrix{T}(I, NC, NC) atol=tolerance rtol=tolerance
                        @test det(matrix) ≈ one(T) atol=tolerance rtol=tolerance
                    end
                end
            end
        end
    end

    @testset "generic SU(N) normalization completes rank-deficient input" begin
        NC = 4
        nw = 1
        lattice_size = (1, 1, 1, 1)
        storage_size = ntuple(d -> lattice_size[d] + 2nw, 4)
        indexer = DIndexer(lattice_size)
        data = ones(ComplexF64, NC, NC, storage_size...)

        LatticeMatrices.kernel_normalize_generic!(
            1, data, indexer, Val(NC), Val(nw))
        indices = delinearize(indexer, 1, nw)
        matrix = @view data[:, :, indices...]
        @test matrix' * matrix ≈ Matrix{ComplexF64}(I, NC, NC) atol=2e-12 rtol=2e-12
        @test det(matrix) ≈ 1 atol=2e-12 rtol=2e-12
    end

    @testset "generic SU(N) normalization public API" begin
        NC = 4
        nw = 1
        lattice_size = (2 * nprocs, 2, 2, 2)
        process_grid = (nprocs, 1, 1, 1)
        data = rand(MersenneTwister(0x53554150), ComplexF64,
            NC, NC, lattice_size...)
        matrix_field = LatticeMatrix(data, 4, process_grid; nw)
        epochs_before = halo_epochs(matrix_field)

        normalize_matrix!(matrix_field)

        @test halo_epochs(matrix_field).core == epochs_before.core + 1
        @test halo_is_dirty(matrix_field)
        host_data = Array(matrix_field.A)
        for site in 1:prod(matrix_field.PN)
            indices = delinearize(matrix_field.indexer, site, nw)
            matrix = @view host_data[:, :, indices...]
            @test matrix' * matrix ≈ Matrix{ComplexF64}(I, NC, NC) atol=2e-12 rtol=2e-12
            @test det(matrix) ≈ 1 atol=2e-12 rtol=2e-12
        end
    end

    @testset "Dirac operator adjoint involution" begin
        lattice_size = (2 * nprocs, 2, 2, 2)
        process_grid = (nprocs, 1, 1, 1)
        gauge_array = ones(ComplexF64, 1, 1, lattice_size...)
        U = [LatticeMatrix(gauge_array, 4, process_grid; nw=1) for _ in 1:4]

        operators = (
            WilsonDiracOperator4D(U, 0.12),
            WilsonDiracOperator4D_Donly(U),
            WilsonDiracCloverOperator4D(U, 0.12, 1.0),
            StaggeredDiracOperator4D(U, 0.01),
            D5DW_MobiusDomainwallOperator5D(U, 2, 0.01, -1.0, 1.0, 1.0),
        )
        for operator in operators
            @test adjoint(adjoint(operator)) === operator
        end

        prototype_array = zeros(ComplexF64, 1, 4, lattice_size...)
        prototype = LatticeMatrix(prototype_array, 4, process_grid; nw=1)
        apply(args...) = nothing
        apply_dag(args...) = nothing
        callback_operator = DiracOp(U, apply, apply_dag, nothing, prototype)
        @test adjoint(adjoint(callback_operator)) === callback_operator

        normal_operator = DdagDOp(callback_operator, similar(prototype))
        @test adjoint(normal_operator) === normal_operator
        @test adjoint(adjoint(normal_operator)) === normal_operator
    end

    @testset "5D nw=0 safety" begin
        L5 = 2
        gauge_size = (2 * nprocs, 2, 2, 2)
        fermion_size = (gauge_size..., L5)
        gauge_grid = (nprocs, 1, 1, 1)
        fermion_grid = (gauge_grid..., 1)

        gauge = ones(ComplexF64, 1, 1, gauge_size...)
        fermion = ones(ComplexF64, 1, 4, fermion_size...)
        U = [LatticeMatrix(gauge, 4, gauge_grid; nw=0) for _ in 1:4]
        ψ = LatticeMatrix(fermion, 5, fermion_grid; nw=0)
        C = similar(ψ)
        D = D5DW_MobiusDomainwallOperator5D(U, L5, 0.1, -1.0, 1.0, 1.0)

        @test_throws ArgumentError mul!(C, D, ψ)
        @test_throws ArgumentError mul!(C, adjoint(D), ψ)
        @test_throws ArgumentError LatticeMatrices.D4x_5D!(C, U, ψ, 1.0)
        @test_throws ArgumentError LatticeMatrices.apply_F_5D!(C, 0.1, L5, ψ)
        @test_throws ArgumentError LatticeMatrices.apply_δF_5D!(C, 0.1, L5, ψ)
    end

    @testset "long shifts with halos" begin
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        global_size = (4 * nprocs,)
        process_grid = (nprocs,)
        A = reshape(ComplexF64.(1:prod(global_size)), 1, 1, global_size...)
        M = LatticeMatrix(A, 1, process_grid; nw=1, numtemps=2)
        C = similar(M)
        initial_pool_size = length(M.temps)

        for amount in (2, -2, global_size[1] + 1, -global_size[1] - 1)
            with_shifted_lattice(M, (amount,)) do shifted
                @test isopen(shifted)
                @test count(M.temps._flagusing) == 1
                substitute!(C, shifted)
            end
            @test count(M.temps._flagusing) == 0
            @test length(M.temps) == initial_pool_size
            result = gather_matrix(C)
            if rank == 0
                expected = circshift(A, (0, 0, -amount))
                @test result == expected
            end
        end

        materialized = Shifted_Lattice(M, (2,))
        @test getfield(materialized, :lease).index in eachindex(M.temps._data)
        release!(materialized)
        release!(materialized)
        @test !isopen(materialized)
        @test_throws ArgumentError materialized.data

        short_shift = Shifted_Lattice(M, (1,))
        release!(short_shift)
        @test isopen(short_shift)
        substitute!(C, short_shift)

        @test_throws ErrorException with_shifted_lattice(M, (2,)) do _
            error("intentional scoped-shift failure")
        end
        @test count(M.temps._flagusing) == 0

        abandoned_lease = _abandon_materialized_shift(M, (2,))
        @test count(M.temps._flagusing) == 1
        GC.gc(true)
        @test abandoned_lease.value === nothing
        @test count(M.temps._flagusing) == 0
        @test length(M.temps) == initial_pool_size

        @test_throws ArgumentError LatticeMatrix(
            1, 1, 1, (2 * nprocs,), process_grid; nw=3)
    end

    @testset "direct multidimensional shifts" begin
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        global_size = (4 * nprocs, 3, 4, 5)
        process_grid = (nprocs, 1, 1, 1)
        phases = (cis(0.17), cis(-0.31), -1.0 + 0im, im)
        values = reshape(
            ComplexF64.(1:(2 * prod(global_size))), 2, 1, global_size...)

        for nw in (0, 1, 2)
            M = LatticeMatrix(values, 4, process_grid; nw, phases, numtemps=2)
            C = similar(M)
            shifts = (
                (nw + 1, 0, 0, 0),
                (-nw - 2, nw + 1, 0, 0),
                (global_size[1], -global_size[2], 2 * global_size[3] + 1, 0),
                (3 * global_size[1] + 2, -2 * global_size[2] - 1,
                    global_size[3] + 3, -global_size[4] - 2),
            )
            for shift in shifts
                with_shifted_lattice(M, shift) do shifted
                    @test isopen(shifted)
                    substitute!(C, shifted)
                end
                @test count(M.temps._flagusing) == 0
                @test length(M.temps) == 2
                result = gather_matrix(C)
                if rank == 0
                    @test result ≈ _reference_direct_shift(values, shift, phases)
                end
            end
        end
    end

    @testset "partial trace uses global positions" begin
        global_size = (4 * nprocs,)
        process_grid = (nprocs,)
        values = zeros(ComplexF64, 2, 2, global_size...)
        for x in 1:global_size[1]
            values[1, 1, x] = x
            values[2, 2, x] = 100 + x
        end

        for nw in (0, 1)
            matrix = LatticeMatrix(values, 1, process_grid; nw)
            for position in (1, global_size[1] ÷ 2, global_size[1])
                @test partial_trace(matrix, 1, position) == 100 + 2position
            end
        end
    end

    @testset "three-lattice parallel_reduce with arguments" begin
        global_size = (4 * nprocs,)
        process_grid = (nprocs,)
        A = fill(1.0 + 0im, 1, 1, global_size...)
        B = fill(2.0 + 0im, 1, 1, global_size...)
        C = fill(3.0 + 0im, 1, 1, global_size...)
        MA = LatticeMatrix(A, 1, process_grid; nw=0)
        MB = LatticeMatrix(B, 1, process_grid; nw=0)
        MC = LatticeMatrix(C, 1, process_grid; nw=0)
        scale = 2.0 + 0im

        result = JACC.parallel_reduce(
            _reduce_three_lattices_with_scale, MA, MB, MC, scale)
        @test result == scale * sum(A + B + C)
    end

    @testset "complex boundary phases are reversible" begin
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        global_size = (4 * nprocs,)
        process_grid = (nprocs,)
        A = reshape(ComplexF64.(1:prod(global_size)), 1, 1, global_size...)

        for nw in (0, 1)
            M = LatticeMatrix(A, 1, process_grid; nw, phases=(im,), numtemps=2)
            shifted = similar(M)
            restored = similar(M)
            with_shifted_lattice(M, (1,)) do view
                substitute!(shifted, view)
            end
            set_halo!(shifted)
            with_shifted_lattice(shifted, (-1,)) do view
                substitute!(restored, view)
            end
            result = gather_matrix(restored)
            if rank == 0
                @test result == A
            end
            @test count(M.temps._flagusing) == 0
            @test count(shifted.temps._flagusing) == 0
        end
    end

    @testset "checkerboard clear uses global coordinates" begin
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        global_size = (3 * nprocs, 2, 2, 2)
        process_grid = (nprocs, 1, 1, 1)
        original = reshape(
            ComplexF64.(1:(4 * prod(global_size))),
            2,
            2,
            global_size...,
        )

        for nw in (0, 1)
            for target_even in (true, false)
                M = LatticeMatrix(original, 4, process_grid; nw)
                clear_matrix!(M, target_even)

                result = gather_matrix(M)
                expected = copy(original)
                for site in CartesianIndices(global_size)
                    global_indices = Tuple(site)
                    if iseven(sum(global_indices)) == target_even
                        @views expected[:, :, global_indices...] .= zero(eltype(expected))
                    end
                end

                if rank == 0
                    @test result == expected
                end

                if nw == 1
                    shifted = similar(M)
                    substitute!(shifted, Shifted_Lattice(M, (1, 0, 0, 0)))
                    shifted_result = gather_matrix(shifted)
                    if rank == 0
                        @test shifted_result == circshift(expected, (0, 0, -1, 0, 0, 0))
                    end
                end
            end
        end
    end

    @testset "checkerboard addition" begin
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        global_size = (3 * nprocs, 2, 2, 2)
        process_grid = (nprocs, 1, 1, 1)
        zero_shift = (0, 0, 0, 0)
        source_shift = (1, 0, 0, 0)

        for nc in (2, 3)
            nvalues = nc * nc * prod(global_size)
            original_a = reshape(
                [complex(Float64(2i + 1), Float64(-i)) for i in 1:nvalues],
                nc,
                nc,
                global_size...,
            )
            initial_c = reshape(
                [complex(Float64(-3i), Float64(i + 2)) for i in 1:nvalues],
                nc,
                nc,
                global_size...,
            )

            A = LatticeMatrix(original_a, 4, process_grid; nw=1)
            shifted_a = Shifted_Lattice(A, source_shift)
            operand_specs = (
                (A, zero_shift, false),
                (shifted_a, source_shift, false),
                (A', zero_shift, true),
                (shifted_a', source_shift, true),
            )

            for (operand, shift, source_adjoint) in operand_specs
                for target_even in (true, false)
                    for α in (1.0, -0.5)
                        masked = LatticeMatrix(initial_c, 4, process_grid; nw=1)
                        add_matrix_evenodd!(masked, operand, target_even, α)
                        result = gather_matrix(masked)

                        if rank == 0
                            expected = copy(initial_c)
                            for site in CartesianIndices(global_size)
                                global_indices = Tuple(site)
                                if iseven(sum(global_indices)) == target_even
                                    source_indices = ntuple(
                                        d -> mod1(
                                            global_indices[d] + shift[d],
                                            global_size[d],
                                        ),
                                        4,
                                    )
                                    for jc in 1:nc
                                        for ic in 1:nc
                                            value = source_adjoint ?
                                                conj(original_a[jc, ic, source_indices...]) :
                                                original_a[ic, jc, source_indices...]
                                            expected[ic, jc, global_indices...] += α * value
                                        end
                                    end
                                end
                            end
                            @test result == expected
                        end
                    end
                end
            end
        end
    end

    @testset "checkerboard site map" begin
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        global_size = (3 * nprocs, 2, 2, 2)
        process_grid = (nprocs, 1, 1, 1)
        shift = (1, 0, 0, 0)

        for nc in (2, 3)
            nvalues = nc * nc * prod(global_size)
            original_u = reshape(
                [complex(Float64(i), Float64(2i + 1)) for i in 1:nvalues],
                nc,
                nc,
                global_size...,
            )
            original_v = reshape(
                [complex(Float64(3i - 1), Float64(-i)) for i in 1:nvalues],
                nc,
                nc,
                global_size...,
            )

            for target_even in (true, false)
                U = LatticeMatrix(original_u, 4, process_grid; nw=1)
                V = LatticeMatrix(original_v, 4, process_grid; nw=1)
                map_matrix_evenodd!(U, V, _deterministic_checkerboard_map!, target_even)

                result = gather_matrix(U)
                expected = copy(original_u)
                for site in CartesianIndices(global_size)
                    global_indices = Tuple(site)
                    if iseven(sum(global_indices)) == target_even
                        for jc in 1:nc
                            for ic in 1:nc
                                expected[ic, jc, global_indices...] =
                                    2 * original_u[ic, jc, global_indices...] -
                                    conj(original_v[jc, ic, global_indices...])
                            end
                        end
                    end
                end

                shifted = similar(U)
                substitute!(shifted, Shifted_Lattice(U, shift))
                shifted_result = gather_matrix(shifted)

                if rank == 0
                    @test result == expected
                    @test shifted_result == circshift(expected, (0, 0, -1, 0, 0, 0))
                end
            end
        end
    end

    @testset "checkerboard shifted multiplication" begin
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        global_size = (3 * nprocs, 2, 2, 2)
        process_grid = (nprocs, 1, 1, 1)
        shift_a = (1, 0, 0, 0)
        shift_b = (-1, 0, 0, 0)

        for nc in (2, 3)
            nvalues = nc * nc * prod(global_size)
            original_a = reshape(
                [complex(Float64(i), Float64(2i + 1)) for i in 1:nvalues],
                nc,
                nc,
                global_size...,
            )
            original_b = reshape(
                [complex(Float64(3i - 1), Float64(-i)) for i in 1:nvalues],
                nc,
                nc,
                global_size...,
            )
            sentinel = complex(-17.0, 5.0)
            initial_c = fill(sentinel, nc, nc, global_size...)

            A = LatticeMatrix(original_a, 4, process_grid; nw=1)
            B = LatticeMatrix(original_b, 4, process_grid; nw=1)
            shifted_a = Shifted_Lattice(A, shift_a)
            shifted_b = Shifted_Lattice(B, shift_b)
            adjoint_pairs = ((false, false), (true, false), (false, true), (true, true))

            for (adjoint_a, adjoint_b) in adjoint_pairs
                operand_a = adjoint_a ? shifted_a' : shifted_a
                operand_b = adjoint_b ? shifted_b' : shifted_b
                reference = zeros(ComplexF64, nc, nc, global_size...)

                for site in CartesianIndices(global_size)
                    global_indices = Tuple(site)
                    source_a = ntuple(
                        d -> mod1(global_indices[d] + shift_a[d], global_size[d]),
                        4,
                    )
                    source_b = ntuple(
                        d -> mod1(global_indices[d] + shift_b[d], global_size[d]),
                        4,
                    )

                    for jc in 1:nc
                        for ic in 1:nc
                            value = zero(eltype(reference))
                            for kc in 1:nc
                                value_a = adjoint_a ?
                                    conj(original_a[kc, ic, source_a...]) :
                                    original_a[ic, kc, source_a...]
                                value_b = adjoint_b ?
                                    conj(original_b[jc, kc, source_b...]) :
                                    original_b[kc, jc, source_b...]
                                value += value_a * value_b
                            end
                            reference[ic, jc, global_indices...] = value
                        end
                    end
                end

                for target_even in (true, false)
                    masked = LatticeMatrix(initial_c, 4, process_grid; nw=1)
                    mul!(masked, operand_a, operand_b, target_even)
                    result = gather_matrix(masked)

                    if rank == 0
                        expected = copy(initial_c)
                        for site in CartesianIndices(global_size)
                            global_indices = Tuple(site)
                            if iseven(sum(global_indices)) == target_even
                                @views expected[:, :, global_indices...] .=
                                    reference[:, :, global_indices...]
                            end
                        end
                        @test result == expected
                    end
                end
            end
        end
    end

    @testset "constructor validation" begin
        @test_throws ArgumentError LatticeMatrix(
            1, 1, 1, (4 * nprocs,), (nprocs + 1,); nw=0)
        @test_throws ArgumentError LatticeMatrix(
            1, 1, 1, (4 * nprocs,), (nprocs,); nw=0, phases=(0,))
        @test_throws ArgumentError LatticeMatrix(
            1, 1, 2, (4 * nprocs,), (nprocs, 1); nw=0)
        if nprocs > 1
            @test_throws ArgumentError LatticeMatrix(
                1, 1, 1, (4 * nprocs + 1,), (nprocs,); nw=0)
        end
    end

    @testset "similar owns communication buffers" begin
        global_size = (4 * nprocs,)
        process_grid = (nprocs,)
        M = LatticeMatrix(1, 1, 1, global_size, process_grid; nw=1)
        S = similar(M)
        @test M.buf !== S.buf
        @test M.buf_host !== S.buf_host
        @test M.shift_buf_host !== S.shift_buf_host
        @test all(M.buf[i] !== S.buf[i] for i in eachindex(M.buf))
        @test all(M.buf_host[i] !== S.buf_host[i] for i in eachindex(M.buf_host))
    end
end
