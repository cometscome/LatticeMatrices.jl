function _uniform_fill_reference(global_size, nc1, nc2, key, algorithm, ::Type{T}) where {T}
    reference = Array{T}(undef, nc1, nc2, global_size...)
    for site in CartesianIndices(global_size)
        global_indices = Tuple(site)
        stream = site_rng(key, global_site_id(global_indices, global_size), algorithm)
        for jc in 1:nc2, ic in 1:nc1
            if T <: Complex
                real_type = typeof(real(zero(T)))
                stream, real_value = rand_uniform(stream, real_type)
                stream, imag_value = rand_uniform(stream, real_type)
                reference[ic, jc, global_indices...] = T(
                    real_value - real_type(0.5),
                    imag_value - real_type(0.5),
                )
            else
                stream, value = rand_uniform(stream, T)
                reference[ic, jc, global_indices...] = value - T(0.5)
            end
        end
    end
    return reference
end

function _gaussian_fill_reference(global_size, nc1, nc2, key, algorithm, sigma, ::Type{T}) where {T}
    reference = Array{T}(undef, nc1, nc2, global_size...)
    for site in CartesianIndices(global_size)
        global_indices = Tuple(site)
        stream = site_rng(key, global_site_id(global_indices, global_size), algorithm)
        use_spare = false
        spare = zero(T)
        for jc in 1:nc2, ic in 1:nc1
            if use_spare
                value = spare
            else
                stream, value, spare = rand_normal_pair(stream, T)
            end
            reference[ic, jc, global_indices...] = T(sigma) * value
            use_spare = !use_spare
        end
    end
    return reference
end

function random_fill_tests()
    @testset "global-site random lattice fills" begin
        nprocs = MPI.Comm_size(MPI.COMM_WORLD)
        process_grid = (nprocs, 1)
        global_size = (8, 3)

        for algorithm in (PCG32(), Xoshiro256PlusPlus(), Philox4x32())
            uniform_key = RNGStreamKey(0x8f3c2a19, 7, 2, 1, 0x55)
            uniform_field = LatticeMatrix(
                3,
                2,
                2,
                global_size,
                process_grid;
                nw=1,
                elementtype=ComplexF64,
            )
            randomize_matrix!(uniform_field, uniform_key; rng_algorithm=algorithm)
            uniform_global = gather_and_bcast_matrix(uniform_field)
            uniform_reference = _uniform_fill_reference(
                global_size,
                3,
                2,
                uniform_key,
                algorithm,
                ComplexF64,
            )
            @test uniform_global == uniform_reference
            @test !halo_is_dirty(uniform_field)

            repeated = similar(uniform_field)
            randomize_matrix!(repeated, uniform_key; rng_algorithm=algorithm)
            @test gather_and_bcast_matrix(repeated) == uniform_global

            changed = similar(uniform_field)
            randomize_matrix!(
                changed,
                RNGStreamKey(uniform_key.seed, uniform_key.sweep, uniform_key.direction + 1, 1, 0x55);
                rng_algorithm=algorithm,
            )
            @test gather_and_bcast_matrix(changed) != uniform_global

            gaussian_key = RNGStreamKey(0x51f15e, 11, 4, 0, 0x66)
            gaussian_field = LatticeMatrix(
                5,
                1,
                2,
                global_size,
                process_grid;
                nw=1,
                elementtype=Float64,
            )
            sigma = 1.75
            randomize_gaussian_matrix!(
                gaussian_field,
                gaussian_key;
                sigma,
                rng_algorithm=algorithm,
            )
            gaussian_global = gather_and_bcast_matrix(gaussian_field)
            gaussian_reference = _gaussian_fill_reference(
                global_size,
                5,
                1,
                gaussian_key,
                algorithm,
                sigma,
                Float64,
            )
            @test gaussian_global == gaussian_reference
            @test !halo_is_dirty(gaussian_field)
        end

        statistical_size = (64, 32)
        statistical_field = LatticeMatrix(
            8,
            1,
            2,
            statistical_size,
            process_grid;
            nw=0,
            elementtype=Float64,
        )
        sigma = 1.6
        randomize_gaussian_matrix!(
            statistical_field;
            sigma,
            seed=0x31415926,
            rng_algorithm=Philox4x32(),
        )
        values = gather_and_bcast_matrix(statistical_field)
        sample_mean = sum(values) / length(values)
        sample_std = sqrt(sum(abs2, values) / length(values) - sample_mean^2)
        @test abs(sample_mean) < 0.04 * sigma
        @test abs(sample_std - sigma) < 0.04 * sigma
    end
end
