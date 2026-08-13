function _site_rng_allocation_probe(key, algorithm::A) where A
    stream = site_rng(key, UInt64(41), algorithm)
    stream, x = rand_uniform(stream, Float64)
    stream, z0, z1 = rand_normal_pair(stream, Float64)
    stream, y = rand_bounded(stream, UInt32(11))
    return x + z0 + z1 + Float64(y)
end

_site_rng_allocated(key::K, algorithm::A) where {K,A} =
    @allocated _site_rng_allocation_probe(key, algorithm)

function site_rng_tests()
@testset "site-local RNG API" begin
    key = RNGStreamKey(0x123456789abcdef0, 17, 3, 1, 2)
    rng = site_rng(key, UInt64(41), PCG32())

    @test isbitstype(PCG32SiteRNG)
    @test isbitstype(Xoshiro256PlusPlusSiteRNG)
    @test isbitstype(Philox4x32SiteRNG)
    @test isbitstype(RNGStreamKey)
    @test isodd(rng.inc)

    words = UInt32[]
    for _ in 1:8
        rng, value = rand_u32(rng)
        push!(words, value)
    end
    @test words == UInt32[
        0xf6bb5d81,
        0x0da05710,
        0xe9beae27,
        0x595296cb,
        0xcb1dff86,
        0xf6f95cdc,
        0x430127fb,
        0x51f359da,
    ]

    for algorithm in (PCG32(), Xoshiro256PlusPlus(), Philox4x32())
        rng32 = site_rng(key, 41, algorithm)
        rng32, value32 = rand_uniform(rng32, Float32)
        @test value32 isa Float32
        @test 0.0f0 <= value32 < 1.0f0

        rng64 = site_rng(key, 41, algorithm)
        rng64, value64 = rand_uniform(rng64, Float64)
        @test value64 isa Float64
        @test 0.0 <= value64 < 1.0

        normal32 = site_rng(key, 41, algorithm)
        normal32, z32a, z32b = rand_normal_pair(normal32, Float32)
        @test z32a isa Float32
        @test z32b isa Float32
        @test isfinite(z32a) && isfinite(z32b)

        normal64 = site_rng(key, 41, algorithm)
        normal64, z64 = rand_normal(normal64, Float64)
        @test z64 isa Float64
        @test isfinite(z64)

        for T in (Float32, Float64)
            rng_open = site_rng(key, 41, algorithm)
            for _ in 1:1000
                rng_open, value = rand_uniform_open(rng_open, T)
                @test zero(T) < value < one(T)
            end
        end
    end

    normal_stream = site_rng(key, 73, Philox4x32())
    normal_sum = 0.0
    normal_sum2 = 0.0
    normal_count = 100_000
    for _ in 1:(normal_count ÷ 2)
        normal_stream, z0, z1 = rand_normal_pair(normal_stream, Float64)
        normal_sum += z0 + z1
        normal_sum2 += abs2(z0) + abs2(z1)
    end
    normal_mean = normal_sum / normal_count
    normal_std = sqrt(normal_sum2 / normal_count - normal_mean^2)
    @test abs(normal_mean) < 0.015
    @test abs(normal_std - 1) < 0.015

    for algorithm in (PCG32(), Xoshiro256PlusPlus(), Philox4x32())
        counts = zeros(Int, 7)
        rng_int = site_rng(key, 41, algorithm)
        for _ in 1:7000
            rng_int, value = rand_bounded(rng_int, UInt32(7))
            @test UInt32(0) <= value < UInt32(7)
            counts[Int(value)+1] += 1
        end
        @test all(>(800), counts)
        @test all(<(1200), counts)
    end

    base_words = let stream = site_rng(key, 41)
        ntuple(4) do _
            stream, value = rand_u32(stream)
            value
        end
    end
    for changed_key in (
        RNGStreamKey(key.seed + 1, key.sweep, key.direction, key.color, key.subgroup),
        RNGStreamKey(key.seed, key.sweep + 1, key.direction, key.color, key.subgroup),
        RNGStreamKey(key.seed, key.sweep, key.direction + 1, key.color, key.subgroup),
        RNGStreamKey(key.seed, key.sweep, key.direction, key.color + 1, key.subgroup),
        RNGStreamKey(key.seed, key.sweep, key.direction, key.color, key.subgroup + 1),
    )
        stream = site_rng(changed_key, 41)
        changed_words = ntuple(4) do _
            stream, value = rand_u32(stream)
            value
        end
        @test changed_words != base_words
    end
    changed_site = site_rng(key, 42)
    changed_site_words = ntuple(4) do _
        changed_site, value = rand_u32(changed_site)
        value
    end
    @test changed_site_words != base_words

    xoshiro_stream = site_rng(key, 41, Xoshiro256PlusPlus())
    xoshiro_words = ntuple(4) do _
        xoshiro_stream, value = rand_u64(xoshiro_stream)
        value
    end
    @test xoshiro_words == (
        0x6ae4378d9cfd95cb,
        0xec28ce8e467f5024,
        0x9a85a2bcb03f1f8a,
        0x7fdd12651519f8a9,
    )
    @test xoshiro_words != UInt64.(base_words)

    # Random123's published Philox4x32-10 zero-counter/zero-key vector.
    @test LatticeMatrices._philox4x32_10(
        UInt32(0), UInt32(0), UInt32(0), UInt32(0), UInt32(0), UInt32(0)
    ) == (
        UInt32(0x6627e8d5),
        UInt32(0xe169c58d),
        UInt32(0xbc57ac4c),
        UInt32(0x9b00dbd8),
    )

    philox_stream = site_rng(key, 41, Philox4x32())
    philox_words = ntuple(8) do _
        philox_stream, value = rand_u32(philox_stream)
        value
    end
    @test philox_words == (
        0xf03945a7,
        0xbbcbc866,
        0xb9b8c5fd,
        0xa4c43ef5,
        0x2344cf49,
        0xfd2ff13d,
        0xdd20337f,
        0x22ebfdf9,
    )

    # These helpers use global one-based coordinates but return a zero-based
    # column-major id.  The two decompositions describe the same global site.
    global_size = (12, 10, 8, 6)
    global_from_x_split = global_site_coordinates((2, 7, 3, 5), (1, 0, 0, 0), (6, 10, 8, 6))
    global_from_y_split = global_site_coordinates((8, 2, 3, 5), (0, 1, 0, 0), (12, 5, 8, 6))
    @test global_from_x_split == global_from_y_split == (8, 7, 3, 5)
    @test global_site_id(global_from_x_split, global_size) ==
          UInt64(LinearIndices(global_size)[global_from_x_split...]-1)
    @test site_rng(key, global_site_id(global_from_x_split, global_size)) ==
          site_rng(key, global_site_id(global_from_y_split, global_size))

    # Warm the compiled paths before checking that state creation and draws do
    # not allocate on the host.  Both values are immutable and passed by value.
    _site_rng_allocation_probe(key, PCG32())
    _site_rng_allocation_probe(key, Xoshiro256PlusPlus())
    _site_rng_allocation_probe(key, Philox4x32())
    @test _site_rng_allocated(key, PCG32()) == 0
    @test _site_rng_allocated(key, Xoshiro256PlusPlus()) == 0
    @test _site_rng_allocated(key, Philox4x32()) == 0
end

@testset "LatticeMatrix global site ids" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (4, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)
    lattice = LatticeMatrix(1, 1, 4, global_size, process_grid; nw=0)
    key = RNGStreamKey(123, 9, 2, 0, 0)

    for site in CartesianIndices(lattice.PN)
        local_indices = Tuple(site)
        global_indices = global_site_coordinates(lattice, local_indices)
        expected_id = UInt64(LinearIndices(global_size)[global_indices...]-1)
        @test global_site_id(lattice, local_indices) == expected_id
        @test site_rng(key, global_site_id(lattice, local_indices)) ==
              site_rng(key, expected_id)
    end
end
end
