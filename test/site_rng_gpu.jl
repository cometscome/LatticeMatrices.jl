import JACC
JACC.@init_backend

using LatticeMatrices
using Test

@inline function kernel_site_rng_gpu!(
    i,
    words,
    uniforms32,
    uniforms64,
    bounded,
    key,
    algorithm,
)
    rng = site_rng(key, UInt64(i - 1), algorithm)
    @inbounds for draw in axes(words, 1)
        rng, words[draw, i] = rand_u32(rng)
    end
    @inbounds for draw in axes(uniforms32, 1)
        rng, uniforms32[draw, i] = rand_uniform(rng, Float32)
    end
    @inbounds for draw in axes(uniforms64, 1)
        rng, uniforms64[draw, i] = rand_uniform(rng, Float64)
    end
    @inbounds for draw in axes(bounded, 1)
        rng, bounded[draw, i] = rand_bounded(rng, UInt32(17))
    end
    return nothing
end

function site_rng_gpu_tests()
    @testset "site RNG CPU/GPU bit identity" begin
        nsites = 257
        nwords = 19
        nfloat32 = 7
        nfloat64 = 7
        nbounded = 11
        key = RNGStreamKey(0x123456789abcdef0, 27, 3, 1, 2)

        for algorithm in (PCG32(), Xoshiro256PlusPlus(), Philox4x32())
            words = JACC.zeros(UInt32, nwords, nsites)
            uniforms32 = JACC.zeros(Float32, nfloat32, nsites)
            uniforms64 = JACC.zeros(Float64, nfloat64, nsites)
            bounded = JACC.zeros(UInt32, nbounded, nsites)
            JACC.parallel_for(
                nsites,
                kernel_site_rng_gpu!,
                words,
                uniforms32,
                uniforms64,
                bounded,
                key,
                algorithm,
            )

            gpu_words = JACC.to_host(words)
            gpu_uniforms32 = JACC.to_host(uniforms32)
            gpu_uniforms64 = JACC.to_host(uniforms64)
            gpu_bounded = JACC.to_host(bounded)

            for i in 1:nsites
                rng = site_rng(key, UInt64(i - 1), algorithm)
                for draw in 1:nwords
                    rng, expected = rand_u32(rng)
                    @test gpu_words[draw, i] == expected
                end
                for draw in 1:nfloat32
                    rng, expected = rand_uniform(rng, Float32)
                    @test reinterpret(UInt32, gpu_uniforms32[draw, i]) ==
                          reinterpret(UInt32, expected)
                end
                for draw in 1:nfloat64
                    rng, expected = rand_uniform(rng, Float64)
                    @test reinterpret(UInt64, gpu_uniforms64[draw, i]) ==
                          reinterpret(UInt64, expected)
                end
                for draw in 1:nbounded
                    rng, expected = rand_bounded(rng, UInt32(17))
                    @test gpu_bounded[draw, i] == expected
                end
            end
        end
    end
end

site_rng_gpu_tests()
