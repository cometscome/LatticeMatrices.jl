# ---------------- Device-safe tiny PRNG (PCG32, stateless per element) ----------------
# All functions are @inline and pure; no dynamic dispatch, no allocations.

# ---------------- PCG32 core (safe on GPU) ----------------
# All integer ops; rotation count as Int to avoid implicit conversions.
# PCG32 step: safe for Julia/CPU & CUDA
@inline function pcg32_step(state::UInt64, inc::UInt64)
    oldstate = state
    # 64-bit LCG update (wraps mod 2^64 automatically for UInt64)
    state = oldstate * 0x5851F42D4C957F2D + (inc | 0x01)

    # Mix and squeeze to 32 bits
    # IMPORTANT: mask to 32 bits before converting to UInt32 to avoid InexactError.
    x = ((oldstate >> 18) ⊻ oldstate) >> 27         # still up to 37 bits here
    xorshifted = UInt32(x & 0xFFFF_FFFF)            # explicit truncation to 32 bits

    # Rotate by high bits of oldstate
    rot = Int(oldstate >> 59)                       # shift count must be Int
    out = (xorshifted >> rot) | (xorshifted << ((32 - rot) & 31))

    return state, out
end

# Map UInt32 -> uniform in [0,1) as Float32
@inline u01_f32(x::UInt32) = Float32(x) * Float32(2.3283064365386963f-10)  # 1/2^32

# Map two UInt32 -> uniform in [0,1) with ~53 bits as Float64
@inline function u01_f64(x::UInt32, y::UInt32)
    # (x<<21 ^ y>>11) gives 53 bits; scale by 2^-53
    mant = (UInt64(x) << 21) ⊻ (UInt64(y) >> 11)
    return Float64(mant) * 1.1102230246251565e-16  # 2^-53
end

# Deterministic per-site stream: combine indices into (state,inc)
@inline function mix_seed(ix,iy,iz,it,ic,jc,seed0::UInt64)
    h = seed0 ⊻ (UInt64(ix) * 0x9E3779B97F4A7C15) ⊻ (UInt64(iy) * 0xBF58476D1CE4E5B9) ⊻
        (UInt64(iz) * 0x94D049BB133111EB) ⊻ (UInt64(it) * 0xD6E8FEB86659FD93) ⊻
        (UInt64(ic) * 0xA24BAED4963EE407) ⊻ (UInt64(jc) * 0x9FB21C651E98DF25)
    state = h ⊻ 0xDA942042E4DD58B5
    inc   = (h >> 1) | 0x1                 # must be odd
    return state, inc
end

# ---------------- Public counter/stream API for site-local kernels ----------------

"""Supertype of device-safe, immutable site-local RNG states."""
abstract type SiteRNG end

"""Supertype of zero-size, compile-time site RNG selectors."""
abstract type SiteRNGAlgorithm end

"""Compile-time selector for the PCG32 site RNG."""
struct PCG32 <: SiteRNGAlgorithm end

"""Compile-time selector for the xoshiro256++ site RNG."""
struct Xoshiro256PlusPlus <: SiteRNGAlgorithm end

"""Compile-time selector for the Philox4x32-10 counter-based site RNG."""
struct Philox4x32 <: SiteRNGAlgorithm end

"""Small immutable PCG32 state used inside a site kernel."""
struct PCG32SiteRNG <: SiteRNG
    state::UInt64
    inc::UInt64
end

"""Small immutable xoshiro256++ state used inside a site kernel."""
struct Xoshiro256PlusPlusSiteRNG <: SiteRNG
    s0::UInt64
    s1::UInt64
    s2::UInt64
    s3::UInt64
end

"""
State for Philox4x32-10.  `block` counts 128-bit result blocks while `site`
occupies the other 64 bits of the Philox counter.  Four generated words are
cached and consumed through `lane`, avoiding ten Philox rounds per scalar draw.
"""
struct Philox4x32SiteRNG <: SiteRNG
    block::UInt64
    site::UInt64
    key::UInt64
    word0::UInt32
    word1::UInt32
    word2::UInt32
    word3::UInt32
    lane::UInt8
end

"""
    RNGStreamKey(seed, sweep=0, direction=0, color=0, subgroup=0)

Tags identifying one deterministic random-number stream.  The site itself is
added by [`site_rng`](@ref).  All fields are fixed-width integers so the key is
an `isbits` value that can be passed to CPU and accelerator kernels.
"""
struct RNGStreamKey
    seed::UInt64
    sweep::UInt64
    direction::UInt32
    color::UInt32
    subgroup::UInt32
end

RNGStreamKey(
    seed::Integer,
    sweep::Integer=0,
    direction::Integer=0,
    color::Integer=0,
    subgroup::Integer=0,
) = RNGStreamKey(
    UInt64(seed),
    UInt64(sweep),
    UInt32(direction),
    UInt32(color),
    UInt32(subgroup),
)

# SplitMix64 finalizer.  `mix_seed` above is retained for source compatibility
# with older kernels; new lattice fills use the global-site API below.
@inline function _splitmix64(x::UInt64)
    z = x + 0x9e3779b97f4a7c15
    z = xor(z, z >> 30) * 0xbf58476d1ce4e5b9
    z = xor(z, z >> 27) * 0x94d049bb133111eb
    return xor(z, z >> 31)
end

@inline function _mix_site_rng_tag(h::UInt64, tag::UInt64, salt::UInt64)
    return _splitmix64(xor(h, _splitmix64(tag + salt)))
end

@inline function _rng_stream_seed(key::RNGStreamKey)
    h = _splitmix64(xor(key.seed, 0x243f6a8885a308d3))
    h = _mix_site_rng_tag(h, key.sweep, 0xa4093822299f31d0)
    h = _mix_site_rng_tag(h, UInt64(key.direction), 0x082efa98ec4e6c89)
    h = _mix_site_rng_tag(h, UInt64(key.color), 0x452821e638d01377)
    h = _mix_site_rng_tag(h, UInt64(key.subgroup), 0xbe5466cf34e90c6c)
    return h
end

"""
    site_rng(key::RNGStreamKey, global_site, algorithm=PCG32())

Construct a deterministic, independent site-local state. `global_site` is the
zero-based global linear site id returned by [`global_site_id`](@ref), not a
local index or MPI rank.  Select `PCG32()` or `Xoshiro256PlusPlus()` as the
third argument, or use counter-based `Philox4x32()`.  The selector is a
zero-size value, so dispatch is resolved at kernel compilation time without a
per-site runtime branch.
"""
@inline function _site_rng_seed(key::RNGStreamKey, global_site::UInt64)
    h = _rng_stream_seed(key)
    h = _mix_site_rng_tag(h, global_site, 0x13198a2e03707344)
    return h
end

@inline function site_rng(
    key::RNGStreamKey,
    global_site::UInt64,
    ::PCG32,
)
    h = _site_rng_seed(key, global_site)
    state = _splitmix64(xor(h, 0xc0ac29b7c97c50dd))
    inc = _splitmix64(xor(h, 0x3f84d5b5b5470917)) | UInt64(1)
    return PCG32SiteRNG(state, inc)
end

@inline function site_rng(
    key::RNGStreamKey,
    global_site::UInt64,
    ::Xoshiro256PlusPlus,
)
    h = _site_rng_seed(key, global_site)
    gamma = UInt64(0x9e3779b97f4a7c15)
    return Xoshiro256PlusPlusSiteRNG(
        _splitmix64(h),
        _splitmix64(h + gamma),
        _splitmix64(h + gamma + gamma),
        _splitmix64(h + gamma + gamma + gamma),
    )
end

@inline function site_rng(
    key::RNGStreamKey,
    global_site::UInt64,
    ::Philox4x32,
)
    stream_key = _splitmix64(xor(_rng_stream_seed(key), 0x9216d5d98979fb1b))
    return Philox4x32SiteRNG(
        UInt64(0),
        global_site,
        stream_key,
        UInt32(0),
        UInt32(0),
        UInt32(0),
        UInt32(0),
        UInt8(4),
    )
end

@inline site_rng(
    key::RNGStreamKey,
    global_site::Integer,
    algorithm::SiteRNGAlgorithm,
) = site_rng(key, UInt64(global_site), algorithm)

@inline site_rng(
    key::RNGStreamKey,
    global_site::Integer;
    algorithm::SiteRNGAlgorithm=PCG32(),
) = site_rng(key, UInt64(global_site), algorithm)

"""
    rand_u32(rng::SiteRNG) -> updated_rng, value

Advance a site-local stream once and return a uniformly distributed `UInt32`.
"""
@inline function rand_u32(rng::PCG32SiteRNG)
    state, value = pcg32_step(rng.state, rng.inc)
    return PCG32SiteRNG(state, rng.inc), value
end

@inline _rotate_left_64(value::UInt64, amount::Int) =
    (value << amount) | (value >> (64 - amount))

@inline function rand_u64(rng::Xoshiro256PlusPlusSiteRNG)
    result = _rotate_left_64(rng.s0 + rng.s3, 23) + rng.s0
    temporary = rng.s1 << 17

    s2 = xor(rng.s2, rng.s0)
    s3 = xor(rng.s3, rng.s1)
    s1 = xor(rng.s1, s2)
    s0 = xor(rng.s0, s3)
    s2 = xor(s2, temporary)
    s3 = _rotate_left_64(s3, 45)

    return Xoshiro256PlusPlusSiteRNG(s0, s1, s2, s3), result
end


@inline function rand_u32(rng::Xoshiro256PlusPlusSiteRNG)
    rng, value = rand_u64(rng)
    return rng, UInt32(value >> 32)
end

@inline function _philox_mulhilo(multiplier::UInt32, value::UInt32)
    product = UInt64(multiplier) * UInt64(value)
    return UInt32(product >> 32), UInt32(product & 0xffff_ffff)
end


@inline function _philox4x32_round(
    c0::UInt32,
    c1::UInt32,
    c2::UInt32,
    c3::UInt32,
    k0::UInt32,
    k1::UInt32,
)
    high0, low0 = _philox_mulhilo(UInt32(0xd2511f53), c0)
    high1, low1 = _philox_mulhilo(UInt32(0xcd9e8d57), c2)
    return xor(high1, c1, k0), low1, xor(high0, c3, k1), low0
end

"""Internal Philox4x32-10 block transform, kept scalar and device-safe."""
@inline function _philox4x32_10(
    c0::UInt32,
    c1::UInt32,
    c2::UInt32,
    c3::UInt32,
    k0::UInt32,
    k1::UInt32,
)
    @inbounds for _ in 1:10
        c0, c1, c2, c3 = _philox4x32_round(c0, c1, c2, c3, k0, k1)
        k0 += UInt32(0x9e3779b9)
        k1 += UInt32(0xbb67ae85)
    end
    return c0, c1, c2, c3
end

@inline function rand_u32(rng::Philox4x32SiteRNG)
    if rng.lane == UInt8(4)
        c0 = UInt32(rng.block & 0xffff_ffff)
        c1 = UInt32(rng.block >> 32)
        c2 = UInt32(rng.site & 0xffff_ffff)
        c3 = UInt32(rng.site >> 32)
        k0 = UInt32(rng.key & 0xffff_ffff)
        k1 = UInt32(rng.key >> 32)
        word0, word1, word2, word3 = _philox4x32_10(c0, c1, c2, c3, k0, k1)
        updated = Philox4x32SiteRNG(
            rng.block + UInt64(1),
            rng.site,
            rng.key,
            word0,
            word1,
            word2,
            word3,
            UInt8(1),
        )
        return updated, word0
    elseif rng.lane == UInt8(1)
        return Philox4x32SiteRNG(
            rng.block, rng.site, rng.key,
            rng.word0, rng.word1, rng.word2, rng.word3, UInt8(2),
        ), rng.word1
    elseif rng.lane == UInt8(2)
        return Philox4x32SiteRNG(
            rng.block, rng.site, rng.key,
            rng.word0, rng.word1, rng.word2, rng.word3, UInt8(3),
        ), rng.word2
    else
        return Philox4x32SiteRNG(
            rng.block, rng.site, rng.key,
            rng.word0, rng.word1, rng.word2, rng.word3, UInt8(4),
        ), rng.word3
    end
end

"""Advance a site-local stream and return a uniformly distributed `UInt64`."""
@inline function rand_u64(rng::PCG32SiteRNG)
    rng, high = rand_u32(rng)
    rng, low = rand_u32(rng)
    return rng, (UInt64(high) << 32) | UInt64(low)
end


@inline function rand_u64(rng::Philox4x32SiteRNG)
    rng, high = rand_u32(rng)
    rng, low = rand_u32(rng)
    return rng, (UInt64(high) << 32) | UInt64(low)
end

"""
    rand_uniform(rng::SiteRNG, Float32/Float64) -> updated_rng, value

Generate a uniform floating-point value in `[0, 1)`. `Float32` uses 24 random
bits and `Float64` uses 53 random bits from the selected generator.
"""
@inline function rand_uniform(rng::SiteRNG, ::Type{Float32})
    rng, value = rand_u32(rng)
    # Use the high 24 bits.  Converting all 32 bits to Float32 can round the
    # largest UInt32 values to 2^32 and incorrectly produce 1.0f0.
    return rng, Float32(value >> 8) * 5.9604645f-8
end

@inline function rand_uniform(rng::SiteRNG, ::Type{Float64})
    rng, value = rand_u64(rng)
    mantissa = value >> 11
    return rng, Float64(mantissa) * 1.1102230246251565e-16
end

"""
    rand_uniform_open(rng::SiteRNG, Float32/Float64) -> updated_rng, value

Generate a uniform floating-point value in `(0, 1)`.  This is the appropriate
variant for logarithms in rejection samplers.
"""
@inline function rand_uniform_open(rng::SiteRNG, ::Type{Float32})
    rng, value = rand_uniform(rng, Float32)
    return rng, ifelse(iszero(value), 2.9802322f-8, value)
end

@inline function rand_uniform_open(rng::SiteRNG, ::Type{Float64})
    rng, value = rand_uniform(rng, Float64)
    return rng, ifelse(iszero(value), 5.551115123125783e-17, value)
end

"""
    rand_normal_pair(rng::SiteRNG, Float32/Float64) -> updated_rng, z0, z1

Generate two independent standard-normal values with the Box--Muller
transform.  The RNG state and all intermediate values are immutable scalars,
so this function is allocation-free and safe to call inside accelerator
kernels.  Uniform draws are bit-identical across supported backends; the last
bits of the normal values may differ because device transcendental functions
need not be bit-identical to their host counterparts.
"""
@inline function rand_normal_pair(rng::SiteRNG, ::Type{T}) where {T<:Union{Float32,Float64}}
    rng, radial_uniform = rand_uniform_open(rng, T)
    rng, angular_uniform = rand_uniform(rng, T)
    radius = sqrt(T(-2) * log(radial_uniform))
    angle = T(2pi) * angular_uniform
    return rng, radius * cos(angle), radius * sin(angle)
end

"""
    rand_normal(rng::SiteRNG, Float32/Float64) -> updated_rng, value

Generate one standard-normal value.  For bulk generation, prefer
[`rand_normal_pair`](@ref), which consumes both Box--Muller outputs.
"""
@inline function rand_normal(rng::SiteRNG, ::Type{T}) where {T<:Union{Float32,Float64}}
    rng, value, _ = rand_normal_pair(rng, T)
    return rng, value
end

"""
    rand_bounded(rng::SiteRNG, upper::UInt32) -> updated_rng, value

Generate an unbiased integer in `0:(upper-1)` using rejection sampling.
`upper` must be nonzero.  The method is branch-safe for device kernels and can
advance the RNG as many times as necessary.
"""
@inline function rand_bounded(rng::SiteRNG, upper::UInt32)
    threshold = (UInt32(0) - upper) % upper
    while true
        rng, value = rand_u32(rng)
        if value >= threshold
            return rng, value % upper
        end
    end
end

export SiteRNG,
    SiteRNGAlgorithm,
    PCG32SiteRNG,
    Xoshiro256PlusPlusSiteRNG,
    Philox4x32SiteRNG,
    PCG32,
    Xoshiro256PlusPlus,
    Philox4x32,
    RNGStreamKey,
    site_rng,
    rand_u32,
    rand_u64,
    rand_uniform,
    rand_uniform_open,
    rand_normal,
    rand_normal_pair,
    rand_bounded
