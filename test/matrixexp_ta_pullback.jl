function _pullback_ta_matrix(::Type{T}, NC, phase) where {T<:Complex}
    RT = typeof(real(zero(T)))
    raw = Matrix{T}(undef, NC, NC)
    for jc = 1:NC, ic = 1:NC
        raw[ic, jc] = complex(
            RT(mod(7ic + 5jc + phase, 17) - 8) / RT(11),
            RT(mod(3ic + 11jc + 2phase, 19) - 9) / RT(13),
        )
    end
    result = (raw - adjoint(raw)) / RT(2)
    result -= tr(result) / RT(NC) * I
    return Matrix{T}(result)
end

function _pullback_cotangent(::Type{T}, NC) where {T<:Complex}
    RT = typeof(real(zero(T)))
    return T[
        complex(
            RT(mod(13ic + 2jc, 23) - 11) / RT(17),
            RT(mod(5ic + 7jc, 29) - 14) / RT(19),
        ) for ic = 1:NC, jc = 1:NC
    ]
end

function _pullback_block_frechet(A, direction, t)
    NC = size(A, 1)
    block = zeros(ComplexF64, 2NC, 2NC)
    block[1:NC, 1:NC] .= t .* ComplexF64.(A)
    block[1:NC, NC+1:2NC] .= t .* ComplexF64.(direction)
    block[NC+1:2NC, NC+1:2NC] .= t .* ComplexF64.(A)
    return exp(block)[1:NC, NC+1:2NC]
end

function _check_exp_ta_pullback_case(
    ::Type{T}, A, direction, cotangent, t, global_size, process_grid;
    atol, rtol,
) where {T<:Complex}
    NC = size(A, 1)
    values_A = Array{T}(undef, NC, NC, global_size...)
    values_C = similar(values_A)
    for site in CartesianIndices(global_size)
        coordinates = Tuple(site)
        @views values_A[:, :, coordinates...] .= A
        @views values_C[:, :, coordinates...] .= cotangent
    end

    lattice_A = LatticeMatrix(values_A, length(global_size), process_grid; nw=1)
    lattice_C = LatticeMatrix(values_C, length(global_size), process_grid; nw=1)
    output = similar(lattice_A)
    exp_ta_pullback!(output, lattice_C, lattice_A, t)
    result = gather_and_bcast_matrix(output)
    first_site = ntuple(_ -> 1, length(global_size))
    pullback = Matrix(@view result[:, :, first_site...])

    derivative = _pullback_block_frechet(A, direction, t)
    lhs = tr(ComplexF64.(cotangent) * derivative)
    rhs = tr(ComplexF64.(pullback) * ComplexF64.(direction))
    @test all(isfinite, pullback)
    @test lhs ≈ rhs atol=atol rtol=rtol
    return nothing
end

function matrixexp_ta_pullback_tests()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (2 * nprocs, 2)
    process_grid = (nprocs, 1)

    @testset "SU(2)/SU(3) TA exponential pullback" begin
        for T in (ComplexF64, ComplexF32), NC in (2, 3)
            RT = typeof(real(zero(T)))
            base_A = _pullback_ta_matrix(T, NC, 3)
            direction = _pullback_ta_matrix(T, NC, 9)
            cotangent = _pullback_cotangent(T, NC)
            atol = T === ComplexF64 ? 5e-10 : 4e-4
            rtol = T === ComplexF64 ? 5e-10 : 4e-4

            for scale in (0.0, 1e-14, 1e-9, 1e-6, 1e-3, 1.0)
                _check_exp_ta_pullback_case(
                    T, RT(scale) .* base_A, direction, cotangent, RT(0.3),
                    global_size, process_grid; atol, rtol,
                )
            end
            for t in (0.0, -0.4)
                _check_exp_ta_pullback_case(
                    T, base_A, direction, cotangent, RT(t),
                    global_size, process_grid; atol, rtol,
                )
            end

            if NC == 3
                unit_imag = complex(zero(RT), one(RT))
                repeated = Matrix{T}(Diagonal(T[unit_imag, unit_imag, -RT(2) * unit_imag]))
                for sign in (-1, 1), scale in (1e-6, 1e-2, 1.0)
                    _check_exp_ta_pullback_case(
                        T, RT(sign * scale) .* repeated, direction, cotangent, RT(0.3),
                        global_size, process_grid; atol, rtol,
                    )
                end
            end
        end
    end
    return nothing
end
