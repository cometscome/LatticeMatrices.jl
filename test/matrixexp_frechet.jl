function _frechet_test_fields(::Type{T}, NC, global_size) where {T<:Complex}
    RT = typeof(real(zero(T)))
    A = Array{T}(undef, NC, NC, global_size...)
    direction = similar(A)
    cotangent = similar(A)

    for site in CartesianIndices(global_size)
        coordinates = Tuple(site)
        n = sum((d + 1) * coordinates[d] for d in eachindex(coordinates))
        raw_A = Matrix{T}(undef, NC, NC)
        raw_direction = similar(raw_A)
        for jc = 1:NC, ic = 1:NC
            raw_A[ic, jc] = complex(
                RT(mod(7ic + 5jc + 3n, 17) - 8) / RT(13),
                RT(mod(3ic + 11jc + 2n, 19) - 9) / RT(15),
            )
            raw_direction[ic, jc] = complex(
                RT(mod(11ic + 3jc + 2n, 23) - 11) / RT(14),
                RT(mod(2ic + 13jc + 5n, 31) - 15) / RT(18),
            )
            cotangent[ic, jc, coordinates...] = complex(
                RT(mod(13ic + 2jc + n, 23) - 11) / RT(17),
                RT(mod(5ic + 7jc + 4n, 29) - 14) / RT(19),
            )
        end
        ta = (raw_A - adjoint(raw_A)) / RT(2)
        ta_direction = (raw_direction - adjoint(raw_direction)) / RT(2)
        trace_part = tr(ta) / RT(NC)
        direction_trace_part = tr(ta_direction) / RT(NC)
        for ic = 1:NC
            ta[ic, ic] -= trace_part
            ta_direction[ic, ic] -= direction_trace_part
        end
        @views A[:, :, coordinates...] .= ta
        @views direction[:, :, coordinates...] .= ta_direction
    end

    first_site = ntuple(_ -> 1, length(global_size))
    @views fill!(A[:, :, first_site...], zero(T))
    return A, direction, cotangent
end

function matrixexp_frechet_tests()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    process_grid = (nprocs, 1)
    global_size = (2 * nprocs, 2)

    @testset "SU(2)/SU(3) exponential TA pullback" begin
        for T in (ComplexF32, ComplexF64), NC in (2, 3)
            RT = typeof(real(zero(T)))
            A, direction, cotangent = _frechet_test_fields(T, NC, global_size)
            lattice_A = LatticeMatrix(A, 2, process_grid; nw=1)
            lattice_cotangent = LatticeMatrix(cotangent, 2, process_grid; nw=1)
            output = similar(lattice_A)
            t = RT(0.3)

            exp_ta_pullback!(output, lattice_cotangent, lattice_A, t)
            result = gather_and_bcast_matrix(output)

            step = T === ComplexF64 ? RT(2e-6) : RT(2e-3)
            tolerance = T === ComplexF64 ? RT(2e-9) : RT(3e-4)
            for site in CartesianIndices(global_size)
                coordinates = Tuple(site)
                matrix_A = Matrix(@view A[:, :, coordinates...])
                matrix_direction = Matrix(@view direction[:, :, coordinates...])
                matrix_cotangent = Matrix(@view cotangent[:, :, coordinates...])
                derivative = (
                    exp(t * (matrix_A + step * matrix_direction)) -
                    exp(t * (matrix_A - step * matrix_direction))
                ) / (RT(2) * step)
                pullback = Matrix(@view result[:, :, coordinates...])
                @test tr(matrix_cotangent * derivative) ≈
                      tr(pullback * matrix_direction) atol=tolerance rtol=tolerance
            end
        end
    end
    return nothing
end
