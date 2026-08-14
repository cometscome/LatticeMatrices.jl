function _su3_test_input(::Type{T}, global_size) where {T<:Complex}
    RT = typeof(real(zero(T)))
    input = Array{T}(undef, 3, 3, global_size...)
    for site in CartesianIndices(global_size)
        coordinates = Tuple(site)
        n = sum((d + 1) * coordinates[d] for d in eachindex(coordinates))
        for j in 1:3, i in 1:3
            re = RT(mod(7i + 11j + 3n, 19) - 9) / RT(7)
            im = RT(mod(13i + 5j + 2n, 23) - 11) / RT(9)
            input[i, j, coordinates...] = complex(re, im)
        end
    end

    first_site = ntuple(_ -> 1, length(global_size))
    @views fill!(input[:, :, first_site...], zero(T))
    if prod(global_size) > 1
        second_site = Tuple(CartesianIndices(global_size)[2])
        @views fill!(input[:, :, second_site...], zero(T))
        input[1, 1, second_site...] = complex(zero(RT), one(RT))
        input[2, 2, second_site...] = complex(zero(RT), one(RT))
        input[3, 3, second_site...] = complex(zero(RT), -RT(2))
    end
    return input
end

function _su3_ta_reference(input, t)
    output = similar(input)
    global_size = size(input)[3:end]
    for site in CartesianIndices(global_size)
        coordinates = Tuple(site)
        matrix = Matrix(@view input[:, :, coordinates...])
        ta = (matrix - matrix') / 2
        ta -= (tr(ta) / 3) * I
        @views output[:, :, coordinates...] .= exp(t * ta)
    end
    return output
end

function _su3_basis_test_input(::Type{RT}, global_size) where {RT<:Real}
    input = Array{RT}(undef, 8, 1, global_size...)
    for site in CartesianIndices(global_size)
        coordinates = Tuple(site)
        n = sum((d + 2) * coordinates[d] for d in eachindex(coordinates))
        for alpha in 1:8
            input[alpha, 1, coordinates...] =
                RT(mod(5alpha + 3n, 17) - 8) / RT(6)
        end
    end
    first_site = ntuple(_ -> 1, length(global_size))
    @views fill!(input[:, :, first_site...], zero(RT))
    return input
end

function _su3_basis_reference(input, t, ::Type{T}) where {T<:Complex}
    RT = typeof(real(zero(T)))
    output = Array{T}(undef, 3, 3, size(input)[3:end]...)
    inv_sqrt_three = inv(sqrt(RT(3)))
    global_size = size(input)[3:end]
    for site in CartesianIndices(global_size)
        coordinates = Tuple(site)
        c = ntuple(alpha -> RT(t) * input[alpha, 1, coordinates...] / RT(2), 8)
        q = T[
            c[3]+inv_sqrt_three*c[8] c[1]-im*c[2] c[4]-im*c[5]
            c[1]+im*c[2] -c[3]+inv_sqrt_three*c[8] c[6]-im*c[7]
            c[4]+im*c[5] c[6]+im*c[7] -RT(2)*inv_sqrt_three*c[8]
        ]
        @views output[:, :, coordinates...] .= exp(im * q)
    end
    return output
end

function matrixexp_su3_tests()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    process_grid = (nprocs, 1)
    global_size = (2 * nprocs, 2)

    @testset "fused SU(3) matrix exponential" begin
        for T in (ComplexF32, ComplexF64)
            RT = typeof(real(zero(T)))
            atol = T === ComplexF64 ? 2e-12 : 5e-5
            input = _su3_test_input(T, global_size)
            lattice = LatticeMatrix(input, 2, process_grid; nw=1)
            output = similar(lattice)

            for t in RT.((0, 1e-5, 0.01, 0.12, 1.0))
                expt_TA!(output, lattice, t)
                result = gather_and_bcast_matrix(output)
                reference = _su3_ta_reference(input, t)
                @test result ≈ reference atol=atol rtol=atol
            end

            basis_input = _su3_basis_test_input(RT, global_size)
            basis = LatticeMatrix(basis_input, 2, process_grid; nw=0)
            basis_output = LatticeMatrix(
                3, 3, 2, global_size, process_grid;
                nw=1, elementtype=T,
            )
            for t in RT.((0, 1e-5, 0.01, 0.12, 1.0))
                expt!(basis_output, basis, t)
                result = gather_and_bcast_matrix(basis_output)
                reference = _su3_basis_reference(basis_input, t, T)
                @test result ≈ reference atol=atol rtol=atol
            end
        end
    end
    return nothing
end
