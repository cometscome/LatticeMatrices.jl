function _hisq_smearing_ad_loss_from_links(U, V, left)
    hisq_fat7_level1!(V, U)
    return real(
        dot(left[1], V[1]) + dot(left[2], V[2]) +
        dot(left[3], V[3]) + dot(left[4], V[4]))
end

function _hisq_smearing_ad_values(
    ::Val{N}, global_size, offset,
) where N
    count = N * N * prod(global_size)
    values = reshape(Float64.(1:count), N, N, global_size...)
    scale = 3count + offset
    return complex.(
        (values .+ offset) ./ scale,
        reverse(values; dims=(1, 2)) ./ (2scale),
    )
end

function _hisq_smearing_ad_loss(U, left)
    V = [similar(link) for link in U]
    hisq_fat7_level1!(V, U)
    return real(
        dot(left[1], V[1]) + dot(left[2], V[2]) +
        dot(left[3], V[3]) + dot(left[4], V[4]))
end

function hisq_smearing_ad_tests(NCs=(2, 3))
    nprocs = test_comm_size()
    process_grid = (nprocs, 1, 1, 1)
    global_size = (3 * nprocs, 3, 3, 3)
    nw = 1

    for NC in NCs
        @testset "NC=$NC" begin
            U = [
                LatticeMatrix(
                    _hisq_smearing_ad_values(Val(NC), global_size, 7mu),
                    4, process_grid; nw,
                ) for mu in 1:4
            ]
            directions = [
                LatticeMatrix(
                    _hisq_smearing_ad_values(
                        Val(NC), global_size, 41 + 11mu),
                    4, process_grid; nw,
                ) for mu in 1:4
            ]
            left = [
                LatticeMatrix(
                    _hisq_smearing_ad_values(
                        Val(NC), global_size, 89 + 13mu),
                    4, process_grid; nw,
                ) for mu in 1:4
            ]
            set_halo!.(U)
            set_halo!.(directions)
            set_halo!.(left)

            dU = [similar(link) for link in U]
            V = [similar(link) for link in U]
            dV = [similar(link) for link in U]
            clear_matrix!.(dU)
            clear_matrix!.(V)
            clear_matrix!.(dV)

            @testset "HISQ level-1 Fat7 Enzyme pullback" begin
                @test Base.get_extension(
                    LatticeMatrices, :LatticeMatricesEnzymeExt) !== nothing
                Enzyme.API.strictAliasing!(false)
                Enzyme.autodiff(
                    Enzyme.Reverse,
                    Enzyme.Const(_hisq_smearing_ad_loss_from_links),
                    Enzyme.Active,
                    enzyme_duplicated(U, dU),
                    enzyme_duplicated(V, dV),
                    Enzyme.Const(Tuple(left)),
                )

                epsilon = 2e-6
                for varied_direction in 1:4
                    U_plus = deepcopy(U)
                    U_minus = deepcopy(U)
                    add_matrix!(
                        U_plus[varied_direction],
                        directions[varied_direction], epsilon)
                    add_matrix!(
                        U_minus[varied_direction],
                        directions[varied_direction], -epsilon)
                    set_halo!.(U_plus)
                    set_halo!.(U_minus)
                    finite_difference = (
                        _hisq_smearing_ad_loss(U_plus, left) -
                        _hisq_smearing_ad_loss(U_minus, left)
                    ) / (2epsilon)
                    enzyme_directional = real(dot(
                        dU[varied_direction],
                        directions[varied_direction]))
                    @test isapprox(
                        enzyme_directional, finite_difference;
                        atol=3e-6, rtol=3e-7)
                end

                @test all(link -> all(iszero, link.A), dV)
            end
        end
    end
end
