using Random: MersenneTwister

@testset "fused product Lie-algebra projection" begin
    nprocs = test_comm_size()
    global_size = (2nprocs,)
    process_grid = (nprocs,)
    rng = MersenneTwister(0x70726f6a)

    for colors in (2, 3)
        generators = colors^2 - 1
        left = LatticeMatrix(
            rand(rng, ComplexF64, colors, colors, global_size...),
            1,
            process_grid;
            nw=1,
        )
        right = LatticeMatrix(
            rand(rng, ComplexF64, colors, colors, global_size...),
            1,
            process_grid;
            nw=1,
        )
        temporary = similar(left)
        separate = LatticeMatrix(
            zeros(Float64, generators, 1, global_size...),
            1,
            process_grid;
            nw=1,
        )
        fused = similar(separate)

        for (left_operand, right_operand) in (
            (left, right),
            (left', right),
            (left, right'),
            (left', right'),
        )
            clear_matrix!(separate)
            clear_matrix!(fused)
            mul!(temporary, left_operand, right_operand)
            traceless_antihermitian_add!(separate, -0.375, temporary)
            traceless_antihermitian_product_add!(
                fused, -0.375, left_operand, right_operand)
            @test maximum(abs, separate.A .- fused.A) < 5e-13
        end
    end
end
