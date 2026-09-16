using Enzyme

function _coefficient_ad_loss(coefficient, output, input)
    clear_matrix!(output)
    add_matrix!(output, input, coefficient)
    return realtrace(output)
end

function coefficient_ad_tests()
    @testset "active real coefficient pullback" begin
        nprocs = test_comm_size()
        global_size = (4 * nprocs,)
        count = 4 * prod(global_size)
        real_values = collect(1.0:count) ./ 11
        imag_values = collect(count:-1:1) ./ 17
        values = reshape(complex.(real_values, imag_values), 2, 2, global_size...)

        input = LatticeMatrix(values, 1, (nprocs,); nw=1, numtemps=2)
        output = similar(input)
        doutput = similar(input)
        coefficient = 0.37

        derivatives = only(Enzyme.autodiff(
            Enzyme.Reverse,
            Enzyme.Const(_coefficient_ad_loss),
            Enzyme.Active,
            Enzyme.Active(coefficient),
            Enzyme.Duplicated(output, doutput),
            Enzyme.Const(input),
        ))
        derivative = first(derivatives)

        # d/dα realtr(α A) = realtr(A). This exercises the JACC reduction
        # used by the custom add_matrix! reverse rule, including MPI allreduce.
        @test derivative ≈ realtrace(input) rtol=5e-13 atol=5e-13
    end
end
