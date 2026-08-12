using Enzyme

_enzyme_test_loss1(U1) = realtrace(U1)
_enzyme_test_loss2(U1, U2) = realtrace(U1) + realtrace(U2)
_enzyme_test_loss3(U1, U2, U3) = realtrace(U1) + realtrace(U2) + realtrace(U3)
_enzyme_test_loss4(U1, U2, U3, U4) =
    realtrace(U1) + realtrace(U2) + realtrace(U3) + realtrace(U4)
_enzyme_test_loss_temp(U1, temp) = realtrace(U1)

function enzymetests()
    @testset "Enzyme extension" begin
        @test Base.get_extension(LatticeMatrices, :LatticeMatricesEnzymeExt) !== nothing

        @testset "rejects nw=0" begin
            nprocs = MPI.Comm_size(MPI.COMM_WORLD)
            global_size = (4 * nprocs,)
            process_grid = (nprocs,)
            values = reshape(ComplexF64.(1:prod(global_size)), 1, 1, global_size...)

            U0 = LatticeMatrix(values, 1, process_grid; nw=0)
            dU0 = similar(U0)
            U1 = LatticeMatrix(values, 1, process_grid; nw=1)
            dU1 = similar(U1)

            cases = (
                (() -> Enzyme_derivative!(_enzyme_test_loss1, U0, dU0), "U1"),
                (() -> Enzyme_derivative!(_enzyme_test_loss2, U1, U0, dU1, dU0), "U2"),
                (() -> Enzyme_derivative!(
                    _enzyme_test_loss3, U1, U1, U0, dU1, dU1, dU0), "U3"),
                (() -> Enzyme_derivative!(
                    _enzyme_test_loss4, U1, U1, U1, U0, dU1, dU1, dU1, dU0), "U4"),
                (() -> Enzyme_derivative!(_enzyme_test_loss1, U1, dU0), "dfdU1"),
                (() -> Enzyme_derivative!(
                    _enzyme_test_loss_temp, U1, dU1; temp=[U0], dtemp=[dU0]), "temp[1]"),
            )

            for (call, label) in cases
                err = try
                    call()
                    nothing
                catch caught
                    caught
                end
                @test err isa ArgumentError
                @test occursin("does not support nw=0", sprint(showerror, err))
                @test occursin(label, sprint(showerror, err))
            end
        end
    end
end
