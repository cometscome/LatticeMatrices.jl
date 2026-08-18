using Random

@testset "domain-wall physical and midpoint projections" begin
    Random.seed!(0x443557414c4c)
    NC = 3
    L5 = 4
    number_of_processes = MPI.Comm_size(MPI.COMM_WORLD)
    physical_size = (2 * number_of_processes, 2, 2, 4)
    process_grid4 = (number_of_processes, 1, 1, 1)
    process_grid5 = (number_of_processes, 1, 1, 1, 1)

    source4 = LatticeMatrix(
        randn(ComplexF64, NC, 4, physical_size...), 4, process_grid4; nw=0)
    source5 = LatticeMatrix(
        zeros(ComplexF64, NC, 4, physical_size..., L5), 5, process_grid5; nw=0)
    domainwall_import_physical_source!(source5, source4)
    source4_local = Array(source4.A)
    source5_local = Array(source5.A)
    expected_import = zeros(ComplexF64, size(source5_local))
    expected_import[:, 1:2, :, :, :, :, 1] .= source4_local[:, 1:2, :, :, :, :]
    expected_import[:, 3:4, :, :, :, :, L5] .= source4_local[:, 3:4, :, :, :, :]
    @test source5_local == expected_import

    raw_arrays = ntuple(
        _ -> randn(ComplexF64, NC, 4, physical_size..., L5), 4)
    propagators5 = ntuple(
        spin -> LatticeMatrix(raw_arrays[spin], 5, process_grid5; nw=0), 4)
    physical4 = ntuple(
        _ -> LatticeMatrix(NC, 4, 4, physical_size, process_grid4; nw=0), 4)
    midpoint4 = ntuple(_ -> similar(physical4[1]), 4)
    for source_spin in 1:4
        domainwall_export_physical_solution!(
            physical4[source_spin], propagators5[source_spin])
        domainwall_export_midpoint!(midpoint4[source_spin], propagators5[source_spin])
        raw_local = Array(propagators5[source_spin].A)
        physical_local = Array(physical4[source_spin].A)
        midpoint_local = Array(midpoint4[source_spin].A)
        @test physical_local[:, 1:2, :, :, :, :] ==
            raw_local[:, 1:2, :, :, :, :, L5]
        @test physical_local[:, 3:4, :, :, :, :] ==
            raw_local[:, 3:4, :, :, :, :, 1]
        @test midpoint_local[:, 1:2, :, :, :, :] ==
            raw_local[:, 1:2, :, :, :, :, L5 ÷ 2]
        @test midpoint_local[:, 3:4, :, :, :, :] ==
            raw_local[:, 3:4, :, :, :, :, L5 ÷ 2 + 1]
    end

    left = randn(ComplexF64, 4, 4)
    right = randn(ComplexF64, 4, 4)
    kwargs = (
        axis=4,
        origin=(2, 1, 2, 3),
        momentum=(1, -1, 0, 0),
        parity_mask=(1, 0, 1, 0),
        coefficient=0.75 - 0.2im,
    )
    for (projection, projected4) in
        ((:physical, physical4), (:midpoint, midpoint4))
        expected = projected_bilinear_slices(
            projected4, projected4, left, right; kwargs...)
        observed = domainwall_projected_bilinear_slices(
            propagators5, propagators5, left, right;
            projection, kwargs...)
        @test observed ≈ expected rtol=2e-13 atol=2e-13
    end

    odd5 = LatticeMatrix(
        zeros(ComplexF64, NC, 4, physical_size..., 3), 5, process_grid5; nw=0)
    @test_throws ArgumentError domainwall_export_midpoint!(physical4[1], odd5)
end
