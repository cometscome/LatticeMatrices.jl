using LatticeMatrices
using Printf

const LATTICE = (4, 4, 4, 4)
const STAGES = ("thin", "ape", "hyp", "hex")

function bridge_arrays(path)
    arrays = Dict(
        stage => [zeros(ComplexF64, 3, 3, LATTICE...) for _ in 1:4]
        for stage in STAGES
    )
    found_header = false
    for line in eachline(path)
        if !found_header
            found_header = startswith(line, "stage\t")
            continue
        end
        columns = split(line, '\t')
        length(columns) == 10 || continue
        stage = columns[1]
        stage in STAGES || continue
        mu, x0, x1, x2, x3, row, col = parse.(Int, columns[2:8])
        arrays[stage][mu][row, col, x0 + 1, x1 + 1, x2 + 1, x3 + 1] =
            complex(parse(Float64, columns[9]), parse(Float64, columns[10]))
    end
    found_header || error("Bridge++ table header was not found in $path")
    return arrays
end

function physical_arrays(links)
    return [begin
        link = links[mu]
        interior = ntuple(
            d -> (link.nw + 1):(link.nw + link.PN[d]), Val(4))
        Array(link.A[:, :, interior...])
    end for mu in 1:4]
end

maximum_difference(left, right) = maximum(
    maximum(abs, left[mu] .- right[mu]) for mu in 1:4)

length(ARGS) == 1 || error("usage: julia bridgepp_uv_smearing_compare.jl TABLE.tsv")
bridge = bridge_arrays(only(ARGS))
thin = [
    LatticeMatrix(array, 4, (1, 1, 1, 1); nw=1)
    for array in bridge["thin"]
]
ape, _ = ape_smear(thin, APEParameters(
    0.6; projection=:max_retr,
    max_retr_iterations=1000, max_retr_tolerance=1e-14))
hyp, _ = hyp_smear(thin, HYPParameters(
    0.75, 0.6, 0.3; projection=:max_retr,
    max_retr_iterations=1000, max_retr_tolerance=1e-14))
hex, _ = hex_smear(thin, HEXParameters(
    alpha_outer=0.125, alpha_middle=0.15, alpha_inner=0.15))

differences = (
    APE=maximum_difference(physical_arrays(ape), bridge["ape"]),
    HYP=maximum_difference(physical_arrays(hyp), bridge["hyp"]),
    HEX=maximum_difference(physical_arrays(hex), bridge["hex"]),
)
for name in keys(differences)
    @printf("%s maximum absolute difference: %.10e\n", name, differences[name])
end

differences.APE < 1e-13 || error("APE comparison failed")
differences.HYP < 1e-13 || error("HYP comparison failed")
differences.HEX < 1e-11 || error("HEX comparison failed")
