using LatticeMatrices
using Printf

const LATTICE = (4, 4, 4, 4)
const STAGES = ("thin", "smeared", "left", "force")

function qex_arrays(path)
    arrays = Dict(
        stage => [zeros(ComplexF64, 3, 3, LATTICE...) for _ in 1:4]
        for stage in STAGES
    )
    open(path, "r") do stream
        startswith(readline(stream), "stage\t") ||
            error("QEX table header was not found in $path")
        for line in eachline(stream)
            columns = split(line, '\t')
            stage = columns[1]
            mu, x0, x1, x2, x3, row, col = parse.(Int, columns[2:8])
            arrays[stage][mu][row, col, x0 + 1, x1 + 1, x2 + 1, x3 + 1] =
                complex(parse(Float64, columns[9]), parse(Float64, columns[10]))
        end
    end
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

length(ARGS) == 1 || error("usage: julia qex_stout_compare.jl TABLE.tsv")
qex = qex_arrays(only(ARGS))
thin = [
    LatticeMatrix(array, 4, (1, 1, 1, 1); nw=1)
    for array in qex["thin"]
]
left = [
    LatticeMatrix(array, 4, (1, 1, 1, 1); nw=1)
    for array in qex["left"]
]
smeared, cache = stout_smear(thin, StoutParameters(0.1))
force = [similar(link) for link in thin]
stout_pullback!(force, left, thin, cache)

forward_difference = maximum_difference(
    physical_arrays(smeared), qex["smeared"])
force_difference = maximum_difference(physical_arrays(force), qex["force"])
@printf("stout forward maximum absolute difference: %.10e\n", forward_difference)
@printf("stout pullback maximum absolute difference: %.10e\n", force_difference)

forward_difference < 1e-13 || error("stout forward comparison failed")
force_difference < 1e-13 || error("stout pullback comparison failed")
