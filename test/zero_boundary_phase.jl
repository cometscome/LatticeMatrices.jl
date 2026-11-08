module ZeroBoundaryPhaseTests
using Test, LinearAlgebra, LatticeMatrices, JACC
JACC.@init_backend
if get(ENV,"LDO_TEST_MPI","false")=="true"
    using MPI
    MPI.Initialized() || MPI.Init()
end
include("communication_helpers.jl")
const snapshots=Dict{String,Vector{Float64}}()
pack(a)=vcat(vec(real.(a)),vec(imag.(a)))

# Independent global zero extension, with the old powers for nonzero phases.
function reference(values, shift, phases)
    dims=size(values)[3:end]; out=similar(values)
    for site in CartesianIndices(dims)
        pos=Tuple(site)
        raw=pos.+shift
        outside=any(d->iszero(phases[d]) && !(1<=raw[d]<=dims[d]),eachindex(dims))
        factor=outside ? 0.0+0im : prod(iszero(phases[d]) ? 1 :
            phases[d]^fld(raw[d]-1,dims[d]) for d in eachindex(dims))
        src=ntuple(d->mod1(raw[d],dims[d]),length(dims))
        out[:,:,pos...].=factor.*values[:,:,src...]
    end
    out
end

@testset "Zero boundary phase and unchanged nonzero shifts" begin
    grid=(test_comm_size(),1); dims=(8,6)
    values=reshape([sin(i)+im*cos(2i) for i in 1:prod(dims)],1,1,dims...)
    phasesets=((0.0,1.0),(1.0,0.0),(0.0,0.0),(-1.0,cis(0.37)),(2.0,0.0),(2.0,cis(-0.21)))
    shifts=((0,0),(1,0),(-1,0),(0,1),(0,-1),(2,-2),(-2,2),(9,-7),(-17,13))
    for nw in (0,1,2), phases in phasesets
        a=LatticeMatrix(values,2,grid;nw,phases,numtemps=4)
        b=similar(a)
        for shift in shifts
            with_shifted_lattice(a,shift) do shifted
                substitute!(b,shifted)
            end
            actual=gather_and_bcast_matrix(b)
            @test actual ≈ reference(values,shift,phases) rtol=3e-13 atol=3e-13
            snapshots[string(nw,phases,shift)]=pack(actual)
            @test !any(a.temps._flagusing)
            if nw==0
                materialized=LatticeMatrices._materialize_periodic_shift(a,shift)
                @test gather_and_bcast_matrix(materialized) ≈ reference(values,shift,phases) rtol=3e-13 atol=3e-13
                if test_comm_size()==1
                    lazy=LatticeMatrices._lazy_shift_nowing(a,shift)
                    mul!(b,lazy,LatticeMatrices._lazy_shift_nowing(a,(0,0)),1.0,0.0)
                    @test gather_and_bcast_matrix(b) ≈ reference(values,shift,phases).*values rtol=5e-13 atol=5e-13
                end
            end
        end
        if nw>0
            fallback=LatticeMatrix(values,2,grid;nw,phases)
            for d in 1:2
                rm,rp=fallback.nbr[d]
                if rm==fallback.myrank && rp==fallback.myrank
                    LatticeMatrices.exchange_dim_local!(fallback,d)
                else
                    LatticeMatrices._exchange_dim_full_buffers!(fallback,d,rm,rp,fallback.myrank)
                end
            end
            @test Array(fallback.A) ≈ Array(a.A) rtol=3e-13 atol=3e-13
        end
    end
    # The transpose of a wall shift uses the opposite displacement, but is not
    # an inverse: boundary data is lost. Preserve complex/non-unit adjoint maps.
    for phases in phasesets
        a=LatticeMatrix(values,2,grid;nw=0,phases,numtemps=4)
        left=similar(a); right=similar(a)
        adjphases=LatticeMatrices._adjoint_shift_phases(a)
        dual=LatticeMatrix(values,2,grid;nw=0,phases=adjphases,numtemps=4)
        with_shifted_lattice(a,(2,-1)) do s;substitute!(left,s);end
        with_shifted_lattice(dual,(-2,1)) do s;substitute!(right,s);end
        @test dot(dual,left) ≈ dot(right,a) rtol=5e-13 atol=5e-13
    end
end
end
