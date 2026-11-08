module ZeroBoundaryADTests
using Test, LinearAlgebra, LatticeMatrices, Enzyme, JACC
JACC.@init_backend
if get(ENV,"LDO_TEST_MPI","false")=="true"
    using MPI
    MPI.Initialized() || MPI.Init()
end
include("communication_helpers.jl")
const snapshots=Dict{String,Vector{Float64}}()
function short_loss(U1,U2,U3,U4,temp)
    mul_AshiftB!(temp[1],U1,U2,(1,0,0,0))
    return realtrace(temp[1])
end
function long_loss(U1,U2,U3,U4,temp)
    shifted=shift_L(U2,(3,-1,0,-2))
    substitute!(temp[1],shifted)
    release!(shifted)
    return realtrace(temp[1])
end
@testset "Zero boundary shift pullbacks" begin
    dims=(8,4,2,4);grid=(test_comm_size(),1,1,1)
    values=reshape([sin(i)+im*cos(2i) for i in 1:prod(dims)],1,1,dims...)
    perturbation=reshape([cos(i)+im*sin(2i) for i in 1:prod(dims)],1,1,dims...)
    for nw in (1,2),phases in ((0.0,cis(0.17),1.0,0.0),(cis(0.37),cis(-0.21),-1.0,im))
        V=[LatticeMatrix(values,4,grid;nw,phases,numtemps=4) for _ in 1:4]
        direction=LatticeMatrix(perturbation,4,grid;nw,phases)
        for loss in (short_loss,long_loss)
            gradients=[similar(v) for v in V];clear_matrix!.(gradients)
            temp=[similar(V[1])];dtemp=[similar(V[1])]
            clear_matrix!.(temp);clear_matrix!.(dtemp)
            Enzyme_derivative!(loss,V...,gradients...;temp,dtemp)
            epsilon=1e-6
            plus=deepcopy(V[2]);minus=deepcopy(V[2])
            add_matrix!(plus,direction,epsilon);add_matrix!(minus,direction,-epsilon)
            numerical=(loss(V[1],plus,V[3],V[4],temp)-loss(V[1],minus,V[3],V[4],temp))/(2epsilon)
            analytic=real(dot(gradients[2],direction))
            @test analytic ≈ numerical rtol=5e-7 atol=5e-6
            @test real(dot(gradients[2],gradients[2]))>0
            @test !any(V[2].temps._flagusing)
            @test !any(gradients[2].temps._flagusing)
            gradient=gather_and_bcast_matrix(gradients[2])
            snapshots[string(nw,phases,loss)]=vcat(vec(real.(gradient)),vec(imag.(gradient)))
        end
    end
end
end
