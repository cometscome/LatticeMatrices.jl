module ZeroBoundaryCloverTests
using Test, LinearAlgebra, LatticeMatrices, JACC
JACC.@init_backend
if get(ENV,"LDO_TEST_MPI","false")=="true"
    using MPI
    MPI.Initialized() || MPI.Init()
end
include("communication_helpers.jl")
include("wilson_clover.jl")
const snapshots=Dict{String,Vector{Float64}}()
pack(a)=vcat(vec(real.(a)),vec(imag.(a)))

# Smooth, noncommuting SU(N) links, identical on every process/backend.
function links_for(nc,dims)
    links=[zeros(ComplexF64,nc,nc,dims...) for _ in 1:4]
    for (sid,site) in enumerate(CartesianIndices(dims)),mu in 1:4
        h=[sin(i+3j+sid+2mu)+im*cos(2i+j+sid+mu) for i in 1:nc,j in 1:nc]
        h=(h-h')/2;h-=tr(h)*I/nc
        links[mu][:,:,Tuple(site)...].=exp(0.2h)
    end
    links
end

@testset "Clover wall retains gauge curvature and force" begin
    dims=(4,2,2,4); grid=(1,1,1,test_comm_size()); kappa=0.1
    for nc in (2,3)
        links=links_for(nc,dims)
        U=[LatticeMatrix(u,4,grid;nw=1) for u in links]
        values=reshape([sin(i)+im*cos(2i) for i in 1:nc*4prod(dims)],nc,4,dims...)
        for phases in ((1,1,1,0),(0,1,1,0)),csw in (-1.0,0.0,1.0)
            psi=LatticeMatrix(values,4,grid;nw=1,phases)
            chi=LatticeMatrix(conj.(values),4,grid;nw=1,phases)
            result=similar(psi);dagger=similar(psi)
            D=WilsonDiracCloverOperator4D(U,kappa,csw)
            expected,fields=_clover_test_reference(links,values,kappa,csw,phases)
            mul!(result,D,psi)
            actual=gather_and_bcast_matrix(result)
            @test actual ≈ expected rtol=3e-12 atol=3e-12
            tag=string(nc,phases,csw)
            snapshots[tag*"forward"]=pack(actual)
            for plane in 1:6
                @test gather_and_bcast_matrix(D.clover[plane]) ≈ fields[plane] rtol=3e-12 atol=3e-12
            end
            mul!(dagger,D',chi)
            @test dot(chi,result) ≈ dot(dagger,psi) rtol=3e-12 atol=3e-12
            gamma5_values=_clover_test_apply_gamma5(values)
            ref5,_=_clover_test_reference(links,gamma5_values,kappa,csw,phases)
            mul!(dagger,D',psi)
            @test gather_and_bcast_matrix(dagger) ≈ _clover_test_apply_gamma5(ref5) rtol=3e-12 atol=3e-12
            # Independent field strength must be nonzero, including near walls.
            @test maximum(maximum(abs,f) for f in fields)>1e-3
            dlinks=[similar(u) for u in U];clear_matrix!.(dlinks)
            wilson_clover_link_pullback!(dlinks,D,U,chi,psi)
            for mu in 1:4
                snapshots[tag*"force$(mu)"]=pack(gather_and_bcast_matrix(dlinks[mu]))
            end
            for t in (2,dims[4])
                directions=[zeros(ComplexF64,size(u)) for u in links]
                direction=Matrix(im*Diagonal([1.0,-1.0,zeros(nc-2)...]))
                directions[4][:,:,1,1,1,t].=direction*links[4][:,:,1,1,1,t]
                delta=[LatticeMatrix(u,4,grid;nw=1) for u in directions]
                analytic=real(sum(dot(dlinks[mu],delta[mu]) for mu in 1:4))
                epsilon=1e-6
                plus=[LatticeMatrix(links[mu].+epsilon.*directions[mu],4,grid;nw=1) for mu in 1:4]
                minus=[LatticeMatrix(links[mu].-epsilon.*directions[mu],4,grid;nw=1) for mu in 1:4]
                yp=similar(psi);ym=similar(psi)
                mul!(yp,WilsonDiracCloverOperator4D(plus,kappa,csw),psi)
                mul!(ym,WilsonDiracCloverOperator4D(minus,kappa,csw),psi)
                numerical=(real(dot(chi,yp))-real(dot(chi,ym)))/(2epsilon)
                @test analytic ≈ numerical rtol=3e-5 atol=3e-7
                if csw==0 && t==dims[4]
                    @test abs(analytic)<1e-13
                end
            end
            @test all(gather_and_bcast_matrix(U[mu])==links[mu] for mu in 1:4)
        end
    end
end
end
