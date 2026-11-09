module Float32PrecisionTests
using Test, LinearAlgebra, LatticeMatrices, JACC
JACC.@init_backend
const comm = LatticeMatrices._default_communicator()
const ranks = LatticeMatrices._comm_size(comm)
const snapshots = Dict{String,Vector{Float64}}()
pack(x) = Float64.(vcat(vec(real.(x)),vec(imag.(x))))

@testset "Padé 3x3 normalized coefficients" begin
    H = [0.10 0.05+0.02im 0.01-0.03im; 0.05-0.02im -0.04 0.02+0.01im; 0.01+0.03im 0.02-0.01im -0.06]
    for T in (ComplexF32,ComplexF64), scale in (0,1e-8,1,40,200)
        A = T.(im.*H.*scale)
        c = @inferred LatticeMatrices.exp3x3_pade(permutedims(A)...,1.0)
        C = permutedims(reshape(collect(c),3,3))
        tol = T == ComplexF32 ? 3e-5 : 2e-13
        @test eltype(C) == T
        @test all(isfinite,C)
        @test C ≈ exp(ComplexF64.(A)) rtol=tol atol=tol
        @test C'*C ≈ I atol=tol
    end
end

@testset "Field projection and exponential precision" begin
    dims=(2,2,2,4); grid=(1,1,1,ranks)
    for T in (ComplexF32,ComplexF64), nc in (2,3,4), nw in (0,1)
        RT=real(T); tol=T==ComplexF32 ? 3e-5 : 3e-12
        values=reshape(T[sin(i)/7+im*cos(2i)/9 for i in 1:nc^2*prod(dims)],nc,nc,dims...)
        values[:,:,1,1,1,1].=0 # zero-field SU(2) sin(R)/R limit
        a=LatticeMatrix(values,4,grid; nw,comm0=comm)
        projected=similar(a); out=similar(a)
        LatticeMatrices.traceless_antihermitian!(projected,a)
        reference=ComplexF64.(values)
        exponent=similar(reference)
        for site in CartesianIndices(dims)
            pos=Tuple(site); m=ComplexF64.(values[:,:,pos...]); q=(m-m')/2; q-=tr(q)/nc*I
            reference[:,:,pos...]=q
            exponent[:,:,pos...]=exp(RT(0.2)*q)
        end
        @test gather_and_bcast_matrix(projected) ≈ reference rtol=tol atol=tol
        LatticeMatrices.expt!(out,projected,0.2)
        actual=gather_and_bcast_matrix(out)
        @test all(isfinite,actual)
        @test actual ≈ exponent rtol=tol atol=tol
        snapshots["exp-$T-$nc-$nw"]=pack(actual)
        LatticeMatrices.expt_TA!(out,a,0.2)
        @test gather_and_bcast_matrix(out) ≈ exponent rtol=tol atol=tol
        LatticeMatrices.traceless_antihermitian!(out,0.3,a)
        @test gather_and_bcast_matrix(out) ≈ RT(0.3).*reference rtol=tol atol=tol
        LatticeMatrices.traceless_antihermitian_add!(out,0.2,a)
        @test gather_and_bcast_matrix(out) ≈ RT(0.5).*reference rtol=tol atol=tol
    end
end
end
