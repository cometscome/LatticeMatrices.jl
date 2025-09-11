using LatticeMatrices
using Test
using MPI
import JACC
using LinearAlgebra
using InteractiveUtils
JACC.@init_backend
using MPI, JACC, StaticArrays

function dotproduct(i,A,B,::Val{NC1}, ::Val{NC2}, ::Val{nw}, ::Val{NC1}, ::Val{NC2}, ::Val{nw},dindexer) where {NC1,NC2,nw}
    indices = delinearize(dindexer, i, nw)
    s = zero(eltype(A))
    for jc = 1:NC2
        for ic=1:NC1
            s += A[ic,jc,indices...]' * B[ic,jc,indices...]
        end
    end
    return s
end


function multtest(NC, dim)
    NX = 16
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    myrank = MPI.Comm_rank(MPI.COMM_WORLD)
    #=
    if dim == 1
        gsize = (NX,)
    elseif dim == 2
        gsize = (NX, NY)
    elseif dim == 3
        gsize = (NX, NY, NZ)
    elseif dim == 4
        gsize = (NX, NY, NZ, NT)
    else
        error("dim should be smaller than 5.")
    end
    =#
    gsize = ntuple(_ -> NX, dim)
    #gsize = (NX, NY)
    nw = 1

    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    if length(ARGS) == 0
        n1 = nprocs ÷ 2
        if n1 == 0
            n1 = 1
        end
        PEs = ntuple(i -> ifelse(i == 1, n1, ifelse(i == 2, nprocs ÷ n1, 1)), dim)
        #PEs = (n1, nprocs ÷ n1, 1, 1)
    else
        PEs = Tuple(parse.(Int64, ARGS))
    end
    PEs = PEs[1:dim]
    M1 = LatticeMatrix(NC, NC, dim, gsize, PEs; nw)
    comm = M1.cart

    A1 = rand(ComplexF64, NC, NC, gsize...)

    A2 = rand(ComplexF64, NC, NC, gsize...)
    M2 = LatticeMatrix(A2, dim, PEs; nw)


    A3 = rand(ComplexF64, NC, NC, gsize...)
    M3 = LatticeMatrix(A3, dim, PEs; nw)

    L = 1
    indexer = DIndexer(gsize)
    indices = delinearize(indexer, L, nw)

    #indices = (ix + nw, iy + nw, iz + nw, it + nw)[1:dim]
    indices_a = delinearize(indexer, L, 0)

    a1 = A1[:, :, indices_a...]
    a2 = A2[:, :, indices_a...]
    a3 = A3[:, :, indices_a...]



    expt!(M1, M2, 1)
    m1 = M1.A[:, :, indices...]
    a1 = exp(a2)
    if myrank == 0
        @test a1 ≈ Array(m1) atol = 1e-6
    end

    a = 1.2
    b = 3.9
    axpby!(
        a,
        M2,
        b,
        M1
    )
    m1 = M1.A[:, :, indices...]
    axpby!(a, a2, b, a1)
    if myrank == 0
        @test a1 ≈ Array(m1) atol = 1e-6
    end

    #NG = 4
    for NG=3:4
        A1g = rand(ComplexF64, NC, NG, gsize...)
        A2g = rand(ComplexF64, NC, NG, gsize...)
        M1g = LatticeMatrix(A1g, dim, PEs; nw)
        M2g = LatticeMatrix(A2g, dim, PEs; nw)
        a1g = A1g[:, :, indices_a...]
        a2g = A2g[:, :, indices_a...]

        c = rand(NG,NG)
        mul!(M1g, c)
        m1 = M1g.A[:, :, indices...]
        atemp = similar(a2g)
        mul!(atemp,a1g,c)
        a1g .= atemp
        #mul!(a1, c)
        if myrank == 0
            #display(a1g)
            #display(Array(m1))
            @test a1g ≈ Array(m1) atol = 1e-6
        end

        mul!(M2g, transpose(c))
        m2 = M2g.A[:, :, indices...]
        atemp = similar(a2g)
        mul!(atemp,a2g,transpose(c))
        a2g .= atemp
        #mul!(a1, c)
        if myrank == 0
            #display(a1g)
            #display(Array(m1))
            @test a2g ≈ Array(m2) atol = 1e-6
        end

        A1g = rand(ComplexF64, NC, NG, gsize...)
        A2g = rand(ComplexF64, NC, NG, gsize...)
        M1g = LatticeMatrix(A1g, dim, PEs; nw)
        M2g = LatticeMatrix(A2g, dim, PEs; nw)
        a1g = A1g[:, :, indices_a...]
        a2g = A2g[:, :, indices_a...]


        c = rand(NG,NG)
        JACC.parallel_for(LatticeMatrices.kernel_Dmatrix_mulA!,M1g,JACC.array(c))
        m1 = M1g.A[:, :, indices...]
        atemp = similar(a2g)
        mul!(atemp,a1g,c)
        a1g .= atemp
        #mul!(a1, c)
        if myrank == 0
            #display(a1g)
            #display(Array(m1))
            @test a1g ≈ Array(m1) atol = 1e-6
        end


        A1g = rand(ComplexF64, NC, NG, gsize...)
        A2g = rand(ComplexF64, NC, NG, gsize...)
        M1g = LatticeMatrix(A1g, dim, PEs; nw)
        M2g = LatticeMatrix(A2g, dim, PEs; nw)

        s = JACC.parallel_reduce(dotproduct,M1g,M2g)
        #mul!(a1, c)
        if myrank == 0
            #display(a1g)
            #display(Array(m1))
            #println(s)
        end



    end



    mul!(a1, a2, a3)
    mul!(M1, M2, M3)
    m1 = M1.A[:, :, indices...]
    if myrank == 0
        @test a1 ≈ Array(m1) atol = 1e-6
    end


    mul!(a1, a2', a3)
    mul!(M1, M2', M3)
    m1 = M1.A[:, :, indices...]
    if myrank == 0
        @test a1 ≈ Array(m1) atol = 1e-6
    end

    mul!(a1, a2, a3')
    mul!(M1, M2, M3')
    m1 = M1.A[:, :, indices...]
    if myrank == 0
        @test a1 ≈ Array(m1) atol = 1e-6
    end

    mul!(a1, a2', a3')
    mul!(M1, M2', M3')
    m1 = M1.A[:, :, indices...]
    if myrank == 0
        @test a1 ≈ Array(m1) atol = 1e-6
    end

    #shift = (1, 0, 0, 0)
    shift = ntuple(i -> ifelse(i == 1, -1, 0), dim)
    indices_p = shiftindices(indices, shift)
    indices_a_p = shiftindices(indices_a, shift)
    indices_a_p = ntuple(i -> ifelse(indices_a_p[i] < 1, indices_a_p[i] + gsize[i], indices_a_p[i]), dim)
    indices_a_p = ntuple(i -> ifelse(indices_a_p[i] > gsize[i], indices_a_p[i] - gsize[i], indices_a_p[i]), dim)
    #=
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]
    ixp = ifelse(ixp < 1, ixp + NX, ixp)
    iyp = ifelse(iyp < 1, iyp + NY, iyp)
    izp = ifelse(izp < 1, izp + NZ, izp)
    itp = ifelse(itp < 1, itp + NT, itp)
    ixp = ifelse(ixp > NX, ixp - NX, ixp)
    iyp = ifelse(iyp > NY, iyp - NY, iyp)
    izp = ifelse(izp > NZ, izp - NZ, izp)
    itp = ifelse(itp > NT, itp - NT, itp)
    shift = shift[1:dim]
    indices_p = (ixp + nw, iyp + nw, izp + nw, itp + nw)[1:dim]
    indices_a_p = (ixp, iyp, izp, itp)[1:dim]
    =#

    M3_p = Shifted_Lattice(M3, shift)
    M2_p = Shifted_Lattice(M2, shift)
    a2_p = A2[:, :, indices_a_p...]
    a3_p = A3[:, :, indices_a_p...]

    mul!(a1, a2, a3_p)
    mul!(M1, M2, M3_p)
    m1 = M1.A[:, :, indices...]
    if myrank == 0
        @test a1 ≈ Array(m1) atol = 1e-6
    end

    mul!(a1, a2_p, a3)
    mul!(M1, M2_p, M3)
    m1 = M1.A[:, :, indices...]
    if myrank == 0
        @test a1 ≈ Array(m1) atol = 1e-6
    end

    mul!(a1, a2_p, a3_p)
    mul!(M1, M2_p, M3_p)
    m1 = M1.A[:, :, indices...]
    if myrank == 0
        @test a1 ≈ Array(m1) atol = 1e-6
    end

    mul!(a1, a2', a3_p)
    mul!(M1, M2', M3_p)
    m1 = M1.A[:, :, indices...]
    if myrank == 0
        @test a1 ≈ Array(m1) atol = 1e-6
    end

    mul!(a1, a2_p', a3)
    mul!(M1, M2_p', M3)
    m1 = M1.A[:, :, indices...]
    if myrank == 0
        @test a1 ≈ Array(m1) atol = 1e-6
    end

    mul!(a1, a2_p', a3_p)
    mul!(M1, M2_p', M3_p)
    m1 = M1.A[:, :, indices...]
    if myrank == 0
        @test a1 ≈ Array(m1) atol = 1e-6
    end

    mul!(a1, a2, a3_p')
    mul!(M1, M2, M3_p')
    m1 = M1.A[:, :, indices...]
    if myrank == 0
        @test a1 ≈ Array(m1) atol = 1e-6
    end

    mul!(a1, a2_p, a3')
    mul!(M1, M2_p, M3')
    m1 = M1.A[:, :, indices...]
    if myrank == 0
        @test a1 ≈ Array(m1) atol = 1e-6
    end

    mul!(a1, a2_p, a3_p')
    mul!(M1, M2_p, M3_p')
    m1 = M1.A[:, :, indices...]
    if myrank == 0
        @test a1 ≈ Array(m1) atol = 1e-6
    end

    mul!(a1, a2', a3_p')
    mul!(M1, M2', M3_p')
    m1 = M1.A[:, :, indices...]
    if myrank == 0
        @test a1 ≈ Array(m1) atol = 1e-6
    end

    mul!(a1, a2_p', a3')
    mul!(M1, M2_p', M3')
    m1 = M1.A[:, :, indices...]
    if myrank == 0
        @test a1 ≈ Array(m1) atol = 1e-6
    end

    mul!(a1, a2_p', a3_p')
    mul!(M1, M2_p', M3_p')
    m1 = M1.A[:, :, indices...]
    if myrank == 0
        @test a1 ≈ Array(m1) atol = 1e-6
    end





end

function indextest(dim)
    NX = 4
    gsize = ntuple(_ -> NX, dim)

    #=
    if dim == 1
        gsize = (NX,)
    elseif dim == 2
        gsize = (NX, NY)
    elseif dim == 3
        gsize = (NX, NY, NZ)
    elseif dim == 4
        gsize = (NX, NY, NZ, NT)
    else
        error("dim should be smaller than 5.")
    end
    =#

    d = DIndexer(gsize)
    println(d)
    N = prod(gsize)
    for i = 1:N
        indices = delinearize(d, i)
        println(indices)
        L = linearize(d, indices)
        println(L)
    end
end

function main()
    MPI.Init()
    #=
    for dim = 1:5
        indextest(dim)
    end
    =#

    for dim = 2:4
        for NC = 2:4
            @testset "NC = $NC, dim = $dim" begin
                println("NC = $NC, dim = $dim")
                multtest(NC, dim)
                @time multtest(NC, dim)
            end
        end
    end
end

@testset "LatticeMatrices.jl" begin

    # Write your tests here.
    #latticetest4D()
    #normalizetest()
    main()

end


