function add!(b, Y, a, X) #b*Y + a*X -> Y
    LinearAlgebra.axpby!(a, X, b, Y) #X*a + Y*b -> Y
end

function add!(Y, a, X) #Y + a*X -> Y
    LinearAlgebra.axpby!(a, X, 1, Y) #X*a + Y -> Y
end




function cg(x, A, b, temps; eps=1e-10, maxsteps=5000, verboselevel=2) #Ax=b
    #temps = get_blockoraryvectors_forCG(A)

    if verboselevel >= 3
        println("--------------------------------------")
        println("cg method")
    end

    bnorm = sqrt(real(b ⋅ b))
    #=
    res = deepcopy(b)
    temp1 = similar(x)
    mul!(temp1,A,x)
    add!(res,-1,temp1)
    q = similar(x)
    p = deepcopy(res)
    =#


    res, it_res = get_block(temps)
    #res = temps[1]
    substitute!(res, b)
    set_halo!(res)
    temp1, it_temp1 = get_block(temps)
    #temp1 = temps[2]
    #println("in CG $(sum(abs.(x.f)))")
    mul!(temp1, A, x)
    set_halo!(temp1)


    add!(res, -1, temp1)
    q, it_q = get_block(temps)
    p, it_p = get_block(temps)
    #q = temps[3]
    #p = temps[4]
    substitute!(p, res)
    set_halo!(p)

    #p = deepcopy(res)

    rnorm = sqrt(real(res ⋅ res)) / bnorm

    #println(rnorm)

    if rnorm < eps
        unused!(temps, it_res)
        unused!(temps, it_temp1)
        unused!(temps, it_q)
        unused!(temps, it_p)
        return
    end

    c1 = p ⋅ p


    for i = 1:maxsteps
        mul!(q, A, p)
        set_halo!(q)
        c2 = real(dot(p, q))
        #println("c2 = $c2")

        #c2 = p ⋅ q

        α = c1 / c2
        #! ...  x   = x   + alpha * p  
        #println("add2")
        add!(x, α, p)
        set_halo!(x)
        #...  res = res - alpha * q 
        #println("add1")
        add!(res, -α, q)
        c3 = res ⋅ res
        rnorm = sqrt(real(c3)) / bnorm
        if verboselevel >= 3
            println("$i-th eps: $rnorm")
        end




        if rnorm < eps
            #println("$i eps: $eps rnorm $rnorm")
            if verboselevel >= 3
                println("Converged at $i-th step. eps: $rnorm")
                println("--------------------------------------")
            end
            unused!(temps, it_res)
            unused!(temps, it_temp1)
            unused!(temps, it_q)
            unused!(temps, it_p)
            return
        end

        β = c3 / c1
        c1 = c3

        #println("add3")
        add!(β, p, 1, res) #p = beta*p + s
        set_halo!(p)

    end



    error("""
    The CG is not converged! with maxsteps = $(maxsteps)
    residual is $rnorm
    maxsteps should be larger.""")


end
