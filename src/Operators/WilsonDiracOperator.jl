
struct WilsonDiracOperator4D{T} <: OperatorOnKernel
    U::Vector{T}
    κ::Base.RefValue{Float64}

    function WilsonDiracOperator4D(U::Vector{T}, κ) where {T<:LatticeMatrix}
        @assert length(U) == 4 "U must be a vector of length 4."
        return new{T}(U, Ref(Float64(κ)))
    end
end

@inline function Base.getproperty(operator::WilsonDiracOperator4D, name::Symbol)
    name === :κ && return getfield(operator, :κ)[]
    return getfield(operator, name)
end

"""
    WilsonDiracOperator4D(U1, U2, U3, U4, κ)

Construct a Wilson operator from four explicit links.  This form is also the
AD-safe constructor for callbacks whose link arguments are differentiated:
it preserves the concrete link type while assembling the internal collection.
"""
function WilsonDiracOperator4D(
    U1::T, U2::T, U3::T, U4::T, κ,
) where {T<:LatticeMatrix}
    links = Vector{T}(undef, 4)
    links[1] = U1
    links[2] = U2
    links[3] = U3
    links[4] = U4
    return WilsonDiracOperator4D(links, κ)
end

export WilsonDiracOperator4D

@inline function _ensure_wilson_halo!(U, psi)
    ensure_halo!(U[1])
    ensure_halo!(U[2])
    ensure_halo!(U[3])
    ensure_halo!(U[4])
    ensure_halo!(psi)
    return nothing
end

struct Adjoint_WilsonDiracOperator4D{T} <: OperatorOnKernel
    parent::T
end

function Base.adjoint(A::T) where {T<:WilsonDiracOperator4D}
    Adjoint_WilsonDiracOperator4D{typeof(A)}(A)
end
Base.adjoint(A::Adjoint_WilsonDiracOperator4D) = A.parent



"""
ψ_n - κ sum_ν U_n[ν](1 - γν)*ψ_{n+ν} + U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
"""
function LinearAlgebra.mul!(C::TC,
    Dirac::TD, ψ::TC) where {T1,AT1,NC1,nw,DI,
    TC<:LatticeMatrix{4,T1,AT1,NC1,4,nw,DI},TD<:WilsonDiracOperator4D}

    _ensure_wilson_halo!(Dirac.U, ψ)
    U1 = get_matrix(Dirac.U[1])
    U2 = get_matrix(Dirac.U[2])
    U3 = get_matrix(Dirac.U[3])
    U4 = get_matrix(Dirac.U[4])
    ψdata = get_matrix(ψ)
    Cdata = get_matrix(C)

    _parallel_for_mutating!(C,
        prod(C.PN), kernel_WilsonDiracOperator4D!, Cdata, U1, U2, U3, U4, Dirac.κ, ψdata,
        Val(NC1), Val(nw), C.indexer)

end



function kernel_WilsonDiracOperator4D!(i, C, U1, U2, U3, U4, κ, ψdata, ::Val{NC1}, ::Val{nw}, dindexer) where {NC1,nw}
    indices = delinearize(dindexer, i, nw)
    indices_1p = shiftindices(indices, shift_1p)
    indices_1m = shiftindices(indices, shift_1m)
    indices_2p = shiftindices(indices, shift_2p)
    indices_2m = shiftindices(indices, shift_2m)
    indices_3p = shiftindices(indices, shift_3p)
    indices_3m = shiftindices(indices, shift_3m)
    indices_4p = shiftindices(indices, shift_4p)
    indices_4m = shiftindices(indices, shift_4m)


    @inbounds for ic = 1:NC1
        for ia = 1:4
            C[ic, ia, indices...] = ψdata[ic, ia, indices...]
        end
    end

    @inbounds for ic = 1:NC1
        for jc = 1:NC1
            #U_n[ν](1 - γν)*ψ_{n+ν} 

            v = mul_op(oneminusγ1, ψdata, jc, indices_1p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U1[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneminusγ2, ψdata, jc, indices_2p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U2[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneminusγ3, ψdata, jc, indices_3p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U3[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneminusγ4, ψdata, jc, indices_4p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U4[ic, jc, indices...] * v[ia]
            end


            # U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
            v = mul_op(oneplusγ1, ψdata, jc, indices_1m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U1[jc, ic, indices_1m...]' * v[ia]
            end

            v = mul_op(oneplusγ2, ψdata, jc, indices_2m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U2[jc, ic, indices_2m...]' * v[ia]
            end

            v = mul_op(oneplusγ3, ψdata, jc, indices_3m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U3[jc, ic, indices_3m...]' * v[ia]
            end


            v = mul_op(oneplusγ4, ψdata, jc, indices_4m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U4[jc, ic, indices_4m...]' * v[ia]
            end


        end
    end


end

@inline function muladdmulti(a1, b1, a2, b2, a3, b3)
    acc = zero(typeof(a1))
    acc = muladd(a1, b1, acc)
    acc = muladd(a2, b2, acc)
    acc = muladd(a3, b3, acc)
    return acc
end

@inline function _wilson_half_project3(ψdata, color, indices,
    ::Val{PM}, ::Val{MU}) where {PM,MU}

    @inbounds begin
        ψ1 = ψdata[color, 1, indices...]
        ψ2 = ψdata[color, 2, indices...]
        ψ3 = ψdata[color, 3, indices...]
        ψ4 = ψdata[color, 4, indices...]
    end
    if MU == 1
        return ψ1 - PM * im * ψ4, ψ2 - PM * im * ψ3
    elseif MU == 2
        return ψ1 - PM * ψ4, ψ2 + PM * ψ3
    elseif MU == 3
        return ψ1 - PM * im * ψ3, ψ2 + PM * im * ψ4
    else
        return ψ1 - PM * ψ3, ψ2 - PM * ψ4
    end
end

@inline function _wilson_half_matvec_forward3(
    U, ψdata, indices, indices_p, pm, mu,
)
    h11, h12 = _wilson_half_project3(ψdata, 1, indices_p, pm, mu)
    h21, h22 = _wilson_half_project3(ψdata, 2, indices_p, pm, mu)
    h31, h32 = _wilson_half_project3(ψdata, 3, indices_p, pm, mu)
    @inbounds begin
        U11 = U[1, 1, indices...]
        U12 = U[1, 2, indices...]
        U13 = U[1, 3, indices...]
        U21 = U[2, 1, indices...]
        U22 = U[2, 2, indices...]
        U23 = U[2, 3, indices...]
        U31 = U[3, 1, indices...]
        U32 = U[3, 2, indices...]
        U33 = U[3, 3, indices...]
    end
    return (
        muladdmulti(U11, h11, U12, h21, U13, h31),
        muladdmulti(U11, h12, U12, h22, U13, h32),
        muladdmulti(U21, h11, U22, h21, U23, h31),
        muladdmulti(U21, h12, U22, h22, U23, h32),
        muladdmulti(U31, h11, U32, h21, U33, h31),
        muladdmulti(U31, h12, U32, h22, U33, h32),
    )
end

@inline function _wilson_half_matvec_backward3(
    U, ψdata, indices_m, pm, mu,
)
    h11, h12 = _wilson_half_project3(ψdata, 1, indices_m, pm, mu)
    h21, h22 = _wilson_half_project3(ψdata, 2, indices_m, pm, mu)
    h31, h32 = _wilson_half_project3(ψdata, 3, indices_m, pm, mu)
    @inbounds begin
        U11 = conj(U[1, 1, indices_m...])
        U12 = conj(U[2, 1, indices_m...])
        U13 = conj(U[3, 1, indices_m...])
        U21 = conj(U[1, 2, indices_m...])
        U22 = conj(U[2, 2, indices_m...])
        U23 = conj(U[3, 2, indices_m...])
        U31 = conj(U[1, 3, indices_m...])
        U32 = conj(U[2, 3, indices_m...])
        U33 = conj(U[3, 3, indices_m...])
    end
    return (
        muladdmulti(U11, h11, U12, h21, U13, h31),
        muladdmulti(U11, h12, U12, h22, U13, h32),
        muladdmulti(U21, h11, U22, h21, U23, h31),
        muladdmulti(U21, h12, U22, h22, U23, h32),
        muladdmulti(U31, h11, U32, h21, U33, h31),
        muladdmulti(U31, h12, U32, h22, U33, h32),
    )
end

@inline function _wilson_reconstruct_add3(
    accumulator, halfspinor, ::Val{PM}, ::Val{MU},
) where {PM,MU}
    c11, c12, c13, c14,
    c21, c22, c23, c24,
    c31, c32, c33, c34 = accumulator
    h11, h12, h21, h22, h31, h32 = halfspinor
    if MU == 1
        return (
            c11 + h11, c12 + h12, c13 + PM * im * h12, c14 + PM * im * h11,
            c21 + h21, c22 + h22, c23 + PM * im * h22, c24 + PM * im * h21,
            c31 + h31, c32 + h32, c33 + PM * im * h32, c34 + PM * im * h31,
        )
    elseif MU == 2
        return (
            c11 + h11, c12 + h12, c13 + PM * h12, c14 - PM * h11,
            c21 + h21, c22 + h22, c23 + PM * h22, c24 - PM * h21,
            c31 + h31, c32 + h32, c33 + PM * h32, c34 - PM * h31,
        )
    elseif MU == 3
        return (
            c11 + h11, c12 + h12, c13 + PM * im * h11, c14 - PM * im * h12,
            c21 + h21, c22 + h22, c23 + PM * im * h21, c24 - PM * im * h22,
            c31 + h31, c32 + h32, c33 + PM * im * h31, c34 - PM * im * h32,
        )
    else
        return (
            c11 + h11, c12 + h12, c13 - PM * h11, c14 - PM * h12,
            c21 + h21, c22 + h22, c23 - PM * h21, c24 - PM * h22,
            c31 + h31, c32 + h32, c33 - PM * h31, c34 - PM * h32,
        )
    end
end

@inline function _wilson_hopping_accumulator3(
    U1, U2, U3, U4, ψdata, indices, ::Val{FORWARD_PM},
) where FORWARD_PM
    zero_value = zero(@inbounds ψdata[1, 1, indices...])
    accumulator = (
        zero_value, zero_value, zero_value, zero_value,
        zero_value, zero_value, zero_value, zero_value,
        zero_value, zero_value, zero_value, zero_value,
    )
    forward_pm = Val(FORWARD_PM)
    backward_pm = Val(-FORWARD_PM)

    indices_p = shiftindices(indices, shift_1p)
    indices_m = shiftindices(indices, shift_1m)
    accumulator = _wilson_reconstruct_add3(accumulator,
        _wilson_half_matvec_forward3(
            U1, ψdata, indices, indices_p, forward_pm, Val(1)),
        forward_pm, Val(1))
    accumulator = _wilson_reconstruct_add3(accumulator,
        _wilson_half_matvec_backward3(
            U1, ψdata, indices_m, backward_pm, Val(1)),
        backward_pm, Val(1))

    indices_p = shiftindices(indices, shift_2p)
    indices_m = shiftindices(indices, shift_2m)
    accumulator = _wilson_reconstruct_add3(accumulator,
        _wilson_half_matvec_forward3(
            U2, ψdata, indices, indices_p, forward_pm, Val(2)),
        forward_pm, Val(2))
    accumulator = _wilson_reconstruct_add3(accumulator,
        _wilson_half_matvec_backward3(
            U2, ψdata, indices_m, backward_pm, Val(2)),
        backward_pm, Val(2))

    indices_p = shiftindices(indices, shift_3p)
    indices_m = shiftindices(indices, shift_3m)
    accumulator = _wilson_reconstruct_add3(accumulator,
        _wilson_half_matvec_forward3(
            U3, ψdata, indices, indices_p, forward_pm, Val(3)),
        forward_pm, Val(3))
    accumulator = _wilson_reconstruct_add3(accumulator,
        _wilson_half_matvec_backward3(
            U3, ψdata, indices_m, backward_pm, Val(3)),
        backward_pm, Val(3))

    indices_p = shiftindices(indices, shift_4p)
    indices_m = shiftindices(indices, shift_4m)
    accumulator = _wilson_reconstruct_add3(accumulator,
        _wilson_half_matvec_forward3(
            U4, ψdata, indices, indices_p, forward_pm, Val(4)),
        forward_pm, Val(4))
    return _wilson_reconstruct_add3(accumulator,
        _wilson_half_matvec_backward3(
            U4, ψdata, indices_m, backward_pm, Val(4)),
        backward_pm, Val(4))
end

@inline function _write_wilson_result3!(
    C, ψdata, indices, hopping, coefficient, ::Val{COPY_INPUT},
) where COPY_INPUT
    @inbounds begin
        C[1, 1, indices...] =
            (COPY_INPUT ? ψdata[1, 1, indices...] : zero(hopping[1])) + coefficient * hopping[1]
        C[1, 2, indices...] =
            (COPY_INPUT ? ψdata[1, 2, indices...] : zero(hopping[2])) + coefficient * hopping[2]
        C[1, 3, indices...] =
            (COPY_INPUT ? ψdata[1, 3, indices...] : zero(hopping[3])) + coefficient * hopping[3]
        C[1, 4, indices...] =
            (COPY_INPUT ? ψdata[1, 4, indices...] : zero(hopping[4])) + coefficient * hopping[4]
        C[2, 1, indices...] =
            (COPY_INPUT ? ψdata[2, 1, indices...] : zero(hopping[5])) + coefficient * hopping[5]
        C[2, 2, indices...] =
            (COPY_INPUT ? ψdata[2, 2, indices...] : zero(hopping[6])) + coefficient * hopping[6]
        C[2, 3, indices...] =
            (COPY_INPUT ? ψdata[2, 3, indices...] : zero(hopping[7])) + coefficient * hopping[7]
        C[2, 4, indices...] =
            (COPY_INPUT ? ψdata[2, 4, indices...] : zero(hopping[8])) + coefficient * hopping[8]
        C[3, 1, indices...] =
            (COPY_INPUT ? ψdata[3, 1, indices...] : zero(hopping[9])) + coefficient * hopping[9]
        C[3, 2, indices...] =
            (COPY_INPUT ? ψdata[3, 2, indices...] : zero(hopping[10])) + coefficient * hopping[10]
        C[3, 3, indices...] =
            (COPY_INPUT ? ψdata[3, 3, indices...] : zero(hopping[11])) + coefficient * hopping[11]
        C[3, 4, indices...] =
            (COPY_INPUT ? ψdata[3, 4, indices...] : zero(hopping[12])) + coefficient * hopping[12]
    end
    return nothing
end

@inline function _wilson_halfspin_link_pullback_row3!(
    dU, dresult, ψdata, indices, coefficient,
    plus_psi, minus_dresult, ::Val{ROW}, ::Val{MU},
) where {ROW,MU}
    dplus1, dplus2 = _wilson_half_project3(
        dresult, ROW, indices, Val(-1), Val(MU))
    minus_psi1, minus_psi2 = _wilson_half_project3(
        ψdata, ROW, indices, Val(1), Val(MU))
    @inbounds begin
        value1 = dplus1 * conj(plus_psi[1]) +
                 dplus2 * conj(plus_psi[2]) +
                 minus_psi1 * conj(minus_dresult[1]) +
                 minus_psi2 * conj(minus_dresult[2])
        value2 = dplus1 * conj(plus_psi[3]) +
                 dplus2 * conj(plus_psi[4]) +
                 minus_psi1 * conj(minus_dresult[3]) +
                 minus_psi2 * conj(minus_dresult[4])
        value3 = dplus1 * conj(plus_psi[5]) +
                 dplus2 * conj(plus_psi[6]) +
                 minus_psi1 * conj(minus_dresult[5]) +
                 minus_psi2 * conj(minus_dresult[6])
        dU[ROW, 1, indices...] += coefficient * value1
        dU[ROW, 2, indices...] += coefficient * value2
        dU[ROW, 3, indices...] += coefficient * value3
    end
    return nothing
end

@inline function _wilson_halfspin_link_pullback_direction3!(
    dU, dresult, ψdata, indices, indices_plus, coefficient, ::Val{MU},
) where MU
    plus11, plus12 = _wilson_half_project3(
        ψdata, 1, indices_plus, Val(-1), Val(MU))
    plus21, plus22 = _wilson_half_project3(
        ψdata, 2, indices_plus, Val(-1), Val(MU))
    plus31, plus32 = _wilson_half_project3(
        ψdata, 3, indices_plus, Val(-1), Val(MU))
    minus_dresult11, minus_dresult12 = _wilson_half_project3(
        dresult, 1, indices_plus, Val(1), Val(MU))
    minus_dresult21, minus_dresult22 = _wilson_half_project3(
        dresult, 2, indices_plus, Val(1), Val(MU))
    minus_dresult31, minus_dresult32 = _wilson_half_project3(
        dresult, 3, indices_plus, Val(1), Val(MU))
    plus_psi = (plus11, plus12, plus21, plus22, plus31, plus32)
    minus_dresult = (
        minus_dresult11, minus_dresult12,
        minus_dresult21, minus_dresult22,
        minus_dresult31, minus_dresult32,
    )
    _wilson_halfspin_link_pullback_row3!(dU, dresult, ψdata, indices,
        coefficient, plus_psi, minus_dresult, Val(1), Val(MU))
    _wilson_halfspin_link_pullback_row3!(dU, dresult, ψdata, indices,
        coefficient, plus_psi, minus_dresult, Val(2), Val(MU))
    _wilson_halfspin_link_pullback_row3!(dU, dresult, ψdata, indices,
        coefficient, plus_psi, minus_dresult, Val(3), Val(MU))
    return nothing
end


function kernel_WilsonDiracOperator4D!(i, C, U1, U2, U3, U4, κ, ψdata, ::Val{3}, ::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)
    hopping = _wilson_hopping_accumulator3(
        U1, U2, U3, U4, ψdata, indices, Val(-1))
    _write_wilson_result3!(C, ψdata, indices, hopping, -κ, Val(true))
    return nothing
end


"""
ψ_n - κ sum_ν U_n[ν](1 + γν)*ψ_{n+ν} + U_{n-ν}[-ν]^+ (1 - γν)*ψ_{n-ν}
"""
function LinearAlgebra.mul!(C::TC,
    Dirac::TD, ψ::TC) where {T1,AT1,NC1,nw,DI,
    TC<:LatticeMatrix{4,T1,AT1,NC1,4,nw,DI},TD<:Adjoint_WilsonDiracOperator4D}

    _ensure_wilson_halo!(Dirac.parent.U, ψ)
    U1 = get_matrix(Dirac.parent.U[1])
    U2 = get_matrix(Dirac.parent.U[2])
    U3 = get_matrix(Dirac.parent.U[3])
    U4 = get_matrix(Dirac.parent.U[4])
    ψdata = get_matrix(ψ)
    Cdata = get_matrix(C)

    _parallel_for_mutating!(C,
        prod(C.PN), kernel_adjoint_WilsonDiracOperator4D!, Cdata, U1, U2, U3, U4, Dirac.parent.κ, ψdata,
        Val(NC1), Val(nw), C.indexer)

end


function kernel_adjoint_WilsonDiracOperator4D!(i, C, U1, U2, U3, U4, κ, ψdata, ::Val{NC1}, ::Val{nw}, dindexer) where {NC1,nw}
    indices = delinearize(dindexer, i, nw)
    indices_1p = shiftindices(indices, shift_1p)
    indices_1m = shiftindices(indices, shift_1m)
    indices_2p = shiftindices(indices, shift_2p)
    indices_2m = shiftindices(indices, shift_2m)
    indices_3p = shiftindices(indices, shift_3p)
    indices_3m = shiftindices(indices, shift_3m)
    indices_4p = shiftindices(indices, shift_4p)
    indices_4m = shiftindices(indices, shift_4m)


    @inbounds for ic = 1:NC1
        for ia = 1:4
            C[ic, ia, indices...] = ψdata[ic, ia, indices...]
        end
    end

    @inbounds for ic = 1:NC1
        for jc = 1:NC1
            #U_n[ν](1 - γν)*ψ_{n+ν} 

            v = mul_op(oneplusγ1, ψdata, jc, indices_1p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U1[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneplusγ2, ψdata, jc, indices_2p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U2[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneplusγ3, ψdata, jc, indices_3p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U3[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneplusγ4, ψdata, jc, indices_4p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U4[ic, jc, indices...] * v[ia]
            end


            # U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
            v = mul_op(oneminusγ1, ψdata, jc, indices_1m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U1[jc, ic, indices_1m...]' * v[ia]
            end

            v = mul_op(oneminusγ2, ψdata, jc, indices_2m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U2[jc, ic, indices_2m...]' * v[ia]
            end

            v = mul_op(oneminusγ3, ψdata, jc, indices_3m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U3[jc, ic, indices_3m...]' * v[ia]
            end


            v = mul_op(oneminusγ4, ψdata, jc, indices_4m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U4[jc, ic, indices_4m...]' * v[ia]
            end


        end
    end


end


function kernel_adjoint_WilsonDiracOperator4D!(i, C, U1, U2, U3, U4, κ, ψdata, ::Val{3}, ::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)
    hopping = _wilson_hopping_accumulator3(
        U1, U2, U3, U4, ψdata, indices, Val(1))
    _write_wilson_result3!(C, ψdata, indices, hopping, -κ, Val(true))
    return nothing
end

struct WilsonDiracOperator4D_Donly{T} <: OperatorOnKernel
    U::Vector{T}

    function WilsonDiracOperator4D_Donly(U::Vector{T}) where {T<:LatticeMatrix}
        @assert length(U) == 4 "U must be a vector of length 4."
        return new{T}(U)
    end
end
export WilsonDiracOperator4D_Donly

"""
0.5 sum_ν U_n[ν](1 - γν)*ψ_{n+ν} + U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
"""
function LinearAlgebra.mul!(C::TC,
    Dirac::TD, ψ::TC) where {T1,AT1,NC1,nw,DI,
    TC<:LatticeMatrix{4,T1,AT1,NC1,4,nw,DI},TD<:WilsonDiracOperator4D_Donly}

    _ensure_wilson_halo!(Dirac.U, ψ)
    U1 = get_matrix(Dirac.U[1])
    U2 = get_matrix(Dirac.U[2])
    U3 = get_matrix(Dirac.U[3])
    U4 = get_matrix(Dirac.U[4])
    ψdata = get_matrix(ψ)
    Cdata = get_matrix(C)

    _parallel_for_mutating!(C,
        prod(C.PN), kernel_WilsonDiracOperator4D_Donly!, Cdata, U1, U2, U3, U4, ψdata,
        Val(NC1), Val(nw), C.indexer)

end


function kernel_WilsonDiracOperator4D_Donly!(i, C, U1, U2, U3, U4, ψdata, ::Val{NC1}, ::Val{nw}, dindexer) where {NC1,nw}
    indices = delinearize(dindexer, i, nw)
    indices_1p = shiftindices(indices, shift_1p)
    indices_1m = shiftindices(indices, shift_1m)
    indices_2p = shiftindices(indices, shift_2p)
    indices_2m = shiftindices(indices, shift_2m)
    indices_3p = shiftindices(indices, shift_3p)
    indices_3m = shiftindices(indices, shift_3m)
    indices_4p = shiftindices(indices, shift_4p)
    indices_4m = shiftindices(indices, shift_4m)


    @inbounds for ic = 1:NC1
        for ia = 1:4
            C[ic, ia, indices...] = zero(ψdata[ic, ia, indices...])
        end
    end

    @inbounds for ic = 1:NC1
        for jc = 1:NC1
            #U_n[ν](1 - γν)*ψ_{n+ν} 

            v = mul_op(oneminusγ1, ψdata, jc, indices_1p)
            for ia = 1:4
                C[ic, ia, indices...] += 0.5 * U1[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneminusγ2, ψdata, jc, indices_2p)
            for ia = 1:4
                C[ic, ia, indices...] += 0.5 * U2[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneminusγ3, ψdata, jc, indices_3p)
            for ia = 1:4
                C[ic, ia, indices...] += 0.5 * U3[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneminusγ4, ψdata, jc, indices_4p)
            for ia = 1:4
                C[ic, ia, indices...] += 0.5 * U4[ic, jc, indices...] * v[ia]
            end


            # U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
            v = mul_op(oneplusγ1, ψdata, jc, indices_1m)
            for ia = 1:4
                C[ic, ia, indices...] += 0.5 * U1[jc, ic, indices_1m...]' * v[ia]
            end

            v = mul_op(oneplusγ2, ψdata, jc, indices_2m)
            for ia = 1:4
                C[ic, ia, indices...] += 0.5 * U2[jc, ic, indices_2m...]' * v[ia]
            end

            v = mul_op(oneplusγ3, ψdata, jc, indices_3m)
            for ia = 1:4
                C[ic, ia, indices...] += 0.5 * U3[jc, ic, indices_3m...]' * v[ia]
            end


            v = mul_op(oneplusγ4, ψdata, jc, indices_4m)
            for ia = 1:4
                C[ic, ia, indices...] += 0.5 * U4[jc, ic, indices_4m...]' * v[ia]
            end


        end
    end


end


function kernel_WilsonDiracOperator4D_Donly!(i, C, U1, U2, U3, U4, ψdata, ::Val{3}, ::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)
    hopping = _wilson_hopping_accumulator3(
        U1, U2, U3, U4, ψdata, indices, Val(-1))
    half = one(real(hopping[1])) / 2
    _write_wilson_result3!(C, ψdata, indices, hopping, half, Val(false))
    return nothing
end

struct Adjoint_WilsonDiracOperator4D_Donly{T} <: OperatorOnKernel
    parent::T
end

function Base.adjoint(A::T) where {T<:WilsonDiracOperator4D_Donly}
    Adjoint_WilsonDiracOperator4D_Donly{typeof(A)}(A)
end
Base.adjoint(A::Adjoint_WilsonDiracOperator4D_Donly) = A.parent


"""
0.5 sum_ν U_n[ν](1 + γν)*ψ_{n+ν} + U_{n-ν}[-ν]^+ (1 - γν)*ψ_{n-ν}
"""
function LinearAlgebra.mul!(C::TC,
    Dirac::TD, ψ::TC) where {T1,AT1,NC1,nw,DI,
    TC<:LatticeMatrix{4,T1,AT1,NC1,4,nw,DI},TD<:Adjoint_WilsonDiracOperator4D_Donly}

    _ensure_wilson_halo!(Dirac.parent.U, ψ)
    U1 = get_matrix(Dirac.parent.U[1])
    U2 = get_matrix(Dirac.parent.U[2])
    U3 = get_matrix(Dirac.parent.U[3])
    U4 = get_matrix(Dirac.parent.U[4])
    ψdata = get_matrix(ψ)
    Cdata = get_matrix(C)

    _parallel_for_mutating!(C,
        prod(C.PN), kernel_adjoint_WilsonDiracOperator4D_Donly!, Cdata, U1, U2, U3, U4, ψdata,
        Val(NC1), Val(nw), C.indexer)

end


function kernel_adjoint_WilsonDiracOperator4D_Donly!(i, C, U1, U2, U3, U4, ψdata, ::Val{NC1}, ::Val{nw}, dindexer) where {NC1,nw}
    indices = delinearize(dindexer, i, nw)
    indices_1p = shiftindices(indices, shift_1p)
    indices_1m = shiftindices(indices, shift_1m)
    indices_2p = shiftindices(indices, shift_2p)
    indices_2m = shiftindices(indices, shift_2m)
    indices_3p = shiftindices(indices, shift_3p)
    indices_3m = shiftindices(indices, shift_3m)
    indices_4p = shiftindices(indices, shift_4p)
    indices_4m = shiftindices(indices, shift_4m)


    @inbounds for ic = 1:NC1
        for ia = 1:4
            C[ic, ia, indices...] = zero(ψdata[ic, ia, indices...])
        end
    end

    κ = -0.5
    @inbounds for ic = 1:NC1
        for jc = 1:NC1
            #U_n[ν](1 - γν)*ψ_{n+ν} 

            v = mul_op(oneplusγ1, ψdata, jc, indices_1p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U1[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneplusγ2, ψdata, jc, indices_2p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U2[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneplusγ3, ψdata, jc, indices_3p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U3[ic, jc, indices...] * v[ia]
            end
            v = mul_op(oneplusγ4, ψdata, jc, indices_4p)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U4[ic, jc, indices...] * v[ia]
            end


            # U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
            v = mul_op(oneminusγ1, ψdata, jc, indices_1m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U1[jc, ic, indices_1m...]' * v[ia]
            end

            v = mul_op(oneminusγ2, ψdata, jc, indices_2m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U2[jc, ic, indices_2m...]' * v[ia]
            end

            v = mul_op(oneminusγ3, ψdata, jc, indices_3m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U3[jc, ic, indices_3m...]' * v[ia]
            end


            v = mul_op(oneminusγ4, ψdata, jc, indices_4m)
            for ia = 1:4
                C[ic, ia, indices...] += -κ * U4[jc, ic, indices_4m...]' * v[ia]
            end


        end
    end


end


function kernel_adjoint_WilsonDiracOperator4D_Donly!(i, C, U1, U2, U3, U4, ψdata, ::Val{3}, ::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)
    hopping = _wilson_hopping_accumulator3(
        U1, U2, U3, U4, ψdata, indices, Val(1))
    half = one(real(hopping[1])) / 2
    _write_wilson_result3!(C, ψdata, indices, hopping, half, Val(false))
    return nothing
end

# ---------------------------------------------------------------------------
# Halo-free Wilson kernels (nw == 0)
# ---------------------------------------------------------------------------

@inline function kernel_initialize_WilsonDiracOperator4D_nowing!(
    i, C, ψ, ::Val{NC1}, dindexer, ::Val{copy_input}) where {NC1,copy_input}
    indices = delinearize(dindexer, i, 0)
    @inbounds for ia in 1:4
        for ic in 1:NC1
            value = ψ[ic, ia, indices...]
            C[ic, ia, indices...] = copy_input ? value : zero(value)
        end
    end
    return nothing
end

@inline function kernel_WilsonDiracOperator4D_direction_nowing!(
    i, C, U, Uminus, ψplus, ψminus, coefficient,
    ::Val{NC1}, dindexer, op_plus, op_minus) where NC1
    indices = delinearize(dindexer, i, 0)

    @inbounds for ic in 1:NC1
        for jc in 1:NC1
            vplus = mul_op(op_plus, ψplus, jc, indices)
            vminus = mul_op(op_minus, ψminus, jc, indices)
            uplus = U[ic, jc, indices...]
            uminus = Uminus[jc, ic, indices...]'
            for ia in 1:4
                C[ic, ia, indices...] += coefficient *
                    (uplus * vplus[ia] + uminus * vminus[ia])
            end
        end
    end
    return nothing
end

function _apply_WilsonDiracOperator4D_nowing!(C, U, ψ, coefficient,
    copy_input::Bool, adjoint_operator::Bool)
    all(u -> iszero(u.nw), U) || throw(ArgumentError(
        "nw=0 Wilson operators require nw=0 gauge fields"))

    _parallel_for_mutating!(C,
        prod(C.PN), kernel_initialize_WilsonDiracOperator4D_nowing!,
        C.A, ψ.A, Val(C.NC1), C.indexer, Val(copy_input))

    plus_operators = adjoint_operator ? oneplusγs : oneminusγs
    minus_operators = adjoint_operator ? oneminusγs : oneplusγs

    for d in 1:4
        ψplus = _materialize_periodic_shift(ψ, shifts_p[d])
        ψminus = _materialize_periodic_shift(ψ, shifts_m[d])
        Uminus = _materialize_periodic_shift(U[d], shifts_m[d])

        _parallel_for_mutating!(C,
            prod(C.PN), kernel_WilsonDiracOperator4D_direction_nowing!,
            C.A, U[d].A, Uminus.A, ψplus.A, ψminus.A, coefficient,
            Val(C.NC1), C.indexer, plus_operators[d], minus_operators[d])
    end
    return C
end

function LinearAlgebra.mul!(C::TC, Dirac::WilsonDiracOperator4D, ψ::TC) where {
    T,AT,NC1,DI,TC<:LatticeMatrix{4,T,AT,NC1,4,0,DI}
}
    return _apply_WilsonDiracOperator4D_nowing!(
        C, Dirac.U, ψ, -Dirac.κ, true, false)
end

function LinearAlgebra.mul!(C::TC, Dirac::Adjoint_WilsonDiracOperator4D, ψ::TC) where {
    T,AT,NC1,DI,TC<:LatticeMatrix{4,T,AT,NC1,4,0,DI}
}
    return _apply_WilsonDiracOperator4D_nowing!(
        C, Dirac.parent.U, ψ, -Dirac.parent.κ, true, true)
end

function LinearAlgebra.mul!(C::TC, Dirac::WilsonDiracOperator4D_Donly, ψ::TC) where {
    T,AT,NC1,DI,TC<:LatticeMatrix{4,T,AT,NC1,4,0,DI}
}
    return _apply_WilsonDiracOperator4D_nowing!(
        C, Dirac.U, ψ, 0.5, false, false)
end

function LinearAlgebra.mul!(C::TC, Dirac::Adjoint_WilsonDiracOperator4D_Donly, ψ::TC) where {
    T,AT,NC1,DI,TC<:LatticeMatrix{4,T,AT,NC1,4,0,DI}
}
    return _apply_WilsonDiracOperator4D_nowing!(
        C, Dirac.parent.U, ψ, 0.5, false, true)
end
