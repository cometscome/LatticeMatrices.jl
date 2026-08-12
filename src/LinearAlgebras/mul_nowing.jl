# Fused periodic shifts for halo-free lattice matrices.
#
# `_lazy_shift_nowing` creates a private wrapper for these kernels. They map
# the shifted site while loading each operand, avoiding a full-volume shifted
# temporary. Public `Shifted_Lattice` and `shift_L` remain materialized so
# unsupported operations cannot accidentally use halo-based indexing.

@inline function _periodic_index_nowing(index, shift, size)
    shifted = index + shift
    if 1 <= shifted <= size
        return shifted
    end
    return mod(shifted - 1, size) + 1
end

@inline function _periodic_site_and_phase_nowing(
    indices, shift, global_size, phases, ::Type{T}, unit_phases::Bool,
) where {T}
    shifted_indices = ntuple(
        d -> _periodic_index_nowing(indices[d], shift[d], global_size[d]),
        length(indices))

    factor = one(T)
    if !unit_phases
        @inbounds for d in eachindex(indices)
            _, wraps = _periodic_shift_index(indices[d], shift[d], global_size[d])
            factor *= phases[d]^wraps
        end
    end
    return shifted_indices, factor
end

@inline _load_left_nowing(A, ic, kc, indices, ::Val{false}) =
    A[ic, kc, indices...]

@inline _load_left_nowing(A, ic, kc, indices, ::Val{true}) =
    conj(A[kc, ic, indices...])

@inline _load_right_nowing(B, kc, jc, indices, ::Val{false}) =
    B[kc, jc, indices...]

@inline _load_right_nowing(B, kc, jc, indices, ::Val{true}) =
    conj(B[jc, kc, indices...])

@inline _adjoint_phase_nowing(factor, ::Val{false}) = factor
@inline _adjoint_phase_nowing(factor, ::Val{true}) = conj(factor)

@inline function kernel_Dmatrix_mul_periodic_nowing!(
    i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, dindexer,
    shiftA, shiftB, global_sizeA, global_sizeB, phasesA, phasesB,
    unit_phasesA, unit_phasesB, adjointA, adjointB, alpha, beta,
) where {NC1,NC2,NC3}
    indices = delinearize(dindexer, i, 0)
    indices_A, factor_A = _periodic_site_and_phase_nowing(
        indices, shiftA, global_sizeA, phasesA, eltype(A), unit_phasesA)
    indices_B, factor_B = _periodic_site_and_phase_nowing(
        indices, shiftB, global_sizeB, phasesB, eltype(B), unit_phasesB)
    factor = _adjoint_phase_nowing(factor_A, adjointA) *
             _adjoint_phase_nowing(factor_B, adjointB)

    @inbounds for jc in 1:NC2
        for ic in 1:NC1
            value = zero(eltype(C))
            for kc in 1:NC3
                value += _load_left_nowing(
                    A, ic, kc, indices_A, adjointA) *
                    _load_right_nowing(
                        B, kc, jc, indices_B, adjointB)
            end
            value *= factor
            if iszero(beta)
                C[ic, jc, indices...] = alpha * value
            else
                C[ic, jc, indices...] =
                    alpha * value + beta * C[ic, jc, indices...]
            end
        end
    end
    return nothing
end

function _mul_periodic_nowing!(
    C::LatticeMatrix{D,T1,AT1,NC1,NC2,0,DI},
    A::LatticeMatrix{D,T2,AT2,NCA1,NCA2,0,DI}, shiftA, adjointA,
    B::LatticeMatrix{D,T3,AT3,NCB1,NCB2,0,DI}, shiftB, adjointB,
    alpha, beta, ::Val{NC3},
) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NCA1,NCA2,NCB1,NCB2,NC3,DI}
    alpha_in = T1(alpha)
    beta_in = T1(beta)
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_periodic_nowing!,
        C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), C.indexer,
        shiftA, shiftB, A.gsize, B.gsize, A.phases, B.phases,
        all(isone, A.phases), all(isone, B.phases),
        adjointA, adjointB, alpha_in, beta_in)
    return C
end

@inline _zero_shift_nowing(::Val{D}) where {D} = ntuple(_ -> 0, D)

# C = A * shift(B)
function LinearAlgebra.mul!(
    C::LatticeMatrix{D,T1,AT1,NC1,NC2,0,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,0,DI},
    B::_LazyShifted_Lattice{L,D},
) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,DI,
    L<:LatticeMatrix{D,T3,AT3,NC3,NC2,0,DI}}
    return _mul_periodic_nowing!(C, A, _zero_shift_nowing(Val(D)), Val(false),
        B.data, get_shift(B), Val(false), one(T1), zero(T1), Val(NC3))
end

function LinearAlgebra.mul!(
    C::LatticeMatrix{D,T1,AT1,NC1,NC2,0,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,0,DI},
    B::_LazyShifted_Lattice{L,D}, alpha::S, beta::S,
) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,DI,S<:Number,
    L<:LatticeMatrix{D,T3,AT3,NC3,NC2,0,DI}}
    return _mul_periodic_nowing!(C, A, _zero_shift_nowing(Val(D)), Val(false),
        B.data, get_shift(B), Val(false), alpha, beta, Val(NC3))
end

# C = A * shift(B)'
function LinearAlgebra.mul!(
    C::LatticeMatrix{D,T1,AT1,NC1,NC2,0,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,0,DI},
    B::Adjoint_Lattice{_LazyShifted_Lattice{L,D}},
) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,DI,
    L<:LatticeMatrix{D,T3,AT3,NC2,NC3,0,DI}}
    shifted_B = B.data
    return _mul_periodic_nowing!(C, A, _zero_shift_nowing(Val(D)), Val(false),
        shifted_B.data, get_shift(B), Val(true), one(T1), zero(T1), Val(NC3))
end

function LinearAlgebra.mul!(
    C::LatticeMatrix{D,T1,AT1,NC1,NC2,0,DI},
    A::LatticeMatrix{D,T2,AT2,NC1,NC3,0,DI},
    B::Adjoint_Lattice{_LazyShifted_Lattice{L,D}}, alpha::S, beta::S,
) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,DI,S<:Number,
    L<:LatticeMatrix{D,T3,AT3,NC2,NC3,0,DI}}
    shifted_B = B.data
    return _mul_periodic_nowing!(C, A, _zero_shift_nowing(Val(D)), Val(false),
        shifted_B.data, get_shift(B), Val(true), alpha, beta, Val(NC3))
end

# C = shift(A) * shift(B)
function LinearAlgebra.mul!(
    C::LatticeMatrix{D,T1,AT1,NC1,NC2,0,DI},
    A::_LazyShifted_Lattice{L1,D}, B::_LazyShifted_Lattice{L2,D},
) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC1,NC3,0,DI},
    L2<:LatticeMatrix{D,T3,AT3,NC3,NC2,0,DI}}
    return _mul_periodic_nowing!(C, A.data, get_shift(A), Val(false),
        B.data, get_shift(B), Val(false), one(T1), zero(T1), Val(NC3))
end

function LinearAlgebra.mul!(
    C::LatticeMatrix{D,T1,AT1,NC1,NC2,0,DI},
    A::_LazyShifted_Lattice{L1,D}, B::_LazyShifted_Lattice{L2,D},
    alpha::S, beta::S,
) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,DI,S<:Number,
    L1<:LatticeMatrix{D,T2,AT2,NC1,NC3,0,DI},
    L2<:LatticeMatrix{D,T3,AT3,NC3,NC2,0,DI}}
    return _mul_periodic_nowing!(C, A.data, get_shift(A), Val(false),
        B.data, get_shift(B), Val(false), alpha, beta, Val(NC3))
end

# C = shift(A)' * shift(B)
function LinearAlgebra.mul!(
    C::LatticeMatrix{D,T1,AT1,NC1,NC2,0,DI},
    A::Adjoint_Lattice{_LazyShifted_Lattice{L1,D}}, B::_LazyShifted_Lattice{L2,D},
) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,0,DI},
    L2<:LatticeMatrix{D,T3,AT3,NC3,NC2,0,DI}}
    shifted_A = A.data
    return _mul_periodic_nowing!(C, shifted_A.data, get_shift(A), Val(true),
        B.data, get_shift(B), Val(false), one(T1), zero(T1), Val(NC3))
end

function LinearAlgebra.mul!(
    C::LatticeMatrix{D,T1,AT1,NC1,NC2,0,DI},
    A::Adjoint_Lattice{_LazyShifted_Lattice{L1,D}}, B::_LazyShifted_Lattice{L2,D},
    alpha::S, beta::S,
) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,DI,S<:Number,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,0,DI},
    L2<:LatticeMatrix{D,T3,AT3,NC3,NC2,0,DI}}
    shifted_A = A.data
    return _mul_periodic_nowing!(C, shifted_A.data, get_shift(A), Val(true),
        B.data, get_shift(B), Val(false), alpha, beta, Val(NC3))
end

# C = shift(A) * shift(B)'
function LinearAlgebra.mul!(
    C::LatticeMatrix{D,T1,AT1,NC1,NC2,0,DI},
    A::_LazyShifted_Lattice{L1,D},
    B::Adjoint_Lattice{_LazyShifted_Lattice{L2,D}},
) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC1,NC3,0,DI},
    L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,0,DI}}
    shifted_B = B.data
    return _mul_periodic_nowing!(C, A.data, get_shift(A), Val(false),
        shifted_B.data, get_shift(B), Val(true), one(T1), zero(T1), Val(NC3))
end

function LinearAlgebra.mul!(
    C::LatticeMatrix{D,T1,AT1,NC1,NC2,0,DI},
    A::_LazyShifted_Lattice{L1,D},
    B::Adjoint_Lattice{_LazyShifted_Lattice{L2,D}}, alpha::S, beta::S,
) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,DI,S<:Number,
    L1<:LatticeMatrix{D,T2,AT2,NC1,NC3,0,DI},
    L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,0,DI}}
    shifted_B = B.data
    return _mul_periodic_nowing!(C, A.data, get_shift(A), Val(false),
        shifted_B.data, get_shift(B), Val(true), alpha, beta, Val(NC3))
end

# C = shift(A)' * shift(B)'
function LinearAlgebra.mul!(
    C::LatticeMatrix{D,T1,AT1,NC1,NC2,0,DI},
    A::Adjoint_Lattice{_LazyShifted_Lattice{L1,D}},
    B::Adjoint_Lattice{_LazyShifted_Lattice{L2,D}},
) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,DI,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,0,DI},
    L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,0,DI}}
    shifted_A = A.data
    shifted_B = B.data
    return _mul_periodic_nowing!(C, shifted_A.data, get_shift(A), Val(true),
        shifted_B.data, get_shift(B), Val(true), one(T1), zero(T1), Val(NC3))
end

function LinearAlgebra.mul!(
    C::LatticeMatrix{D,T1,AT1,NC1,NC2,0,DI},
    A::Adjoint_Lattice{_LazyShifted_Lattice{L1,D}},
    B::Adjoint_Lattice{_LazyShifted_Lattice{L2,D}}, alpha::S, beta::S,
) where {D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,DI,S<:Number,
    L1<:LatticeMatrix{D,T2,AT2,NC3,NC1,0,DI},
    L2<:LatticeMatrix{D,T3,AT3,NC2,NC3,0,DI}}
    shifted_A = A.data
    shifted_B = B.data
    return _mul_periodic_nowing!(C, shifted_A.data, get_shift(A), Val(true),
        shifted_B.data, get_shift(B), Val(true), alpha, beta, Val(NC3))
end

@inline function kernel_Dsubstitute_periodic_nowing!(
    i, C, A, ::Val{NC1}, ::Val{NC2}, dindexer, shift,
    global_size, phases, adjointA,
) where {NC1,NC2}
    indices = delinearize(dindexer, i, 0)
    indices_A, factor_A = _periodic_site_and_phase_nowing(
        indices, shift, global_size, phases, eltype(A), false)

    @inbounds for jc in 1:NC2
        for ic in 1:NC1
            C[ic, jc, indices...] = _load_left_nowing(
                A, ic, jc, indices_A, adjointA) *
                _adjoint_phase_nowing(factor_A, adjointA)
        end
    end
    return nothing
end

function substitute!(
    C::LatticeMatrix{D,T1,AT1,NC1,NC2,0,DI},
    A::_LazyShifted_Lattice{L,D},
) where {D,T1,T2,AT1,AT2,NC1,NC2,DI,
    L<:LatticeMatrix{D,T2,AT2,NC1,NC2,0,DI}}
    JACC.parallel_for(
        prod(C.PN), kernel_Dsubstitute_periodic_nowing!,
        C.A, A.data.A, Val(NC1), Val(NC2), C.indexer, get_shift(A),
        A.data.gsize, A.data.phases, Val(false))
    return C
end


function substitute!(
    C::LatticeMatrix{D,T1,AT1,NC1,NC2,0,DI},
    A::Adjoint_Lattice{_LazyShifted_Lattice{L,D}},
) where {D,T1,T2,AT1,AT2,NC1,NC2,DI,
    L<:LatticeMatrix{D,T2,AT2,NC2,NC1,0,DI}}
    shifted_A = A.data
    JACC.parallel_for(
        prod(C.PN), kernel_Dsubstitute_periodic_nowing!,
        C.A, shifted_A.data.A, Val(NC1), Val(NC2), C.indexer, get_shift(A),
        shifted_A.data.gsize, shifted_A.data.phases, Val(true))
    return C
end
