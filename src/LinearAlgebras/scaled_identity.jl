"""
    ScaledIdentityLattice(colors, scalar)

Represent `scalar(x) * I_colors` without storing the diagonal matrix. The
backing `scalar` is a 1×1 `LatticeMatrix` (or its shifted/adjoint view), held
by reference. It need not be a unit phase or a group center element.

`mul!(C, A, B)` and `mul!(C, B, A)` use scalar multiplication when B has
this type. `shift_L(B, shift)` and `B'` preserve the representation. Release
shifted views with `release!` or use `with_shifted_lattice`. Mutate the
coefficient through `B.scalar`; arbitrary matrix-entry writes are not
supported. `substitute!(dense, B)` explicitly expands to full matrices.

Multiplication may overwrite unshifted, non-adjoint A, but not the scalar
storage or a shifted/adjointed A. Operations mark destination halos dirty.
"""
struct ScaledIdentityLattice{NC,S} <: AbstractLattice
    scalar::S
    function ScaledIdentityLattice(NC::Integer, scalar::S) where {S}
        NC > 0 || throw(ArgumentError("colors must be positive"))
        data, _, dagger = _center_operand(scalar)
        _center_shape(data, dagger) == (1, 1) || throw(DimensionMismatch(
            "ScaledIdentityLattice requires a 1×1 scalar lattice"))
        return new{Int(NC),S}(scalar)
    end
end

Base.adjoint(B::ScaledIdentityLattice{NC}) where {NC} =
    ScaledIdentityLattice(NC, B.scalar')

function shift_L(B::ScaledIdentityLattice{NC,<:LatticeMatrix}, shift::NTuple{D,Int}) where {NC,D}
    return ScaledIdentityLattice(NC, shift_L(B.scalar, shift))
end

function shift_L(B::ScaledIdentityLattice{NC,<:Adjoint_Lattice{<:LatticeMatrix}},
    shift::NTuple{D,Int}) where {NC,D}
    return ScaledIdentityLattice(NC, shift_L(B.scalar.data, shift)')
end

function with_shifted_lattice(f::F, B::ScaledIdentityLattice, shift) where {F}
    shifted = shift_L(B, shift)
    try
        return f(shifted)
    finally
        release!(shifted)
    end
end

release!(B::ScaledIdentityLattice{NC,<:LatticeMatrix}) where {NC} = nothing
release!(B::ScaledIdentityLattice{NC,<:Adjoint_Lattice{<:LatticeMatrix}}) where {NC} = nothing
release!(B::ScaledIdentityLattice) = release!(B.scalar)
Base.isopen(B::ScaledIdentityLattice{NC,<:LatticeMatrix}) where {NC} = true
Base.isopen(B::ScaledIdentityLattice{NC,<:Adjoint_Lattice{<:LatticeMatrix}}) where {NC} = true
Base.isopen(B::ScaledIdentityLattice) = isopen(B.scalar)

Base.similar(B::ScaledIdentityLattice{NC,<:LatticeMatrix}) where {NC} =
    ScaledIdentityLattice(NC, similar(B.scalar))

function substitute!(C::ScaledIdentityLattice{NC,<:LatticeMatrix},
    B::ScaledIdentityLattice{NC}) where {NC}
    _check_center_layout(C.scalar, first(_center_operand(B.scalar)))
    substitute!(C.scalar, B.scalar)
    return C
end

const _ScalarMatrixOperand = Union{LatticeMatrix,Shifted_Lattice,Adjoint_Lattice}

# The CUDA extension selects one thread per contiguous color component.
# Keep the site-wise CPU kernel, which amortizes index calculation over colors.
@inline _center_component_threads(::Any) = false

function LinearAlgebra.mul!(C::LatticeMatrix, A::_ScalarMatrixOperand,
    B::ScaledIdentityLattice{NC}) where {NC}
    _center_shape(C, Val(false))[2] == NC || throw(DimensionMismatch("right identity size mismatch"))
    return _mul_scalar_field!(C, A, B.scalar)
end

function LinearAlgebra.mul!(C::LatticeMatrix, A::_ScalarMatrixOperand,
    B::ScaledIdentityLattice{NC}, alpha::Number, beta::Number) where {NC}
    _center_shape(C, Val(false))[2] == NC || throw(DimensionMismatch("right identity size mismatch"))
    return _mul_scalar_field!(C, A, B.scalar, alpha, beta)
end

function LinearAlgebra.mul!(C::LatticeMatrix, B::ScaledIdentityLattice{NC},
    A::_ScalarMatrixOperand) where {NC}
    _center_shape(C, Val(false))[1] == NC || throw(DimensionMismatch("left identity size mismatch"))
    return _mul_scalar_field!(C, A, B.scalar)
end

function LinearAlgebra.mul!(C::LatticeMatrix, B::ScaledIdentityLattice{NC},
    A::_ScalarMatrixOperand, alpha::Number, beta::Number) where {NC}
    _center_shape(C, Val(false))[1] == NC || throw(DimensionMismatch("left identity size mismatch"))
    return _mul_scalar_field!(C, A, B.scalar, alpha, beta)
end

function substitute!(C::LatticeMatrix{D,T,AT,NC,NC},
    B::ScaledIdentityLattice{NC}) where {D,T,AT,NC}
    s, shift, dagger = _center_operand(B.scalar)
    _check_center_layout(C, s)
    Base.mightalias(C.A, s.A) && throw(ArgumentError("dense expansion must not alias its scalar source"))
    _parallel_for_mutating!(C, prod(C.PN), _kernel_expand_identity!,
        C.A, s.A, C.indexer, Val(NC), C.nw, s.nw, shift, dagger)
    return C
end

@inline function _kernel_expand_identity!(i, C, s, indexer, ::Val{NC},
    nc, ns, shift, ::Val{DAG}) where {NC,DAG}
    site = delinearize(indexer, i, 0)
    ic = ntuple(d -> site[d] + nc, length(site))
    ix = ntuple(d -> site[d] + ns + shift[d], length(site))
    @inbounds begin
        value = s[1, 1, ix...]
        value = DAG ? conj(value) : value
        for col in 1:NC, row in 1:NC
            C[row, col, ic...] = row == col ? value : zero(value)
        end
    end
    return nothing
end

@inline _mul_scalar_field!(C::LatticeMatrix{D,TC}, A, B) where {D,TC} =
    _mul_scalar_field!(C, A, B, one(TC), zero(TC), Val(true), Val(true))

function _mul_scalar_field!(C::LatticeMatrix{D,TC}, A, B, alpha, beta) where {D,TC}
    alpha_in, beta_in = TC(alpha), TC(beta)
    return _mul_scalar_field!(C, A, B, alpha_in, beta_in,
        Val(isone(alpha_in)), Val(iszero(beta_in)))
end

function _mul_scalar_field!(C::LatticeMatrix{D,TC,ATC,NR,NC}, A, B,
    alpha, beta, unit_alpha::Val{UA}, zero_beta::Val{ZB}) where {D,TC,ATC,NR,NC,UA,ZB}
    a, sa, da = _center_operand(A)
    b, sb, db = _center_operand(B)
    _check_center_layout(C, a)
    _check_center_layout(C, b)
    _center_shape(a, da) == (NR, NC) || throw(DimensionMismatch("A and C matrix sizes differ"))
    _center_shape(b, db) == (1, 1) || throw(DimensionMismatch("s must have exactly one internal component"))
    Base.mightalias(C.A, b.A) && throw(ArgumentError("destination must not alias the scalar field"))
    if Base.mightalias(C.A, a.A) && (any(!iszero, sa) || da === Val(true))
        throw(ArgumentError("scalar multiplication cannot overwrite shifted or adjointed A"))
    end
    component_threads = _center_component_threads(C.A)
    count = component_threads ? prod(C.PN) * NR * NC : prod(C.PN)
    kernel = component_threads ? _kernel_mul_center_component! : _kernel_mul_center!
    _parallel_for_mutating!(C, count, kernel,
        C.A, a.A, b.A, C.indexer, Val(NR), Val(NC),
        Val(C.nw), Val(a.nw), Val(b.nw), sa, sb, da, db,
        alpha, beta, unit_alpha, zero_beta)
    return C
end

@inline _center_operand(A::LatticeMatrix{D}) where {D} =
    (A, ntuple(_ -> 0, D), Val(false))

function _center_operand(A::Shifted_Lattice)
    _assert_shift_open(A)
    data, shift = A.data, get_shift(A)
    _ensure_halo_for_shift!(data, shift)
    return data, shift, Val(false)
end

@inline function _center_operand(A::Adjoint_Lattice)
    data, shift, _ = _center_operand(A.data)
    return data, shift, Val(true)
end

@inline _center_shape(::LatticeMatrix{D,T,AT,NR,NC}, ::Val{false}) where {D,T,AT,NR,NC} = (NR, NC)
@inline _center_shape(::LatticeMatrix{D,T,AT,NR,NC}, ::Val{true}) where {D,T,AT,NR,NC} = (NC, NR)

function _check_center_layout(C, A)
    (C.gsize == A.gsize && C.PN == A.PN && C.coords == A.coords && C.dims == A.dims) ||
        throw(DimensionMismatch("operands must have the same lattice decomposition"))
    return nothing
end

@inline function _kernel_mul_center!(i, C, A, B, indexer, ::Val{NR}, ::Val{NC},
    ::Val{nc}, ::Val{na}, ::Val{nb}, sa, sb, ::Val{DA}, ::Val{DB},
    alpha, beta, ::Val{UNIT_ALPHA}, ::Val{ZERO_BETA}) where {NR,NC,nc,na,nb,DA,DB,UNIT_ALPHA,ZERO_BETA}
    site = delinearize(indexer, i, 0)
    ic = ntuple(d -> site[d] + nc, length(site))
    ia = ntuple(d -> site[d] + na + sa[d], length(site))
    ib = ntuple(d -> site[d] + nb + sb[d], length(site))
    @inbounds begin
        phase = B[1, 1, ib...]
        phase = DB ? conj(phase) : phase
        # Resolve each site's base index once. Repeated multidimensional
        # indexing in the color loop prevents effective CPU optimization.
        # LinearIndices also respects each operand's own shape/halo width.
        abase = LinearIndices(A)[1, 1, ia...]
        cbase = LinearIndices(C)[1, 1, ic...]
        for k in 0:(NR * NC - 1)
            # For A', source rows are NC and its column-major offset is
            # (output row - 1) * NC + (output column - 1).
            aoffset = DA ? (k % NR) * NC + k ÷ NR : k
            value = DA ? conj(A[abase + aoffset]) : A[abase + aoffset]
            product = value * phase
            product = UNIT_ALPHA ? product : alpha * product
            C[cbase + k] = ZERO_BETA ? product : product + beta * C[cbase + k]
        end
    end
    return nothing
end

@inline function _kernel_mul_center_component!(i, C, A, B, indexer, ::Val{NR}, ::Val{NC},
    ::Val{nc}, ::Val{na}, ::Val{nb}, sa, sb, ::Val{DA}, ::Val{DB},
    alpha, beta, ::Val{UNIT_ALPHA}, ::Val{ZERO_BETA}) where {NR,NC,nc,na,nb,DA,DB,UNIT_ALPHA,ZERO_BETA}
    site_index, k = divrem(i - 1, NR * NC)
    site = delinearize(indexer, site_index + 1, 0)
    ic = ntuple(d -> site[d] + nc, length(site))
    ia = ntuple(d -> site[d] + na + sa[d], length(site))
    ib = ntuple(d -> site[d] + nb + sb[d], length(site))
    @inbounds begin
        phase = B[1, 1, ib...]
        phase = DB ? conj(phase) : phase
        abase = LinearIndices(A)[1, 1, ia...]
        cbase = LinearIndices(C)[1, 1, ic...]
        aoffset = DA ? (k % NR) * NC + k ÷ NR : k
        value = DA ? conj(A[abase + aoffset]) : A[abase + aoffset]
        # Preserve the site-wise kernel's arithmetic order. Adjacent threads
        # now load/store adjacent color components instead of striding by NR*NC.
        product = value * phase
        product = UNIT_ALPHA ? product : alpha * product
        C[cbase + k] = ZERO_BETA ? product : product + beta * C[cbase + k]
    end
    return nothing
end

export ScaledIdentityLattice
