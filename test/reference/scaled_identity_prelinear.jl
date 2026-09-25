# Frozen reference for the initial scalar kernel (before linear-index reuse).
# Kept outside src: used only for numerical regression tests and benchmarks.
module PrelinearScaledIdentity
using LatticeMatrices
const LM = LatticeMatrices

function mul_scalar!(C::LatticeMatrix{D,TC,ATC,NR,NC}, A, B,
    alpha=one(TC), beta=zero(TC)) where {D,TC,ATC,NR,NC}
    a, sa, da = LM._center_operand(A)
    b, sb, db = LM._center_operand(B)
    LM._check_center_layout(C, a)
    LM._check_center_layout(C, b)
    LM._center_shape(a, da) == (NR, NC) || throw(DimensionMismatch("A and C matrix sizes differ"))
    LM._center_shape(b, db) == (1, 1) || throw(DimensionMismatch("s must have exactly one internal component"))
    Base.mightalias(C.A, b.A) && throw(ArgumentError("destination must not alias the scalar field"))
    if Base.mightalias(C.A, a.A) && (any(!iszero, sa) || da === Val(true))
        throw(ArgumentError("scalar multiplication cannot overwrite shifted or adjointed A"))
    end
    LM._parallel_for_mutating!(C, prod(C.PN), kernel!,
        C.A, a.A, b.A, C.indexer, Val(NR), Val(NC),
        Val(C.nw), Val(a.nw), Val(b.nw), sa, sb, da, db,
        TC(alpha), TC(beta), Val(isone(TC(alpha))), Val(iszero(TC(beta))))
    return C
end

@inline function kernel!(i, C, A, B, indexer, ::Val{NR}, ::Val{NC},
    ::Val{nc}, ::Val{na}, ::Val{nb}, sa, sb, ::Val{DA}, ::Val{DB},
    alpha, beta, ::Val{UNIT_ALPHA}, ::Val{ZERO_BETA}) where {NR,NC,nc,na,nb,DA,DB,UNIT_ALPHA,ZERO_BETA}
    site = delinearize(indexer, i, 0)
    ic = ntuple(d -> site[d] + nc, length(site))
    ia = ntuple(d -> site[d] + na + sa[d], length(site))
    ib = ntuple(d -> site[d] + nb + sb[d], length(site))
    @inbounds begin
        phase = B[1, 1, ib...]
        phase = DB ? conj(phase) : phase
        for col in 1:NC, row in 1:NR
            value = DA ? conj(A[col, row, ia...]) : A[row, col, ia...]
            product = value * phase
            product = UNIT_ALPHA ? product : alpha * product
            C[row, col, ic...] = ZERO_BETA ? product : product + beta * C[row, col, ic...]
        end
    end
    return nothing
end
end
