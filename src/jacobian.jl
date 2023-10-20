@concrete struct JacobianWrapper{iip} <: Function
    f
    p
end

# Previous Implementation did not hold onto `iip`, but this causes problems in packages
# where we check for the presence of function signatures to check which dispatch to call
(uf::JacobianWrapper{false})(u) = uf.f(u, uf.p)
(uf::JacobianWrapper{false})(res, u) = (vec(res) .= vec(uf.f(u, uf.p)))
(uf::JacobianWrapper{true})(res, u) = uf.f(res, u, uf.p)

sparsity_detection_alg(_, _) = NoSparsityDetection()
function sparsity_detection_alg(f, ad::AbstractSparseADType)
    if f.sparsity === nothing
        if f.jac_prototype === nothing
            return SymbolicsSparsityDetection()
        else
            jac_prototype = f.jac_prototype
        end
    else
        jac_prototype = f.sparsity
    end

    if SciMLBase.has_colorvec(f)
        return PrecomputedJacobianColorvec(; jac_prototype, f.colorvec,
            partition_by_rows = ad isa ADTypes.AbstractSparseReverseMode)
    else
        return JacPrototypeSparsityDetection(; jac_prototype)
    end
end

# NoOp for Jacobian if it is not a Abstract Array -- For eg, JacVec Operator
jacobian!!(J, _) = J
# `!!` notation is from BangBang.jl since J might be jacobian in case of oop `f.jac`
# and we don't want wasteful `copyto!`
function jacobian!!(J::Union{AbstractMatrix{<:Number}, Nothing}, cache)
    @unpack f, uf, u, p, jac_cache, alg, fu2 = cache
    iip = isinplace(cache)
    if iip
        has_jac(f) ? f.jac(J, u, p) :
        sparse_jacobian!(J, alg.ad, jac_cache, uf, fu2, _maybe_mutable(u, alg.ad))
    else
        return has_jac(f) ? f.jac(u, p) :
               sparse_jacobian!(J, alg.ad, jac_cache, uf, _maybe_mutable(u, alg.ad))
    end
    return J
end
# Scalar case
jacobian!!(::Number, cache) = last(value_derivative(cache.uf, cache.u))

# Build Jacobian Caches
function jacobian_caches(alg::AbstractNonlinearSolveAlgorithm, f, u, p, ::Val{iip};
    linsolve_kwargs = (;),
    linsolve_with_JᵀJ::Val{needsJᵀJ} = Val(false)) where {iip, needsJᵀJ}
    uf = JacobianWrapper{iip}(f, p)

    haslinsolve = hasfield(typeof(alg), :linsolve)

    has_analytic_jac = has_jac(f)
    linsolve_needs_jac = (concrete_jac(alg) === nothing &&
                          (!haslinsolve || (haslinsolve && (alg.linsolve === nothing ||
                             needs_concrete_A(alg.linsolve)))))
    alg_wants_jac = (concrete_jac(alg) !== nothing && concrete_jac(alg))

    # NOTE: The deepcopy is needed here since we are using the resid_prototype elsewhere
    fu = f.resid_prototype === nothing ? (iip ? _mutable_zero(u) : _mutable(f(u, p))) :
         (iip ? deepcopy(f.resid_prototype) : f.resid_prototype)
    if !has_analytic_jac && (linsolve_needs_jac || alg_wants_jac || needsJᵀJ)
        sd = sparsity_detection_alg(f, alg.ad)
        ad = alg.ad
        jac_cache = iip ? sparse_jacobian_cache(ad, sd, uf, fu, _maybe_mutable(u, ad)) :
                    sparse_jacobian_cache(ad, sd, uf, _maybe_mutable(u, ad); fx = fu)
    else
        jac_cache = nothing
    end

    # FIXME: To properly support needsJᵀJ without Jacobian, we need to implement
    #        a reverse diff operation with the seed being `Jx`, this is not yet implemented
    J = if !(linsolve_needs_jac || alg_wants_jac || needsJᵀJ)
        # We don't need to construct the Jacobian
        JacVec(uf, u; autodiff = __get_nonsparse_ad(alg.ad))
    else
        if has_analytic_jac
            f.jac_prototype === nothing ? undefmatrix(u) : f.jac_prototype
        else
            f.jac_prototype === nothing ? init_jacobian(jac_cache) : f.jac_prototype
        end
    end

    du = _mutable_zero(u)

    if needsJᵀJ
        JᵀJ = __init_JᵀJ(J)
        # FIXME: This needs to be handled better for JacVec Operator
        Jᵀfu = J' * _vec(fu)
    end

    linprob = LinearProblem(needsJᵀJ ? __maybe_symmetric(JᵀJ) : J,
        needsJᵀJ ? _vec(Jᵀfu) : _vec(fu); u0 = _vec(du))

    if alg isa PseudoTransient
        alpha = convert(eltype(u), alg.alpha_initial)
        J_new = J - (1 / alpha) * I
        linprob = LinearProblem(J_new, _vec(fu); u0 = _vec(du))
    end

    weight = similar(u)
    recursivefill!(weight, true)

    Pl, Pr = wrapprecs(alg.precs(needsJᵀJ ? __maybe_symmetric(JᵀJ) : J, nothing, u, p,
            nothing, nothing, nothing, nothing, nothing)..., weight)
    linsolve = init(linprob, alg.linsolve; alias_A = true, alias_b = true, Pl, Pr,
        linsolve_kwargs...)

    needsJᵀJ && return uf, linsolve, J, fu, jac_cache, du, JᵀJ, Jᵀfu
    return uf, linsolve, J, fu, jac_cache, du
end

__get_nonsparse_ad(::AutoSparseForwardDiff) = AutoForwardDiff()
__get_nonsparse_ad(::AutoSparseFiniteDiff) = AutoFiniteDiff()
__get_nonsparse_ad(::AutoSparseZygote) = AutoZygote()
__get_nonsparse_ad(ad) = ad

__init_JᵀJ(J::Number) = zero(J)
__init_JᵀJ(J::AbstractArray) = J' * J
__init_JᵀJ(J::StaticArray) = MArray{Tuple{size(J, 2), size(J, 2)}, eltype(J)}(undef)

__maybe_symmetric(x) = Symmetric(x)
__maybe_symmetric(x::Number) = x
# LinearSolve with `nothing` doesn't dispatch correctly here
__maybe_symmetric(x::StaticArray) = x
__maybe_symmetric(x::SparseArrays.AbstractSparseMatrix) = x

## Special Handling for Scalars
function jacobian_caches(alg::AbstractNonlinearSolveAlgorithm, f, u::Number, p,
    ::Val{false}; linsolve_with_JᵀJ::Val{needsJᵀJ} = Val(false),
    kwargs...) where {needsJᵀJ}
    # NOTE: Scalar `u` assumes scalar output from `f`
    uf = JacobianWrapper{false}(f, p)
    needsJᵀJ && return uf, nothing, u, nothing, nothing, u, u, u
    return uf, nothing, u, nothing, nothing, u
end
