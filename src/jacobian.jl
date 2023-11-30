@concrete struct KrylovJᵀJ
    JᵀJ
    Jᵀ
end

__maybe_symmetric(x::KrylovJᵀJ) = x.JᵀJ

isinplace(JᵀJ::KrylovJᵀJ) = isinplace(JᵀJ.Jᵀ)

# Select if we are going to use sparse differentiation or not
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
    @unpack f, uf, u, p, jac_cache, alg, fu_cache = cache
    iip = isinplace(cache)
    if iip
        if has_jac(f)
            f.jac(J, u, p)
        else
            sparse_jacobian!(J, alg.ad, jac_cache, uf, fu_cache, u)
        end
        return J
    else
        if has_jac(f)
            return f.jac(u, p)
        elseif can_setindex(typeof(J))
            return sparse_jacobian!(J, alg.ad, jac_cache, uf, u)
        else
            return sparse_jacobian(alg.ad, jac_cache, uf, u)
        end
    end
end
# Scalar case
jacobian!!(::Number, cache) = last(value_derivative(cache.uf, cache.u))

# Build Jacobian Caches
function jacobian_caches(alg::AbstractNonlinearSolveAlgorithm, f::F, u, p, ::Val{iip};
        linsolve_kwargs = (;), lininit::Val{linsolve_init} = Val(true),
        linsolve_with_JᵀJ::Val{needsJᵀJ} = Val(false)) where {iip, needsJᵀJ, linsolve_init, F}
    uf = SciMLBase.JacobianWrapper{iip}(f, p)

    haslinsolve = hasfield(typeof(alg), :linsolve)

    has_analytic_jac = has_jac(f)
    linsolve_needs_jac = (concrete_jac(alg) === nothing &&
                          (!haslinsolve || (haslinsolve && (alg.linsolve === nothing ||
                             needs_concrete_A(alg.linsolve)))))
    alg_wants_jac = (concrete_jac(alg) !== nothing && concrete_jac(alg))

    # NOTE: The deepcopy is needed here since we are using the resid_prototype elsewhere
    fu = f.resid_prototype === nothing ? (iip ? zero(u) : f(u, p)) :
         (iip ? deepcopy(f.resid_prototype) : f.resid_prototype)
    if !has_analytic_jac && (linsolve_needs_jac || alg_wants_jac)
        sd = sparsity_detection_alg(f, alg.ad)
        ad = alg.ad
        jac_cache = iip ? sparse_jacobian_cache(ad, sd, uf, fu, u) :
                    sparse_jacobian_cache(ad, sd, uf, __maybe_mutable(u, ad); fx = fu)
    else
        jac_cache = nothing
    end

    J = if !(linsolve_needs_jac || alg_wants_jac)
        if f.jvp === nothing
            # We don't need to construct the Jacobian
            JacVec(uf, u; fu, autodiff = __get_nonsparse_ad(alg.ad))
        else
            if iip
                jvp = (_, u, v) -> (du_ = similar(fu); f.jvp(du_, v, u, p); du_)
                jvp! = (du_, _, u, v) -> f.jvp(du_, v, u, p)
            else
                jvp = (_, u, v) -> f.jvp(v, u, p)
                jvp! = (du_, _, u, v) -> (du_ .= f.jvp(v, u, p))
            end
            op = SparseDiffTools.FwdModeAutoDiffVecProd(f, u, (), jvp, jvp!)
            FunctionOperator(op, u, fu; isinplace = Val(true), outofplace = Val(false),
                p, islinear = true)
        end
    else
        if has_analytic_jac
            f.jac_prototype === nothing ? undefmatrix(u) : f.jac_prototype
        elseif f.jac_prototype === nothing
            init_jacobian(jac_cache; preserve_immutable = Val(true))
        else
            f.jac_prototype
        end
    end

    du = copy(u)

    if needsJᵀJ
        JᵀJ, Jᵀfu = __init_JᵀJ(J, _vec(fu), uf, u; f,
            vjp_autodiff = __get_nonsparse_ad(__getproperty(alg, Val(:vjp_autodiff))),
            jvp_autodiff = __get_nonsparse_ad(alg.ad))
    else
        JᵀJ, Jᵀfu = nothing, nothing
    end

    if linsolve_init
        if alg isa PseudoTransient && J isa SciMLOperators.AbstractSciMLOperator
            linprob_A = J - inv(convert(eltype(u), alg.alpha_initial)) * I
        else
            linprob_A = needsJᵀJ ? __maybe_symmetric(JᵀJ) : J
        end
        linsolve = linsolve_caches(linprob_A, needsJᵀJ ? Jᵀfu : fu, du, p, alg;
            linsolve_kwargs)
    else
        linsolve = nothing
    end

    return uf, linsolve, J, fu, jac_cache, du, JᵀJ, Jᵀfu
end

## Special Handling for Scalars
function jacobian_caches(alg::AbstractNonlinearSolveAlgorithm, f::F, u::Number, p,
        ::Val{false}; linsolve_with_JᵀJ::Val{needsJᵀJ} = Val(false),
        kwargs...) where {needsJᵀJ, F}
    # NOTE: Scalar `u` assumes scalar output from `f`
    uf = SciMLBase.JacobianWrapper{false}(f, p)
    needsJᵀJ && return uf, nothing, u, nothing, nothing, u, u, u
    return uf, FakeLinearSolveJLCache(u, u), u, nothing, nothing, u
end

# Linear Solve Cache
function linsolve_caches(A, b, u, p, alg; linsolve_kwargs = (;))
    if alg.linsolve === nothing && A isa SMatrix && linsolve_kwargs === (;)
        # Default handling for SArrays in LinearSolve is not great. Some parts are patched
        # but there are quite a few unnecessary allocations
        return FakeLinearSolveJLCache(A, b)
    end

    linprob = LinearProblem(A, _vec(b); u0 = _vec(u), linsolve_kwargs...)

    weight = __init_ones(u)

    Pl, Pr = wrapprecs(alg.precs(A, nothing, u, p, nothing, nothing, nothing, nothing,
            nothing)..., weight)
    return init(linprob, alg.linsolve; alias_A = true, alias_b = true, Pl, Pr)
end
linsolve_caches(A::KrylovJᵀJ, b, u, p, alg) = linsolve_caches(A.JᵀJ, b, u, p, alg)

__init_JᵀJ(J::Number, args...; kwargs...) = zero(J), zero(J)
function __init_JᵀJ(J::AbstractArray, fu, args...; kwargs...)
    JᵀJ = J' * J
    Jᵀfu = J' * fu
    return JᵀJ, Jᵀfu
end
function __init_JᵀJ(J::StaticArray, fu, args...; kwargs...)
    JᵀJ = MArray{Tuple{size(J, 2), size(J, 2)}, eltype(J)}(undef)
    return JᵀJ, J' * fu
end
function __init_JᵀJ(J::FunctionOperator, fu, uf, u, args...; f = nothing,
        vjp_autodiff = nothing, jvp_autodiff = nothing, kwargs...)
    # FIXME: Proper fix to this requires the FunctionOperator patch
    if f !== nothing && f.vjp !== nothing
        @warn "Currently we don't make use of user provided `jvp`. This is planned to be \
               fixed in the near future."
    end
    autodiff = __concrete_vjp_autodiff(vjp_autodiff, jvp_autodiff, uf)
    Jᵀ = VecJac(uf, u; fu, autodiff)
    JᵀJ_op = SciMLOperators.cache_operator(Jᵀ * J, u)
    JᵀJ = KrylovJᵀJ(JᵀJ_op, Jᵀ)
    Jᵀfu = Jᵀ * fu
    return JᵀJ, Jᵀfu
end

function __concrete_vjp_autodiff(vjp_autodiff, jvp_autodiff, uf)
    if vjp_autodiff === nothing
        if isinplace(uf)
            # VecJac can be only FiniteDiff
            return AutoFiniteDiff()
        else
            # Short circuit if we see that FiniteDiff was used for J computation
            jvp_autodiff isa AutoFiniteDiff && return jvp_autodiff
            # Check if Zygote is loaded then use Zygote else use FiniteDiff
            is_extension_loaded(Val{:Zygote}()) && return AutoZygote()
            return AutoFiniteDiff()
        end
    else
        ad = __get_nonsparse_ad(vjp_autodiff)
        if isinplace(uf) && ad isa AutoZygote
            @warn "Attempting to use Zygote.jl for linesearch on an in-place problem. \
                Falling back to finite differencing."
            return AutoFiniteDiff()
        end
        return ad
    end
end

# Generic Handling of Krylov Methods for Normal Form Linear Solves
function __update_JᵀJ!(iip::Val, cache, sym::Symbol, J)
    return __update_JᵀJ!(iip, cache, sym, getproperty(cache, sym), J)
end
__update_JᵀJ!(::Val{false}, cache, sym::Symbol, _, J) = setproperty!(cache, sym, J' * J)
__update_JᵀJ!(::Val{true}, cache, sym::Symbol, _, J) = mul!(getproperty(cache, sym), J', J)
__update_JᵀJ!(::Val{false}, cache, sym::Symbol, H::KrylovJᵀJ, J) = H
__update_JᵀJ!(::Val{true}, cache, sym::Symbol, H::KrylovJᵀJ, J) = H

function __update_Jᵀf!(iip::Val, cache, sym1::Symbol, sym2::Symbol, J, fu)
    return __update_Jᵀf!(iip, cache, sym1, sym2, getproperty(cache, sym2), J, fu)
end
function __update_Jᵀf!(::Val{false}, cache, sym1::Symbol, sym2::Symbol, _, J, fu)
    return setproperty!(cache, sym1, _restructure(getproperty(cache, sym1), J' * fu))
end
function __update_Jᵀf!(::Val{true}, cache, sym1::Symbol, sym2::Symbol, _, J, fu)
    return mul!(_vec(getproperty(cache, sym1)), J', fu)
end
function __update_Jᵀf!(::Val{false}, cache, sym1::Symbol, sym2::Symbol, H::KrylovJᵀJ, J, fu)
    return setproperty!(cache, sym1, _restructure(getproperty(cache, sym1), H.Jᵀ * fu))
end
function __update_Jᵀf!(::Val{true}, cache, sym1::Symbol, sym2::Symbol, H::KrylovJᵀJ, J, fu)
    return mul!(_vec(getproperty(cache, sym1)), H.Jᵀ, fu)
end

# Left-Right Multiplication
__lr_mul(::Val, H, g) = dot(g, H, g)
## TODO: Use a cache here to avoid allocations
__lr_mul(::Val{false}, H::KrylovJᵀJ, g) = dot(g, H.JᵀJ, g)
function __lr_mul(::Val{true}, H::KrylovJᵀJ, g)
    c = similar(g)
    mul!(c, H.JᵀJ, g)
    return dot(g, c)
end
