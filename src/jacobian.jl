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
            if is_extension_loaded(Val(:Symbolics))
                return SymbolicsSparsityDetection()
            else
                return ApproximateJacobianSparsity()
            end
        else
            jac_prototype = f.jac_prototype
        end
    elseif f.sparsity isa SparseDiffTools.AbstractSparsityDetection
        if f.jac_prototype === nothing
            return f.sparsity
        else
            jac_prototype = f.jac_prototype
        end
    elseif f.sparsity isa AbstractMatrix
        jac_prototype = f.sparsity
    elseif f.jac_prototype isa AbstractMatrix
        jac_prototype = f.jac_prototype
    else
        error("`sparsity::typeof($(typeof(f.sparsity)))` & \
               `jac_prototype::typeof($(typeof(f.jac_prototype)))` is not supported. \
               Use `sparsity::AbstractMatrix` or `sparsity::AbstractSparsityDetection` or \
               set to `nothing`. `jac_prototype` can be set to `nothing` or an \
               `AbstractMatrix`.")
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
    cache.stats.njacs += 1
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
function jacobian!!(::Number, cache)
    cache.stats.njacs += 1
    return last(value_derivative(cache.uf, cache.u))
end

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
    return uf, FakeLinearSolveJLCache(u, u), u, zero(u), nothing, u, u, u
end

# Linear Solve Cache
function linsolve_caches(A, b, u, p, alg; linsolve_kwargs = (;))
    if A isa Number ||
       (alg.linsolve === nothing && A isa SMatrix && linsolve_kwargs === (;))
        # Default handling for SArrays in LinearSolve is not great. Some parts are patched
        # but there are quite a few unnecessary allocations
        return FakeLinearSolveJLCache(A, _vec(b))
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

# jvp fallback scalar
function __gradient_operator(uf, u; autodiff, kwargs...)
    if !(autodiff isa AutoFiniteDiff || autodiff isa AutoZygote)
        _ad = autodiff
        number_ad = ifelse(ForwardDiff.can_dual(eltype(u)), AutoForwardDiff(),
            AutoFiniteDiff())
        if u isa Number
            autodiff = number_ad
        else
            if isinplace(uf)
                autodiff = AutoFiniteDiff()
            else
                autodiff = ifelse(is_extension_loaded(Val{:Zygote}()), AutoZygote(),
                    AutoFiniteDiff())
            end
        end
        if _ad !== nothing && _ad !== autodiff
            @warn "$(_ad) not supported for VecJac. Using $(autodiff) instead."
        end
    end
    return u isa Number ? GradientScalar(uf, u, autodiff) :
           VecJac(uf, u; autodiff, kwargs...)
end

@concrete mutable struct GradientScalar
    uf
    u
    autodiff
end

function Base.:*(jvp::GradientScalar, v::Number)
    if jvp.autodiff isa AutoForwardDiff
        T = typeof(ForwardDiff.Tag(typeof(jvp.uf), typeof(jvp.u)))
        out = jvp.uf(ForwardDiff.Dual{T}(jvp.u, one(v)))
        return ForwardDiff.extract_derivative(T, out)
    elseif jvp.autodiff isa AutoFiniteDiff
        J = FiniteDiff.finite_difference_derivative(jvp.uf, jvp.u, jvp.autodiff.fdtype)
        return J
    else
        error("Only ForwardDiff & FiniteDiff is currently supported.")
    end
end

# Generic Handling of Krylov Methods for Normal Form Linear Solves
function __update_JᵀJ!(cache::AbstractNonlinearSolveCache, J = nothing)
    if !(cache.JᵀJ isa KrylovJᵀJ)
        J_ = ifelse(J === nothing, cache.J, J)
        @bb cache.JᵀJ = transpose(J_) × J_
    end
end

function __update_Jᵀf!(cache::AbstractNonlinearSolveCache, J = nothing)
    if cache.JᵀJ isa KrylovJᵀJ
        @bb cache.Jᵀf = cache.JᵀJ.Jᵀ × cache.fu
    else
        J_ = ifelse(J === nothing, cache.J, J)
        @bb cache.Jᵀf = transpose(J_) × vec(cache.fu)
    end
end

# Left-Right Multiplication
__lr_mul(cache::AbstractNonlinearSolveCache) = __lr_mul(cache, cache.JᵀJ, cache.Jᵀf)
function __lr_mul(cache::AbstractNonlinearSolveCache, JᵀJ::KrylovJᵀJ, Jᵀf)
    @bb cache.lr_mul_cache = JᵀJ.JᵀJ × vec(Jᵀf)
    return dot(_vec(Jᵀf), _vec(cache.lr_mul_cache))
end
function __lr_mul(cache::AbstractNonlinearSolveCache, JᵀJ, Jᵀf)
    @bb cache.lr_mul_cache = JᵀJ × vec(Jᵀf)
    return dot(_vec(Jᵀf), _vec(cache.lr_mul_cache))
end
