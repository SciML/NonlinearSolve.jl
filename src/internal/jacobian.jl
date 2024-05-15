"""
    JacobianCache(prob, alg, f::F, fu, u, p; autodiff = nothing,
        vjp_autodiff = nothing, jvp_autodiff = nothing, linsolve = missing) where {F}

Construct a cache for the Jacobian of `f` w.r.t. `u`.

### Arguments

  - `prob`: A [`NonlinearProblem`](@ref) or a [`NonlinearLeastSquaresProblem`](@ref).
  - `alg`: A [`AbstractNonlinearSolveAlgorithm`](@ref). Used to check for
    [`concrete_jac`](@ref).
  - `f`: The function to compute the Jacobian of.
  - `fu`: The evaluation of `f(u, p)` or `f(_, u, p)`. Used to determine the size of the
    result cache and Jacobian.
  - `u`: The current value of the state.
  - `p`: The current value of the parameters.

### Keyword Arguments

  - `autodiff`: Automatic Differentiation or Finite Differencing backend for computing the
    jacobian. By default, selects a backend based on sparsity parameters, type of state,
    function properties, etc.
  - `vjp_autodiff`: Automatic Differentiation or Finite Differencing backend for computing
    the vector-Jacobian product.
  - `jvp_autodiff`: Automatic Differentiation or Finite Differencing backend for computing
    the Jacobian-vector product.
  - `linsolve`: Linear Solver Algorithm used to determine if we need a concrete jacobian
    or if possible we can just use a [`NonlinearSolve.JacobianOperator`](@ref) instead.
"""
@concrete mutable struct JacobianCache{iip} <: AbstractNonlinearSolveJacobianCache{iip}
    J
    f
    uf
    fu
    u
    p
    jac_cache
    alg
    njacs::Int
    autodiff
    vjp_autodiff
    jvp_autodiff
end

function reinit_cache!(cache::JacobianCache{iip}, args...; p = cache.p,
        u0 = cache.u, kwargs...) where {iip}
    cache.njacs = 0
    cache.u = u0
    cache.p = p
    cache.uf = JacobianWrapper{iip}(cache.f, p)
end

function JacobianCache(
        prob, alg, f::F, fu_, u, p; autodiff = nothing, vjp_autodiff = nothing,
        jvp_autodiff = nothing, linsolve = missing) where {F}
    iip = isinplace(prob)
    uf = JacobianWrapper{iip}(f, p)

    autodiff = get_concrete_forward_ad(autodiff, prob; check_reverse_mode = false)
    jvp_autodiff = get_concrete_forward_ad(
        jvp_autodiff, prob, Val(false); check_reverse_mode = true)
    vjp_autodiff = get_concrete_reverse_ad(
        vjp_autodiff, prob, Val(false); check_forward_mode = false)

    has_analytic_jac = SciMLBase.has_jac(f)
    linsolve_needs_jac = concrete_jac(alg) === nothing && (linsolve === missing ||
                          (linsolve === nothing || __needs_concrete_A(linsolve)))
    alg_wants_jac = concrete_jac(alg) !== nothing && concrete_jac(alg)
    needs_jac = linsolve_needs_jac || alg_wants_jac

    @bb fu = similar(fu_)

    if !has_analytic_jac && needs_jac
        sd = __sparsity_detection_alg(f, autodiff)
        jac_cache = iip ? sparse_jacobian_cache(autodiff, sd, uf, fu, u) :
                    sparse_jacobian_cache(
            autodiff, sd, uf, __maybe_mutable(u, autodiff); fx = fu)
    else
        jac_cache = nothing
    end

    J = if !needs_jac
        JacobianOperator(prob, fu, u; jvp_autodiff, vjp_autodiff)
    else
        if has_analytic_jac
            f.jac_prototype === nothing ?
            similar(fu, promote_type(eltype(fu), eltype(u)), length(fu), length(u)) :
            copy(f.jac_prototype)
        elseif f.jac_prototype === nothing
            init_jacobian(jac_cache; preserve_immutable = Val(true))
        else
            f.jac_prototype
        end
    end

    return JacobianCache{iip}(
        J, f, uf, fu, u, p, jac_cache, alg, 0, autodiff, vjp_autodiff, jvp_autodiff)
end

function JacobianCache(
        prob, alg, f::F, ::Number, u::Number, p; autodiff = nothing, kwargs...) where {F}
    uf = JacobianWrapper{false}(f, p)
    autodiff = get_concrete_forward_ad(autodiff, prob; check_reverse_mode = false)
    if !(autodiff isa AutoForwardDiff ||
         autodiff isa AutoPolyesterForwardDiff ||
         autodiff isa AutoFiniteDiff)
        autodiff = AutoFiniteDiff()
        # Other cases are not properly supported so we fallback to finite differencing
        @warn "Scalar AD is supported only for AutoForwardDiff and AutoFiniteDiff. \
               Detected $(autodiff). Falling back to AutoFiniteDiff."
    end
    return JacobianCache{false}(
        u, f, uf, u, u, p, nothing, alg, 0, autodiff, nothing, nothing)
end

@inline (cache::JacobianCache)(u = cache.u) = cache(cache.J, u, cache.p)
@inline function (cache::JacobianCache)(::Nothing)
    J = cache.J
    J isa JacobianOperator && return StatefulJacobianOperator(J, cache.u, cache.p)
    return J
end

function (cache::JacobianCache)(J::JacobianOperator, u, p = cache.p)
    return StatefulJacobianOperator(J, u, p)
end
function (cache::JacobianCache)(::Number, u, p = cache.p) # Scalar
    cache.njacs += 1
    J = last(__value_derivative(cache.autodiff, cache.uf, u))
    return J
end
# Compute the Jacobian
function (cache::JacobianCache{iip})(
        J::Union{AbstractMatrix, Nothing}, u, p = cache.p) where {iip}
    cache.njacs += 1
    if iip
        if has_jac(cache.f)
            cache.f.jac(J, u, p)
        else
            sparse_jacobian!(J, cache.autodiff, cache.jac_cache, cache.uf, cache.fu, u)
        end
        J_ = J
    else
        J_ = if has_jac(cache.f)
            cache.f.jac(u, p)
        elseif __can_setindex(typeof(J))
            sparse_jacobian!(J, cache.autodiff, cache.jac_cache, cache.uf, u)
            J
        else
            sparse_jacobian(cache.autodiff, cache.jac_cache, cache.uf, u)
        end
    end
    return J_
end

# Sparsity Detection Choices
@inline __sparsity_detection_alg(_, _) = NoSparsityDetection()
@inline function __sparsity_detection_alg(f::NonlinearFunction, ad::AutoSparse)
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
    elseif f.sparsity isa AbstractSparsityDetection
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
            partition_by_rows = (ad isa AutoSparse && ADTypes.mode(ad) isa ADTypes.ReverseMode))
    else
        return JacPrototypeSparsityDetection(; jac_prototype)
    end
end

@inline function __value_derivative(
        ::Union{AutoForwardDiff, AutoPolyesterForwardDiff}, f::F, x::R) where {F, R}
    T = typeof(ForwardDiff.Tag(f, R))
    out = f(ForwardDiff.Dual{T}(x, one(x)))
    return ForwardDiff.value(out), ForwardDiff.extract_derivative(T, out)
end

@inline function __value_derivative(ad::AutoFiniteDiff, f::F, x::R) where {F, R}
    return f(x), FiniteDiff.finite_difference_derivative(f, x, ad.fdtype)
end

@inline function __scalar_jacvec(f::F, x::R, v::V) where {F, R, V}
    T = typeof(ForwardDiff.Tag(f, R))
    out = f(ForwardDiff.Dual{T}(x, v))
    return ForwardDiff.value(out), ForwardDiff.extract_derivative(T, out)
end
