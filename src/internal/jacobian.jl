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
    or if possible we can just use a [`SciMLJacobianOperators.JacobianOperator`](@ref)
    instead.
"""
@concrete mutable struct JacobianCache{iip} <: AbstractNonlinearSolveJacobianCache{iip}
    J
    f
    fu
    u
    p
    stats::NLStats
    autodiff
    di_extras
    sdifft_extras
end

function reinit_cache!(cache::JacobianCache{iip}, args...; p = cache.p,
        u0 = cache.u, kwargs...) where {iip}
    cache.u = u0
    cache.p = p
end

function JacobianCache(prob, alg, f::F, fu_, u, p; stats, autodiff = nothing,
        vjp_autodiff = nothing, jvp_autodiff = nothing, linsolve = missing) where {F}
    iip = isinplace(prob)

    has_analytic_jac = SciMLBase.has_jac(f)
    linsolve_needs_jac = concrete_jac(alg) === nothing && (linsolve === missing ||
                          (linsolve === nothing || __needs_concrete_A(linsolve)))
    alg_wants_jac = concrete_jac(alg) !== nothing && concrete_jac(alg)
    needs_jac = linsolve_needs_jac || alg_wants_jac

    @bb fu = similar(fu_)

    autodiff = get_concrete_forward_ad(autodiff, prob; check_forward_mode = false)

    if !has_analytic_jac && needs_jac
        sd = sparsity_detection_alg(f, autodiff)
        sparse_jac = !(sd isa NoSparsityDetection)
        # Eventually we want to do everything via DI. But for now, we just do the dense via DI
        if sparse_jac
            di_extras = nothing
            uf = JacobianWrapper{iip}(f, p)
            sdifft_extras = if iip
                sparse_jacobian_cache(autodiff, sd, uf, fu, u)
            else
                sparse_jacobian_cache(
                    autodiff, sd, uf, __maybe_mutable(u, autodiff); fx = fu)
            end
        else
            sdifft_extras = nothing
            di_extras = if iip
                DI.prepare_jacobian(f, fu, autodiff, u, Constant(p))
            else
                DI.prepare_jacobian(f, autodiff, u, Constant(p))
            end
        end
    else
        sparse_jac = false
        di_extras = nothing
        sdifft_extras = nothing
    end

    J = if !needs_jac
        jvp_autodiff = get_concrete_forward_ad(
            jvp_autodiff, prob, Val(false); check_forward_mode = true)
        vjp_autodiff = get_concrete_reverse_ad(
            vjp_autodiff, prob, Val(false); check_reverse_mode = false)
        JacobianOperator(prob, fu, u; jvp_autodiff, vjp_autodiff)
    else
        if f.jac_prototype === nothing
            if !sparse_jac
                # While this is technically wasteful, it gives out the type of the Jacobian
                # which is needed to create the linear solver cache
                stats.njacs += 1
                if iip
                    DI.jacobian(f, fu, di_extras, autodiff, u, Constant(p))
                else
                    DI.jacobian(f, autodiff, u, Constant(p))
                end
            else
                zero(init_jacobian(sdifft_extras; preserve_immutable = Val(true)))
            end
        else
            similar(f.jac_prototype)
        end
    end

    return JacobianCache{iip}(
        J, f, fu, u, p, stats, autodiff, di_extras, sdifft_extras)
end

function JacobianCache(prob, alg, f::F, ::Number, u::Number, p; stats,
        autodiff = nothing, kwargs...) where {F}
    fu = f(u, p)
    if SciMLBase.has_jac(f) || SciMLBase.has_vjp(f) || SciMLBase.has_jvp(f)
        return JacobianCache{false}(u, f, fu, u, p, stats, autodiff, nothing)
    end
    autodiff = get_concrete_forward_ad(autodiff, prob; check_forward_mode = false)
    di_extras = DI.prepare_derivative(f, get_dense_ad(autodiff), u, Constant(prob.p))
    return JacobianCache{false}(u, f, fu, u, p, stats, autodiff, di_extras, nothing)
end

(cache::JacobianCache)(u = cache.u) = cache(cache.J, u, cache.p)
function (cache::JacobianCache)(::Nothing)
    cache.J isa JacobianOperator &&
        return StatefulJacobianOperator(cache.J, cache.u, cache.p)
    return cache.J
end

# Operator
function (cache::JacobianCache)(J::JacobianOperator, u, p = cache.p)
    return StatefulJacobianOperator(J, u, p)
end
# Numbers
function (cache::JacobianCache)(::Number, u, p = cache.p) # Scalar
    cache.stats.njacs += 1
    if SciMLBase.has_jac(cache.f)
        return cache.f.jac(u, p)
    elseif SciMLBase.has_vjp(cache.f)
        return cache.f.vjp(one(u), u, p)
    elseif SciMLBase.has_jvp(cache.f)
        return cache.f.jvp(one(u), u, p)
    end
    return DI.derivative(cache.f, cache.di_extras, cache.autodiff, u, Constant(p))
end
# Actually Compute the Jacobian
function (cache::JacobianCache{iip})(
        J::Union{AbstractMatrix, Nothing}, u, p = cache.p) where {iip}
    cache.stats.njacs += 1
    if iip
        if SciMLBase.has_jac(cache.f)
            cache.f.jac(J, u, p)
        elseif cache.di_extras !== nothing
            DI.jacobian!(
                cache.f, cache.fu, J, cache.di_extras, cache.autodiff, u, Constant(p))
        else
            uf = JacobianWrapper{iip}(cache.f, p)
            sparse_jacobian!(J, cache.autodiff, cache.sdifft_extras, uf, cache.fu, u)
        end
        return J
    else
        if SciMLBase.has_jac(cache.f)
            return cache.f.jac(u, p)
        elseif cache.di_extras !== nothing
            return DI.jacobian(cache.f, cache.di_extras, cache.autodiff, u, Constant(p))
        else
            uf = JacobianWrapper{iip}(cache.f, p)
            if __can_setindex(typeof(J))
                sparse_jacobian!(J, cache.autodiff, cache.sdifft_extras, uf, u)
                return J
            else
                return sparse_jacobian(cache.autodiff, cache.sdifft_extras, uf, u)
            end
        end
    end
end

function sparsity_detection_alg(f::NonlinearFunction, ad::AbstractADType)
    # TODO: Also handle case where colorvec is provided
    f.sparsity === nothing && return NoSparsityDetection()
    return sparsity_detection_alg(f, AutoSparse(ad; sparsity_detector = f.sparsity))
end

function sparsity_detection_alg(f::NonlinearFunction, ad::AutoSparse)
    if f.sparsity === nothing
        if f.jac_prototype === nothing
            is_extension_loaded(Val(:Symbolics)) && return SymbolicsSparsityDetection()
            return ApproximateJacobianSparsity()
        else
            jac_prototype = f.jac_prototype
        end
    elseif f.sparsity isa AbstractSparsityDetection
        f.jac_prototype === nothing && return f.sparsity
        jac_prototype = f.jac_prototype
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
            partition_by_rows = ADTypes.mode(ad) isa ADTypes.ReverseMode)
    else
        return JacPrototypeSparsityDetection(; jac_prototype)
    end
end

get_dense_ad(ad) = ad
get_dense_ad(ad::AutoSparse) = ADTypes.dense_ad(ad)
