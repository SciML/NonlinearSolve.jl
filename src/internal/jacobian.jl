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
    or if possible we can just use a `SciMLJacobianOperators.JacobianOperator` instead.
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

    if !has_analytic_jac && needs_jac
        autodiff = construct_concrete_adtype(f, autodiff)
        di_extras = if iip
            DI.prepare_jacobian(f, fu, autodiff, u, Constant(prob.p))
        else
            DI.prepare_jacobian(f, autodiff, u, Constant(prob.p))
        end
    else
        di_extras = nothing
    end

    J = if !needs_jac
        JacobianOperator(prob, fu, u; jvp_autodiff, vjp_autodiff)
    else
        if f.jac_prototype === nothing
            # While this is technically wasteful, it gives out the type of the Jacobian
            # which is needed to create the linear solver cache
            stats.njacs += 1
            if has_analytic_jac
                __similar(
                    fu, promote_type(eltype(fu), eltype(u)), length(fu), length(u))
            else
                if iip
                    DI.jacobian(f, fu, di_extras, autodiff, u, Constant(p))
                else
                    DI.jacobian(f, di_extras, autodiff, u, Constant(p))
                end
            end
        else
            if eltype(f.jac_prototype) <: Bool
                similar(f.jac_prototype, promote_type(eltype(fu), eltype(u)))
            else
                similar(f.jac_prototype)
            end
        end
    end

    return JacobianCache{iip}(J, f, fu, u, p, stats, autodiff, di_extras)
end

function JacobianCache(prob, alg, f::F, ::Number, u::Number, p; stats,
        autodiff = nothing, kwargs...) where {F}
    fu = f(u, p)
    if SciMLBase.has_jac(f) || SciMLBase.has_vjp(f) || SciMLBase.has_jvp(f)
        return JacobianCache{false}(u, f, fu, u, p, stats, autodiff, nothing)
    end
    di_extras = DI.prepare_derivative(f, get_dense_ad(autodiff), u, Constant(prob.p))
    return JacobianCache{false}(u, f, fu, u, p, stats, autodiff, di_extras)
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
        else
            DI.jacobian!(
                cache.f, cache.fu, J, cache.di_extras, cache.autodiff, u, Constant(p))
        end
        return J
    else
        if SciMLBase.has_jac(cache.f)
            return cache.f.jac(u, p)
        else
            return DI.jacobian(cache.f, cache.di_extras, cache.autodiff, u, Constant(p))
        end
    end
end

function construct_concrete_adtype(f::NonlinearFunction, ad::AbstractADType)
    @assert !(ad isa AutoSparse) "This shouldn't happen. Open an issue."
    if f.sparsity === nothing
        if f.jac_prototype === nothing
            if SciMLBase.has_colorvec(f)
                @warn "`colorvec` is provided but `sparsity` and `jac_prototype` is not \
                       specified. `colorvec` will be ignored."
            end
            return ad # No sparse AD
        else
            if !sparse_or_structured_prototype(f.jac_prototype)
                if SciMLBase.has_colorvec(f)
                    @warn "`colorvec` is provided but `jac_prototype` is not a sparse \
                           or structured matrix. `colorvec` will be ignored."
                end
                return ad
            end
            return AutoSparse(
                ad;
                sparsity_detector = KnownJacobianSparsityDetector(f.jac_prototype),
                coloring_algorithm = select_fastest_coloring_algorithm(
                    f.jac_prototype, f, ad)
            )
        end
    else
        if f.sparsity isa AbstractMatrix
            if f.jac_prototype !== f.sparsity
                if f.jac_prototype !== nothing &&
                   sparse_or_structured_prototype(f.jac_prototype)
                    throw(ArgumentError("`sparsity::AbstractMatrix` and a sparse or \
                                         structured `jac_prototype` cannot be both \
                                         provided. Pass only `jac_prototype`."))
                end
            end
            return AutoSparse(
                ad;
                sparsity_detector = KnownJacobianSparsityDetector(f.sparsity),
                coloring_algorithm = select_fastest_coloring_algorithm(
                    f.sparsity, f, ad)
            )
        end

        @assert f.sparsity isa ADTypes.AbstractSparsityDetector
        sparsity_detector = f.sparsity
        if f.jac_prototype === nothing
            if SciMLBase.has_colorvec(f)
                @warn "`colorvec` is provided but `jac_prototype` is not specified. \
                       `colorvec` will be ignored."
            end
            return AutoSparse(
                ad;
                sparsity_detector,
                coloring_algorithm = GreedyColoringAlgorithm(LargestFirst())
            )
        else
            if sparse_or_structured_prototype(f.jac_prototype)
                if !(sparsity_detector isa NoSparsityDetector)
                    @warn "`jac_prototype` is a sparse matrix but sparsity = $(f.sparsity) \
                           has also been specified. Ignoring sparsity field and using \
                           `jac_prototype` sparsity."
                end
                sparsity_detector = KnownJacobianSparsityDetector(f.jac_prototype)
            end

            return AutoSparse(
                ad;
                sparsity_detector,
                coloring_algorithm = select_fastest_coloring_algorithm(
                    f.jac_prototype, f, ad)
            )
        end
    end
end

function select_fastest_coloring_algorithm(
        prototype, f::NonlinearFunction, ad::AbstractADType)
    if SciMLBase.has_colorvec(f)
        return ConstantColoringAlgorithm{ifelse(
            ADTypes.mode(ad) isa ADTypes.ReverseMode, :row, :column)}(
            prototype, f.colorvec)
    end
    return GreedyColoringAlgorithm(LargestFirst())
end

function construct_concrete_adtype(::NonlinearFunction, ad::AutoSparse)
    error("Specifying a sparse AD type for Nonlinear Problems was removed in v4. \
           Instead use the `sparsity`, `jac_prototype`, and `colorvec` to specify \
           the right sparsity pattern and coloring algorithm. Ignoring the sparsity \
           detection algorithm and coloring algorithm present in $(ad).")
end

get_dense_ad(ad) = ad
get_dense_ad(ad::AutoSparse) = ADTypes.dense_ad(ad)

sparse_or_structured_prototype(::AbstractSparseMatrix) = true
function sparse_or_structured_prototype(prototype::AbstractMatrix)
    return ArrayInterface.isstructured(prototype)
end
