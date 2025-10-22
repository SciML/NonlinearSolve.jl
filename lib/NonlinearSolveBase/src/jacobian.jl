
"""
    construct_jacobian_cache(
        prob, alg, f, fu, u = prob.u0, p = prob.p;
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing,
        linsolve = missing
    )

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
    or if possible we can just use a `JacobianOperator` instead.
"""
function construct_jacobian_cache(
        prob, alg, f::NonlinearFunction, fu, u = prob.u0, p = prob.p; stats,
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing,
        linsolve = missing
)
    has_analytic_jac = SciMLBase.has_jac(f)
    linsolve_needs_jac = !concrete_jac(alg) && (linsolve === missing ||
                          (linsolve === nothing || needs_concrete_A(linsolve)))
    needs_jac = linsolve_needs_jac || concrete_jac(alg)

    fu_cache = Utils.safe_similar(fu)

    if !has_analytic_jac && needs_jac
        if autodiff === nothing
            throw(ArgumentError("`autodiff` argument to `construct_jacobian_cache` must be \
                                 specified and cannot be `nothing`. Use \
                                 `NonlinearSolveBase.select_jacobian_autodiff` for \
                                 automatic backend selection."))
        end
        autodiff = construct_concrete_adtype(f, autodiff)
        di_extras = if SciMLBase.isinplace(f)
            DI.prepare_jacobian(f, fu_cache, autodiff, u, Constant(p), strict = Val(false))
        else
            DI.prepare_jacobian(f, autodiff, u, Constant(p), strict = Val(false))
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
                Utils.safe_similar(
                    fu, promote_type(eltype(fu), eltype(u)), length(fu), length(u)
                )
            else
                if SciMLBase.isinplace(f)
                    DI.jacobian(f, fu_cache, di_extras, autodiff, u, Constant(p))
                else
                    DI.jacobian(f, di_extras, autodiff, u, Constant(p))
                end
            end
        else
            if eltype(f.jac_prototype) <: Bool
                Utils.safe_similar(f.jac_prototype, promote_type(eltype(fu), eltype(u)))
            else
                Utils.safe_similar(f.jac_prototype)
            end
        end
    end

    return JacobianCache(J, f, fu, p, stats, autodiff, di_extras)
end

function construct_jacobian_cache(
        prob, alg, f::NonlinearFunction, fu::Number, u::Number = prob.u0, p = prob.p; stats,
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing,
        linsolve = missing
)
    if SciMLBase.has_jac(f) || SciMLBase.has_vjp(f) || SciMLBase.has_jvp(f)
        return JacobianCache(fu, f, fu, p, stats, autodiff, nothing)
    end
    if autodiff === nothing
        throw(ArgumentError("`autodiff` argument to `construct_jacobian_cache` must be \
                             specified and cannot be `nothing`. Use \
                             `NonlinearSolveBase.select_jacobian_autodiff` for \
                             automatic backend selection."))
    end
    @assert !(autodiff isa AutoSparse) "`autodiff` cannot be `AutoSparse` for scalar \
                                        nonlinear problems."
    di_extras = DI.prepare_derivative(f, autodiff, u, Constant(prob.p))
    return JacobianCache(u, f, fu, p, stats, autodiff, di_extras)
end

@concrete mutable struct JacobianCache <: AbstractJacobianCache
    J
    f <: NonlinearFunction
    fu
    p
    stats::NLStats
    autodiff
    di_extras
end

function InternalAPI.reinit!(cache::JacobianCache; p = cache.p, kwargs...)
    cache.p = p
end

# Deprecations
(cache::JacobianCache{<:Number})(::Nothing) = error("Please report a bug to NonlinearSolve.jl")
(cache::JacobianCache{<:JacobianOperator})(::Nothing) = error("Please report a bug to NonlinearSolve.jl")
(cache::JacobianCache)(::Nothing) = error("Please report a bug to NonlinearSolve.jl")

reused_jacobian(cache::JacobianCache, u) = cache.J
reused_jacobian(cache::JacobianCache{<:JacobianOperator}, u) = StatefulJacobianOperator(cache.J, u, cache.p)

# Core Computation
## Numbers
function (cache::JacobianCache{<:Number})(u)
    cache.stats.njacs += 1
    
    (; f, J, p) = cache
    cache.J = if SciMLBase.has_jac(f)
        f.jac(u, p)
    elseif SciMLBase.has_vjp(f)
        f.vjp(one(u), u, p)
    elseif SciMLBase.has_jvp(f)
        f.jvp(one(u), u, p)
    else
        DI.derivative(f, cache.di_extras, cache.autodiff, u, Constant(p))
    end
    return cache.J
end

## Actually Compute the Jacobian
function (cache::JacobianCache)(u)
    cache.stats.njacs += 1
    (; f, J, p) = cache
    if SciMLBase.isinplace(f)
        if SciMLBase.has_jac(f)
            f.jac(J, u, p)
        else
            DI.jacobian!(
                f, cache.fu, J, cache.di_extras, cache.autodiff, u, Constant(p)
            )
        end
        return J
    else
        if SciMLBase.has_jac(cache.f)
            cache.J = f.jac(u, p)
        else
            cache.J = DI.jacobian(f, cache.di_extras, cache.autodiff, u, Constant(p))
        end
        return cache.J
    end
end

function (cache::JacobianCache{<:JacobianOperator})(u)
    return StatefulJacobianOperator(cache.J, u, cache.p)
end

# Sparse Automatic Differentiation
function construct_concrete_adtype(f::NonlinearFunction, ad::AbstractADType)
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
            coloring_algorithm = select_fastest_coloring_algorithm(f.jac_prototype, f, ad)
            coloring_algorithm === nothing && return ad
            return AutoSparse(
                ad;
                sparsity_detector = KnownJacobianSparsityDetector(f.jac_prototype),
                coloring_algorithm
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
            coloring_algorithm = select_fastest_coloring_algorithm(f.sparsity, f, ad)
            coloring_algorithm === nothing && return ad
            return AutoSparse(
                ad;
                sparsity_detector = KnownJacobianSparsityDetector(f.sparsity),
                coloring_algorithm
            )
        end

        @assert f.sparsity isa ADTypes.AbstractSparsityDetector
        sparsity_detector = f.sparsity
        if f.jac_prototype === nothing
            if SciMLBase.has_colorvec(f)
                @warn "`colorvec` is provided but `jac_prototype` is not specified. \
                       `colorvec` will be ignored."
            end
            coloring_algorithm = select_fastest_coloring_algorithm(nothing, f, ad)
            coloring_algorithm === nothing && return ad
            return AutoSparse(ad; sparsity_detector, coloring_algorithm)
        else
            if sparse_or_structured_prototype(f.jac_prototype)
                if !(sparsity_detector isa NoSparsityDetector)
                    @warn lazy"`jac_prototype` is a sparse matrix but sparsity = $(f.sparsity) \
                           has also been specified. Ignoring sparsity field and using \
                           `jac_prototype` sparsity."
                end
                sparsity_detector = KnownJacobianSparsityDetector(f.jac_prototype)
            end
            coloring_algorithm = select_fastest_coloring_algorithm(f.jac_prototype, f, ad)
            coloring_algorithm === nothing && return ad
            return AutoSparse(ad; sparsity_detector, coloring_algorithm)
        end
    end
end

function construct_concrete_adtype(::NonlinearFunction, ad::AutoSparse)
    error(lazy"Specifying a sparse AD type for Nonlinear Problems was removed in v4. \
           Instead use the `sparsity`, `jac_prototype`, and `colorvec` to specify \
           the right sparsity pattern and coloring algorithm. Ignoring the sparsity \
           detection algorithm and coloring algorithm present in $(ad).")
end

function select_fastest_coloring_algorithm(
        prototype, f::NonlinearFunction, ad::AbstractADType)
    if !Utils.is_extension_loaded(Val(:SparseMatrixColorings))
        @warn "`SparseMatrixColorings` must be explicitly imported for sparse automatic \
               differentiation to work. Proceeding with Dense Automatic Differentiation."
        return nothing
    end
    return select_fastest_coloring_algorithm(Val(:SparseMatrixColorings), prototype, f, ad)
end

function sparse_or_structured_prototype(prototype::AbstractMatrix)
    return ArrayInterface.isstructured(prototype)
end
