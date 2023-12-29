"""
    NewtonDescent(; linsolve = nothing, precs = DEFAULT_PRECS)

Compute the descent direction as ``J δu = -fu``. For non-square Jacobian problems, this is
commonly refered to as the Gauss-Newton Descent.

### Keyword Arguments

  - `linsolve`: the [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) used for the
    linear solves within the Newton method. Defaults to `nothing`, which means it uses the
    LinearSolve.jl default algorithm choice. For more information on available algorithm
    choices, see the
    [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/).
  - `precs`: the choice of preconditioners for the linear solver. Defaults to using no
    preconditioners. For more information on specifying preconditioners for LinearSolve
    algorithms, consult the
    [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/).

See also [`Dogleg`](@ref), [`SteepestDescent`](@ref), [`DampedNewtonDescent`](@ref).
"""
@kwdef @concrete struct NewtonDescent <: AbstractDescentAlgorithm
    linsolve = nothing
    precs = DEFAULT_PRECS
end

supports_line_search(::NewtonDescent) = true

@concrete mutable struct NewtonDescentCache{pre_inverted, normalform} <:
                         AbstractDescentCache
    δu
    lincache
    # For normal form else nothing
    JᵀJ_cache
    Jᵀfu_cache
end

function SciMLBase.init(prob::NonlinearProblem, alg::NewtonDescent, J, fu, u;
        pre_inverted::Val{INV} = False, linsolve_kwargs = (;), abstol = nothing,
        reltol = nothing, kwargs...) where {INV}
    @bb δu = similar(u)
    INV && return NewtonDescentCache{true, false}(δu, nothing, nothing, nothing)
    lincache = LinearSolverCache(alg, alg.linsolve, J, _vec(fu), _vec(u); abstol, reltol,
        linsolve_kwargs...)
    return NewtonDescentCache{false, false}(δu, lincache, nothing, nothing)
end

function SciMLBase.init(prob::NonlinearLeastSquaresProblem, alg::NewtonDescent, J, fu, u;
        pre_inverted::Val{INV} = False, linsolve_kwargs = (;),
        abstol = nothing, reltol = nothing, kwargs...) where {INV}
    @assert !INV "Precomputed Inverse for Non-Square Jacobian doesn't make sense."

    normal_form = __needs_square_A(alg.linsolve, u)
    if normal_form
        JᵀJ = transpose(J) * J
        Jᵀfu = transpose(J) * _vec(fu)
        A, b = __maybe_symmetric(JᵀJ), Jᵀfu
    else
        JᵀJ, Jᵀfu = nothing, nothing
        A, b = J, _vec(fu)
    end
    lincache = LinearSolverCache(alg, alg.linsolve, A, b, _vec(u); abstol, reltol,
        linsolve_kwargs...)
    @bb δu = similar(u)
    return NewtonDescentCache{false, normal_form}(δu, lincache, JᵀJ, Jᵀfu)
end

function SciMLBase.solve!(cache::NewtonDescentCache{INV, false}, J, fu;
        skip_solve::Bool = false, kwargs...) where {INV}
    skip_solve && return cache.δu
    if INV
        @assert J!==nothing "`J` must be provided when `pre_inverted = Val(true)`."
        @bb cache.δu = J × vec(fu)
    else
        δu = cache.lincache(; A = J, b = _vec(fu), kwargs..., linu = _vec(cache.δu),
            du = _vec(cache.δu))
        cache.δu = _restructure(cache.δu, δu)
    end
    @bb @. cache.δu *= -1
    return cache.δu
end

function SciMLBase.solve!(cache::NewtonDescentCache{false, true}, J, fu;
        skip_solve::Bool = false, kwargs...)
    skip_solve && return cache.δu
    @bb cache.JᵀJ_cache = transpose(J) × J
    @bb cache.Jᵀfu_cache = transpose(J) × fu
    δu = cache.lincache(; A = __maybe_symmetric(cache.JᵀJ_cache), b = cache.Jᵀfu_cache,
        kwargs..., linu = _vec(cache.δu), du = _vec(cache.δu))
    cache.δu = _restructure(cache.δu, δu)
    @bb @. cache.δu *= -1
    return cache.δu
end
