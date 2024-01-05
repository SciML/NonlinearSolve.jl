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
    δus
    lincache
    JᵀJ_cache  # For normal form else nothing
    Jᵀfu_cache
    timer::TimerOutput
end

@internal_caches NewtonDescentCache :lincache

function SciMLBase.init(prob::NonlinearProblem, alg::NewtonDescent, J, fu, u;
        shared::Val{N} = Val(1), pre_inverted::Val{INV} = False, linsolve_kwargs = (;),
        abstol = nothing, reltol = nothing, timer = TimerOutput(), kwargs...) where {INV, N}
    @bb δu = similar(u)
    δus = N ≤ 1 ? nothing : map(2:N) do i
        @bb δu_ = similar(u)
    end
    INV && return NewtonDescentCache{true, false}(δu, δus, nothing, nothing, nothing, timer)
    lincache = LinearSolverCache(alg, alg.linsolve, J, _vec(fu), _vec(u); abstol, reltol,
        linsolve_kwargs...)
    return NewtonDescentCache{false, false}(δu, δus, lincache, nothing, nothing, timer)
end

function SciMLBase.init(prob::NonlinearLeastSquaresProblem, alg::NewtonDescent, J, fu, u;
        pre_inverted::Val{INV} = False, linsolve_kwargs = (;), shared::Val{N} = Val(1),
        abstol = nothing, reltol = nothing, timer = TimerOutput(), kwargs...) where {INV, N}
    length(fu) != length(u) &&
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
    δus = N ≤ 1 ? nothing : map(2:N) do i
        @bb δu_ = similar(u)
    end
    return NewtonDescentCache{false, normal_form}(δu, δus, lincache, JᵀJ, Jᵀfu, timer)
end

function SciMLBase.solve!(cache::NewtonDescentCache{INV, false}, J, fu, u,
        idx::Val = Val(1); skip_solve::Bool = false, kwargs...) where {INV}
    δu = get_du(cache, idx)
    skip_solve && return δu
    if INV
        @assert J!==nothing "`J` must be provided when `pre_inverted = Val(true)`."
        @bb δu = J × vec(fu)
    else
        @timeit_debug cache.timer "linear solve" begin
            δu = cache.lincache(; A = J, b = _vec(fu), kwargs..., linu = _vec(δu),
                du = _vec(δu))
            δu = _restructure(get_du(cache, idx), δu)
        end
    end
    @bb @. δu *= -1
    set_du!(cache, δu, idx)
    return δu, true, (;)
end

function SciMLBase.solve!(cache::NewtonDescentCache{false, true}, J, fu, u,
        idx::Val = Val(1); skip_solve::Bool = false, kwargs...)
    δu = get_du(cache, idx)
    skip_solve && return δu
    @bb cache.JᵀJ_cache = transpose(J) × J
    @bb cache.Jᵀfu_cache = transpose(J) × fu
    @timeit_debug cache.timer "linear solve" begin
        δu = cache.lincache(; A = __maybe_symmetric(cache.JᵀJ_cache), b = cache.Jᵀfu_cache,
            kwargs..., linu = _vec(δu), du = _vec(δu))
        δu = _restructure(get_du(cache, idx), δu)
    end
    @bb @. δu *= -1
    set_du!(cache, δu, idx)
    return δu, true, (;)
end
