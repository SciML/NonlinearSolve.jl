"""
    SteepestDescent(; linsolve = nothing, precs = DEFAULT_PRECS)

Compute the descent direction as ``δu = -Jᵀfu``.

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

The linear solver and preconditioner are only used if `J` is provided in the inverted form.

See also [`Dogleg`](@ref), [`NewtonDescent`](@ref), [`DampedNewtonDescent`](@ref).
"""
@kwdef @concrete struct SteepestDescent <: AbstractDescentAlgorithm
    linsolve = nothing
    precs = DEFAULT_PRECS
end

supports_line_search(::SteepestDescent) = true

@concrete mutable struct SteepestDescentCache{pre_inverted} <: AbstractDescentCache
    δu
    δus
    lincache
    timer::TimerOutput
end

@internal_caches SteepestDescentCache :lincache

@inline function SciMLBase.init(prob::AbstractNonlinearProblem, alg::SteepestDescent, J, fu,
        u; shared::Val{N} = Val(1), pre_inverted::Val{INV} = False, linsolve_kwargs = (;),
        abstol = nothing, reltol = nothing, timer = TimerOutput(), kwargs...) where {INV, N}
    INV && @assert length(fu)==length(u) "Non-Square Jacobian Inverse doesn't make sense."
    @bb δu = similar(u)
    δus = N ≤ 1 ? nothing : map(2:N) do i
        @bb δu_ = similar(u)
    end
    if INV
        lincache = LinearSolverCache(alg, alg.linsolve, transpose(J), _vec(fu), _vec(u);
            abstol, reltol, linsolve_kwargs...)
    else
        lincache = nothing
    end
    return SteepestDescentCache{INV}(δu, δus, lincache, timer)
end

function SciMLBase.solve!(cache::SteepestDescentCache{INV}, J, fu, u, idx::Val = Val(1);
        kwargs...) where {INV}
    δu = get_du(cache, idx)
    if INV
        A = J === nothing ? nothing : transpose(J)
        @timeit_debug cache.timer "linear solve" begin
            δu = cache.lincache(; A, b = _vec(fu), kwargs..., linu = _vec(δu),
                du = _vec(δu))
            δu = _restructure(get_du(cache, idx), δu)
        end
    else
        @assert J!==nothing "`J` must be provided when `pre_inverted = Val(false)`."
        @bb δu = transpose(J) × vec(fu)
    end
    @bb @. δu *= -1
    set_du!(cache, δu, idx)
    return δu, true, (;)
end
