"""
    SimpleLimitedMemoryBroyden(; threshold::Int = 27)
    SimpleLimitedMemoryBroyden(; threshold::Val = Val(27))

A limited memory implementation of Broyden. This method applies the L-BFGS scheme to
Broyden's method. This Alogrithm unfortunately cannot non-allocating for StaticArrays
without compromising on the "simple" aspect.

If the threshold is larger than the problem size, then this method will use `SimpleBroyden`.

!!! warning

    This method is not very stable and can diverge even for very simple problems. This has
    mostly been tested for neural networks in DeepEquilibriumNetworks.jl.
"""
struct SimpleLimitedMemoryBroyden{threshold} <: AbstractSimpleNonlinearSolveAlgorithm end

__get_threshold(::SimpleLimitedMemoryBroyden{threshold}) where {threshold} = Val(threshold)

function SimpleLimitedMemoryBroyden(; threshold::Union{Val, Int} = Val(27))
    return SimpleLimitedMemoryBroyden{SciMLBase._unwrap_val(threshold)}()
end

@views function SciMLBase.__solve(prob::NonlinearProblem, alg::SimpleLimitedMemoryBroyden,
        args...; abstol = nothing, reltol = nothing, maxiters = 1000, alias_u0 = false,
        termination_condition = nothing, kwargs...)
    x = __maybe_unaliased(prob.u0, alias_u0)
    threshold = __get_threshold(alg)
    η = min(SciMLBase._unwrap_val(threshold), maxiters)

    # For scalar problems / if the threshold is larger than problem size just use Broyden
    if x isa Number || length(x) ≤ η
        return SciMLBase.__solve(prob, SimpleBroyden(), args...;
            abstol, reltol, maxiters, termination_condition, kwargs...)
    end

    fx = _get_fx(prob, x)

    U, Vᵀ = __init_low_rank_jacobian(x, fx, threshold)

    abstol, reltol, tc_cache = init_termination_cache(abstol, reltol, fx, x,
        termination_condition)

    @bb xo = copy(x)
    @bb δx = copy(fx)
    @bb δx .*= -1
    @bb fo = copy(fx)
    @bb δf = copy(fx)

    @bb vᵀ_cache = copy(x)
    Tcache = __lbroyden_threshold_cache(x, threshold)
    @bb mat_cache = copy(x)

    for i in 1:maxiters
        @bb @. x = xo + δx
        fx = __eval_f(prob, fx, x)
        @bb @. δf = fx - fo

        # Termination Checks
        tc_sol = check_termination(tc_cache, fx, x, xo, prob, alg)
        tc_sol !== nothing && return tc_sol

        _U = selectdim(U, 2, 1:min(η, i - 1))
        _Vᵀ = selectdim(Vᵀ, 1, 1:min(η, i - 1))

        vᵀ = _rmatvec!!(vᵀ_cache, Tcache, _U, _Vᵀ, δx)
        mvec = _matvec!!(mat_cache, Tcache, _U, _Vᵀ, δf)
        d = dot(vᵀ, δf)
        @bb @. δx = (δx - mvec) / d

        selectdim(U, 2, mod1(i, η)) .= _vec(δx)
        selectdim(Vᵀ, 1, mod1(i, η)) .= _vec(vᵀ)

        _U = selectdim(U, 2, 1:min(η, i))
        _Vᵀ = selectdim(Vᵀ, 1, 1:min(η, i))
        δx = _matvec!!(δx, Tcache, _U, _Vᵀ, fx)
        @bb @. δx *= -1

        @bb copyto!(xo, x)
        @bb copyto!(fo, fx)
    end

    return build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end

function _rmatvec!!(y, xᵀU, U, Vᵀ, x)
    # xᵀ × (-I + UVᵀ)
    η = size(U, 2)
    if η == 0
        @bb @. y = -x
        return y
    end
    x_ = vec(x)
    xᵀU_ = view(xᵀU, 1:η)
    @bb xᵀU_ = transpose(U) × x_
    @bb y = transpose(Vᵀ) × vec(xᵀU_)
    @bb @. y -= x
    return y
end

function _matvec!!(y, Vᵀx, U, Vᵀ, x)
    # (-I + UVᵀ) × x
    η = size(U, 2)
    if η == 0
        @bb @. y = -x
        return y
    end
    x_ = vec(x)
    Vᵀx_ = view(Vᵀx, 1:η)
    @bb Vᵀx_ = Vᵀ × x_
    @bb y = U × vec(Vᵀx_)
    @bb @. y -= x
    return y
end

__lbroyden_threshold_cache(x, ::Val{threshold}) where {threshold} = similar(x, threshold)
function __lbroyden_threshold_cache(x::SArray, ::Val{threshold}) where {threshold}
    return SArray{Tuple{threshold}, eltype(x)}(ntuple(_ -> zero(eltype(x)), threshold))
end
