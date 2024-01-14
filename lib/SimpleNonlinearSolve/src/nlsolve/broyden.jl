"""
    SimpleBroyden(; linesearch = Val(false), alpha = nothing)

A low-overhead implementation of Broyden. This method is non-allocating on scalar
and static array problems.

### Keyword Arguments

  * `linesearch`: If `linesearch` is `Val(true)`, then we use the `LiFukushimaLineSearch`
    [1] line search else no line search is used. For advanced customization of the line
    search, use the [`Broyden`](@ref) algorithm in `NonlinearSolve.jl`.
  * `alpha`: Scale the initial jacobian initialization with `alpha`. If it is `nothing`, we
    will compute the scaling using `2 * norm(fu) / max(norm(u), true)`.

### References

[1] Li, Dong-Hui, and Masao Fukushima. "A derivative-free line search and global convergence
of Broyden-like method for nonlinear equations." Optimization methods and software 13.3
(2000): 181-201.
"""
@concrete struct SimpleBroyden{linesearch} <: AbstractSimpleNonlinearSolveAlgorithm
    alpha
end

function SimpleBroyden(; linesearch = Val(false), alpha = nothing)
    return SimpleBroyden{_unwrap_val(linesearch)}(alpha)
end

__get_linesearch(::SimpleBroyden{LS}) where {LS} = Val(LS)

function SciMLBase.__solve(prob::NonlinearProblem, alg::SimpleBroyden, args...;
        abstol = nothing, reltol = nothing, maxiters = 1000, alias_u0 = false,
        termination_condition = nothing, kwargs...)
    x = __maybe_unaliased(prob.u0, alias_u0)
    fx = _get_fx(prob, x)
    T = promote_type(eltype(x), eltype(fx))

    @bb xo = copy(x)
    @bb δx = copy(x)
    @bb δf = copy(fx)
    @bb fprev = copy(fx)

    if alg.alpha === nothing
        fx_norm = NONLINEARSOLVE_DEFAULT_NORM(fx)
        x_norm = NONLINEARSOLVE_DEFAULT_NORM(x)
        init_α = ifelse(fx_norm ≥ 1e-5, max(x_norm, T(true)) / (2 * fx_norm), T(true))
    else
        init_α = inv(alg.alpha)
    end

    J⁻¹ = __init_identity_jacobian(fx, x, init_α)
    @bb J⁻¹δf = copy(x)
    @bb xᵀJ⁻¹ = copy(x)
    @bb δJ⁻¹n = copy(x)
    @bb δJ⁻¹ = copy(J⁻¹)

    abstol, reltol, tc_cache = init_termination_cache(abstol, reltol, fx, x,
        termination_condition)

    ls_cache = __get_linesearch(alg) === Val(true) ?
               LiFukushimaLineSearch()(prob, fx, x) : nothing

    for _ in 1:maxiters
        @bb δx = J⁻¹ × vec(fprev)
        @bb δx .*= -1

        α = ls_cache === nothing ? true : ls_cache(xo, δx)
        @bb @. x = xo + α * δx
        fx = __eval_f(prob, fx, x)
        @bb @. δf = fx - fprev

        # Termination Checks
        tc_sol = check_termination(tc_cache, fx, x, xo, prob, alg)
        tc_sol !== nothing && return tc_sol

        @bb J⁻¹δf = J⁻¹ × vec(δf)
        d = dot(δx, J⁻¹δf)
        @bb xᵀJ⁻¹ = transpose(J⁻¹) × vec(δx)

        @bb @. δJ⁻¹n = (δx - J⁻¹δf) / d

        δJ⁻¹n_ = _vec(δJ⁻¹n)
        xᵀJ⁻¹_ = _vec(xᵀJ⁻¹)
        @bb δJ⁻¹ = δJ⁻¹n_ × transpose(xᵀJ⁻¹_)
        @bb J⁻¹ .+= δJ⁻¹

        @bb copyto!(xo, x)
        @bb copyto!(fprev, fx)
    end

    return build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end
