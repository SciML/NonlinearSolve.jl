"""
    DFSane(; σ_min::Real = 1e-10, σ_max::Real = 1e10, σ_1::Real = 1.0, M::Int = 10,
        γ::Real = 1e-4, τ_min::Real = 0.1, τ_max::Real = 0.5, n_exp::Int = 2,
        η_strategy::Function = (fn_1, n, x_n, f_n) -> fn_1 / n^2,
        max_inner_iterations::Int = 100)

A low-overhead and allocation-free implementation of the df-sane method for solving large-scale nonlinear
systems of equations. For in depth information about all the parameters and the algorithm,
see the paper: [W LaCruz, JM Martinez, and M Raydan (2006), Spectral Residual Method without
Gradient Information for Solving Large-Scale Nonlinear Systems of Equations, Mathematics of
Computation, 75, 1429-1448.](https://www.researchgate.net/publication/220576479_Spectral_Residual_Method_without_Gradient_Information_for_Solving_Large-Scale_Nonlinear_Systems_of_Equations)

### Keyword Arguments

  - `σ_min`: the minimum value of the spectral coefficient `σₙ` which is related to the step
    size in the algorithm. Defaults to `1e-10`.
  - `σ_max`: the maximum value of the spectral coefficient `σₙ` which is related to the step
    size in the algorithm. Defaults to `1e10`.
  - `σ_1`: the initial value of the spectral coefficient `σₙ` which is related to the step
    size in the algorithm.. Defaults to `1.0`.
  - `M`: The monotonicity of the algorithm is determined by a this positive integer.
    A value of 1 for `M` would result in strict monotonicity in the decrease of the L2-norm
    of the function `f`. However, higher values allow for more flexibility in this reduction.
    Despite this, the algorithm still ensures global convergence through the use of a
    non-monotone line-search algorithm that adheres to the Grippo-Lampariello-Lucidi
    condition. Values in the range of 5 to 20 are usually sufficient, but some cases may call
    for a higher value of `M`. The default setting is 10.
  - `γ`: a parameter that influences if a proposed step will be accepted. Higher value of `γ`
    will make the algorithm more restrictive in accepting steps. Defaults to `1e-4`.
  - `τ_min`: if a step is rejected the new step size will get multiplied by factor, and this
    parameter is the minimum value of that factor. Defaults to `0.1`.
  - `τ_max`: if a step is rejected the new step size will get multiplied by factor, and this
    parameter is the maximum value of that factor. Defaults to `0.5`.
  - `n_exp`: the exponent of the loss, i.e. ``f_n=||F(x_n)||^{n_exp}``. The paper uses
    `n_exp ∈ {1,2}`. Defaults to `2`.
  - `η_strategy`:  function to determine the parameter `η`, which enables growth
    of ``||f_n||^2``. Called as ``η = η_strategy(fn_1, n, x_n, f_n)`` with `fn_1` initialized as
    ``fn_1=||f(x_1)||^{n_exp}``, `n` is the iteration number, `x_n` is the current `x`-value and
    `f_n` the current residual. Should satisfy ``η > 0`` and ``∑ₖ ηₖ < ∞``. Defaults to
    ``fn_1 / n^2``.
  - `max_inner_iterations`: the maximum number of iterations allowed for the inner loop of the
    algorithm. Defaults to `100`.
"""
@kwdef @concrete struct DFSane <: AbstractNonlinearSolveAlgorithm
    σ_min = 1e-10
    σ_max = 1e10
    σ_1 = 1.0
    M::Int = 10
    γ = 1e-4
    τ_min = 0.1
    τ_max = 0.5
    n_exp::Int = 2
    η_strategy = (fn_1, n, x_n, f_n) -> fn_1 / n^2
    max_inner_iterations::Int = 100
end

@concrete mutable struct DFSaneCache{iip} <: AbstractNonlinearSolveCache{iip}
    alg
    u
    u_cache
    fu
    fu_cache
    du
    history
    f_norm
    f_norm_0
    M
    σ_n
    σ_min
    σ_max
    α_1
    γ
    τ_min
    τ_max
    n_exp::Int
    p
    force_stop::Bool
    maxiters::Int
    internalnorm
    retcode::SciMLBase.ReturnCode.T
    abstol
    reltol
    prob
    stats::NLStats
    tc_cache
    trace
end

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg::DFSane, args...;
        alias_u0 = false, maxiters = 1000, abstol = nothing, reltol = nothing,
        termination_condition = nothing, internalnorm::F = DEFAULT_NORM,
        kwargs...) where {uType, iip, F}
    u = __maybe_unaliased(prob.u0, alias_u0)
    T = eltype(u)

    @bb du = similar(u)
    @bb u_cache = copy(u)

    fu = evaluate_f(prob, u)
    @bb fu_cache = copy(fu)

    f_norm = internalnorm(fu)^alg.n_exp
    f_norm_0 = f_norm

    history = fill(f_norm, alg.M)

    abstol, reltol, tc_cache = init_termination_cache(abstol, reltol, fu, u_cache,
        termination_condition)
    trace = init_nonlinearsolve_trace(alg, u, fu, nothing, du; kwargs...)

    return DFSaneCache{iip}(alg, u, u_cache, fu, fu_cache, du, history, f_norm, f_norm_0,
        alg.M, T(alg.σ_1), T(alg.σ_min), T(alg.σ_max), one(T), T(alg.γ), T(alg.τ_min),
        T(alg.τ_max), alg.n_exp, prob.p, false, maxiters, internalnorm, ReturnCode.Default,
        abstol, reltol, prob, NLStats(1, 0, 0, 0, 0), tc_cache, trace)
end

function perform_step!(cache::DFSaneCache{iip}) where {iip}
    @unpack alg, f_norm, σ_n, σ_min, σ_max, α_1, γ, τ_min, τ_max, n_exp, M, prob = cache
    T = eltype(cache.u)
    f_norm_old = f_norm

    # Spectral parameter range check
    σ_n = sign(σ_n) * clamp(abs(σ_n), σ_min, σ_max)

    # Line search direction
    @bb @. cache.du = -σ_n * cache.fu

    η = alg.η_strategy(cache.f_norm_0, cache.stats.nsteps, cache.u, cache.fu)

    f_bar = maximum(cache.history)
    α₊ = α_1
    α₋ = α_1

    @bb axpy!(α₊, cache.du, cache.u)

    evaluate_f(cache, cache.u, cache.p)
    f_norm = cache.internalnorm(cache.fu)^n_exp
    α = α₊

    inner_converged = false
    for k in 1:(cache.alg.max_inner_iterations)
        if f_norm ≤ f_bar + η - γ * α₊^2 * f_norm_old
            α = α₊
            inner_converged = true
            break
        end

        α₊ = α₊ * clamp(α₊ * f_norm_old / (f_norm + (T(2) * α₊ - T(1)) * f_norm_old),
            τ_min, τ_max)
        @bb axpy!(-α₋, cache.du, cache.u)

        evaluate_f(cache, cache.u, cache.p)
        f_norm = cache.internalnorm(cache.fu)^n_exp

        if f_norm ≤ f_bar + η - γ * α₋^2 * f_norm_old
            α = α₋
            inner_converged = true
            break
        end

        α₋ = α₋ * clamp(α₋ * f_norm_old / (f_norm + (T(2) * α₋ - T(1)) * f_norm_old),
            τ_min, τ_max)
        @bb axpy!(α₊, cache.du, cache.u)

        evaluate_f(cache, cache.u, cache.p)
        f_norm = cache.internalnorm(cache.fu)^n_exp
    end

    if !inner_converged
        cache.retcode = ReturnCode.ConvergenceFailure
        cache.force_stop = true
    end

    update_trace!(cache, α)
    check_and_update!(cache, cache.fu, cache.u, cache.u_cache)

    # Update spectral parameter
    @bb @. cache.u_cache = cache.u - cache.u_cache
    @bb @. cache.fu_cache = cache.fu - cache.fu_cache

    α₊ = sum(abs2, cache.u_cache)
    @bb @. cache.u_cache *= cache.fu_cache
    α₋ = sum(cache.u_cache)
    cache.σ_n = α₊ / α₋

    # Spectral parameter bounds check
    if !(σ_min ≤ abs(cache.σ_n) ≤ σ_max)
        test_norm = sqrt(sum(abs2, cache.fuprev))
        cache.σ_n = clamp(inv(test_norm), T(1), T(1e5))
    end

    # Take step
    @bb copyto!(cache.u_cache, cache.u)
    @bb copyto!(cache.fu_cache, cache.fu)
    cache.f_norm = f_norm

    # Update history
    cache.history[cache.stats.nsteps % M + 1] = f_norm
    cache.stats.nf += 1
    return nothing
end

function __reinit_internal!(cache::DFSaneCache; kwargs...)
    cache.f_norm = cache.internalnorm(cache.fu)^cache.n_exp
    cache.f_norm_0 = cache.f_norm
    return
end
