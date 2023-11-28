"""
    SimpleDFSane(; σ_min::Real = 1e-10, σ_max::Real = 1e10, σ_1::Real = 1.0,
        M::Int = 10, γ::Real = 1e-4, τ_min::Real = 0.1, τ_max::Real = 0.5,
        nexp::Int = 2, η_strategy::Function = (f_1, k, x, F) -> f_1 ./ k^2)

A low-overhead implementation of the df-sane method for solving large-scale nonlinear
systems of equations. For in depth information about all the parameters and the algorithm,
see the paper: [W LaCruz, JM Martinez, and M Raydan (2006), Spectral residual mathod without
gradient information for solving large-scale nonlinear systems of equations, Mathematics of
Computation, 75, 1429-1448.](https://www.researchgate.net/publication/220576479_Spectral_Residual_Method_without_Gradient_Information_for_Solving_Large-Scale_Nonlinear_Systems_of_Equations)

### Keyword Arguments

  - `σ_min`: the minimum value of the spectral coefficient `σ_k` which is related to the step
    size in the algorithm. Defaults to `1e-10`.
  - `σ_max`: the maximum value of the spectral coefficient `σ_k` which is related to the step
    size in the algorithm. Defaults to `1e10`.
  - `σ_1`: the initial value of the spectral coefficient `σ_k` which is related to the step
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
  - `nexp`: the exponent of the loss, i.e. ``f_k=||F(x_k)||^{nexp}``. The paper uses
    `nexp ∈ {1,2}`. Defaults to `2`.
  - `η_strategy`:  function to determine the parameter `η_k`, which enables growth
    of ``||F||^2``. Called as ``η_k = η_strategy(f_1, k, x, F)`` with `f_1` initialized as
    ``f_1=||F(x_1)||^{nexp}``, `k` is the iteration number, `x` is the current `x`-value and
    `F` the current residual. Should satisfy ``η_k > 0`` and ``∑ₖ ηₖ < ∞``. Defaults to
    ``||F||^2 / k^2``.
"""
@kwdef @concrete struct SimpleDFSane <: AbstractSimpleNonlinearSolveAlgorithm
    σ_min = 1e-10
    σ_max = 1e10
    σ_1 = 1.0
    M::Int = 10
    γ = 1e-4
    τ_min = 0.1
    τ_max = 0.5
    nexp::Int = 2
    η_strategy = (f_1, k, x, F) -> f_1 ./ k^2
end

function SciMLBase.__solve(prob::NonlinearProblem, alg::SimpleDFSane, args...;
        abstol = nothing, reltol = nothing, maxiters = 1000,
        termination_condition = nothing, kwargs...)
    x = float(copy(prob.u0))
    fx = _get_fx(prob, x)
    T = eltype(x)

    σ_min = T(alg.σ_min)
    σ_max = T(alg.σ_max)
    σ_k = T(alg.σ_1)

    (; M, nexp, η_strategy) = alg
    γ = T(alg.γ)
    τ_min = T(alg.τ_min)
    τ_max = T(alg.τ_max)

    abstol, reltol, tc_cache = init_termination_cache(abstol, reltol, fx, x,
        termination_condition)

    fx_norm = NONLINEARSOLVE_DEFAULT_NORM(fx)^nexp
    α_1 = one(T)
    f_1 = fx_norm

    history_f_k = fill(fx_norm, M)

    # Generate the cache
    @bb d = copy(x)
    @bb xo = copy(x)
    @bb x_cache = copy(x)
    @bb δx = copy(x)
    @bb fxo = copy(fx)
    @bb δf = copy(fx)

    k = 0
    while k < maxiters
        # Spectral parameter range check
        σ_k = sign(σ_k) * clamp(abs(σ_k), σ_min, σ_max)

        # Line search direction
        @bb @. d = -σ_k * fx

        η = η_strategy(f_1, k, x, fx)
        f_bar = maximum(history_f_k)
        α_p = α_1
        α_m = α_1

        @bb @. x += α_p * d

        fx = __eval_f(prob, fx, x)
        fx_norm_new = NONLINEARSOLVE_DEFAULT_NORM(fx)^nexp

        while k < maxiters
            fx_norm_new ≤ (f_bar + η - γ * α_p^2 * fx_norm) && break

            α_p = α_p^2 * fx_norm / (fx_norm_new + (T(2) * α_p - T(1)) * fx_norm)
            @bb @. x -= α_m * d

            fx = __eval_f(prob, fx, x)
            fx_norm_new = NONLINEARSOLVE_DEFAULT_NORM(fx)^nexp

            fx_norm_new ≤ (f_bar + η - γ * α_m^2 * fx_norm) && break

            α_tm = α_m^2 * fx_norm / (fx_norm_new + (T(2) * α_m - T(1)) * fx_norm)
            α_p = clamp(α_p, τ_min * α_p, τ_max * α_p)
            α_m = clamp(α_tm, τ_min * α_m, τ_max * α_m)
            @bb @. x += α_p * d

            fx = __eval_f(prob, fx, x)
            fx_norm_new = NONLINEARSOLVE_DEFAULT_NORM(fx)^nexp

            k += 1
        end

        tc_sol = check_termination(tc_cache, fx, x, xo, prob, alg)
        tc_sol !== nothing && return tc_sol

        # Update spectral parameter
        @bb @. δx = x - xo
        @bb @. δf = fx - fxo

        σ_k = dot(δx, δx) / dot(δx, δf)

        # Take step
        @bb copyto!(xo, x)
        @bb copyto!(fxo, fx)
        fx_norm = fx_norm_new

        # Store function value
        history_f_k[mod1(k, M)] = fx_norm_new
        k += 1
    end

    return build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end
