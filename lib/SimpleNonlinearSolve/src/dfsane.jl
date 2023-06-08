"""
```julia
SimpleDFSane(; σ_min::Real = 1e-10, σ_max::Real = 1e10, σ_1::Real = 1.0,
             M::Int = 10, γ::Real = 1e-4, τ_min::Real = 0.1, τ_max::Real = 0.5,
             nexp::Int = 2, η_strategy::Function = (f_1, k, x, F) -> f_1 / k^2)
```

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
struct SimpleDFSane{T} <: AbstractSimpleNonlinearSolveAlgorithm
    σ_min::T
    σ_max::T
    σ_1::T
    M::Int
    γ::T
    τ_min::T
    τ_max::T
    nexp::Int
    η_strategy::Function

    function SimpleDFSane(; σ_min::Real = 1e-10, σ_max::Real = 1e10, σ_1::Real = 1.0,
        M::Int = 10, γ::Real = 1e-4, τ_min::Real = 0.1, τ_max::Real = 0.5,
        nexp::Int = 2, η_strategy::Function = (f_1, k, x, F) -> f_1 / k^2)
        new{typeof(σ_min)}(σ_min, σ_max, σ_1, M, γ, τ_min, τ_max, nexp, η_strategy)
    end
end

function SciMLBase.__solve(prob::NonlinearProblem, alg::SimpleDFSane,
    args...; abstol = nothing, reltol = nothing, maxiters = 1000,
    kwargs...)
    f = Base.Fix2(prob.f, prob.p)
    x = float(prob.u0)
    T = eltype(x)
    σ_min = float(alg.σ_min)
    σ_max = float(alg.σ_max)
    σ_k = float(alg.σ_1)
    M = alg.M
    γ = float(alg.γ)
    τ_min = float(alg.τ_min)
    τ_max = float(alg.τ_max)
    nexp = alg.nexp
    η_strategy = alg.η_strategy

    if SciMLBase.isinplace(prob)
        error("SimpleDFSane currently only supports out-of-place nonlinear problems")
    end

    atol = abstol !== nothing ? abstol :
           real(oneunit(eltype(T))) * (eps(real(one(eltype(T)))))^(4 // 5)
    rtol = reltol !== nothing ? reltol : eps(real(one(eltype(T))))^(4 // 5)

    function ff(x)
        F = f(x)
        f_k = norm(F)^nexp
        return f_k, F
    end

    f_k, F_k = ff(x)
    α_1 = convert(T, 1.0)
    f_1 = f_k
    history_f_k = fill(f_k, M)

    for k in 1:maxiters
        iszero(F_k) &&
            return SciMLBase.build_solution(prob, alg, x, F_k;
                retcode = ReturnCode.Success)

        # Spectral parameter range check
        if abs(σ_k) > σ_max
            σ_k = sign(σ_k) * σ_max
        elseif abs(σ_k) < σ_min
            σ_k = sign(σ_k) * σ_min
        end

        # Line search direction
        d = -σ_k * F_k

        η = η_strategy(f_1, k, x, F_k)
        f̄ = maximum(history_f_k)
        α_p = α_1
        α_m = α_1
        x_new = x + α_p * d
        f_new, F_new = ff(x_new)
        while true
            if f_new ≤ f̄ + η - γ * α_p^2 * f_k
                break
            end

            α_tp = α_p^2 * f_k / (f_new + (2 * α_p - 1) * f_k)
            x_new = x - α_m * d
            f_new, F_new = ff(x_new)

            if f_new ≤ f̄ + η - γ * α_m^2 * f_k
                break
            end

            α_tm = α_m^2 * f_k / (f_new + (2 * α_m - 1) * f_k)
            α_p = min(τ_max * α_p, max(α_tp, τ_min * α_p))
            α_m = min(τ_max * α_m, max(α_tm, τ_min * α_m))
            x_new = x + α_p * d
            f_new, F_new = ff(x_new)
        end

        if isapprox(x_new, x, atol = atol, rtol = rtol)
            return SciMLBase.build_solution(prob, alg, x_new, F_new;
                retcode = ReturnCode.Success)
        end
        # Update spectral parameter
        s_k = x_new - x
        y_k = F_new - F_k
        σ_k = (s_k' * s_k) / (s_k' * y_k)

        # Take step
        x = x_new
        F_k = F_new
        f_k = f_new

        # Store function value
        history_f_k[k % M + 1] = f_new
    end
    return SciMLBase.build_solution(prob, alg, x, F_k; retcode = ReturnCode.MaxIters)
end
