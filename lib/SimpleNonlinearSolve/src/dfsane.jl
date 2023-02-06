"""
```julia
SimpleDFSane(; σ_min::Real = 1e-10, σ_0::Real = 1.0, M::Int = 10,
            γ::Real = 1e-4, τ_min::Real = 0.1, τ_max::Real = 0.5,
            nexp::Int = 2)
```

A low-overhead implementation of the df-sane method. For more information, see [1].
References:
    W LaCruz, JM Martinez, and M Raydan (2006), Spectral residual mathod without gradient information for solving large-scale nonlinear systems of equations, Mathematics of Computation, 75, 1429-1448.
### Keyword Arguments


- `σ_min`: the minimum value of `σ_k`. Defaults to `1e-10`. # TODO write about this...
- `σ_0`: the initial value of `σ_k`. Defaults to `1.0`. # TODO write about this...
- `M`: the value of `M` in the paper. Defaults to `10`. # TODO write about this...
- `γ`: the value of `γ` in the paper. Defaults to `1e-4`. # TODO write about this...
- `τ_min`: the minimum value of `τ_k`. Defaults to `0.1`. # TODO write about this...
- `τ_max`: the maximum value of `τ_k`. Defaults to `0.5`. # TODO write about this...
- `nexp`: the value of `nexp` in the paper. Defaults to `2`. # TODO write about this...

"""
struct SimpleDFSane{T} <: AbstractSimpleNonlinearSolveAlgorithm
    σ_min::T
    σ_0::T
    M::Int
    γ::T
    τ_min::T
    τ_max::T
    nexp::Int

    function SimpleDFSane(; σ_min::Real = 1e-10, σ_0::Real = 1.0, M::Int = 10,
                          γ::Real = 1e-4, τ_min::Real = 0.1, τ_max::Real = 0.5,
                          nexp::Int = 2)
        new{typeof(σ_min)}(σ_min, σ_0, M, γ, τ_min, τ_max, nexp)
    end
end

function SciMLBase.__solve(prob::NonlinearProblem,
                           alg::SimpleDFSane, args...; abstol = nothing,
                           reltol = nothing,
                           maxiters = 1000, kwargs...)
    f = Base.Fix2(prob.f, prob.p)
    x = float(prob.u0)
    T = eltype(x)
    σ_min = float(alg.σ_min)
    σ_k = float(alg.σ_0)
    M = alg.M
    γ = float(alg.γ)
    τ_min = float(alg.τ_min)
    τ_max = float(alg.τ_max)
    nexp = alg.nexp

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
    α_0 = convert(T, 1.0)
    f_0 = f_k
    prev_fs = fill(f_k, M)

    for k in 1:maxiters
        iszero(F_k) &&
            return SciMLBase.build_solution(prob, alg, x, F_k;
                                            retcode = ReturnCode.Success)

        # Control spectral parameter
        if abs(σ_k) > 1 / σ_min
            σ_k = 1 / σ_min * sign(σ_k)
        elseif abs(σ_k) < σ_min
            σ_k = σ_min
        end

        # Line search direction
        d = -σ_k * F_k

        # Nonmonotone line search
        η = f_0 / k^2

        f_bar = maximum(prev_fs)
        α_p = α_0
        α_m = α_0
        xp = x + α_p * d
        fp, Fp = ff(xp)
        while true
            if fp ≤ f_bar + η - γ * α_p^2 * f_k
                break
            end

            α_tp = α_p^2 * f_k / (fp + (2 * α_p - 1) * f_k)
            xp = x - α_m * d
            fp, Fp = ff(xp)

            if fp ≤ f_bar + η - γ * α_m^2 * f_k
                break
            end

            α_tm = α_m^2 * f_k / (fp + (2 * α_m - 1) * f_k)
            α_p = min(τ_max * α_p, max(α_tp, τ_min * α_p))
            α_m = min(τ_max * α_m, max(α_tm, τ_min * α_m))
            xp = x + α_p * d
            fp, Fp = ff(xp)
        end

        if isapprox(xp, x, atol = atol, rtol = rtol)
            return SciMLBase.build_solution(prob, alg, xp, Fp;
                                            retcode = ReturnCode.Success)
        end
        # Update spectral parameter
        s_k = xp - x
        y_k = Fp - F_k
        σ_k = dot(s_k, s_k) / dot(s_k, y_k)

        # Take step
        x = xp
        F_k = Fp
        f_k = fp

        # Store function value
        idx_to_replace = k % M + 1
        prev_fs[idx_to_replace] = fp
    end
    return SciMLBase.build_solution(prob, alg, x, F_k; retcode = ReturnCode.MaxIters)
end
