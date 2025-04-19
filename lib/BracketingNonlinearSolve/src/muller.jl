"""
    Muller(; middle = nothing)

Muller's method for determining a root of a univariate, scalar function.

The algorithm, described in Sec. 9.5.2 of
[Press et al. (2007)](https://numerical.recipes/book.html), is a generalization
of the secant method, using quadratic interpolation of three points to find the
next estimate for the root. Due to the quadratic interpolation, the method is
well suited for obtaining complex roots.

This method requires three initial guesses `(left, middle, right)` for the
solution. The guesses `(left, right) = tspan` are provided by the
`IntervalNonlinearProblem`, while the `middle` guess may be specified as an
optional keyword argument. In notable contrast to the other
`BracketingNonlinearSolve.jl` solvers, the `Muller` algorithm does not need
`(left, right)` to bracket the root.

### Keyword Arguments

  - `middle`: the initial guess for the middle point. If not provided, the
    midpoint of the interval `(left, right)` is used.
"""
struct Muller{T} <: AbstractBracketingAlgorithm
    middle::T
end

Muller() = Muller(nothing)

function CommonSolve.solve(prob::IntervalNonlinearProblem, alg::Muller, args...;
        abstol = nothing, maxiters = 1000, kwargs...)
    @assert !SciMLBase.isinplace(prob) "`Muller` only supports out-of-place problems."
    xᵢ₋₂, xᵢ = prob.tspan
    xᵢ₋₁ = isnothing(alg.middle) ? (xᵢ₋₂ + xᵢ) / 2 : alg.middle
    xᵢ₋₂, xᵢ₋₁, xᵢ = promote(xᵢ₋₂, xᵢ₋₁, xᵢ)
    @assert xᵢ₋₂ ≠ xᵢ₋₁ ≠ xᵢ ≠ xᵢ₋₂
    f = Base.Fix2(prob.f, prob.p)
    fxᵢ₋₂, fxᵢ₋₁, fxᵢ = f(xᵢ₋₂), f(xᵢ₋₁), f(xᵢ)

    xᵢ₊₁, fxᵢ₊₁ = xᵢ₋₂, fxᵢ₋₂

    abstol = abs(NonlinearSolveBase.get_tolerance(
        xᵢ₋₂, abstol, promote_type(eltype(xᵢ₋₂), eltype(xᵢ))))

    for _ in 1:maxiters
        q = (xᵢ - xᵢ₋₁) / (xᵢ₋₁ - xᵢ₋₂)
        A = q * fxᵢ - q * (1 + q) * fxᵢ₋₁ + q^2 * fxᵢ₋₂
        B = (2 * q + 1) * fxᵢ - (1 + q)^2 * fxᵢ₋₁ + q^2 * fxᵢ₋₂
        C = (1 + q) * fxᵢ

        denom₊ = B + √(B^2 - 4 * A * C)
        denom₋ = B - √(B^2 - 4 * A * C)

        if abs(denom₊) ≥ abs(denom₋)
            xᵢ₊₁ = xᵢ - (xᵢ - xᵢ₋₁) * 2 * C / denom₊
        else
            xᵢ₊₁ = xᵢ - (xᵢ - xᵢ₋₁) * 2 * C / denom₋
        end

        fxᵢ₊₁ = f(xᵢ₊₁)

        # Termination Check
        if abstol ≥ abs(fxᵢ₊₁)
            return SciMLBase.build_solution(prob, alg, xᵢ₊₁, fxᵢ₊₁;
                retcode = ReturnCode.Success,
                left = xᵢ₊₁, right = xᵢ₊₁)
        end

        xᵢ₋₂, xᵢ₋₁, xᵢ = xᵢ₋₁, xᵢ, xᵢ₊₁
        fxᵢ₋₂, fxᵢ₋₁, fxᵢ = fxᵢ₋₁, fxᵢ, fxᵢ₊₁
    end

    return SciMLBase.build_solution(prob, alg, xᵢ₊₁, fxᵢ₊₁;
        retcode = ReturnCode.MaxIters,
        left = xᵢ₊₁, right = xᵢ₊₁)
end
