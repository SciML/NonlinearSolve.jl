"""
    ModABNewton(; autodiff = nothing)

[`ModAB`](@ref)'s bisection phase followed by [`NewtonBisection`](@ref)'s Newton steps.

The method bisects, without derivatives, until `f` looks linear on the bracket by ModAB's
test, then switches to Newton's method safeguarded by bisection. The derivative is the `jac`
of the `IntervalNonlinearFunction` when it has one, otherwise it comes from automatic
differentiation with the `autodiff` backend, for example `AutoForwardDiff()` after
`using ForwardDiff`. Without either it is an error.

Derivatives are only used near the root, where Newton's method converges quadratically, so
the method needs fewer evaluations of `f'` than `NewtonBisection`, and falls back to plain
bisection (like ModAB) where Newton's method would converge slowly, as at multiple roots.
"""
@concrete struct ModABNewton <: AbstractBracketingAlgorithm
    autodiff
end

function ModABNewton(; autodiff::Union{Nothing, ADTypes.AbstractADType} = nothing)
    return ModABNewton(autodiff)
end

function SciMLBase.__solve(
        prob::IntervalNonlinearProblem, alg::ModABNewton, args...;
        maxiters = 1000, abstol = nothing, verbose::NonlinearVerbosity = NonlinearVerbosity(), kwargs...
    )
    @assert !SciMLBase.isinplace(prob) "`ModABNewton` only supports out-of-place problems."

    left, right = minmax(promote(prob.tspan...)...)
    fun = interval_derivative(prob, alg, left)
    fl, fr = value(fun, left), value(fun, right)

    abstol = NonlinearSolveBase.get_tolerance(
        left, abstol, promote_type(eltype(left), eltype(right))
    )

    if iszero(fl)
        return build_exact_solution(prob, alg, left, fl, ReturnCode.ExactSolutionLeft)
    end

    if iszero(fr)
        return build_exact_solution(prob, alg, right, fr, ReturnCode.ExactSolutionRight)
    end

    if sign(fl) == sign(fr)
        @SciMLMessage(
            "The interval is not an enclosing interval, opposite signs at the \
        boundaries are required.",
            verbose, :non_enclosing_interval
        )
        return build_bracketing_solution(prob, alg, left, fl, left, right, ReturnCode.InitialFailure)
    end

    for i in 1:maxiters
        if nextfloat(left) == right
            return newton_bisection_solution(prob, alg, left, right, fl, fr, ReturnCode.FloatingPointLimit)
        end
        if (right - left) / 2 < abstol
            return newton_bisection_solution(prob, alg, left, right, fl, fr, ReturnCode.Success)
        end
        mid = (left + right) / 2
        fm = value(fun, mid)
        iszero(fm) && return build_exact_solution(prob, alg, mid, fm, ReturnCode.Success)
        # ModAB's test: f at the midpoint is close to the chord
        chord = (fl + fr) / 2
        r = 1 - abs(chord / (fr - fl))
        linear = abs(chord - fm) < r * r * (abs(chord) + abs(fm))
        if sign(fm) == sign(fl)
            left, fl = mid, fm
        else
            right, fr = mid, fm
        end
        if linear
            return newton_bracket(
                prob, alg, fun, left, right, fl, fr, zero(fl), zero(fr), false, false,
                abstol, maxiters - i
            )
        end
    end

    return newton_bisection_solution(prob, alg, left, right, fl, fr, ReturnCode.MaxIters)
end
