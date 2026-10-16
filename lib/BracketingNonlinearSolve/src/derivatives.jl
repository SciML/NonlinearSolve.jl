# NewtonBisection and ModABNewton get `f'` from the `jac` of the `IntervalNonlinearFunction`
# (`UserDerivative`) or, without one, by automatic differentiation with the `autodiff`
# backend of the algorithm (`ADDerivative`), where one pass gives `f` and `f'` together.

struct UserDerivative{F, J}
    f::F
    jac::J
end

struct ADDerivative{F, B, P, C}
    f::F
    backend::B
    prep::P
    p::C
end

function interval_derivative(prob::IntervalNonlinearProblem, alg, x)
    SciMLBase.has_jac(prob.f) &&
        return UserDerivative(Base.Fix2(prob.f, prob.p), Base.Fix2(prob.f.jac, prob.p))
    backend = alg.autodiff
    backend === nothing && throw(
        ArgumentError(
            "`$(nameof(typeof(alg)))` needs the derivative of `f`, given as \
            `IntervalNonlinearFunction(f; jac)`, or an automatic differentiation backend, \
            for example `$(nameof(typeof(alg)))(; autodiff = AutoForwardDiff())`."
        )
    )
    DI.check_available(backend) || throw(
        ArgumentError(
            "The automatic differentiation backend $(backend) is not available. Load its \
            package, for example `using ForwardDiff`."
        )
    )
    # Not strict: the Newton points take the type of `x - f(x) / f'(x)`, which differs from
    # the type of the bracket when `f` promotes (a Float32 bracket and a Float64 `f`).
    prep = DI.prepare_derivative(prob.f, backend, x, DI.Constant(prob.p); strict = Val(false))
    return ADDerivative(prob.f, backend, prep, prob.p)
end

value(d::UserDerivative, x) = d.f(x)
value(d::ADDerivative, x) = d.f(x, d.p)

derivative(d::UserDerivative, x) = d.jac(x)
derivative(d::ADDerivative, x) = last(value_and_derivative(d, x))

function value_and_derivative(d::ADDerivative, x)
    return DI.value_and_derivative(d.f, d.prep, d.backend, x, DI.Constant(d.p))
end
