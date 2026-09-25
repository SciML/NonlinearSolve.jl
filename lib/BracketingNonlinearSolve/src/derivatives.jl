# The derivative-using methods get `f'` from the `jac` of the `IntervalNonlinearFunction`.

struct UserDerivative{F, J}
    f::F
    jac::J
end

function interval_derivative(prob::IntervalNonlinearProblem, alg)
    SciMLBase.has_jac(prob.f) || throw(
        ArgumentError(
            "`$(nameof(typeof(alg)))` needs the derivative of `f`, given as \
            `IntervalNonlinearFunction(f; jac)`."
        )
    )
    return UserDerivative(Base.Fix2(prob.f, prob.p), Base.Fix2(prob.f.jac, prob.p))
end

value(d::UserDerivative, x) = d.f(x)
derivative(d::UserDerivative, x) = d.jac(x)
