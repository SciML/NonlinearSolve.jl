using NonlinearSolve
using SciMLBase

using ForwardDiff, ReverseDiff, SciMLSensitivity, Tracker, Zygote, Enzyme, Mooncake

ff(u, p) = u .^ 2 .- p

function solve_nlprob(p)
    prob = NonlinearProblem{false}(ff, [1.0, 2.0], p)
    sol = solve(prob, NewtonRaphson())
    res = sol isa AbstractArray ? sol : sol.u
    return sum(abs2, res)
end

function solve_despecialized_nlprob(p)
    f = NonlinearFunction{false, SciMLBase.AutoDespecialize}(ff)
    prob = NonlinearSolveBase.get_concrete_problem(NonlinearProblem(f, [1.0, 2.0], p))
    return sum(abs2, solve(prob, NewtonRaphson()).u)
end

function solve_despecialized_structured_nlprob(p)
    f = NonlinearFunction{false, SciMLBase.AutoDespecialize}(
        (u, p) -> u .^ 2 .- (p.values .+ p.coeffs.shift)
    )
    prob = NonlinearSolveBase.get_concrete_problem(NonlinearProblem(f, [1.0, 2.0], p))
    return sum(abs2, solve(prob, NewtonRaphson()).u)
end

function zygote_mooncake_grad(f, p)
    zygote_grad = only(Zygote.gradient(f, p))
    cache = Mooncake.prepare_gradient_cache(f, p)
    mooncake_grad = Mooncake.value_and_gradient!!(cache, f, p)[2][2]
    return zygote_grad, mooncake_grad
end

p = [3.0, 2.0]

∂p_zygote = only(Zygote.gradient(solve_nlprob, p))
∂p_forwarddiff = ForwardDiff.gradient(solve_nlprob, p)
∂p_tracker = Tracker.data(only(Tracker.gradient(solve_nlprob, p)))
∂p_reversediff = ReverseDiff.gradient(solve_nlprob, p)
∂p_enzyme = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), solve_nlprob, p)[1]

cache = Mooncake.prepare_gradient_cache(solve_nlprob, p)
∂p_mooncake = Mooncake.value_and_gradient!!(cache, solve_nlprob, p)[2][2]

@test ∂p_zygote ≈ ∂p_tracker ≈ ∂p_reversediff ≈ ∂p_enzyme
@test ∂p_zygote ≈ ∂p_forwarddiff ≈ ∂p_tracker ≈ ∂p_reversediff ≈ ∂p_enzyme
@test ∂p_forwarddiff ≈ ∂p_mooncake

∂p_despecialized_zygote, ∂p_despecialized_mooncake = zygote_mooncake_grad(
    solve_despecialized_nlprob, p
)

@test ∂p_despecialized_zygote ≈ ∂p_despecialized_mooncake

p_structured = (; values = p, coeffs = (; shift = 1.0))
∂p_structured_zygote, ∂p_structured_mooncake = zygote_mooncake_grad(
    solve_despecialized_structured_nlprob, p_structured
)

@test ∂p_structured_zygote.values ≈ ∂p_structured_mooncake.values
@test ∂p_structured_zygote.coeffs.shift ≈ ∂p_structured_mooncake.coeffs.shift
