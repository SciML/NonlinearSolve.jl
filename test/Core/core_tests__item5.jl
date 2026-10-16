using NonlinearSolve

using NonlinearProblemLibrary

# Use a problem that the initial solvers cannot solve and cause the initial value to
# diverge. If we don't alias correctly, all the subsequent algorithms will also fail.
prob = NonlinearProblemLibrary.nlprob_23_testcases["Generalized Rosenbrock function"].prob
u0 = copy(prob.u0)
prob = remake(prob; u0 = copy(u0))

# If aliasing is not handled properly this will diverge
sol = solve(
    prob; abstol = 1.0e-6, alias_u0 = true,
    termination_condition = AbsNormTerminationMode(Base.Fix1(maximum, abs))
)

@test sol.u === prob.u0
@test SciMLBase.successful_retcode(sol.retcode)

prob = remake(prob; u0 = copy(u0))

cache = init(
    prob; abstol = 1.0e-6, alias_u0 = true,
    termination_condition = AbsNormTerminationMode(Base.Fix1(maximum, abs))
)
sol = solve!(cache)

@test sol.u === prob.u0
@test SciMLBase.successful_retcode(sol.retcode)

# `reinit!` of an aliased polyalgorithm cache whose ladder starts past index 1: the
# unattempted subalgorithms must not be fallback candidates, and the new `u0` must seed
# the attempts. Compared against a fresh solve of the same reinitialized problem.
function sq_resid!(du, u, p)
    du .= u .* u .- p
    return nothing
end
alg = NonlinearSolvePolyAlgorithm((NewtonRaphson(), Broyden()); start_index = 2)
solve_kwargs = (; maxiters = 1, abstol = 1.0e-14, reltol = 1.0e-14)
ref = solve(NonlinearProblem(sq_resid!, [2.0, 3.0], 5.0), alg; solve_kwargs...)
@test ref.retcode == ReturnCode.MaxIters

for alias_u0 in (false, true), presolve in (false, true)
    local cache = init(
        NonlinearProblem(sq_resid!, [1.0, 2.0], 3.0), alg; solve_kwargs...,
        alias = SciMLBase.NonlinearAliasSpecifier(; alias_u0)
    )
    presolve && solve!(cache)
    reinit!(cache, [2.0, 3.0]; p = 5.0)
    local sol = solve!(cache)
    @test sol.retcode == ref.retcode
    @test sol.u ≈ ref.u
    @test sol.resid ≈ ref.resid
end
