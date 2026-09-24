using NonlinearSolveQuasiNewton, SciMLBase, Test

# Started at the root of a nearby parameter value the residual is tiny, so the seeded
# `alpha * I` inverse Jacobian sets the whole first step. Seeded with the wrong sign that
# step walks away from the root the solve started at.
quadratic_f!(du, u, p) = (du .= u .^ 2 .- p; nothing)

p_range = range(0.01, 2, length = 200)
failures = map(2:length(p_range)) do i
    prob = NonlinearProblem{true}(quadratic_f!, [sqrt(p_range[i - 1])], p_range[i])
    sol = solve(prob, LimitedMemoryBroyden(); maxiters = 100, abstol = 1.0e-10)
    converged = SciMLBase.successful_retcode(sol) &&
        isapprox(sol.u[1], sqrt(p_range[i]); rtol = 1.0e-6)
    return converged ? nothing : (; i, p = p_range[i], u = sol.u[1], sol.retcode)
end
@test isempty(filter(!isnothing, failures))
