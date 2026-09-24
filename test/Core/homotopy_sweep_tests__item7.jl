using NonlinearSolve

using SciMLBase

# kwargs (incl. the alias specifier) are forwarded to every inner solve; with
# aliasing on, the inner solver iterates directly in its u0 buffer, so without
# copy protection the returned u would be a diverged buffer rather than the
# last converged iterate. Identical step sequences make exact equality against
# the no-aliasing reference run valid.
H(u, p, λ) = [(1 - λ) * u[1] + λ * (u[1]^2 + 1.0)]
prob = HomotopyProblem(H, [0.0]; λspan = (0.0, 1.0))
sol = solve(
    prob, HomotopySweep(; inner = NewtonRaphson(), min_dλ = 1.0e-2);
    alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = true)
)
ref = solve(prob, HomotopySweep(; inner = NewtonRaphson(), min_dλ = 1.0e-2))
@test !SciMLBase.successful_retcode(sol)
@test sol.u == ref.u    # last CONVERGED iterate, not the aliased diverged buffer
