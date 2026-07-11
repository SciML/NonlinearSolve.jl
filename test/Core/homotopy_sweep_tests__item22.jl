using NonlinearSolve

using SciMLBase

import NonlinearSolveBase

# The continuation drivers assign `last_sol` from both the concretely-inferred inner
# `solve!(cache)` and a fresh `solve(inner_prob, inner)` (the exempt/anchor/landing path,
# whose return type inference gives up and yields `Any`), which widens the driver's local
# to `Any`. Reading `last_sol.u` / `.resid` / `.stats` inline over an `Any` local then
# boxes each field value; a profiler attributes the box to SciMLBase's solution
# `getproperty` (src/solutions/solution_interface.jl). The `_sol_*` accessors take the
# solution as a typed argument, so each is specialized on the concrete runtime solution
# type and its field read is a non-boxing `getfield`. This test pins that invariant.

H!(du, u, p, λ) = (du[1] = (1 - λ) * (u[1] - p[1]) + λ * (u[1]^2 - p[1]); nothing)
prob = HomotopyProblem(H!, [4.0], [4.0])

sol = solve(prob, HomotopySweep())
@test SciMLBase.successful_retcode(sol)
@test sol.u[1] ≈ 2.0 atol = 1.0e-6

# A concrete `NonlinearSolution` produced through the same inner path the drivers use.
inner_sol = solve(NonlinearProblem((u, p) -> u .* u .- p, [1.0], 2.0), NewtonRaphson())
@test inner_sol isa SciMLBase.AbstractNonlinearSolution

# The accessors must not allocate on a concretely-typed solution. Each read is done inside
# a loop and its result folded into a running scalar (so nothing escapes to be boxed at the
# measurement boundary), and the per-iteration cost is isolated by differencing two loop
# lengths — this cancels the fixed one-time boxing `@allocated` charges on a specialized
# method's first execution, which is otherwise a flaky non-zero on a single call.
function probe_reads(s, n)
    acc = 0.0
    for _ in 1:n
        acc += NonlinearSolveBase._sol_u(s)[1]
        acc += NonlinearSolveBase._sol_resid(s)[1]
        acc += SciMLBase.successful_retcode(NonlinearSolveBase._sol_retcode(s)) ? 1.0 : 0.0
        acc += NonlinearSolveBase._sol_nsteps(s)
    end
    return acc
end
probe_reads(inner_sol, 10)  # warm up
per_iter = @allocated(probe_reads(inner_sol, 2000)) - @allocated(probe_reads(inner_sol, 1000))
@test per_iter == 0

# The accessors must be type-stable on a concrete solution (the property that makes the
# field read a non-boxing `getfield` even when the caller's binding is `Any`-typed).
@test (@inferred NonlinearSolveBase._sol_u(inner_sol)) === inner_sol.u
@test (@inferred NonlinearSolveBase._sol_resid(inner_sol)) === inner_sol.resid
@test (@inferred NonlinearSolveBase._sol_retcode(inner_sol)) === inner_sol.retcode
@test (@inferred NonlinearSolveBase._sol_nsteps(inner_sol)) isa Int
