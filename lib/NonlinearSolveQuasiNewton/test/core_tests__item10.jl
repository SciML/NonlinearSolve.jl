using NonlinearSolveQuasiNewton
using SciMLBase
using Test

function generalized_rosenbrock!(out, x, p)
    out[1] = 1.0 - x[1]
    @views @. out[2:end] = 10.0 * (x[2:end] - x[1:(end - 1)]^2)
    return nothing
end

x0 = ones(10)
x0[1] = -1.2
prob = NonlinearProblem(generalized_rosenbrock!, x0)
alg = Broyden(; init_jacobian = Val(:true_jacobian), update_rule = Val(:bad_broyden))
sol = solve(prob, alg; maxiters = 10_000)
residual = similar(sol.u)
generalized_rosenbrock!(residual, sol.u, nothing)

@test SciMLBase.successful_retcode(sol)
@test maximum(abs, residual) ≤ 1.0e-3
