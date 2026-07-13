<<<<<<< HEAD
using NonlinearSolveQuasiNewton, SciMLBase

# The `postcondition` iterate corrector applied at the quasi-Newton commit points: a step
# clamp keeps the secant iteration inside a trusted move per iteration.
f! = (du, u, p) -> (du .= u .^ 2 .- 2; nothing)
Hclamp = (up, uprev, p, cache) -> (up .= clamp.(up, uprev .- 0.5, uprev .+ 0.5); nothing)
prob = NonlinearProblem(f!, [3.0, 3.0])
for alg in (Broyden(), Klement())
    sol = solve(prob, alg; postcondition = Hclamp)
    @test SciMLBase.successful_retcode(sol)
    @test isapprox(sol.u, fill(sqrt(2), 2); atol = 1.0e-6)
end
=======
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
>>>>>>> 846ebe62 (Preserve triangular Broyden initialization)
