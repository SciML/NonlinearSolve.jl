using NonlinearSolveQuasiNewton, SciMLBase

# `NonlinearFunction.postcondition` iterate correction applied at the quasi-Newton
# iterate-commit points: a step clamp keeps the secant iteration inside a trusted move
# per iteration and must not break convergence.
f! = (du, u, p) -> (du .= u .^ 2 .- 2; nothing)
Hclamp = (up, uprev, p) -> (up .= clamp.(up, uprev .- 0.5, uprev .+ 0.5); nothing)
prob = NonlinearProblem(NonlinearFunction(f!; postcondition = Hclamp), [3.0, 3.0])
for alg in (Broyden(), Klement())
    sol = solve(prob, alg)
    @test SciMLBase.successful_retcode(sol)
    @test isapprox(sol.u, fill(sqrt(2), 2); atol = 1.0e-6)
end
