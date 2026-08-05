using NonlinearSolveQuasiNewton, SciMLBase

# The `postcondition` iterate corrector applied at the quasi-Newton commit points: a step
# clamp keeps the secant iteration inside a trusted move per iteration.
f! = (du, u, p) -> (du .= u .^ 2 .- 2; nothing)
Hclamp = (up, uprev, p) -> (up .= clamp.(up, uprev .- 0.5, uprev .+ 0.5); nothing)
prob = NonlinearProblem(f!, [3.0, 3.0])
for alg in (Broyden(), Klement())
    sol = solve(prob, alg; postcondition = Hclamp)
    @test SciMLBase.successful_retcode(sol)
    @test isapprox(sol.u, fill(sqrt(2), 2); atol = 1.0e-6)
end
