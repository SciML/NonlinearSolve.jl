using NonlinearSolveSpectralMethods, SciMLBase

# `NonlinearFunction.postcondition` iterate correction applied at the DFSane
# iterate-commit point.
f! = (du, u, p) -> (du .= u .^ 2 .- 2; nothing)
Hclamp = (up, uprev, p) -> (up .= clamp.(up, uprev .- 0.5, uprev .+ 0.5); nothing)
prob = NonlinearProblem(NonlinearFunction(f!; postcondition = Hclamp), [3.0, 3.0])
sol = solve(prob, DFSane())
@test SciMLBase.successful_retcode(sol)
@test isapprox(sol.u, fill(sqrt(2), 2); atol = 1.0e-6)
