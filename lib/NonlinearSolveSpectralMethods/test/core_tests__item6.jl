using NonlinearSolveSpectralMethods, SciMLBase

# The `postcondition` iterate corrector applied at the DF-SANE commit point.
f! = (du, u, p) -> (du .= u .^ 2 .- 2; nothing)
Hclamp = (up, uprev, p) -> (up .= clamp.(up, uprev .- 0.5, uprev .+ 0.5); nothing)
sol = solve(NonlinearProblem(f!, [3.0, 3.0]), DFSane(); postcondition = Hclamp)
@test SciMLBase.successful_retcode(sol)
@test isapprox(sol.u, fill(sqrt(2), 2); atol = 1.0e-6)
