using NonlinearSolveSpectralMethods
include("setup_corerootfindtesting.jl")

u0 = [-10.0, -1.0, 1.0, 2.0, 3.0, 4.0, 10.0]
p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
sol = solve_oop(newton_fails, u0, p; solver = DFSane())
@test SciMLBase.successful_retcode(sol)
@test all(abs.(newton_fails(sol.u, p)) .< 1.0e-9)
