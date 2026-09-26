using Test, Enzyme, ADTypes, LinearAlgebra, SciMLBase, NonlinearSolveFirstOrder

@testset "Bounded LevenbergMarquardt with AutoEnzyme (#1288)" begin
    residual(u, _) = [u[1]^2 - 2, u[2] - 0.25]
    problem = NonlinearLeastSquaresProblem(
        residual, [1.0, 0.0]; lb = [0.0, -1.0], ub = [2.0, 1.0]
    )
    enzyme = AutoEnzyme(
        mode = Enzyme.set_runtime_activity(Enzyme.Forward),
        function_annotation = Enzyme.Const
    )
    enzyme_sol = solve(
        problem, LevenbergMarquardt(; autodiff = enzyme, disable_geodesic = Val(true))
    )
    forwarddiff_sol = solve(
        problem,
        LevenbergMarquardt(; autodiff = AutoForwardDiff(), disable_geodesic = Val(true))
    )

    @test SciMLBase.successful_retcode(enzyme_sol)
    @test enzyme_sol.u ≈ forwarddiff_sol.u
    @test LinearAlgebra.norm(enzyme_sol.resid) < 1.0e-6
end
