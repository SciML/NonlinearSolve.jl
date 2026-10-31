using Test, Enzyme, ADTypes, LinearAlgebra, SciMLBase, NonlinearSolveFirstOrder

const exact_u = [sqrt(2), 0.25]

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

    @test enzyme_sol.u ≈ exact_u atol = 1.0e-6
    @test LinearAlgebra.norm(enzyme_sol.resid) < 1.0e-6
end

@testset "Bounded Enzyme IIP postcondition uses nothing caches (#1288)" begin
    # Default postcondition space is Original: the IIP path maps through
    # `_bounds_tmp(bw.u_cache, …)`. Enzyme wrappers store `nothing` there so residual
    # AD allocates; `_bounds_tmp(::Nothing, u)` must still allocate for this path.
    function residual!(du, u, _)
        du[1] = u[1]^2 - 2
        du[2] = u[2] - 0.25
        return nothing
    end
    problem = NonlinearProblem(
        NonlinearFunction(residual!), [1.0, 0.0]; lb = [0.0, -1.0], ub = [2.0, 1.0]
    )
    enzyme = AutoEnzyme(
        mode = Enzyme.set_runtime_activity(Enzyme.Forward),
        function_annotation = Enzyme.Const
    )
    H! = (up, uprev, p, cache) -> nothing
    sol = solve(
        problem, NewtonRaphson(; autodiff = enzyme); postcondition = H!, maxiters = 100
    )
    @test sol.u ≈ exact_u atol = 1.0e-6
end
