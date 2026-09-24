using SimpleNonlinearSolve
include("setup_rootfindtestsnippet.jl")

@testset "$(nameof(typeof(alg)))" for alg in (
        SimpleBroyden(),
        SimpleKlement(),
        SimpleDFSane(),
        SimpleLimitedMemoryBroyden(),
        SimpleBroyden(; linesearch = LiFukushimaLineSearch(; nan_maxiters = nothing)),
        SimpleLimitedMemoryBroyden(;
            linesearch = LiFukushimaLineSearch(; nan_maxiters = nothing)
        ),
    )
    @testset "[OOP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0], @SVector[1.0, 1.0], 1.0)
        sol = run_nlsolve_oop(quadratic_f, u0; solver = alg)
        @test SciMLBase.successful_retcode(sol)
        @test maximum(abs, quadratic_f(sol.u, 2.0)) < 1.0e-9
    end

    @testset "[IIP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0],)
        sol = run_nlsolve_iip(quadratic_f!, u0; solver = alg)
        @test SciMLBase.successful_retcode(sol)
        @test maximum(abs, quadratic_f(sol.u, 2.0)) < 1.0e-9
    end

    @testset "Termination Condition: $(nameof(typeof(termination_condition))) u0: $(nameof(typeof(u0)))" for termination_condition in
            TERMINATION_CONDITIONS,
            u0 in (1.0, [1.0, 1.0], @SVector[1.0, 1.0])

        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(solve(probN, alg; termination_condition).u .≈ sqrt(2.0))
    end
end
