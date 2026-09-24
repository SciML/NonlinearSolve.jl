using SimpleNonlinearSolve
include("setup_rootfindtestsnippet.jl")

@testset for alg in (
        SimpleHalley,
    )
    @testset for autodiff in (
            AutoForwardDiff(),
            AutoFiniteDiff(),
            AutoReverseDiff(),
            nothing,
        )
        @testset "[OOP] u0: $(typeof(u0))" for u0 in (
                [1.0, 1.0], @SVector[1.0, 1.0], 1.0,
            )
            broken_inferred = u0 isa StaticArray && (
                autodiff isa AutoFiniteDiff ||
                    (autodiff isa AutoReverseDiff && VERSION < v"1.11")
            )
            sol = run_nlsolve_oop(
                quadratic_f, u0; solver = alg(; autodiff),
                broken_inferred
            )
            @test SciMLBase.successful_retcode(sol)
            @test maximum(abs, quadratic_f(sol.u, 2.0)) < 1.0e-9
        end
    end

    @testset "Termination Condition: $(nameof(typeof(termination_condition))) u0: $(nameof(typeof(u0)))" for termination_condition in
            TERMINATION_CONDITIONS,
            u0 in (1.0, [1.0, 1.0], @SVector[1.0, 1.0])

        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(
            solve(
                probN, alg(; autodiff = AutoForwardDiff()); termination_condition
            ).u .≈
                sqrt(2.0)
        )
    end
end
