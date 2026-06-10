using SimpleNonlinearSolve
include("setup_rootfindtestsnippet.jl")

@testset for alg in (
        SimpleNewtonRaphson,
        SimpleTrustRegion,
        (; kwargs...) -> SimpleTrustRegion(; kwargs..., nlsolve_update_rule = Val(true)),
    )
    # Filter autodiff backends based on Julia version
    autodiff_backends = [
        AutoForwardDiff(),
        AutoFiniteDiff(),
        AutoReverseDiff(),
        nothing,
    ]
    if isempty(VERSION.prerelease) && VERSION < v"1.12"
        push!(autodiff_backends, AutoEnzyme())
    end

    @testset for autodiff in autodiff_backends
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

        @testset "[IIP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0],)
            sol = run_nlsolve_iip(quadratic_f!, u0; solver = alg(; autodiff))
            @test SciMLBase.successful_retcode(sol)
            @test maximum(abs, quadratic_f(sol.u, 2.0)) < 1.0e-9
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
end
