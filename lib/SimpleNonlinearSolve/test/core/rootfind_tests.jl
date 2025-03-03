@testsnippet RootfindTestSnippet begin
    using StaticArrays, Random, LinearAlgebra, ForwardDiff, NonlinearSolveBase, SciMLBase
    using ADTypes, PolyesterForwardDiff, Enzyme, ReverseDiff

    quadratic_f(u, p) = u .* u .- p
    quadratic_f!(du, u, p) = (du .= u .* u .- p)

    function newton_fails(u, p)
        return 0.010000000000000002 .+
               10.000000000000002 ./ (1 .+
                (0.21640425613334457 .+
                 216.40425613334457 ./ (1 .+
                  (0.21640425613334457 .+
                   216.40425613334457 ./ (1 .+ 0.0006250000000000001(u .^ 2.0))) .^ 2.0)) .^
                2.0) .- 0.0011552453009332421u .- p
    end

    const TERMINATION_CONDITIONS = [
        NormTerminationMode(Base.Fix1(maximum, abs)),
        RelTerminationMode(),
        RelNormTerminationMode(Base.Fix1(maximum, abs)),
        RelNormSafeTerminationMode(Base.Fix1(maximum, abs)),
        RelNormSafeBestTerminationMode(Base.Fix1(maximum, abs)),
        AbsTerminationMode(),
        AbsNormTerminationMode(Base.Fix1(maximum, abs)),
        AbsNormSafeTerminationMode(Base.Fix1(maximum, abs)),
        AbsNormSafeBestTerminationMode(Base.Fix1(maximum, abs))
    ]

    function run_nlsolve_oop(f::F, u0, p = 2.0; solver) where {F}
        return @inferred solve(NonlinearProblem{false}(f, u0, p), solver; abstol = 1e-9)
    end
    function run_nlsolve_iip(f!::F, u0, p = 2.0; solver) where {F}
        return @inferred solve(NonlinearProblem{true}(f!, u0, p), solver; abstol = 1e-9)
    end
end

@testitem "First Order Methods" setup=[RootfindTestSnippet] tags=[:core] begin
    for alg in (
        SimpleNewtonRaphson,
        SimpleTrustRegion,
        (; kwargs...) -> SimpleTrustRegion(; kwargs..., nlsolve_update_rule = Val(true))
    )
        @testset for autodiff in (
            AutoForwardDiff(),
            AutoFiniteDiff(),
            AutoReverseDiff(),
            AutoEnzyme(),
            nothing
        )
            @testset "[OOP] u0: $(typeof(u0))" for u0 in (
                [1.0, 1.0], @SVector[1.0, 1.0], 1.0)
                sol = run_nlsolve_oop(quadratic_f, u0; solver = alg(; autodiff))
                @test SciMLBase.successful_retcode(sol)
                @test maximum(abs, quadratic_f(sol.u, 2.0)) < 1e-9
            end

            @testset "[IIP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0],)
                sol = run_nlsolve_iip(quadratic_f!, u0; solver = alg(; autodiff))
                @test SciMLBase.successful_retcode(sol)
                @test maximum(abs, quadratic_f(sol.u, 2.0)) < 1e-9
            end

            @testset "Termination Condition: $(nameof(typeof(termination_condition))) u0: $(nameof(typeof(u0)))" for termination_condition in TERMINATION_CONDITIONS,
                u0 in (1.0, [1.0, 1.0], @SVector[1.0, 1.0])

                probN = NonlinearProblem(quadratic_f, u0, 2.0)
                @test all(solve(
                    probN, alg(; autodiff = AutoForwardDiff()); termination_condition).u .≈
                          sqrt(2.0))
            end
        end
    end
end

@testitem "Second Order Methods" setup=[RootfindTestSnippet] tags=[:core] begin
    @testset for alg in (
        SimpleHalley,
    )
        @testset for autodiff in (
            AutoForwardDiff(),
            AutoFiniteDiff(),
            AutoReverseDiff(),
            nothing
        )
            @testset "[OOP] u0: $(typeof(u0))" for u0 in (
                [1.0, 1.0], @SVector[1.0, 1.0], 1.0)
                sol = run_nlsolve_oop(quadratic_f, u0; solver = alg(; autodiff))
                @test SciMLBase.successful_retcode(sol)
                @test maximum(abs, quadratic_f(sol.u, 2.0)) < 1e-9
            end
        end

        @testset "Termination Condition: $(nameof(typeof(termination_condition))) u0: $(nameof(typeof(u0)))" for termination_condition in TERMINATION_CONDITIONS,
            u0 in (1.0, [1.0, 1.0], @SVector[1.0, 1.0])

            probN = NonlinearProblem(quadratic_f, u0, 2.0)
            @test all(solve(
                probN, alg(; autodiff = AutoForwardDiff()); termination_condition).u .≈
                      sqrt(2.0))
        end
    end
end

@testitem "Derivative Free Methods" setup=[RootfindTestSnippet] tags=[:core] begin
    @testset "$(nameof(typeof(alg)))" for alg in (
        SimpleBroyden(),
        SimpleKlement(),
        SimpleDFSane(),
        SimpleLimitedMemoryBroyden(),
        SimpleBroyden(; linesearch = Val(true)),
        SimpleLimitedMemoryBroyden(; linesearch = Val(true))
    )
        @testset "[OOP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0], @SVector[1.0, 1.0], 1.0)
            sol = run_nlsolve_oop(quadratic_f, u0; solver = alg)
            @test SciMLBase.successful_retcode(sol)
            @test maximum(abs, quadratic_f(sol.u, 2.0)) < 1e-9
        end

        @testset "[IIP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0],)
            sol = run_nlsolve_iip(quadratic_f!, u0; solver = alg)
            @test SciMLBase.successful_retcode(sol)
            @test maximum(abs, quadratic_f(sol.u, 2.0)) < 1e-9
        end

        @testset "Termination Condition: $(nameof(typeof(termination_condition))) u0: $(nameof(typeof(u0)))" for termination_condition in TERMINATION_CONDITIONS,
            u0 in (1.0, [1.0, 1.0], @SVector[1.0, 1.0])

            probN = NonlinearProblem(quadratic_f, u0, 2.0)
            @test all(solve(probN, alg; termination_condition).u .≈ sqrt(2.0))
        end
    end
end

@testitem "Newton Fails" setup=[RootfindTestSnippet] tags=[:core] begin
    u0 = [-10.0, -1.0, 1.0, 2.0, 3.0, 4.0, 10.0]
    p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    @testset "$(nameof(typeof(alg)))" for alg in (
        SimpleDFSane(),
        SimpleTrustRegion(),
        SimpleHalley(),
        SimpleTrustRegion(; nlsolve_update_rule = Val(true))
    )
        sol = run_nlsolve_oop(newton_fails, u0, p; solver = alg)
        @test SciMLBase.successful_retcode(sol)
        @test maximum(abs, newton_fails(sol.u, p)) < 1e-9
    end
end

@testitem "Kwargs Propagation" setup=[RootfindTestSnippet] tags=[:core] begin
    prob = NonlinearProblem(quadratic_f, ones(4), 2.0; maxiters = 2)
    sol = solve(prob, SimpleNewtonRaphson())
    @test sol.retcode === ReturnCode.MaxIters
end
