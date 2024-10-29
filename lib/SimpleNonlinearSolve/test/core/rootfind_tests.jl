@testsetup module RootfindingTesting
using Reexport
@reexport using AllocCheck, StaticArrays, Random, LinearAlgebra, ForwardDiff, DiffEqBase,
                TaylorDiff
import PolyesterForwardDiff

quadratic_f(u, p) = u .* u .- p
quadratic_f!(du, u, p) = (du .= u .* u .- p)
quadratic_f2(u, p) = @. p[1] * u * u - p[2]

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
    NormTerminationMode(), RelTerminationMode(), RelNormTerminationMode(),
    AbsTerminationMode(), AbsNormTerminationMode(), RelSafeTerminationMode(),
    AbsSafeTerminationMode(), RelSafeBestTerminationMode(), AbsSafeBestTerminationMode()]

function benchmark_nlsolve_oop(f::F, u0, p = 2.0; solver) where {F}
    prob = NonlinearProblem{false}(f, u0, p)
    return solve(prob, solver; abstol = 1e-9)
end
function benchmark_nlsolve_iip(f!::F, u0, p = 2.0; solver) where {F}
    prob = NonlinearProblem{true}(f!, u0, p)
    return solve(prob, solver; abstol = 1e-9)
end

export quadratic_f, quadratic_f!, quadratic_f2, newton_fails, TERMINATION_CONDITIONS,
       benchmark_nlsolve_oop, benchmark_nlsolve_iip
end

@testitem "First Order Methods" setup=[RootfindingTesting] tags=[:core] begin
    @testset "$(alg)" for alg in (SimpleNewtonRaphson,
        SimpleTrustRegion,
        (args...; kwargs...) -> SimpleTrustRegion(
            args...; nlsolve_update_rule = Val(true), kwargs...))
        @testset "AutoDiff: $(nameof(typeof(autodiff)))" for autodiff in (
            AutoFiniteDiff(), AutoForwardDiff(), AutoPolyesterForwardDiff())
            @testset "[OOP] u0: $(typeof(u0))" for u0 in (
                [1.0, 1.0], @SVector[1.0, 1.0], 1.0)
                u0 isa SVector && autodiff isa AutoPolyesterForwardDiff && continue
                sol = benchmark_nlsolve_oop(quadratic_f, u0; solver = alg(; autodiff))
                @test SciMLBase.successful_retcode(sol)
                @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)
            end

            @testset "[IIP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0],)
                sol = benchmark_nlsolve_iip(quadratic_f!, u0; solver = alg(; autodiff))
                @test SciMLBase.successful_retcode(sol)
                @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)
            end
        end

        @testset "Termination condition: $(nameof(typeof(termination_condition))) u0: $(nameof(typeof(u0)))" for termination_condition in TERMINATION_CONDITIONS,
            u0 in (1.0, [1.0, 1.0], @SVector[1.0, 1.0])

            probN = NonlinearProblem(quadratic_f, u0, 2.0)
            @test all(solve(probN, alg(); termination_condition).u .≈ sqrt(2.0))
        end
    end
end

@testitem "SimpleHalley" setup=[RootfindingTesting] tags=[:core] begin
    @testset "AutoDiff: $(nameof(typeof(autodiff)))" for autodiff in (
        AutoFiniteDiff(), AutoForwardDiff())
        @testset "[OOP] u0: $(nameof(typeof(u0)))" for u0 in (
            [1.0, 1.0], @SVector[1.0, 1.0], 1.0)
            sol = benchmark_nlsolve_oop(quadratic_f, u0; solver = SimpleHalley(; autodiff))
            @test SciMLBase.successful_retcode(sol)
            @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)
        end

        @testset "[IIP] u0: $(nameof(typeof(u0)))" for u0 in ([1.0, 1.0],)
            sol = benchmark_nlsolve_iip(quadratic_f!, u0; solver = SimpleHalley(; autodiff))
            @test SciMLBase.successful_retcode(sol)
            @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)
        end
    end

    @testset "Termination condition: $(nameof(typeof(termination_condition))) u0: $(nameof(typeof(u0)))" for termination_condition in TERMINATION_CONDITIONS,
        u0 in (1.0, [1.0, 1.0], @SVector[1.0, 1.0])

        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(solve(probN, SimpleHalley(); termination_condition).u .≈ sqrt(2.0))
    end
end

@testitem "SimpleHouseholder" setup=[RootfindingTesting] tags=[:core] begin
    @testset "AutoDiff: TaylorDiff.jl" for order in (2, 3, 4)
        @testset "[OOP] u0: $(nameof(typeof(u0)))" for u0 in ([1.0], @SVector[1.0], 1.0)
            sol = benchmark_nlsolve_oop(
                quadratic_f, u0; solver = SimpleHouseholder{order}())
            @test SciMLBase.successful_retcode(sol)
            @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)
        end

        @testset "[IIP] u0: $(nameof(typeof(u0)))" for u0 in ([1.0],)
            sol = benchmark_nlsolve_iip(
                quadratic_f!, u0; solver = SimpleHouseholder{order}())
            @test SciMLBase.successful_retcode(sol)
            @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)
        end
    end

    @testset "Termination condition: $(nameof(typeof(termination_condition))) u0: $(nameof(typeof(u0)))" for termination_condition in TERMINATION_CONDITIONS,
        u0 in (1.0, [1.0], @SVector[1.0])

        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(solve(probN, SimpleHouseholder{2}(); termination_condition).u .≈
                  sqrt(2.0))
    end
end

@testitem "Derivative Free Metods" setup=[RootfindingTesting] tags=[:core] begin
    @testset "$(nameof(typeof(alg)))" for alg in [
        SimpleBroyden(), SimpleKlement(), SimpleDFSane(),
        SimpleLimitedMemoryBroyden(), SimpleBroyden(; linesearch = Val(true)),
        SimpleLimitedMemoryBroyden(; linesearch = Val(true))]
        @testset "[OOP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0], @SVector[1.0, 1.0], 1.0)
            sol = benchmark_nlsolve_oop(quadratic_f, u0; solver = alg)
            @test SciMLBase.successful_retcode(sol)
            @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)
        end

        @testset "[IIP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0],)
            sol = benchmark_nlsolve_iip(quadratic_f!, u0; solver = alg)
            @test SciMLBase.successful_retcode(sol)
            @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)
        end

        @testset "Termination condition: $(nameof(typeof(termination_condition))) u0: $(nameof(typeof(u0)))" for termination_condition in TERMINATION_CONDITIONS,
            u0 in (1.0, [1.0, 1.0], @SVector[1.0, 1.0])

            probN = NonlinearProblem(quadratic_f, u0, 2.0)
            @test all(solve(probN, alg; termination_condition).u .≈ sqrt(2.0))
        end
    end
end

@testitem "Newton Fails" setup=[RootfindingTesting] tags=[:core] begin
    u0 = [-10.0, -1.0, 1.0, 2.0, 3.0, 4.0, 10.0]
    p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    @testset "$(nameof(typeof(alg)))" for alg in (
        SimpleDFSane(), SimpleTrustRegion(), SimpleHalley(),
        SimpleTrustRegion(; nlsolve_update_rule = Val(true)))
        sol = benchmark_nlsolve_oop(newton_fails, u0, p; solver = alg)
        @test SciMLBase.successful_retcode(sol)
        @test all(abs.(newton_fails(sol.u, p)) .< 1e-9)
    end
end

@testitem "Kwargs Propagation" setup=[RootfindingTesting] tags=[:core] begin
    prob = NonlinearProblem(quadratic_f, ones(4), 2.0; maxiters = 2)
    sol = solve(prob, SimpleNewtonRaphson())
    @test sol.retcode === ReturnCode.MaxIters
end

@testitem "Allocation Checks" setup=[RootfindingTesting] tags=[:core] begin
    if Sys.islinux()  # Very slow on other OS
        @testset "$(nameof(typeof(alg)))" for alg in (
            SimpleNewtonRaphson(), SimpleHalley(), SimpleBroyden(),
            SimpleKlement(), SimpleLimitedMemoryBroyden(), SimpleTrustRegion(),
            SimpleTrustRegion(; nlsolve_update_rule = Val(true)),
            SimpleDFSane(), SimpleBroyden(; linesearch = Val(true)),
            SimpleLimitedMemoryBroyden(; linesearch = Val(true)))
            @check_allocs nlsolve(prob, alg) = SciMLBase.solve(prob, alg; abstol = 1e-9)

            nlprob_scalar = NonlinearProblem{false}(quadratic_f, 1.0, 2.0)
            nlprob_sa = NonlinearProblem{false}(quadratic_f, @SVector[1.0, 1.0], 2.0)

            try
                nlsolve(nlprob_scalar, alg)
                @test true
            catch e
                @error e
                @test false
            end

            # ForwardDiff allocates for hessian since we don't propagate the chunksize
            try
                nlsolve(nlprob_sa, alg)
                @test true
            catch e
                @error e
                @test false broken=(alg isa SimpleHalley)
            end
        end
    end
end

@testitem "Interval Nonlinear Problems" setup=[RootfindingTesting] tags=[:core] begin
    @testset "$(nameof(typeof(alg)))" for alg in (
        Bisection(), Falsi(), Ridder(), Brent(), ITP(), Alefeld())
        tspan = (1.0, 20.0)

        function g(p)
            probN = IntervalNonlinearProblem{false}(quadratic_f, typeof(p).(tspan), p)
            sol = solve(probN, alg; abstol = 1e-9)
            return sol.left
        end

        for p in 1.1:0.1:100.0
            @test g(p)≈sqrt(p) atol=1e-3 rtol=1e-3
            @test ForwardDiff.derivative(g, p)≈1 / (2 * sqrt(p)) atol=1e-3 rtol=1e-3
        end

        t = (p) -> [sqrt(p[2] / p[1])]
        p = [0.9, 50.0]

        function g2(p)
            probN = IntervalNonlinearProblem{false}((u, p) -> p[1] * u * u - p[2], tspan, p)
            sol = solve(probN, alg; abstol = 1e-9)
            return [sol.u]
        end

        @test g2(p)≈[sqrt(p[2] / p[1])] atol=1e-3 rtol=1e-3
        @test ForwardDiff.jacobian(g2, p)≈ForwardDiff.jacobian(t, p) atol=1e-3 rtol=1e-3

        probB = IntervalNonlinearProblem{false}(quadratic_f, (1.0, 2.0), 2.0)
        sol = solve(probB, alg; abstol = 1e-9)
        @test sol.left≈sqrt(2.0) atol=1e-3 rtol=1e-3

        if !(alg isa Bisection || alg isa Falsi)
            probB = IntervalNonlinearProblem{false}(quadratic_f, (sqrt(2.0), 10.0), 2.0)
            sol = solve(probB, alg; abstol = 1e-9)
            @test sol.left≈sqrt(2.0) atol=1e-3 rtol=1e-3

            probB = IntervalNonlinearProblem{false}(quadratic_f, (0.0, sqrt(2.0)), 2.0)
            sol = solve(probB, alg; abstol = 1e-9)
            @test sol.left≈sqrt(2.0) atol=1e-3 rtol=1e-3
        end
    end
end

@testitem "Tolerance Tests Interval Methods" setup=[RootfindingTesting] tags=[:core] begin
    @testset "$(nameof(typeof(alg)))" for alg in (Bisection(), Falsi(), ITP())
        tspan = (1.0, 20.0)
        probB = IntervalNonlinearProblem(quadratic_f, tspan, 2.0)
        tols = [0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7]
        ϵ = eps(1.0) #least possible tol for all methods

        for atol in tols
            sol = solve(probB, alg; abstol = atol)
            @test abs(sol.u - sqrt(2)) < atol
            @test abs(sol.u - sqrt(2)) > ϵ #test that the solution is not calculated upto max precision
        end
    end
end

@testitem "Tolerance Tests Interval Methods 2" setup=[RootfindingTesting] tags=[:core] begin
    @testset "$(nameof(typeof(alg)))" for alg in (Ridder(), Brent())
        tspan = (1.0, 20.0)
        probB = IntervalNonlinearProblem(quadratic_f, tspan, 2.0)
        tols = [0.1] # Ridder and Brent converge rapidly so as we lower tolerance below 0.01, it converges with max precision to the solution
        ϵ = eps(1.0) #least possible tol for all methods

        for atol in tols
            sol = solve(probB, alg; abstol = atol)
            @test abs(sol.u - sqrt(2)) < atol
            @test abs(sol.u - sqrt(2)) > ϵ #test that the solution is not calculated upto max precision
        end
    end
end

@testitem "Flipped Signs and Reversed Tspan" setup=[RootfindingTesting] tags=[:core] begin
    @testset "$(nameof(typeof(alg)))" for alg in (
        Alefeld(), Bisection(), Falsi(), Brent(), ITP(), Ridder())
        f1(u, p) = u * u - p
        f2(u, p) = p - u * u

        for p in 1:4
            inp1 = IntervalNonlinearProblem(f1, (1.0, 2.0), p)
            inp2 = IntervalNonlinearProblem(f2, (1.0, 2.0), p)
            inp3 = IntervalNonlinearProblem(f1, (2.0, 1.0), p)
            inp4 = IntervalNonlinearProblem(f2, (2.0, 1.0), p)
            @test abs.(solve(inp1, alg).u) ≈ sqrt.(p)
            @test abs.(solve(inp2, alg).u) ≈ sqrt.(p)
            @test abs.(solve(inp3, alg).u) ≈ sqrt.(p)
            @test abs.(solve(inp4, alg).u) ≈ sqrt.(p)
        end
    end
end
