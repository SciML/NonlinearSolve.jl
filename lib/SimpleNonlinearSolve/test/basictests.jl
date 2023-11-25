using BenchmarkTools, LinearSolve, NonlinearSolve, StaticArrays, Random, LinearAlgebra,
    Test, ForwardDiff, DiffEqBase

_nameof(x) = applicable(nameof, x) ? nameof(x) : _nameof(typeof(x))

quadratic_f(u, p) = u .* u .- p
quadratic_f!(du, u, p) = (du .= u .* u .- p)
quadratic_f2(u, p) = @. p[1] * u * u - p[2]

function newton_fails(u, p)
    return 0.010000000000000002 .+
           10.000000000000002 ./ (1 .+
            (0.21640425613334457 .+
             216.40425613334457 ./ (1 .+
              (0.21640425613334457 .+
               216.40425613334457 ./
               (1 .+ 0.0006250000000000001(u .^ 2.0))) .^ 2.0)) .^ 2.0) .-
           0.0011552453009332421u .- p
end

const TERMINATION_CONDITIONS = [
    NormTerminationMode(), RelTerminationMode(), RelNormTerminationMode(),
    AbsTerminationMode(), AbsNormTerminationMode(), RelSafeTerminationMode(),
    AbsSafeTerminationMode(), RelSafeBestTerminationMode(), AbsSafeBestTerminationMode(),
]

# --- SimpleNewtonRaphson tests ---

@testset "$(alg)" for alg in (SimpleNewtonRaphson, SimpleTrustRegion)
    # Eval else the alg is type unstable
    @eval begin
        function benchmark_nlsolve_oop(f, u0, p = 2.0; autodiff = AutoForwardDiff())
            prob = NonlinearProblem{false}(f, u0, p)
            return solve(prob, $(alg)(; autodiff), abstol = 1e-9)
        end

        function benchmark_nlsolve_iip(f, u0, p = 2.0; autodiff = AutoForwardDiff())
            prob = NonlinearProblem{true}(f, u0, p)
            return solve(prob, $(alg)(; autodiff), abstol = 1e-9)
        end
    end

    @testset "AutoDiff: $(_nameof(autodiff))" for autodiff in (AutoFiniteDiff(),
        AutoForwardDiff())
        @testset "[OOP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0], @SVector[1.0, 1.0], 1.0)
            sol = benchmark_nlsolve_oop(quadratic_f, u0; autodiff)
            @test SciMLBase.successful_retcode(sol)
            @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)
        end

        @testset "[IIP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0],)
            sol = benchmark_nlsolve_iip(quadratic_f!, u0; autodiff)
            @test SciMLBase.successful_retcode(sol)
            @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)
        end
    end

    @testset "Allocations: Static Array and Scalars" begin
        @test (@ballocated $(benchmark_nlsolve_oop)($quadratic_f, $(@SVector[1.0, 1.0]),
            2.0; autodiff = AutoForwardDiff())) < 200
        @test (@ballocated $(benchmark_nlsolve_oop)($quadratic_f, 1.0, 2.0;
            autodiff = AutoForwardDiff())) == 0
    end

    @testset "[OOP] Immutable AD" begin
        for p in [1.0, 100.0]
            @test begin
                res = benchmark_nlsolve_oop(quadratic_f, @SVector[1.0, 1.0], p)
                res_true = sqrt(p)
                all(res.u .≈ res_true)
            end
            @test ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f,
                @SVector[1.0, 1.0], p).u[end], p) ≈ 1 / (2 * sqrt(p))
        end
    end

    @testset "[OOP] Scalar AD" begin
        for p in 1.0:0.1:100.0
            @test begin
                res = benchmark_nlsolve_oop(quadratic_f, 1.0, p)
                res_true = sqrt(p)
                res.u ≈ res_true
            end
            @test ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f, 1.0, p).u,
                p) ≈ 1 / (2 * sqrt(p))
        end
    end

    t = (p) -> [sqrt(p[2] / p[1])]
    p = [0.9, 50.0]
    @test benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u ≈ sqrt(p[2] / p[1])
    @test ForwardDiff.jacobian(p -> [benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u],
        p) ≈ ForwardDiff.jacobian(t, p)

    @testset "Termination condition: $(termination_condition) u0: $(_nameof(u0))" for termination_condition in TERMINATION_CONDITIONS,
        u0 in (1.0, [1.0, 1.0], @SVector[1.0, 1.0])

        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(solve(probN, alg(); termination_condition).u .≈ sqrt(2.0))
    end
end

# --- SimpleHalley tests ---

@testset "SimpleHalley" begin
    function benchmark_nlsolve_oop(f, u0, p = 2.0; autodiff = AutoForwardDiff())
        prob = NonlinearProblem{false}(f, u0, p)
        return solve(prob, SimpleHalley(; autodiff), abstol = 1e-9)
    end

    @testset "AutoDiff: $(_nameof(autodiff))" for autodiff in (AutoFiniteDiff(),
        AutoForwardDiff())
        @testset "[OOP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0], @SVector[1.0, 1.0], 1.0)
            sol = benchmark_nlsolve_oop(quadratic_f, u0; autodiff)
            @test SciMLBase.successful_retcode(sol)
            @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)
        end
    end

    @testset "Allocations: Static Array and Scalars" begin
        @test (@ballocated $(benchmark_nlsolve_oop)($quadratic_f, 1.0, 2.0;
            autodiff = AutoForwardDiff())) == 0
    end

    @testset "[OOP] Immutable AD" begin
        for p in [1.0, 100.0]
            @test begin
                res = benchmark_nlsolve_oop(quadratic_f, @SVector[1.0, 1.0], p)
                res_true = sqrt(p)
                all(res.u .≈ res_true)
            end
            @test ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f,
                @SVector[1.0, 1.0], p).u[end], p) ≈ 1 / (2 * sqrt(p))
        end
    end

    @testset "[OOP] Scalar AD" begin
        for p in 1.0:0.1:100.0
            @test begin
                res = benchmark_nlsolve_oop(quadratic_f, 1.0, p)
                res_true = sqrt(p)
                res.u ≈ res_true
            end
            @test ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f, 1.0, p).u,
                p) ≈ 1 / (2 * sqrt(p))
        end
    end

    t = (p) -> [sqrt(p[2] / p[1])]
    p = [0.9, 50.0]
    @test benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u ≈ sqrt(p[2] / p[1])
    @test ForwardDiff.jacobian(p -> [benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u],
        p) ≈ ForwardDiff.jacobian(t, p)

    @testset "Termination condition: $(termination_condition) u0: $(_nameof(u0))" for termination_condition in TERMINATION_CONDITIONS,
        u0 in (1.0, [1.0, 1.0], @SVector[1.0, 1.0])

        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @show solve(probN, SimpleHalley(); termination_condition).u
        @test all(solve(probN, SimpleHalley(); termination_condition).u .≈ sqrt(2.0))
    end
end

# --- SimpleBroyden / SimpleKlement / SimpleLimitedMemoryBroyden tests ---

@testset "$(alg)" for alg in [SimpleBroyden(), SimpleKlement(), SimpleDFSane(),
        SimpleLimitedMemoryBroyden()]
    function benchmark_nlsolve_oop(f, u0, p = 2.0)
        prob = NonlinearProblem{false}(f, u0, p)
        return solve(prob, alg, abstol = 1e-9)
    end

    function benchmark_nlsolve_iip(f, u0, p = 2.0)
        prob = NonlinearProblem{true}(f, u0, p)
        return solve(prob, alg, abstol = 1e-9)
    end

    @testset "[OOP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0], @SVector[1.0, 1.0], 1.0)
        sol = benchmark_nlsolve_oop(quadratic_f, u0)
        @test SciMLBase.successful_retcode(sol)
        @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)
    end

    @testset "[IIP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0],)
        sol = benchmark_nlsolve_iip(quadratic_f!, u0)
        @test SciMLBase.successful_retcode(sol)
        @test all(abs.(sol.u .* sol.u .- 2) .< 1e-9)
    end

    @testset "Allocations: Static Array and Scalars" begin
        @test (@ballocated $(benchmark_nlsolve_oop)($quadratic_f, $(@SVector[1.0, 1.0]),
            2.0)) < 200
        @test (@ballocated $(benchmark_nlsolve_oop)($quadratic_f, 1.0, 2.0)) == 0
    end

    @testset "[OOP] Immutable AD" begin
        for p in [1.0, 100.0]
            @test begin
                res = benchmark_nlsolve_oop(quadratic_f, @SVector[1.0, 1.0], p)
                res_true = sqrt(p)
                all(res.u .≈ res_true)
            end
            @test ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f,
                @SVector[1.0, 1.0], p).u[end], p) ≈ 1 / (2 * sqrt(p))
        end
    end

    @testset "[OOP] Scalar AD" begin
        for p in 1.0:0.1:100.0
            @test begin
                res = benchmark_nlsolve_oop(quadratic_f, 1.0, p)
                res_true = sqrt(p)
                res.u ≈ res_true
            end
            @test ForwardDiff.derivative(p -> benchmark_nlsolve_oop(quadratic_f, 1.0, p).u,
                p) ≈ 1 / (2 * sqrt(p))
        end
    end

    t = (p) -> [sqrt(p[2] / p[1])]
    p = [0.9, 50.0]
    @test benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u ≈ sqrt(p[2] / p[1])
    @test ForwardDiff.jacobian(p -> [benchmark_nlsolve_oop(quadratic_f2, 0.5, p).u],
        p) ≈ ForwardDiff.jacobian(t, p)

    @testset "Termination condition: $(termination_condition) u0: $(_nameof(u0))" for termination_condition in TERMINATION_CONDITIONS,
        u0 in (1.0, [1.0, 1.0], @SVector[1.0, 1.0])

        probN = NonlinearProblem(quadratic_f, u0, 2.0)
        @test all(solve(probN, alg; termination_condition).u .≈ sqrt(2.0))
    end
end


1 + 1 + 1

# tspan = (1.0, 20.0)
# # Falsi
# g = function (p)
#     probN = IntervalNonlinearProblem{false}(f, typeof(p).(tspan), p)
#     sol = solve(probN, Falsi())
#     return sol.left
# end

# for p in 1.1:0.1:100.0
#     @test g(p) ≈ sqrt(p)
#     @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
# end

# # Ridder
# g = function (p)
#     probN = IntervalNonlinearProblem{false}(f, typeof(p).(tspan), p)
#     sol = solve(probN, Ridder())
#     return sol.left
# end

# for p in 1.1:0.1:100.0
#     @test g(p) ≈ sqrt(p)
#     @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
# end

# # Brent
# g = function (p)
#     probN = IntervalNonlinearProblem{false}(f, typeof(p).(tspan), p)
#     sol = solve(probN, Brent())
#     return sol.left
# end

# for p in 1.1:0.1:100.0
#     @test g(p) ≈ sqrt(p)
#     @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
# end

# # ITP
# g = function (p)
#     probN = IntervalNonlinearProblem{false}(f, typeof(p).(tspan), p)
#     sol = solve(probN, ITP())
#     return sol.u
# end

# for p in 1.1:0.1:100.0
#     @test g(p) ≈ sqrt(p)
#     @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
# end

# # Alefeld
# g = function (p)
#     probN = IntervalNonlinearProblem{false}(f, typeof(p).(tspan), p)
#     sol = solve(probN, Alefeld())
#     return sol.u
# end

# for p in 1.1:0.1:100.0
#     @test g(p) ≈ sqrt(p)
#     @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p))
# end

# f, tspan = (u, p) -> p[1] * u * u - p[2], (1.0, 100.0)
# t = (p) -> [sqrt(p[2] / p[1])]
# p = [0.9, 50.0]
# g = function (p)
#     probN = IntervalNonlinearProblem{false}(f, tspan, p)
#     sol = solve(probN, Alefeld())
#     return [sol.u]
# end

# @test g(p) ≈ [sqrt(p[2] / p[1])]
# @test ForwardDiff.jacobian(g, p) ≈ ForwardDiff.jacobian(t, p)

# f, tspan = (u, p) -> p[1] * u * u - p[2], (1.0, 100.0)
# t = (p) -> [sqrt(p[2] / p[1])]
# p = [0.9, 50.0]
# for alg in [Bisection(), Falsi(), Ridder(), Brent(), ITP()]
#     global g, p
#     g = function (p)
#         probN = IntervalNonlinearProblem{false}(f, tspan, p)
#         sol = solve(probN, alg)
#         return [sol.left]
#     end

#     @test g(p) ≈ [sqrt(p[2] / p[1])]
#     @test ForwardDiff.jacobian(g, p) ≈ ForwardDiff.jacobian(t, p)
# end

# # Bisection Tests
# f, tspan = (u, p) -> u .* u .- 2.0, (1.0, 2.0)
# probB = IntervalNonlinearProblem(f, tspan)

# # Falsi
# sol = solve(probB, Falsi())
# @test sol.left ≈ sqrt(2.0)

# # Bisection
# sol = solve(probB, Bisection())
# @test sol.left ≈ sqrt(2.0)

# # Ridder
# sol = solve(probB, Ridder())
# @test sol.left ≈ sqrt(2.0)
# tspan = (sqrt(2.0), 10.0)
# probB = IntervalNonlinearProblem(f, tspan)
# sol = solve(probB, Ridder())
# @test sol.left ≈ sqrt(2.0)
# tspan = (0.0, sqrt(2.0))
# probB = IntervalNonlinearProblem(f, tspan)
# sol = solve(probB, Ridder())
# @test sol.left ≈ sqrt(2.0)

# # Brent
# sol = solve(probB, Brent())
# @test sol.left ≈ sqrt(2.0)
# tspan = (sqrt(2.0), 10.0)
# probB = IntervalNonlinearProblem(f, tspan)
# sol = solve(probB, Brent())
# @test sol.left ≈ sqrt(2.0)
# tspan = (0.0, sqrt(2.0))
# probB = IntervalNonlinearProblem(f, tspan)
# sol = solve(probB, Brent())
# @test sol.left ≈ sqrt(2.0)

# # Alefeld
# sol = solve(probB, Alefeld())
# @test sol.u ≈ sqrt(2.0)
# tspan = (sqrt(2.0), 10.0)
# probB = IntervalNonlinearProblem(f, tspan)
# sol = solve(probB, Alefeld())
# @test sol.u ≈ sqrt(2.0)
# tspan = (0.0, sqrt(2.0))
# probB = IntervalNonlinearProblem(f, tspan)
# sol = solve(probB, Alefeld())
# @test sol.u ≈ sqrt(2.0)

# # ITP
# sol = solve(probB, ITP())
# @test sol.u ≈ sqrt(2.0)
# tspan = (sqrt(2.0), 10.0)
# probB = IntervalNonlinearProblem(f, tspan)
# sol = solve(probB, ITP())
# @test sol.u ≈ sqrt(2.0)
# tspan = (0.0, sqrt(2.0))
# probB = IntervalNonlinearProblem(f, tspan)
# sol = solve(probB, ITP())
# @test sol.u ≈ sqrt(2.0)

# # Tolerance tests for Interval methods
# f, tspan = (u, p) -> u .* u .- 2.0, (1.0, 10.0)
# probB = IntervalNonlinearProblem(f, tspan)
# tols = [0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7]
# ϵ = eps(1.0) #least possible tol for all methods

# for atol in tols
#     sol = solve(probB, Bisection(), abstol = atol)
#     @test abs(sol.u - sqrt(2)) < atol
#     @test abs(sol.u - sqrt(2)) > ϵ #test that the solution is not calculated upto max precision
#     sol = solve(probB, Falsi(), abstol = atol)
#     @test abs(sol.u - sqrt(2)) < atol
#     @test abs(sol.u - sqrt(2)) > ϵ
#     sol = solve(probB, ITP(), abstol = atol)
#     @test abs(sol.u - sqrt(2)) < atol
#     @test abs(sol.u - sqrt(2)) > ϵ
# end

# tols = [0.1] # Ridder and Brent converge rapidly so as we lower tolerance below 0.01, it converges with max precision to the solution
# for atol in tols
#     sol = solve(probB, Ridder(), abstol = atol)
#     @test abs(sol.u - sqrt(2)) < atol
#     @test abs(sol.u - sqrt(2)) > ϵ
#     sol = solve(probB, Brent(), abstol = atol)
#     @test abs(sol.u - sqrt(2)) < atol
#     @test abs(sol.u - sqrt(2)) > ϵ
# end

# # Garuntee Tests for Bisection
# f = function (u, p)
#     if u < 2.0
#         return u - 2.0
#     elseif u > 3.0
#         return u - 3.0
#     else
#         return 0.0
#     end
# end
# probB = IntervalNonlinearProblem(f, (0.0, 4.0))

# sol = solve(probB, Bisection(; exact_left = true))
# @test f(sol.left, nothing) < 0.0
# @test f(nextfloat(sol.left), nothing) >= 0.0

# sol = solve(probB, Bisection(; exact_right = true))
# @test f(sol.right, nothing) >= 0.0
# @test f(prevfloat(sol.right), nothing) <= 0.0

# sol = solve(probB, Bisection(; exact_left = true, exact_right = true); immutable = false)
# @test f(sol.left, nothing) < 0.0
# @test f(nextfloat(sol.left), nothing) >= 0.0
# @test f(sol.right, nothing) >= 0.0
# @test f(prevfloat(sol.right), nothing) <= 0.0

# # Flipped signs & reversed tspan test for bracketing algorithms
# f1(u, p) = u * u - p
# f2(u, p) = p - u * u

# for alg in (Alefeld(), Bisection(), Falsi(), Brent(), ITP(), Ridder())
#     for p in 1:4
#         inp1 = IntervalNonlinearProblem(f1, (1.0, 2.0), p)
#         inp2 = IntervalNonlinearProblem(f2, (1.0, 2.0), p)
#         inp3 = IntervalNonlinearProblem(f1, (2.0, 1.0), p)
#         inp4 = IntervalNonlinearProblem(f2, (2.0, 1.0), p)
#         @test abs.(solve(inp1, alg).u) ≈ sqrt.(p)
#         @test abs.(solve(inp2, alg).u) ≈ sqrt.(p)
#         @test abs.(solve(inp3, alg).u) ≈ sqrt.(p)
#         @test abs.(solve(inp4, alg).u) ≈ sqrt.(p)
#     end
# end
