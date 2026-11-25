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

@testset "Accepted-point Jacobian" begin
    f!(du, u, p) = (du .= u .^ 2 .- 2)
    jac!(J, u, p) = (J[1, 1] = 2 * only(u))
    prob = NonlinearProblem(NonlinearFunction{true}(f!; jac = jac!), [0.5])
    sol = solve(prob, SimpleTrustRegion(; initial_trust_radius = 5.0, max_trust_radius = 100.0); maxiters = 40, abstol = 1.0e-10)
    @test sol.retcode == ReturnCode.Success
    @test only(sol.u) ≈ sqrt(2)
end

@testset "Exact initial root" begin
    f(u, p) = abs.(u)
    jac(u, p) = error("Jacobian undefined at zero")
    prob = NonlinearProblem(NonlinearFunction{false}(f; jac), [0.0])
    @testset for alg in (SimpleNewtonRaphson(), SimpleHalley())
        sol = solve(prob, alg)
        @test sol.retcode == ReturnCode.Success
        @test sol.u == [0.0]
    end
end

@testset "Shrink limit at a converged trial" begin
    prob = NonlinearProblem((u, p) -> u - 1, 0.0)
    alg = SimpleTrustRegion(; initial_trust_radius = 2.0, shrink_threshold = 2.0, max_shrink_times = 0)
    @test solve(prob, alg).retcode == ReturnCode.ShrinkThresholdExceeded
end

@testset "Converged trust-region trial" begin
    f(u, p) = u
    jac(u, p) = iszero(u) ? error("Jacobian requested at the converged trial") : one(u)
    prob = NonlinearProblem(NonlinearFunction{false}(f; jac), 1.0)
    sol = solve(prob, SimpleTrustRegion(; initial_trust_radius = 1.0))
    @test sol.retcode == ReturnCode.Success
    @test iszero(sol.u)
end

@testset "SimpleBroyden exact initial root" begin
    identity_prob(u) = NonlinearProblem((x, p) -> x, u)
    @testset for u in (0, 0 // 1)
        sol = solve(identity_prob(u), SimpleBroyden())
        @test sol.retcode == ReturnCode.Success
        @test sol.u === u
    end
    prob = identity_prob(zeros(512))
    solve(prob, SimpleBroyden())
    @test (@allocated solve(prob, SimpleBroyden())) < length(prob.u0) * sizeof(prob.u0)
end

@testset "Simple Newton checks convergence before refreshing the Jacobian" begin
    f(x, p) = 0.01f0 * (x^2 - 1)
    for (problem, alg) in (
            (NonlinearProblem, SimpleNewtonRaphson()),
            (NonlinearLeastSquaresProblem, SimpleGaussNewton()),
        )
        for (initial, expected, calls, budget, code) in (
                (0.01f0, 0.01f0, 0, 1, ReturnCode.Success),
                (2.0f0, 1.25f0, 1, 2, ReturnCode.Success),
                (0.1f0, 0.1f0, 0, 0, ReturnCode.Success),
                (4.0f0, 2.125f0, 1, 1, ReturnCode.MaxIters),
            )
            evaluations = Ref(0)
            function jac(x, p)
                evaluations[] += 1
                return 0.02f0 * x
            end
            prob = problem(NonlinearFunction{false}(f; jac), initial)
            sol = solve(prob, alg; abstol = 0.01f0, maxiters = budget)
            @test sol.retcode == code
            @test sol.u ≈ expected
            @test sol.resid ≈ f(sol.u, nothing)
            @test evaluations[] == calls
        end
    end
end
