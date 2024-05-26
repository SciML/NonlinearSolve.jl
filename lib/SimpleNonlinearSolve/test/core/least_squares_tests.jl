@testitem "Nonlinear Least Squares" tags=[:core] begin
    using LinearAlgebra

    true_function(x, θ) = @. θ[1] * exp(θ[2] * x) * cos(θ[3] * x + θ[4])

    θ_true = [1.0, 0.1, 2.0, 0.5]
    x = [-1.0, -0.5, 0.0, 0.5, 1.0]
    y_target = true_function(x, θ_true)

    function loss_function(θ, p)
        ŷ = true_function(p, θ)
        return ŷ .- y_target
    end

    function loss_function!(resid, θ, p)
        ŷ = true_function(p, θ)
        @. resid = ŷ - y_target
        return
    end

    θ_init = θ_true .+ 0.1
    prob_oop = NonlinearLeastSquaresProblem{false}(loss_function, θ_init, x)

    @testset "Solver: $(nameof(typeof(solver)))" for solver in [
        SimpleNewtonRaphson(AutoForwardDiff()), SimpleGaussNewton(AutoForwardDiff()),
        SimpleNewtonRaphson(AutoFiniteDiff()), SimpleGaussNewton(AutoFiniteDiff())]
        sol = solve(prob_oop, solver)
        @test norm(sol.resid, Inf) < 1e-12
    end

    prob_iip = NonlinearLeastSquaresProblem(
        NonlinearFunction{true}(loss_function!, resid_prototype = zeros(length(y_target))),
        θ_init, x)

    @testset "Solver: $(nameof(typeof(solver)))" for solver in [
        SimpleNewtonRaphson(AutoForwardDiff()), SimpleGaussNewton(AutoForwardDiff()),
        SimpleNewtonRaphson(AutoFiniteDiff()), SimpleGaussNewton(AutoFiniteDiff())]
        sol = solve(prob_iip, solver)
        @test norm(sol.resid, Inf) < 1e-12
    end
end
