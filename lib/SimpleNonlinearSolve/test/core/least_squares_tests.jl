@testitem "Nonlinear Least Squares" tags = [:core] begin
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
            SimpleNewtonRaphson(AutoFiniteDiff()), SimpleGaussNewton(AutoFiniteDiff()),
        ]
        sol = solve(prob_oop, solver)
        @test norm(sol.resid, Inf) < 1.0e-12
    end

    prob_iip = NonlinearLeastSquaresProblem(
        NonlinearFunction{true}(loss_function!, resid_prototype = zeros(length(y_target))),
        θ_init, x
    )

    @testset "Solver: $(nameof(typeof(solver)))" for solver in [
            SimpleNewtonRaphson(AutoForwardDiff()), SimpleGaussNewton(AutoForwardDiff()),
            SimpleNewtonRaphson(AutoFiniteDiff()), SimpleGaussNewton(AutoFiniteDiff()),
        ]
        sol = solve(prob_iip, solver)
        @test norm(sol.resid, Inf) < 1.0e-12
    end
end

@testitem "Null u0 NonlinearLeastSquaresProblem" tags = [:core] begin
    using LinearAlgebra, SciMLBase

    # OOP: unsatisfiable residual (issue #842)
    f_unsat(u, p) = [1.0]
    unsat_f = NonlinearFunction{false}(f_unsat; resid_prototype = zeros(1))
    prob = NonlinearLeastSquaresProblem(unsat_f, nothing)
    sol = solve(prob, SimpleNewtonRaphson())
    @test sol.retcode == ReturnCode.Failure
    @test sol.resid == [1.0]
    @test sol.u == Float64[]

    # OOP: satisfiable residual
    f_sat(u, p) = [0.0]
    sat_f = NonlinearFunction{false}(f_sat; resid_prototype = zeros(1))
    prob_sat = NonlinearLeastSquaresProblem(sat_f, nothing)
    sol_sat = solve(prob_sat, SimpleNewtonRaphson())
    @test sol_sat.retcode == ReturnCode.Success
    @test sol_sat.resid == [0.0]

    # IIP: unsatisfiable residual
    function f_unsat!(du, u, p)
        du[1] = 1.0
        return
    end
    unsat_f_iip = NonlinearFunction{true}(f_unsat!; resid_prototype = zeros(1))
    prob_iip = NonlinearLeastSquaresProblem(unsat_f_iip, nothing)
    sol_iip = solve(prob_iip, SimpleNewtonRaphson())
    @test sol_iip.retcode == ReturnCode.Failure
    @test sol_iip.resid == [1.0]
end
