using NonlinearSolve, LinearSolve, LinearAlgebra, Test, Random

true_function(x, θ) = @. θ[1] * exp(θ[2] * x) * cos(θ[3] * x + θ[4])
true_function(y, x, θ) = (@. y = θ[1] * exp(θ[2] * x) * cos(θ[3] * x + θ[4]))

θ_true = [1.0, 0.1, 2.0, 0.5]

x = [-1.0, -0.5, 0.0, 0.5, 1.0]

y_target = true_function(x, θ_true)

function loss_function(θ, p)
    ŷ = true_function(p, θ)
    return abs2.(ŷ .- y_target)
end

function loss_function(resid, θ, p)
    true_function(resid, p, θ)
    resid .= abs2.(resid .- y_target)
    return resid
end

θ_init = θ_true .+ randn!(similar(θ_true)) * 0.1
prob_oop = NonlinearLeastSquaresProblem{false}(loss_function, θ_init, x)
prob_iip = NonlinearLeastSquaresProblem(NonlinearFunction(loss_function;
        resid_prototype = zero(y_target)), θ_init, x)

sol = solve(prob_oop, GaussNewton(); maxiters = 1000, abstol = 1e-8)
@test SciMLBase.successful_retcode(sol)
@test norm(sol.resid) < 1e-6

sol = solve(prob_iip, GaussNewton(); maxiters = 1000, abstol = 1e-8)
@test SciMLBase.successful_retcode(sol)
@test norm(sol.resid) < 1e-6

sol = solve(prob_oop, LevenbergMarquardt(); maxiters = 1000, abstol = 1e-8)
@test SciMLBase.successful_retcode(sol)
@test norm(sol.resid) < 1e-6

sol = solve(prob_iip, LevenbergMarquardt(; linsolve = NormalCholeskyFactorization());
    maxiters = 1000, abstol = 1e-8)
@test SciMLBase.successful_retcode(sol)
@test norm(sol.resid) < 1e-6
