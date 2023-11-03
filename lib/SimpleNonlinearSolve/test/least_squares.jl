using SimpleNonlinearSolve, LinearAlgebra, Test

true_function(x, θ) = @. θ[1] * exp(θ[2] * x) * cos(θ[3] * x + θ[4])

θ_true = [1.0, 0.1, 2.0, 0.5]
x = [-1.0, -0.5, 0.0, 0.5, 1.0]
y_target = true_function(x, θ_true)

function loss_function(θ, p)
    ŷ = true_function(p, θ)
    return abs2.(ŷ .- y_target)
end

θ_init = θ_true .+ 0.1
prob_oop = NonlinearLeastSquaresProblem{false}(loss_function, θ_init, x)
sol = solve(prob_oop, SimpleNewtonRaphson())
sol = solve(prob_oop, SimpleGaussNewton())

@test norm(sol.resid) < 1e-12
