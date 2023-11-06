using NonlinearSolve, LinearSolve, LinearAlgebra, Test, Random, ForwardDiff
import FastLevenbergMarquardt, LeastSquaresOptim

true_function(x, θ) = @. θ[1] * exp(θ[2] * x) * cos(θ[3] * x + θ[4])
true_function(y, x, θ) = (@. y = θ[1] * exp(θ[2] * x) * cos(θ[3] * x + θ[4]))

θ_true = [1.0, 0.1, 2.0, 0.5]

x = [-1.0, -0.5, 0.0, 0.5, 1.0]

y_target = true_function(x, θ_true)

function loss_function(θ, p)
    ŷ = true_function(p, θ)
    return ŷ .- y_target
end

function loss_function(resid, θ, p)
    true_function(resid, p, θ)
    resid .= resid .- y_target
    return resid
end

θ_init = θ_true .+ randn!(similar(θ_true)) * 0.1
prob_oop = NonlinearLeastSquaresProblem{false}(loss_function, θ_init, x)
prob_iip = NonlinearLeastSquaresProblem(NonlinearFunction(loss_function;
        resid_prototype = zero(y_target)), θ_init, x)

nlls_problems = [prob_oop, prob_iip]
solvers = vec(Any[GaussNewton(; linsolve, linesearch)
                  for linsolve in [nothing, LUFactorization()],
linesearch in [Static(), BackTracking(), HagerZhang(), StrongWolfe(), MoreThuente()]])
append!(solvers,
    [
        LevenbergMarquardt(),
        LevenbergMarquardt(; linsolve = LUFactorization()),
        LeastSquaresOptimJL(:lm),
        LeastSquaresOptimJL(:dogleg),
        nothing,
    ])

for prob in nlls_problems, solver in solvers
    @time sol = solve(prob, solver; maxiters = 10000, abstol = 1e-8)
    @test SciMLBase.successful_retcode(sol)
    @test norm(sol.resid) < 1e-6
end

function jac!(J, θ, p)
    resid = zeros(length(p))
    ForwardDiff.jacobian!(J, (resid, θ) -> loss_function(resid, θ, p), resid, θ)
    return J
end

prob = NonlinearLeastSquaresProblem(NonlinearFunction(loss_function;
        resid_prototype = zero(y_target), jac = jac!), θ_init, x)

solvers = [FastLevenbergMarquardtJL(:cholesky), FastLevenbergMarquardtJL(:qr)]

for solver in solvers
    @time sol = solve(prob, solver; maxiters = 10000, abstol = 1e-8)
    @test norm(sol.resid) < 1e-6
end
