using NonlinearSolve
include("setup_wrappernllssetup.jl")

using StaticArrays, FastLevenbergMarquardt

x_sa = SA[-1.0, -0.5, 0.0, 0.5, 1.0]

const y_target_sa = true_function(x_sa, θ_true)

function loss_function_sa(θ, p)
    ŷ = true_function(p, θ)
    return ŷ .- y_target_sa
end

θ_init_sa = SVector{4}(θ_init)
prob_sa = NonlinearLeastSquaresProblem{false}(loss_function_sa, θ_init_sa, x)

sol = solve(prob_sa, FastLevenbergMarquardtJL())
@test maximum(abs, sol.resid) < 1.0e-6
