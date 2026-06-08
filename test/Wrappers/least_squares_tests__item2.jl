using NonlinearSolve
include("setup_wrappernllssetup.jl")

import FastLevenbergMarquardt, MINPACK
using ForwardDiff

function jac!(J, θ, p)
    resid = zeros(length(p))
    ForwardDiff.jacobian!(J, (resid, θ) -> loss_function(resid, θ, p), resid, θ)
    return J
end

jac(θ, p) = ForwardDiff.jacobian(θ -> loss_function(θ, p), θ)

probs = [
    NonlinearLeastSquaresProblem(
        NonlinearFunction{true}(
            loss_function; resid_prototype = zero(y_target), jac = jac!
        ),
        θ_init, x
    ),
    NonlinearLeastSquaresProblem(
        NonlinearFunction{false}(
            loss_function; resid_prototype = zero(y_target), jac = jac
        ),
        θ_init, x
    ),
    NonlinearLeastSquaresProblem(
        NonlinearFunction{false}(loss_function; jac), θ_init, x
    ),
]

solvers = Any[FastLevenbergMarquardtJL(linsolve) for linsolve in (:cholesky, :qr)]
Sys.isapple()||push!(solvers, CMINPACK())

for solver in solvers, prob in probs

    sol = solve(prob, solver; maxiters = 10000, abstol = 1.0e-8)
    @test maximum(abs, sol.resid) < 1.0e-6
end
