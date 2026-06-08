using NonlinearSolve
include("setup_wrappernllssetup.jl")

import FastLevenbergMarquardt, MINPACK

probs = [
    NonlinearLeastSquaresProblem(
        NonlinearFunction{true}(loss_function; resid_prototype = zero(y_target)),
        θ_init, x
    ),
    NonlinearLeastSquaresProblem(
        NonlinearFunction{false}(loss_function; resid_prototype = zero(y_target)),
        θ_init, x
    ),
    NonlinearLeastSquaresProblem(NonlinearFunction{false}(loss_function), θ_init, x),
]

solvers = []
for linsolve in (:cholesky, :qr),
        autodiff in (nothing, AutoForwardDiff(), AutoFiniteDiff())

    push!(solvers, FastLevenbergMarquardtJL(linsolve; autodiff))
end
if !Sys.isapple()
    for method in (:auto, :lm, :lmdif)
        push!(solvers, CMINPACK(; method))
    end
end

for solver in solvers, prob in probs

    sol = solve(prob, solver; maxiters = 10000, abstol = 1.0e-8)
    @test maximum(abs, sol.resid) < 1.0e-6
end
