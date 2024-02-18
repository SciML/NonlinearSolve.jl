@testsetup module WrapperNLLSSetup
using Reexport
@reexport using LinearAlgebra, StableRNGs, StaticArrays, Random, ForwardDiff, Zygote
import FastLevenbergMarquardt, LeastSquaresOptim, MINPACK

true_function(x, θ) = @. θ[1] * exp(θ[2] * x) * cos(θ[3] * x + θ[4])
true_function(y, x, θ) = (@. y = θ[1] * exp(θ[2] * x) * cos(θ[3] * x + θ[4]))

θ_true = [1.0, 0.1, 2.0, 0.5]

x = [-1.0, -0.5, 0.0, 0.5, 1.0]

const y_target = true_function(x, θ_true)

function loss_function(θ, p)
    ŷ = true_function(p, θ)
    return ŷ .- y_target
end

function loss_function(resid, θ, p)
    true_function(resid, p, θ)
    resid .= resid .- y_target
    return resid
end

θ_init = θ_true .+ randn!(StableRNG(0), similar(θ_true)) * 0.1

export loss_function, θ_init, y_target, true_function, x, θ_true
end

@testitem "LeastSquaresOptim.jl" setup=[WrapperNLLSSetup] begin
    prob_oop = NonlinearLeastSquaresProblem{false}(loss_function, θ_init, x)
    prob_iip = NonlinearLeastSquaresProblem(
        NonlinearFunction(loss_function; resid_prototype = zero(y_target)), θ_init, x)

    nlls_problems = [prob_oop, prob_iip]

    solvers = [LeastSquaresOptimJL(alg; autodiff)
               for alg in (:lm, :dogleg),
    autodiff in (nothing, AutoForwardDiff(), AutoFiniteDiff(), :central, :forward)]

    for prob in nlls_problems, solver in solvers
        sol = solve(prob, solver; maxiters = 10000, abstol = 1e-8)
        @test SciMLBase.successful_retcode(sol)
        @test norm(sol.resid, Inf) < 1e-6
    end
end

@testitem "FastLevenbergMarquardt.jl + CMINPACK: Jacobian Provided" setup=[WrapperNLLSSetup] begin
    function jac!(J, θ, p)
        resid = zeros(length(p))
        ForwardDiff.jacobian!(J, (resid, θ) -> loss_function(resid, θ, p), resid, θ)
        return J
    end

    jac(θ, p) = ForwardDiff.jacobian(θ -> loss_function(θ, p), θ)

    probs = [
        NonlinearLeastSquaresProblem(
            NonlinearFunction{true}(
                loss_function; resid_prototype = zero(y_target), jac = jac!),
            θ_init,
            x),
        NonlinearLeastSquaresProblem(
            NonlinearFunction{false}(
                loss_function; resid_prototype = zero(y_target), jac = jac),
            θ_init,
            x),
        NonlinearLeastSquaresProblem(
            NonlinearFunction{false}(loss_function; jac), θ_init, x)]

    solvers = Any[FastLevenbergMarquardtJL(linsolve) for linsolve in (:cholesky, :qr)]
    push!(solvers, CMINPACK())
    for solver in solvers, prob in probs
        sol = solve(prob, solver; maxiters = 10000, abstol = 1e-8)
        @test maximum(abs, sol.resid) < 1e-6
    end
end

@testitem "FastLevenbergMarquardt.jl + CMINPACK: Jacobian Not Provided" setup=[WrapperNLLSSetup] begin
    probs = [
        NonlinearLeastSquaresProblem(
            NonlinearFunction{true}(loss_function; resid_prototype = zero(y_target)),
            θ_init, x),
        NonlinearLeastSquaresProblem(
            NonlinearFunction{false}(loss_function; resid_prototype = zero(y_target)),
            θ_init, x),
        NonlinearLeastSquaresProblem(NonlinearFunction{false}(loss_function), θ_init, x)]

    solvers = vec(Any[FastLevenbergMarquardtJL(linsolve; autodiff)
                      for linsolve in (:cholesky, :qr),
    autodiff in (nothing, AutoForwardDiff(), AutoFiniteDiff())])
    append!(solvers, [CMINPACK(; method) for method in (:auto, :lm, :lmdif)])

    for solver in solvers, prob in probs
        sol = solve(prob, solver; maxiters = 10000, abstol = 1e-8)
        @test norm(sol.resid, Inf) < 1e-6
    end
end

@testitem "FastLevenbergMarquardt.jl + StaticArrays" setup=[WrapperNLLSSetup] begin
    x_sa = SA[-1.0, -0.5, 0.0, 0.5, 1.0]

    const y_target_sa = true_function(x_sa, θ_true)

    function loss_function_sa(θ, p)
        ŷ = true_function(p, θ)
        return ŷ .- y_target_sa
    end

    θ_init_sa = SVector{4}(θ_init)
    prob_sa = NonlinearLeastSquaresProblem{false}(loss_function_sa, θ_init_sa, x)

    sol = solve(prob_sa, FastLevenbergMarquardtJL())
    @test norm(sol.resid, Inf) < 1e-6
end
