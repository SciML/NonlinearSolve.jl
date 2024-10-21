@testsetup module CoreNLLSTesting
using Reexport
@reexport using NonlinearSolve, LinearSolve, LinearAlgebra, StableRNGs, Random, ForwardDiff,
                Zygote
using LineSearches: LineSearches, Static, HagerZhang, MoreThuente, StrongWolfe

linesearches = []
for ls in (Static(), HagerZhang(), MoreThuente(), StrongWolfe(), LineSearches.BackTracking())
    push!(linesearches, LineSearchesJL(; method = ls))
end
push!(linesearches, BackTracking())

true_function(x, θ) = @. θ[1] * exp(θ[2] * x) * cos(θ[3] * x + θ[4])
true_function(y, x, θ) = (@. y = θ[1] * exp(θ[2] * x) * cos(θ[3] * x + θ[4]))

const θ_true = [1.0, 0.1, 2.0, 0.5]

const x = [-1.0, -0.5, 0.0, 0.5, 1.0]

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

const θ_init = θ_true .+ randn!(StableRNG(0), similar(θ_true)) * 0.1

solvers = []
for linsolve in [nothing, LUFactorization(), KrylovJL_GMRES(), KrylovJL_LSMR()]
    vjp_autodiffs = linsolve isa KrylovJL ? [nothing, AutoZygote(), AutoFiniteDiff()] :
                    [nothing]
    for linesearch in linesearches, vjp_autodiff in vjp_autodiffs

        push!(solvers, GaussNewton(; linsolve, linesearch, vjp_autodiff))
    end
end
append!(solvers,
    [LevenbergMarquardt(), LevenbergMarquardt(; linsolve = LUFactorization()),
        LevenbergMarquardt(; linsolve = KrylovJL_GMRES()),
        LevenbergMarquardt(; linsolve = KrylovJL_LSMR()), nothing])
for radius_update_scheme in [RadiusUpdateSchemes.Simple, RadiusUpdateSchemes.NocedalWright,
    RadiusUpdateSchemes.NLsolve, RadiusUpdateSchemes.Hei,
    RadiusUpdateSchemes.Yuan, RadiusUpdateSchemes.Fan, RadiusUpdateSchemes.Bastin]
    push!(solvers, TrustRegion(; radius_update_scheme))
end

export solvers, θ_init, x, y_target, true_function, θ_true, loss_function
end

@testitem "General NLLS Solvers" setup=[CoreNLLSTesting] tags=[:core] begin
    prob_oop = NonlinearLeastSquaresProblem{false}(loss_function, θ_init, x)
    prob_iip = NonlinearLeastSquaresProblem(
        NonlinearFunction(loss_function; resid_prototype = zero(y_target)), θ_init, x)

    nlls_problems = [prob_oop, prob_iip]

    for prob in nlls_problems, solver in solvers
        sol = solve(prob, solver; maxiters = 10000, abstol = 1e-6)
        @test SciMLBase.successful_retcode(sol)
        @test norm(sol.resid, 2) < 1e-6
    end
end

@testitem "Custom VJP" setup=[CoreNLLSTesting] tags=[:core] begin
    # This is just for testing that we can use vjp provided by the user
    function vjp(v, θ, p)
        resid = zeros(length(p))
        J = ForwardDiff.jacobian((resid, θ) -> loss_function(resid, θ, p), resid, θ)
        return vec(v' * J)
    end

    function vjp!(Jv, v, θ, p)
        resid = zeros(length(p))
        J = ForwardDiff.jacobian((resid, θ) -> loss_function(resid, θ, p), resid, θ)
        mul!(vec(Jv), transpose(J), v)
        return nothing
    end

    probs = [
        NonlinearLeastSquaresProblem(
            NonlinearFunction{true}(
                loss_function; resid_prototype = zero(y_target), vjp = vjp!),
            θ_init,
            x),
        NonlinearLeastSquaresProblem(
            NonlinearFunction{false}(
                loss_function; resid_prototype = zero(y_target), vjp = vjp),
            θ_init,
            x)]

    for prob in probs, solver in solvers
        sol = solve(prob, solver; maxiters = 10000, abstol = 1e-6)
        @test SciMLBase.successful_retcode(sol)
        @test norm(sol.resid, 2) < 1e-6
    end
end

@testitem "NLLS Analytic Jacobian" tags=[:core] begin
    dataIn = 1:10
    f(x, p) = x[1] * dataIn .^ 2 .+ x[2] * dataIn .+ x[3]
    dataOut = f([1, 2, 3], nothing) + 0.1 * randn(10, 1)

    resid(x, p) = f(x, p) - dataOut
    jac(x, p) = [dataIn .^ 2 dataIn ones(10, 1)]
    x0 = [1, 1, 1]

    prob = NonlinearLeastSquaresProblem(resid, x0)
    sol1 = solve(prob)

    nlfunc = NonlinearFunction(resid; jac)
    prob = NonlinearLeastSquaresProblem(nlfunc, x0)
    sol2 = solve(prob)

    @test sol1.u ≈ sol2.u
end
