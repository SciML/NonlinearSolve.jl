using NonlinearSolve,
    LinearSolve, LinearAlgebra, Test, StableRNGs, Random, ForwardDiff, Zygote

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

θ_init = θ_true .+ randn!(StableRNG(0), similar(θ_true)) * 0.1
prob_oop = NonlinearLeastSquaresProblem{false}(loss_function, θ_init, x)
prob_iip = NonlinearLeastSquaresProblem(NonlinearFunction(loss_function;
        resid_prototype = zero(y_target)), θ_init, x)

nlls_problems = [prob_oop, prob_iip]

solvers = []
for linsolve in [nothing, LUFactorization(), KrylovJL_GMRES(), KrylovJL_LSMR()]
    vjp_autodiffs = linsolve isa KrylovJL ? [nothing, AutoZygote(), AutoFiniteDiff()] :
                    [nothing]
    for linesearch in [Static(), BackTracking(), HagerZhang(), StrongWolfe(), MoreThuente()],
        vjp_autodiff in vjp_autodiffs

        push!(solvers, GaussNewton(; linsolve, linesearch, vjp_autodiff))
    end
end
append!(solvers,
    [
        LevenbergMarquardt(),
        LevenbergMarquardt(; linsolve = LUFactorization()),
        LevenbergMarquardt(; linsolve = KrylovJL_GMRES()),
        LevenbergMarquardt(; linsolve = KrylovJL_LSMR()),
        nothing,
    ])
for radius_update_scheme in [RadiusUpdateSchemes.Simple, RadiusUpdateSchemes.NocedalWright,
    RadiusUpdateSchemes.NLsolve, RadiusUpdateSchemes.Hei, RadiusUpdateSchemes.Yuan,
    RadiusUpdateSchemes.Fan, RadiusUpdateSchemes.Bastin]
    push!(solvers, TrustRegion(; radius_update_scheme))
end

for prob in nlls_problems, solver in solvers
    @time sol = solve(prob, solver; maxiters = 10000, abstol = 1e-8)
    @test SciMLBase.successful_retcode(sol)
    @test maximum(abs, sol.resid) < 1e-6
end

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
    NonlinearLeastSquaresProblem(NonlinearFunction{true}(loss_function;
            resid_prototype = zero(y_target), vjp = vjp!), θ_init, x),
    NonlinearLeastSquaresProblem(NonlinearFunction{false}(loss_function;
            resid_prototype = zero(y_target), vjp = vjp), θ_init, x),
]

for prob in probs, solver in solvers
    sol = solve(prob, solver; maxiters = 10000, abstol = 1e-8)
    @test maximum(abs, sol.resid) < 1e-6
end
