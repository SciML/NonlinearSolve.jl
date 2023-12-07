using NonlinearSolve,
    LinearSolve, LinearAlgebra, Test, StableRNGs, Random, ForwardDiff, Zygote
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

θ_init = θ_true .+ randn!(StableRNG(0), similar(θ_true)) * 0.1
prob_oop = NonlinearLeastSquaresProblem{false}(loss_function, θ_init, x)
prob_iip = NonlinearLeastSquaresProblem(NonlinearFunction(loss_function;
        resid_prototype = zero(y_target)), θ_init, x)

nlls_problems = [prob_oop, prob_iip]
solvers = []
for linsolve in [nothing, LUFactorization(), KrylovJL_GMRES()]
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
        LeastSquaresOptimJL(:lm),
        LeastSquaresOptimJL(:dogleg),
        nothing,
    ])

for prob in nlls_problems, solver in solvers
    @time sol = solve(prob, solver; maxiters = 10000, abstol = 1e-8)
    @test SciMLBase.successful_retcode(sol)
    @test norm(sol.resid) < 1e-6
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
    mul!(vec(Jv), v', J)
    return nothing
end

probs = [
    NonlinearLeastSquaresProblem(NonlinearFunction{true}(loss_function;
            resid_prototype = zero(y_target), vjp = vjp!), θ_init, x),
    NonlinearLeastSquaresProblem(NonlinearFunction{false}(loss_function;
            resid_prototype = zero(y_target), vjp = vjp), θ_init, x),
]

for prob in probs, solver in solvers
    !(solver isa GaussNewton) && continue
    !(solver.linsolve isa KrylovJL) && continue
    @test_warn "Currently we don't make use of user provided `jvp`. This is planned to be \
    fixed in the near future." sol=solve(prob, solver; maxiters = 10000, abstol = 1e-8)
    sol = solve(prob, solver; maxiters = 10000, abstol = 1e-8)
    @test norm(sol.resid) < 1e-6
end

function jac!(J, θ, p)
    resid = zeros(length(p))
    ForwardDiff.jacobian!(J, (resid, θ) -> loss_function(resid, θ, p), resid, θ)
    return J
end

jac(θ, p) = ForwardDiff.jacobian(θ -> loss_function(θ, p), θ)

probs = [
    NonlinearLeastSquaresProblem(NonlinearFunction{true}(loss_function;
            resid_prototype = zero(y_target), jac = jac!), θ_init, x),
    NonlinearLeastSquaresProblem(NonlinearFunction{false}(loss_function;
            resid_prototype = zero(y_target), jac = jac), θ_init, x),
    NonlinearLeastSquaresProblem(NonlinearFunction{false}(loss_function; jac), θ_init, x),
]

solvers = [FastLevenbergMarquardtJL(linsolve) for linsolve in (:cholesky, :qr)]

for solver in solvers, prob in probs
    @time sol = solve(prob, solver; maxiters = 10000, abstol = 1e-8)
    @test norm(sol.resid) < 1e-6
end

probs = [
    NonlinearLeastSquaresProblem(NonlinearFunction{true}(loss_function;
            resid_prototype = zero(y_target)), θ_init, x),
    NonlinearLeastSquaresProblem(NonlinearFunction{false}(loss_function;
            resid_prototype = zero(y_target)), θ_init, x),
    NonlinearLeastSquaresProblem(NonlinearFunction{false}(loss_function), θ_init, x),
]

solvers = [FastLevenbergMarquardtJL(linsolve; autodiff) for linsolve in (:cholesky, :qr),
autodiff in (nothing, AutoForwardDiff(), AutoFiniteDiff())]

for solver in solvers, prob in probs
    @time sol = solve(prob, solver; maxiters = 10000, abstol = 1e-8)
    @test norm(sol.resid) < 1e-6
end
