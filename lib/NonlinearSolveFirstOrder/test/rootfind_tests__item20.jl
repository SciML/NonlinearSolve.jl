using NonlinearSolveFirstOrder
include("setup_corerootfindtesting.jl")

using LinearAlgebra, LinearSolve, ADTypes

function F(u::Vector{Float64}, p::Vector{Float64})
    Δ = Tridiagonal(-ones(99), 2 * ones(100), -ones(99))
    return u + 0.1 * u .* Δ * u - p
end

function F!(du::Vector{Float64}, u::Vector{Float64}, p::Vector{Float64})
    Δ = Tridiagonal(-ones(99), 2 * ones(100), -ones(99))
    du .= u + 0.1 * u .* Δ * u - p
    return nothing
end

function JVP(v::Vector{Float64}, u::Vector{Float64}, p::Vector{Float64})
    Δ = Tridiagonal(-ones(99), 2 * ones(100), -ones(99))
    return v + 0.1 * (u .* Δ * v + v .* Δ * u)
end

function JVP!(
        du::Vector{Float64}, v::Vector{Float64}, u::Vector{Float64}, p::Vector{Float64}
    )
    Δ = Tridiagonal(-ones(99), 2 * ones(100), -ones(99))
    du .= v + 0.1 * (u .* Δ * v + v .* Δ * u)
    return nothing
end

u0 = rand(100)

prob = NonlinearProblem(NonlinearFunction{false}(F; jvp = JVP), u0, u0)
sol = solve(prob, NewtonRaphson(; linsolve = KrylovJL_GMRES()); abstol = 1.0e-13)
err = maximum(abs, sol.resid)
@test err < 1.0e-6

sol = solve(
    prob, TrustRegion(; linsolve = KrylovJL_GMRES(), vjp_autodiff = AutoFiniteDiff());
    abstol = 1.0e-13
)
err = maximum(abs, sol.resid)
@test err < 1.0e-6

prob = NonlinearProblem(NonlinearFunction{true}(F!; jvp = JVP!), u0, u0)
sol = solve(prob, NewtonRaphson(; linsolve = KrylovJL_GMRES()); abstol = 1.0e-13)
err = maximum(abs, sol.resid)
@test err < 1.0e-6

sol = solve(
    prob, TrustRegion(; linsolve = KrylovJL_GMRES(), vjp_autodiff = AutoFiniteDiff());
    abstol = 1.0e-13
)
err = maximum(abs, sol.resid)
@test err < 1.0e-6
