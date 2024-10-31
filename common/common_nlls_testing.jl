using NonlinearSolveBase, SciMLBase, StableRNGs, ForwardDiff, Random, LinearAlgebra

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

prob_oop = NonlinearLeastSquaresProblem{false}(loss_function, θ_init, x)
prob_iip = NonlinearLeastSquaresProblem{true}(
    NonlinearFunction(loss_function; resid_prototype = zero(y_target)), θ_init, x
)
prob_oop_vjp = NonlinearLeastSquaresProblem(
    NonlinearFunction{false}(loss_function; vjp), θ_init, x
)
prob_iip_vjp = NonlinearLeastSquaresProblem(
    NonlinearFunction{true}(loss_function; resid_prototype = zero(y_target), vjp = vjp!),
    θ_init, x
)

export prob_oop, prob_iip, prob_oop_vjp, prob_iip_vjp
export true_function, θ_true, x, y_target, loss_function, θ_init
