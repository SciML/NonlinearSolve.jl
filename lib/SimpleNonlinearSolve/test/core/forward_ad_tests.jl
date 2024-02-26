@testsetup module ForwardADRootfindingTesting
using Reexport
@reexport using ForwardDiff, SimpleNonlinearSolve, StaticArrays, LinearAlgebra
import SimpleNonlinearSolve: AbstractSimpleNonlinearSolveAlgorithm

test_f!(du, u, p) = (@. du = u^2 - p)
test_f(u, p) = (@. u^2 - p)

jacobian_f(::Number, p) = 1 / (2 * √p)
jacobian_f(::Number, p::Number) = 1 / (2 * √p)
jacobian_f(u, p::Number) = one.(u) .* (1 / (2 * √p))
jacobian_f(u, p::AbstractArray) = diagm(vec(@. 1 / (2 * √p)))

function solve_with(::Val{mode}, u, alg) where {mode}
    f = if mode === :iip
        solve_iip(p) = solve(NonlinearProblem(test_f!, u, p), alg).u
    elseif mode === :oop
        solve_oop(p) = solve(NonlinearProblem(test_f, u, p), alg).u
    end
    return f
end

__compatible(::Any, ::Val{:oop}) = true
__compatible(::Number, ::Val{:iip}) = false
__compatible(::AbstractArray, ::Val{:iip}) = true
__compatible(::StaticArray, ::Val{:iip}) = false

__compatible(::Any, ::Number) = true
__compatible(::Number, ::AbstractArray) = false
__compatible(u::AbstractArray, p::AbstractArray) = size(u) == size(p)

__compatible(u::Number, ::AbstractSimpleNonlinearSolveAlgorithm) = true
__compatible(u::AbstractArray, ::AbstractSimpleNonlinearSolveAlgorithm) = true
__compatible(u::StaticArray, ::AbstractSimpleNonlinearSolveAlgorithm) = true

__compatible(::AbstractSimpleNonlinearSolveAlgorithm, ::Val{:iip}) = true
__compatible(::AbstractSimpleNonlinearSolveAlgorithm, ::Val{:oop}) = true
__compatible(::SimpleHalley, ::Val{:iip}) = false

export test_f, test_f!, jacobian_f, solve_with, __compatible
end

@testitem "ForwardDiff.jl Integration: Rootfinding" setup=[ForwardADRootfindingTesting] begin
    @testset "$(nameof(typeof(alg)))" for alg in (SimpleNewtonRaphson(),
        SimpleTrustRegion(), SimpleTrustRegion(; nlsolve_update_rule = Val(true)),
        SimpleHalley(), SimpleBroyden(), SimpleKlement(), SimpleDFSane())
        us = (2.0, @SVector[1.0, 1.0], [1.0, 1.0], ones(2, 2), @SArray ones(2, 2))

        @testset "Scalar AD" begin
            for p in 1.0:0.1:100.0, u0 in us, mode in (:iip, :oop)
                __compatible(u0, alg) || continue
                __compatible(u0, Val(mode)) || continue
                __compatible(alg, Val(mode)) || continue

                sol = solve(NonlinearProblem(test_f, u0, p), alg)
                if SciMLBase.successful_retcode(sol)
                    gs = abs.(ForwardDiff.derivative(solve_with(Val{mode}(), u0, alg), p))
                    gs_true = abs.(jacobian_f(u0, p))
                    if !(isapprox(gs, gs_true, atol = 1e-5))
                        @show sol.retcode, sol.u
                        @error "ForwardDiff Failed for u0=$(u0) and p=$(p) with $(alg)" forwardiff_gradient=gs true_gradient=gs_true
                    else
                        @test abs.(gs)≈abs.(gs_true) atol=1e-5
                    end
                end
            end
        end

        @testset "Jacobian" begin
            for u0 in us, p in ([2.0, 1.0], [2.0 1.0; 3.0 4.0]), mode in (:iip, :oop)
                __compatible(u0, p) || continue
                __compatible(u0, alg) || continue
                __compatible(u0, Val(mode)) || continue
                __compatible(alg, Val(mode)) || continue

                sol = solve(NonlinearProblem(test_f, u0, p), alg)
                if SciMLBase.successful_retcode(sol)
                    gs = abs.(ForwardDiff.jacobian(solve_with(Val{mode}(), u0, alg), p))
                    gs_true = abs.(jacobian_f(u0, p))
                    if !(isapprox(gs, gs_true, atol = 1e-5))
                        @show sol.retcode, sol.u
                        @error "ForwardDiff Failed for u0=$(u0) and p=$(p) with $(alg)" forwardiff_jacobian=gs true_jacobian=gs_true
                    else
                        @test abs.(gs)≈abs.(gs_true) atol=1e-5
                    end
                end
            end
        end
    end
end

@testsetup module ForwardADNLLSTesting
using Reexport
@reexport using ForwardDiff, FiniteDiff, SimpleNonlinearSolve, StaticArrays, LinearAlgebra,
                Zygote

true_function(x, θ) = @. θ[1] * exp(θ[2] * x) * cos(θ[3] * x + θ[4])

const θ_true = [1.0, 0.1, 2.0, 0.5]
const x = [-1.0, -0.5, 0.0, 0.5, 1.0]
const y_target = true_function(x, θ_true)

function loss_function(θ, p)
    ŷ = true_function(p, θ)
    return ŷ .- y_target
end

function loss_function_jac(θ, p)
    return ForwardDiff.jacobian(θ -> loss_function(θ, p), θ)
end

loss_function_vjp(v, θ, p) = reshape(vec(v)' * loss_function_jac(θ, p), size(θ))

function loss_function!(resid, θ, p)
    ŷ = true_function(p, θ)
    @. resid = ŷ - y_target
    return
end

function loss_function_jac!(J, θ, p)
    J .= ForwardDiff.jacobian(θ -> loss_function(θ, p), θ)
    return
end

function loss_function_vjp!(vJ, v, θ, p)
    vec(vJ) .= reshape(vec(v)' * loss_function_jac(θ, p), size(θ))
    return
end

θ_init = θ_true .+ 0.1

export loss_function, loss_function!, loss_function_jac, loss_function_vjp,
       loss_function_jac!, loss_function_vjp!, θ_init, x, y_target
end

@testitem "ForwardDiff.jl Integration: NLLS" setup=[ForwardADNLLSTesting] begin
    @testset "$(nameof(typeof(alg)))" for alg in (
        SimpleNewtonRaphson(), SimpleGaussNewton(),
        SimpleNewtonRaphson(AutoFiniteDiff()), SimpleGaussNewton(AutoFiniteDiff()))
        function obj_1(p)
            prob_oop = NonlinearLeastSquaresProblem{false}(loss_function, θ_init, p)
            sol = solve(prob_oop, alg)
            return sum(abs2, sol.u)
        end

        function obj_2(p)
            ff = NonlinearFunction{false}(loss_function; jac = loss_function_jac)
            prob_oop = NonlinearLeastSquaresProblem{false}(ff, θ_init, p)
            sol = solve(prob_oop, alg)
            return sum(abs2, sol.u)
        end

        function obj_3(p)
            ff = NonlinearFunction{false}(loss_function; vjp = loss_function_vjp)
            prob_oop = NonlinearLeastSquaresProblem{false}(ff, θ_init, p)
            sol = solve(prob_oop, alg)
            return sum(abs2, sol.u)
        end

        finitediff = FiniteDiff.finite_difference_gradient(obj_1, x)

        fdiff1 = ForwardDiff.gradient(obj_1, x)
        fdiff2 = ForwardDiff.gradient(obj_2, x)
        fdiff3 = ForwardDiff.gradient(obj_3, x)

        @test finitediff≈fdiff1 atol=1e-5
        @test finitediff≈fdiff2 atol=1e-5
        @test finitediff≈fdiff3 atol=1e-5
        @test fdiff1 ≈ fdiff2 ≈ fdiff3

        function obj_4(p)
            prob_iip = NonlinearLeastSquaresProblem(
                NonlinearFunction{true}(
                    loss_function!; resid_prototype = zeros(length(y_target))), θ_init, p)
            sol = solve(prob_iip, alg)
            return sum(abs2, sol.u)
        end

        function obj_5(p)
            ff = NonlinearFunction{true}(
                loss_function!; resid_prototype = zeros(length(y_target)), jac = loss_function_jac!)
            prob_iip = NonlinearLeastSquaresProblem(
                ff, θ_init, p)
            sol = solve(prob_iip, alg)
            return sum(abs2, sol.u)
        end

        function obj_6(p)
            ff = NonlinearFunction{true}(
                loss_function!; resid_prototype = zeros(length(y_target)), vjp = loss_function_vjp!)
            prob_iip = NonlinearLeastSquaresProblem(
                ff, θ_init, p)
            sol = solve(prob_iip, alg)
            return sum(abs2, sol.u)
        end

        finitediff = FiniteDiff.finite_difference_gradient(obj_4, x)

        fdiff4 = ForwardDiff.gradient(obj_4, x)
        fdiff5 = ForwardDiff.gradient(obj_5, x)
        fdiff6 = ForwardDiff.gradient(obj_6, x)

        @test finitediff≈fdiff4 atol=1e-5
        @test finitediff≈fdiff5 atol=1e-5
        @test finitediff≈fdiff6 atol=1e-5
        @test fdiff4 ≈ fdiff5 ≈ fdiff6
    end
end
