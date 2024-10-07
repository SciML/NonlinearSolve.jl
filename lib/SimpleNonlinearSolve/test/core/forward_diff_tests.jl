@testitem "ForwardDiff.jl Integration NonlinearLeastSquaresProblem" tags=[:core] begin
    using ForwardDiff, FiniteDiff, SimpleNonlinearSolve, StaticArrays, LinearAlgebra,
          Zygote, ReverseDiff
    using DifferentiationInterface

    const DI = DifferentiationInterface

    true_function(x, θ) = @. θ[1] * exp(θ[2] * x) * cos(θ[3] * x + θ[4])

    θ_true = [1.0, 0.1, 2.0, 0.5]
    x = [-1.0, -0.5, 0.0, 0.5, 1.0]
    y_target = true_function(x, θ_true)

    loss_function(θ, p) = true_function(p, θ) .- y_target

    loss_function_jac(θ, p) = ForwardDiff.jacobian(Base.Fix2(loss_function, p), θ)

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

    @testset for alg in (
        SimpleGaussNewton(),
        SimpleGaussNewton(; autodiff = AutoForwardDiff()),
        SimpleGaussNewton(; autodiff = AutoFiniteDiff()),
        SimpleGaussNewton(; autodiff = AutoReverseDiff())
    )
        function obj_1(p)
            prob_oop = NonlinearLeastSquaresProblem{false}(loss_function, θ_init, p)
            sol = solve(prob_oop, alg)
            return sum(abs2, sol.u)
        end

        function obj_2(p)
            ff = NonlinearFunction{false}(
                loss_function; resid_prototype = zeros(length(y_target)))
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

        finitediff = DI.gradient(obj_1, AutoFiniteDiff(), x)

        fdiff1 = DI.gradient(obj_1, AutoForwardDiff(), x)
        fdiff2 = DI.gradient(obj_2, AutoForwardDiff(), x)
        fdiff3 = DI.gradient(obj_3, AutoForwardDiff(), x)

        @test finitediff≈fdiff1 atol=1e-5
        @test finitediff≈fdiff2 atol=1e-5
        @test finitediff≈fdiff3 atol=1e-5
        @test fdiff1 ≈ fdiff2 ≈ fdiff3

        function obj_4(p)
            prob_iip = NonlinearLeastSquaresProblem(
                NonlinearFunction{true}(
                    loss_function!; resid_prototype = zeros(length(y_target))),
                θ_init,
                p)
            sol = solve(prob_iip, alg)
            return sum(abs2, sol.u)
        end

        function obj_5(p)
            ff = NonlinearFunction{true}(
                loss_function!; resid_prototype = zeros(length(y_target)),
                jac = loss_function_jac!)
            prob_iip = NonlinearLeastSquaresProblem(ff, θ_init, p)
            sol = solve(prob_iip, alg)
            return sum(abs2, sol.u)
        end

        function obj_6(p)
            ff = NonlinearFunction{true}(
                loss_function!; resid_prototype = zeros(length(y_target)),
                vjp = loss_function_vjp!)
            prob_iip = NonlinearLeastSquaresProblem(ff, θ_init, p)
            sol = solve(prob_iip, alg)
            return sum(abs2, sol.u)
        end

        finitediff = DI.gradient(obj_4, AutoFiniteDiff(), x)

        fdiff4 = DI.gradient(obj_4, AutoForwardDiff(), x)
        fdiff5 = DI.gradient(obj_5, AutoForwardDiff(), x)
        fdiff6 = DI.gradient(obj_6, AutoForwardDiff(), x)

        @test finitediff≈fdiff4 atol=1e-5
        @test finitediff≈fdiff5 atol=1e-5
        @test finitediff≈fdiff6 atol=1e-5
        @test fdiff4 ≈ fdiff5 ≈ fdiff6
    end
end
