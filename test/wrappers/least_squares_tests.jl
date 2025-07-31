@testsetup module WrapperNLLSSetup

include("../../common/common_nlls_testing.jl")

end

@testitem "LeastSquaresOptim.jl" setup=[WrapperNLLSSetup] tags=[:wrappers] begin
    import LeastSquaresOptim

    nlls_problems=[prob_oop, prob_iip]

    solvers=[]
    for alg in (:lm, :dogleg),
        autodiff in (nothing, AutoForwardDiff(), AutoFiniteDiff(), :central, :forward)

        push!(solvers, LeastSquaresOptimJL(alg; autodiff))
    end

    for prob in nlls_problems, solver in solvers

        sol=solve(prob, solver; maxiters = 10000, abstol = 1e-8)
        @test SciMLBase.successful_retcode(sol)
        @test maximum(abs, sol.resid) < 1e-6
    end
end

@testitem "FastLevenbergMarquardt.jl + CMINPACK: Jacobian Provided" setup=[WrapperNLLSSetup] tags=[:wrappers] begin
    import FastLevenbergMarquardt, MINPACK
    using ForwardDiff

    function jac!(J, θ, p)
        resid=zeros(length(p))
        ForwardDiff.jacobian!(J, (resid, θ)->loss_function(resid, θ, p), resid, θ)
        return J
    end

    jac(θ, p)=ForwardDiff.jacobian(θ->loss_function(θ, p), θ)

    probs=[
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
        )
    ]

    solvers=Any[FastLevenbergMarquardtJL(linsolve) for linsolve in (:cholesky, :qr)]
    Sys.isapple()||push!(solvers, CMINPACK())

    for solver in solvers, prob in probs

        sol=solve(prob, solver; maxiters = 10000, abstol = 1e-8)
        @test maximum(abs, sol.resid) < 1e-6
    end
end

@testitem "FastLevenbergMarquardt.jl + CMINPACK: Jacobian Not Provided" setup=[WrapperNLLSSetup] tags=[:wrappers] begin
    import FastLevenbergMarquardt, MINPACK

    probs=[
        NonlinearLeastSquaresProblem(
            NonlinearFunction{true}(loss_function; resid_prototype = zero(y_target)),
            θ_init, x
        ),
        NonlinearLeastSquaresProblem(
            NonlinearFunction{false}(loss_function; resid_prototype = zero(y_target)),
            θ_init, x
        ),
        NonlinearLeastSquaresProblem(NonlinearFunction{false}(loss_function), θ_init, x)
    ]

    solvers=[]
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

        sol=solve(prob, solver; maxiters = 10000, abstol = 1e-8)
        @test maximum(abs, sol.resid) < 1e-6
    end
end

@testitem "FastLevenbergMarquardt.jl + StaticArrays" setup=[WrapperNLLSSetup] tags=[:wrappers] begin
    using StaticArrays, FastLevenbergMarquardt

    x_sa=SA[-1.0, -0.5, 0.0, 0.5, 1.0]

    const y_target_sa=true_function(x_sa, θ_true)

    function loss_function_sa(θ, p)
        ŷ=true_function(p, θ)
        return ŷ .- y_target_sa
    end

    θ_init_sa=SVector{4}(θ_init)
    prob_sa=NonlinearLeastSquaresProblem{false}(loss_function_sa, θ_init_sa, x)

    sol=solve(prob_sa, FastLevenbergMarquardtJL())
    @test maximum(abs, sol.resid) < 1e-6
end
