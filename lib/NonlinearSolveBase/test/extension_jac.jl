using NonlinearSolveBase
using SciMLBase
using ADTypes
using ForwardDiff
using LinearAlgebra
using Test

# f(u, p) = u.^2 .- p has Jacobian 2u (elementwise), so a diagonal of 2 .* vec(u).
f_oop(u, p) = @. u^2 - p
f_iip(du, u, p) = (du .= u .^ 2 .- p; du)

function eval_extension_jac(prob, u0, resid; autodiff = AutoForwardDiff(), kwargs...)
    J! = NonlinearSolveBase.construct_extension_jac(
        prob, nothing, u0, resid; autodiff, kwargs...
    )
    uvec = u0 isa Number ? [u0] : vec(copy(u0))
    Jout = zeros(eltype(uvec), length(resid), length(uvec))
    J!(Jout, uvec)
    return Jout
end

@testset "construct_extension_jac scalar and matrix u0" begin
    @testset "$ad" for ad in (AutoForwardDiff(),)
        @testset "scalar oop" begin
            prob = NonlinearProblem(f_oop, 2.0, 4.0)
            _, u0, resid = NonlinearSolveBase.construct_extension_function_wrapper(prob)
            J = eval_extension_jac(prob, u0, resid; autodiff = ad)
            @test J ≈ fill(4.0, 1, 1)

            J_scalar = NonlinearSolveBase.construct_extension_jac(
                prob, nothing, 2.0, 0.0; autodiff = ad, can_handle_scalar = Val(true),
                can_handle_oop = Val(true)
            )
            @test J_scalar(2.0) ≈ 4.0
        end

        @testset "2×2 oop" begin
            prob = NonlinearProblem(f_oop, ones(2, 2), [2.0 1.0; 3.0 4.0])
            _, u0, resid = NonlinearSolveBase.construct_extension_function_wrapper(prob)
            J = eval_extension_jac(prob, u0, resid; autodiff = ad)
            @test J ≈ 2 * I(4)
        end

        @testset "vector oop" begin
            prob = NonlinearProblem(f_oop, ones(2), [2.0, 1.0])
            _, u0, resid = NonlinearSolveBase.construct_extension_function_wrapper(prob)
            J = eval_extension_jac(prob, u0, resid; autodiff = ad)
            @test J ≈ 2 * I(2)
        end

        @testset "2×2 iip" begin
            prob = NonlinearProblem(f_iip, ones(2, 2), [2.0 1.0; 3.0 4.0])
            _, u0, resid = NonlinearSolveBase.construct_extension_function_wrapper(prob)
            J = eval_extension_jac(prob, u0, resid; autodiff = ad)
            @test J ≈ 2 * I(4)
        end

        @testset "analytic jac scalar" begin
            f = NonlinearFunction{false}(f_oop; jac = (u, p) -> 2u)
            prob = NonlinearProblem(f, 2.0, 4.0)
            _, u0, resid = NonlinearSolveBase.construct_extension_function_wrapper(prob)
            J = eval_extension_jac(prob, u0, resid; autodiff = ad)
            @test J ≈ fill(4.0, 1, 1)
        end

        @testset "initial Jacobian uses supplied u0, not prob.u0" begin
            prob = NonlinearProblem((u, p) -> u .^ 2, [1.0, 2.0])
            _, J0 = NonlinearSolveBase.construct_extension_jac(
                prob, nothing, [3.0, 4.0], [9.0, 16.0];
                autodiff = ad, initial_jacobian = Val(true)
            )
            @test J0 ≈ Diagonal([6.0, 8.0])

            # Stored guess is out of domain; the supplied state must be used instead.
            prob_domain = NonlinearProblem((u, p) -> sqrt.(u), [-1.0, -4.0])
            J! = NonlinearSolveBase.construct_extension_jac(
                prob_domain, nothing, [1.0, 4.0], [1.0, 2.0]; autodiff = ad
            )
            Jout = zeros(2, 2)
            J!(Jout, [1.0, 4.0])
            @test Jout ≈ [0.5 0.0; 0.0 0.25]
        end
    end
end
