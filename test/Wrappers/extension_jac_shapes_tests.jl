using NonlinearSolve
using ADTypes
using Test

import MINPACK, NLsolve

f(u, p) = @. u^2 - p

scalar = NonlinearProblem(f, 2.0, 4.0)
matrix = NonlinearProblem(f, ones(2, 2), [2.0 1.0; 3.0 4.0])
vector = NonlinearProblem(f, ones(2), [2.0, 1.0])

ads = (
    AutoFiniteDiff(; fdtype = Val(:central)),
    AutoForwardDiff(),
)

@testset "CMINPACK explicit ADTypes backend" begin
    Sys.isapple() && return
    @testset "$ad $label" for ad in ads,
            (label, prob, expected) in (
                ("scalar", scalar, 2.0),
                ("2x2", matrix, [sqrt(2) 1.0; sqrt(3) 2.0]),
                ("vector", vector, [sqrt(2), 1.0]),
            )
        sol = solve(prob, CMINPACK(autodiff = ad))
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ expected atol = 1.0e-8
        @test maximum(abs, sol.resid) < 1.0e-10
    end
end

# NLsolveJL short-circuits `construct_extension_jac` unless a Jacobian is supplied.
nlsolve_scalar = NonlinearProblem(
    NonlinearFunction{false}(f; jac_prototype = zeros(1, 1)), 2.0, 4.0
)
nlsolve_matrix = NonlinearProblem(
    NonlinearFunction{false}(f; jac_prototype = zeros(4, 4)),
    ones(2, 2), [2.0 1.0; 3.0 4.0]
)
nlsolve_vector = NonlinearProblem(
    NonlinearFunction{false}(f; jac_prototype = zeros(2, 2)), ones(2), [2.0, 1.0]
)

@testset "NLsolveJL explicit ADTypes backend with jac_prototype" begin
    @testset "$ad $label" for ad in ads,
            (label, prob, expected) in (
                ("scalar", nlsolve_scalar, 2.0),
                ("2x2", nlsolve_matrix, [sqrt(2) 1.0; sqrt(3) 2.0]),
                ("vector", nlsolve_vector, [sqrt(2), 1.0]),
            )
        sol = solve(prob, NLsolveJL(autodiff = ad))
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ expected atol = 1.0e-8
        @test maximum(abs, sol.resid) < 1.0e-10
    end
end
