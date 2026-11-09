using ADTypes: AutoForwardDiff
using ForwardDiff
using NonlinearSolveBase
using SciMLBase
using Test

struct NotAJacobian end

# An `AbstractMatrix` with no `size`, so `convert(Matrix, ·)` has no working method.
struct UnsizedMatrix <: AbstractMatrix{Float64} end

function jacobian_cache(jac)
    f = NonlinearFunction{false}((u, p) -> u .^ 2 .- p; jac)
    u = [2.0, 3.0]
    prob = NonlinearProblem(f, u, 0.5)
    cache = NonlinearSolveBase.construct_jacobian_cache(
        prob, nothing, f, f(u, 0.5);
        stats = SciMLBase.NLStats(0, 0, 0, 0, 0), autodiff = AutoForwardDiff()
    )
    return cache, u
end

@testset "an out-of-place jac must return something convertible to the stored Jacobian" begin
    @testset "$(nameof(T))" for T in (NotAJacobian, UnsizedMatrix)
        cache, u = jacobian_cache((u, p) -> T())
        @test_throws r"cannot be converted.*AbstractSciMLOperator" cache(u)
        @test_throws ArgumentError cache(u)
    end

    cache, u = jacobian_cache((u, p) -> [2 * u[1] 0.0; 0.0 2 * u[2]])
    @test cache(u) == [4.0 0.0; 0.0 6.0]
end
