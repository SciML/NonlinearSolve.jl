using Test, ForwardDiff
include("bounded_problems.jl")
include("bounded_validation_problems.jl")

@testset "Benchmark analytic Jacobians" begin
    for case in vcat(bounded_benchmark_cases(), bounded_validation_cases())
        prob = case.prob
        f = function (u)
            r = prob.f.resid_prototype === nothing ? similar(u) : similar(u, length(prob.f.resid_prototype))
            prob.f(r, u, prob.p)
            return r
        end
        expected = ForwardDiff.jacobian(f, prob.u0)
        actual = zero(expected)
        case.jacobian(actual, prob.u0, prob.p)
        @test all(isapprox.(actual, expected; atol = 1.0e-10, rtol = 1.0e-10))
    end
end
