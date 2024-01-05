using LinearAlgebra, NonlinearSolve, Test

@testset "[IIP] no AD" begin
    f_iip = Base.Experimental.@opaque (du, u, p) -> du .= u .* u .- p
    u0 = [0.0]
    prob = NonlinearProblem(f_iip, u0, 1.0)
    for alg in [RobustMultiNewton(autodiff = AutoFiniteDiff()())]
        sol = solve(prob, alg)
        @test isapprox(only(sol.u), 1.0)
        @test SciMLBase.successful_retcode(sol.retcode)
    end
end

@testset "[OOP] no AD" begin
    f_oop = Base.Experimental.@opaque (u, p) -> u .* u .- p
    u0 = [0.0]
    prob = NonlinearProblem{false}(f_oop, u0, 1.0)
    for alg in [RobustMultiNewton(autodiff = AutoFiniteDiff())]
        sol = solve(prob, alg)
        @test isapprox(only(sol.u), 1.0)
        @test SciMLBase.successful_retcode(sol.retcode)
    end
end
