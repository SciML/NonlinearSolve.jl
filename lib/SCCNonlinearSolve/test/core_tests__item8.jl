using SCCNonlinearSolve
include("setup_corerootfindtesting.jl")

using NonlinearSolveFirstOrder

function f1(du, u, p)
    du[1] = cos(u[2]) - u[1]
    return du[2] = sin(u[1] + u[2]) + u[2]
end
explicitfun1(p, sols) = nothing
prob1 = NonlinearProblem(f1, zeros(2))
sccprob = SciMLBase.SCCNonlinearProblem((prob1,), (explicitfun1,))

@testset "init is not supported for SCCNonlinearProblem" begin
    @test_throws ArgumentError init(sccprob)
    @test_throws ArgumentError init(sccprob, SCCNonlinearSolve.SCCAlg())
    @test_throws ArgumentError init(sccprob, NewtonRaphson())
end
