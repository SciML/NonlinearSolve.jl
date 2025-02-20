#runtests
using ImplicitDiscreteSolve
using OrdinaryDiffEqCore
using OrdinaryDiffEqSDIRK
using SimpleNonlinearSolve

# Test implicit Euler using ImplicitDiscreteProblem
@testset "Implicit Discrete System" begin
    function lotkavolterra(u, p, t) 
        [1.5*u[1] - u[1]*u[2], -3.0*u[2] + u[1]*u[2]]
    end

    function f!(resid, u_next, u, p, t)
        @. resid = u_next - 0.01*lotkavolterra(u_next, p, t)
    end
    u0 = [1., 1.]
    tspan = (0., 1.)

    idprob = ImplicitDiscreteProblem(f!, u0, tspan, []; dt = 0.01)
    idsol = solve(idprob, IDSolve(SimpleNewtonRaphson()))

    oprob = ODEProblem(lotkavolterra, u0, tspan)
    osol = solve(oprob, ImplicitEuler())

    @test idsol[end] ≈ osol[end]
end
