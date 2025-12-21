@testsetup module CoreRootfindTesting

include("../../../common/common_rootfind_testing.jl")

end

@testitem "Manual SCC" setup=[CoreRootfindTesting] tags=[:core] begin
    using NonlinearSolveFirstOrder
    function f(du, u, p)
        du[1]=cos(u[2])-u[1]
        du[2]=sin(u[1]+u[2])+u[2]
        du[3]=2u[4]+u[3]+1.0
        du[4]=u[5]^2+u[4]
        du[5]=u[3]^2+u[5]
        du[6]=u[1]+u[2]+u[3]+u[4]+u[5]+2.0u[6]+2.5u[7]+1.5u[8]
        du[7]=u[1]+u[2]+u[3]+2.0u[4]+u[5]+4.0u[6]-1.5u[7]+1.5u[8]
        du[8]=u[1]+2.0u[2]+3.0u[3]+5.0u[4]+6.0u[5]+u[6]-u[7]-u[8]
    end
    prob=NonlinearProblem(f, zeros(8))
    sol=solve(prob, NewtonRaphson())

    u0=zeros(2)
    p=zeros(3)

    function f1(du, u, p)
        du[1]=cos(u[2])-u[1]
        du[2]=sin(u[1]+u[2])+u[2]
    end
    explicitfun1(p, sols)=nothing
    prob1=NonlinearProblem(
        NonlinearFunction{true, SciMLBase.NoSpecialize}(f1), zeros(2), p)
    sol1=solve(prob1, NewtonRaphson())

    function f2(du, u, p)
        du[1]=2u[2]+u[1]+1.0
        du[2]=u[3]^2+u[2]
        du[3]=u[1]^2+u[3]
    end
    explicitfun2(p, sols)=nothing
    prob2=NonlinearProblem(
        NonlinearFunction{true, SciMLBase.NoSpecialize}(f2), zeros(3), p)
    sol2=solve(prob2, NewtonRaphson())

    # Convert f3 to a LinearProblem since it's linear in u
    # du = Au + b where A is the coefficient matrix and b is from parameters
    A3=[2.0 2.5 1.5; 4.0 -1.5 1.5; 1.0 -1.0 -1.0]
    b3=p  # b will be updated by explicitfun3
    prob3=LinearProblem(A3, b3, zeros(3))
    function explicitfun3(p, sols)
        p[1]=-(sols[1][1]+sols[1][2]+sols[2][1]+sols[2][2]+sols[2][3])
        p[2]=-(sols[1][1]+sols[1][2]+sols[2][1]+2.0sols[2][2]+sols[2][3])
        p[3]=-(sols[1][1]+2.0sols[1][2]+3.0sols[2][1]+5.0sols[2][2]+
               6.0sols[2][3])
    end
    explicitfun3(p, [sol1, sol2])
    sol3=solve(prob3)  # LinearProblem uses default linear solver
    manualscc=reduce(vcat, (sol1, sol2, sol3))

    sccprob=SciMLBase.SCCNonlinearProblem((prob1, prob2, prob3),
        SciMLBase.Void{Any}.([explicitfun1, explicitfun2, explicitfun3]))

    # Test with SCCAlg that handles both nonlinear and linear problems
    using SCCNonlinearSolve
    scc_alg=SCCNonlinearSolve.SCCAlg(nlalg = NewtonRaphson(), linalg = nothing)
    scc_sol=solve(sccprob, scc_alg)
    @test sol ≈ manualscc ≈ scc_sol

    # Backwards compat of alg choice
    scc_sol=solve(sccprob, NewtonRaphson())
    @test sol ≈ manualscc ≈ scc_sol

    import NonlinearSolve # Required for Default

    # Test default interface
    scc_sol_default=solve(sccprob)
    @test sol ≈ manualscc ≈ scc_sol_default
end

@testitem "SCCNonlinearProblem solve without explicit u0 (issue #758)" setup=[CoreRootfindTesting] tags=[:core] begin
    # Regression test for https://github.com/SciML/NonlinearSolve.jl/issues/758
    # SCCNonlinearProblem does not have a u0 field, so calling solve() without
    # explicit u0 should not try to access prob.u0
    using SciMLBase
    using SCCNonlinearSolve
    import NonlinearSolve

    # Create simple nonlinear subproblems (OOP style, returning vectors)
    f1(u, p) = [u[1]^2 - 2.0]
    f2(u, p) = [u[1] - 1.0]

    prob1 = NonlinearProblem(f1, [1.0])
    prob2 = NonlinearProblem(f2, [1.0])

    # Create explicit functions (identity - just pass through)
    explicitfun1!(p, sols) = nothing
    explicitfun2!(p, sols) = nothing

    # Create the SCC problem using the same pattern as existing tests
    scc_prob = SciMLBase.SCCNonlinearProblem(
        (prob1, prob2),
        SciMLBase.Void{Any}.([explicitfun1!, explicitfun2!])
    )

    # This should not throw an error about prob.u0 field
    sol = SciMLBase.solve(scc_prob)

    @test SciMLBase.successful_retcode(sol)
    # First subproblem: u^2 = 2 → u = sqrt(2)
    @test sol.u[1] ≈ sqrt(2.0)
    # Second subproblem: u = 1
    @test sol.u[2] ≈ 1.0
end

@testitem "SCC Residuals Transfer" setup=[CoreRootfindTesting] tags=[:core] begin
    using NonlinearSolveFirstOrder
    using LinearAlgebra

    # Create a simple SCC problem with both nonlinear and linear components
    # to test that residuals are properly computed and transferred

    # Nonlinear problem
    function f1(du, u, p)
        du[1] = u[1]^2 - 2.0
        du[2] = u[2] - u[1]
    end
    explicitfun1(p, sols) = nothing
    prob1 = NonlinearProblem(
        NonlinearFunction{true, SciMLBase.NoSpecialize}(f1), [1.0, 1.0], nothing)

    # Linear problem
    A2 = [1.0 0.5; 0.5 1.0]
    b2 = [1.0, 2.0]
    prob2 = LinearProblem(A2, b2)
    explicitfun2(p, sols) = nothing

    # Another nonlinear problem
    function f3(du, u, p)
        du[1] = u[1] + u[2] - 3.0
        du[2] = u[1] * u[2] - 2.0
    end
    explicitfun3(p, sols) = nothing
    prob3 = NonlinearProblem(
        NonlinearFunction{true, SciMLBase.NoSpecialize}(f3), [1.0, 2.0], nothing)

    # Create SCC problem
    sccprob = SciMLBase.SCCNonlinearProblem((prob1, prob2, prob3),
        SciMLBase.Void{Any}.([explicitfun1, explicitfun2, explicitfun3]))

    # Solve with SCCAlg
    using SCCNonlinearSolve
    scc_alg = SCCNonlinearSolve.SCCAlg(nlalg = NewtonRaphson(), linalg = nothing)
    scc_sol = solve(sccprob, scc_alg)

    # Test that solution was successful
    @test SciMLBase.successful_retcode(scc_sol)

    # Test that residuals are not nothing
    @test scc_sol.resid !== nothing
    @test !any(isnothing, scc_sol.resid)

    # Test that residuals have the correct length
    expected_length = length(prob1.u0) + length(prob2.b) + length(prob3.u0)
    @test length(scc_sol.resid) == expected_length

    # Test that residuals are small (near zero for converged solution)
    @test norm(scc_sol.resid) < 1e-10

    # Manually compute residuals to verify correctness
    u1 = scc_sol.u[1:2]
    u2 = scc_sol.u[3:4]
    u3 = scc_sol.u[5:6]

    # Compute residuals for each component
    resid1 = zeros(2)
    f1(resid1, u1, nothing)

    resid2 = A2 * u2 - b2

    resid3 = zeros(2)
    f3(resid3, u3, nothing)

    expected_resid = vcat(resid1, resid2, resid3)
    @test scc_sol.resid ≈ expected_resid atol=1e-10
end
