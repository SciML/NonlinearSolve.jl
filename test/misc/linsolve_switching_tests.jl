@testitem "Singular Systems -- Auto Linear Solve Switching" tags=[:misc] begin
    using LinearSolve, NonlinearSolve

    function f!(du, u, p)
        du[1] = 2u[1] - 2
        du[2] = (u[1] - 4u[2])^2 + 0.1
    end

    u0 = [0.0, 0.0] # Singular Jacobian at u0

    prob = NonlinearProblem(f!, u0)

    sol = solve(prob)  # This doesn't have a root so let's just test the switching
    @test sol.u≈[1.0, 0.25] atol=1e-3 rtol=1e-3

    function nlls!(du, u, p)
        du[1] = 2u[1] - 2
        du[2] = (u[1] - 4u[2])^2 + 0.1
        du[3] = 0
    end

    u0 = [0.0, 0.0]

    prob = NonlinearProblem(NonlinearFunction(nlls!, resid_prototype = zeros(3)), u0)

    solve(prob)
    @test sol.u≈[1.0, 0.25] atol=1e-3 rtol=1e-3
end
