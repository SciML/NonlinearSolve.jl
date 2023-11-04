using NonlinearSolve, MINPACK, Test

function f_iip(du, u, p)
    du[1] = 2 - 2u[1]
    du[2] = u[1] - 4u[2]
end
u0 = zeros(2)
prob_iip = NonlinearProblem{true}(f_iip, u0)
abstol = 1e-8
for alg in [CMINPACK()]
    local sol
    sol = solve(prob_iip, alg)
    @test sol.retcode == ReturnCode.Success
    p = nothing

    du = zeros(2)
    f_iip(du, sol.u, nothing)
    @test maximum(du) < 1e-6
end

# OOP Tests
f_oop(u, p) = [2 - 2u[1], u[1] - 4u[2]]
u0 = zeros(2)
prob_oop = NonlinearProblem{false}(f_oop, u0)
for alg in [CMINPACK()]
    local sol
    sol = solve(prob_oop, alg)
    @test sol.retcode == ReturnCode.Success

    du = zeros(2)
    du = f_oop(sol.u, nothing)
    @test maximum(du) < 1e-6
end
