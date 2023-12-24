using NonlinearSolve, SIAMFANLEquations, LinearAlgebra, Test

# IIP Tests
function f_iip(du, u, p, t)
    du[1] = 2 - 2u[1]
    du[2] = u[1] - 4u[2]
end
u0 = zeros(2)
prob_iip = SteadyStateProblem(f_iip, u0)
abstol = 1e-8

for alg in [SIAMFANLEquationsJL()]
    sol = solve(prob_iip, alg)
    @test sol.retcode == ReturnCode.Success
    p = nothing

    du = zeros(2)
    f_iip(du, sol.u, nothing, 0)
    @test maximum(du) < 1e-6
end

# OOP Tests
f_oop(u, p, t) = [2 - 2u[1], u[1] - 4u[2]]
u0 = zeros(2)
prob_oop = SteadyStateProblem(f_oop, u0)

for alg in [SIAMFANLEquationsJL()]
    sol = solve(prob_oop, alg)
    @test sol.retcode == ReturnCode.Success
    # test the solver is doing reasonable things for linear solve
    # and that the stats are working properly
    @test 1 <= sol.stats.nf < 10

    du = zeros(2)
    du = f_oop(sol.u, nothing, 0)
    @test maximum(du) < 1e-6
end

# NonlinearProblem Tests

function f_iip(du, u, p)
    du[1] = 2 - 2u[1]
    du[2] = u[1] - 4u[2]
end
u0 = zeros(2)
prob_iip = NonlinearProblem{true}(f_iip, u0)
abstol = 1e-8
for alg in [SIAMFANLEquationsJL()]
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
for alg in [SIAMFANLEquationsJL()]
    local sol
    sol = solve(prob_oop, alg)
    @test sol.retcode == ReturnCode.Success

    du = zeros(2)
    du = f_oop(sol.u, nothing)
    @test maximum(du) < 1e-6
end

# tolerance tests for scalar equation solvers
f_tol(u, p) = u^2 - 2
prob_tol = NonlinearProblem(f_tol, 1.0)
for tol in [1e-1, 1e-3, 1e-6, 1e-10, 1e-11]
    for method = [:newton, :pseudotransient, :secant]
        sol = solve(prob_tol, SIAMFANLEquationsJL(method = method), abstol = tol)
        @test abs(sol.u[1] - sqrt(2)) < tol
    end
end

# Test the JFNK technique
f_jfnk(u, p) = u^2 - 2
prob_jfnk = NonlinearProblem(f_jfnk, 1.0)
for tol in [1e-1, 1e-3, 1e-6, 1e-10, 1e-11]
    sol = solve(prob_jfnk, SIAMFANLEquationsJL(linsolve = :gmres), abstol = tol)
    @test abs(sol.u[1] - sqrt(2)) < tol
end

# Test the finite differencing technique
function f!(fvec, x, p)
    fvec[1] = (x[1] + 3) * (x[2]^3 - 7) + 18
    fvec[2] = sin(x[2] * exp(x[1]) - 1)
end

prob = NonlinearProblem{true}(f!, [0.1; 1.2])
sol = solve(prob, SIAMFANLEquationsJL())

du = zeros(2)
f!(du, sol.u, nothing)
@test maximum(du) < 1e-6

# Test the autodiff technique
function f!(fvec, x, p)
    fvec[1] = (x[1] + 3) * (x[2]^3 - 7) + 18
    fvec[2] = sin(x[2] * exp(x[1]) - 1)
end

prob = NonlinearProblem{true}(f!, [0.1; 1.2])
sol = solve(prob, SIAMFANLEquationsJL())

du = zeros(2)
f!(du, sol.u, nothing)
@test maximum(du) < 1e-6

function problem(x, A)
    return x .^ 2 - A
end

function problemJacobian(x, A)
    return diagm(2 .* x)
end

function f!(F, u, p)
    F[1:152] = problem(u, p)
end

function j!(J, u, p)
    J[1:152, 1:152] = problemJacobian(u, p)
end

f = NonlinearFunction(f!)

init = ones(152);
A = ones(152);
A[6] = 0.8

f = NonlinearFunction(f!, jac = j!)

p = A

ProbN = NonlinearProblem(f, init, p)
for method = [:newton, :pseudotransient]
    sol = solve(ProbN, SIAMFANLEquationsJL(method = method), reltol = 1e-8, abstol = 1e-8)
end

#= doesn't support complex numbers handling
init = ones(Complex{Float64}, 152);
ProbN = NonlinearProblem(f, init, p)
sol = solve(ProbN, SIAMFANLEquationsJL(), reltol = 1e-8, abstol = 1e-8)
=#