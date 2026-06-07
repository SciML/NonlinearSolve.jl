using SCCNonlinearSolve
include("setup_corerootfindtesting.jl")

using NonlinearSolveFirstOrder
using LinearAlgebra

# Create a simple SCC problem with both nonlinear and linear components
# to test that residuals are properly computed and transferred

# Nonlinear problem
function f1(du, u, p)
    du[1] = u[1]^2 - 2.0
    return du[2] = u[2] - u[1]
end
explicitfun1(p, sols) = nothing
prob1 = NonlinearProblem(
    NonlinearFunction{true, SciMLBase.NoSpecialize}(f1), [1.0, 1.0], nothing
)

# Linear problem
A2 = [1.0 0.5; 0.5 1.0]
b2 = [1.0, 2.0]
prob2 = LinearProblem(A2, b2)
explicitfun2(p, sols) = nothing

# Another nonlinear problem
function f3(du, u, p)
    du[1] = u[1] + u[2] - 3.0
    return du[2] = u[1] * u[2] - 2.0
end
explicitfun3(p, sols) = nothing
prob3 = NonlinearProblem(
    NonlinearFunction{true, SciMLBase.NoSpecialize}(f3), [1.0, 2.0], nothing
)

# Create SCC problem
sccprob = SciMLBase.SCCNonlinearProblem(
    (prob1, prob2, prob3),
    SciMLBase.Void{Any}.([explicitfun1, explicitfun2, explicitfun3])
)

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
@test norm(scc_sol.resid) < 1.0e-10

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
@test scc_sol.resid ≈ expected_resid atol = 1.0e-10
