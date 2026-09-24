using SCCNonlinearSolve
include("setup_corerootfindtesting.jl")

using NonlinearSolveFirstOrder
using SCCNonlinearSolve
using FunctionWrappers: FunctionWrapper
using ADTypes: AutoFiniteDiff

# Define two nonlinear SCC subproblems with actual parameter coupling.
# Problem 1: cos(u2) - u1 = 0, sin(u1 + u2) + u2 = 0
#   (no dependencies on other components)
function f1_raw(du, u, p)
    du[1] = cos(u[2]) - u[1]
    du[2] = sin(u[1] + u[2]) + u[2]
    return nothing
end

# Problem 2: 2u2 + u1 + p[1] = 0, u3^2 + u2 = 0, u1^2 + u3 = 0
#   p[1] receives u1 from component 1's solution via explicitfun2
function f2_raw(du, u, p)
    du[1] = 2u[2] + u[1] + p[1]
    du[2] = u[3]^2 + u[2]
    du[3] = u[1]^2 + u[3]
    return nothing
end

# Wrap the residual functions with FunctionWrapper to unify their types.
FW = FunctionWrapper{Nothing, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}
f1_wrapped = FW(f1_raw)
f2_wrapped = FW(f2_raw)
@test typeof(f1_wrapped) === typeof(f2_wrapped)

# Create NonlinearFunctions with FullSpecialize (FunctionWrapper handles type unification)
nf1 = NonlinearFunction{true, SciMLBase.FullSpecialize}(f1_wrapped)
nf2 = NonlinearFunction{true, SciMLBase.FullSpecialize}(f2_wrapped)
@test typeof(nf1) === typeof(nf2)

# Create problems with the same p type so all problems have identical concrete type
prob1 = NonlinearProblem(nf1, zeros(2), zeros(3))
prob2 = NonlinearProblem(nf2, zeros(3), zeros(3))
@test typeof(prob1) === typeof(prob2)

# Store as Vector (not Tuple) to exercise the vector branch of iteratively_build_sols
probs = [prob1, prob2]

# Explicit transfer functions with actual parameter coupling:
# - explicitfun1: no-op (first component has no dependencies)
# - explicitfun2: transfers u[1] from component 1's solution into component 2's p[1]
explicitfun1_raw(p, sols) = nothing
function explicitfun2_raw(p, sols)
    p[1] = sols[1].u[1]
    return nothing
end

# Wrap explicitfuns with FunctionWrapper for type unification.
# The stripped solution type is deterministic — compute it from u0 type.
uType = Vector{Float64}
SSol = SciMLBase.NonlinearSolution{
    Float64, 1, uType, uType,
    NamedTuple{(:p,), Tuple{Nothing}}, Nothing, Nothing, Nothing, Nothing, Nothing,
}
SolsView = SubArray{SSol, 1, Vector{SSol}, Tuple{UnitRange{Int64}}, true}
EFW = FunctionWrapper{Nothing, Tuple{Vector{Float64}, SolsView}}
ef1_wrapped = EFW(explicitfun1_raw)
ef2_wrapped = EFW(explicitfun2_raw)

# Verify both FunctionWrapper types are identical -- the whole point of wrapping
@test typeof(ef1_wrapped) === typeof(ef2_wrapped)

# Pass wrapped explicitfuns directly as a Vector (no Void{Any} wrapping)
explicitfuns = [ef1_wrapped, ef2_wrapped]

# Create SCCNonlinearProblem with Vector inputs
sccprob = SciMLBase.SCCNonlinearProblem(probs, explicitfuns)
@test sccprob.probs isa AbstractVector

# Use AutoFiniteDiff since FunctionWrapper enforces strict Float64 signatures
# and cannot accept ForwardDiff.Dual arguments
scc_alg = SCCNonlinearSolve.SCCAlg(
    nlalg = NewtonRaphson(; autodiff = AutoFiniteDiff()), linalg = nothing
)

# First solve
scc_sol = solve(sccprob, scc_alg)
@test SciMLBase.successful_retcode(scc_sol)

# Default: original is nothing (store_original = Val(false))
@test scc_sol.original === nothing

# With store_original = Val(true): original contains sub-solutions
scc_alg_debug = SCCNonlinearSolve.SCCAlg(
    nlalg = NewtonRaphson(; autodiff = AutoFiniteDiff()),
    linalg = nothing,
    store_original = Val(true),
)
scc_sol_debug = solve(sccprob, scc_alg_debug)
@test SciMLBase.successful_retcode(scc_sol_debug)
@test scc_sol_debug.original !== nothing
@test eltype(scc_sol_debug.original) !== Any
@test isconcretetype(eltype(scc_sol_debug.original))

# Verify correctness against a reference full-system solve.
# The coupled system is:
#   du[1] = cos(u[2]) - u[1]
#   du[2] = sin(u[1] + u[2]) + u[2]
#   du[3] = 2u[4] + u[3] + u[1]   (p[1] = u[1] from component 1)
#   du[4] = u[5]^2 + u[4]
#   du[5] = u[3]^2 + u[5]
function f_full(du, u, p)
    du[1] = cos(u[2]) - u[1]
    du[2] = sin(u[1] + u[2]) + u[2]
    du[3] = 2u[4] + u[3] + u[1]
    du[4] = u[5]^2 + u[4]
    return du[5] = u[3]^2 + u[5]
end
ref_prob = NonlinearProblem(f_full, zeros(5))
ref_sol = solve(ref_prob, NewtonRaphson())
@test scc_sol.u ≈ ref_sol.u atol = 1.0e-10

# Second solve should not trigger recompilation (compile_time requires Julia 1.11+)
if VERSION >= v"1.11"
    stats = @timed solve(sccprob, scc_alg)
    @test stats.compile_time == 0.0
end
