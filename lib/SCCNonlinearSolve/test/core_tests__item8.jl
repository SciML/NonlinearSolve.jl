using SCCNonlinearSolve
include("setup_corerootfindtesting.jl")

using NonlinearSolveFirstOrder
using SCCNonlinearSolve
using FunctionWrappers: FunctionWrapper
using ADTypes: AutoFiniteDiff

# Vector-form SCC with an explicitfun FunctionWrapper signature written against
# the ten-parameter NonlinearSolution spelling. The buffer element type must
# match the declared view element type for the callback to keep converting.
function f1_raw(du, u, p)
    du[1] = cos(u[2]) - u[1]
    du[2] = sin(u[1] + u[2]) + u[2]
    return nothing
end

function f2_raw(du, u, p)
    du[1] = 2u[2] + u[1] + p[1]
    du[2] = u[3]^2 + u[2]
    du[3] = u[1]^2 + u[3]
    return nothing
end

FW = FunctionWrapper{Nothing, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}
f1_wrapped = FW(f1_raw)
f2_wrapped = FW(f2_raw)
@test typeof(f1_wrapped) === typeof(f2_wrapped)

nf1 = NonlinearFunction{true, SciMLBase.FullSpecialize}(f1_wrapped)
nf2 = NonlinearFunction{true, SciMLBase.FullSpecialize}(f2_wrapped)
@test typeof(nf1) === typeof(nf2)

prob1 = NonlinearProblem(nf1, zeros(2), zeros(3))
prob2 = NonlinearProblem(nf2, zeros(3), zeros(3))
@test typeof(prob1) === typeof(prob2)

probs = [prob1, prob2]

explicitfun1_raw(p, sols) = nothing
function explicitfun2_raw(p, sols)
    p[1] = sols[1].u[1]
    return nothing
end

# The spelled type omits NonlinearSolution's trailing retcode-carrier parameter
# when one exists, so it may denote a UnionAll; the declared element type is
# still honored as-is, and concreteness is asserted in the updated-signature
# test only (core_tests__item4.jl).
uType = Vector{Float64}
SSol = SciMLBase.NonlinearSolution{
    Float64, 1, uType, uType,
    NamedTuple{(:p,), Tuple{Nothing}}, Nothing, Nothing, Nothing, Nothing, Nothing,
}
SolsView = SubArray{SSol, 1, Vector{SSol}, Tuple{UnitRange{Int64}}, true}
EFW = FunctionWrapper{Nothing, Tuple{Vector{Float64}, SolsView}}
ef1_wrapped = EFW(explicitfun1_raw)
ef2_wrapped = EFW(explicitfun2_raw)
@test typeof(ef1_wrapped) === typeof(ef2_wrapped)

explicitfuns = [ef1_wrapped, ef2_wrapped]

sccprob = SciMLBase.SCCNonlinearProblem(probs, explicitfuns)
@test sccprob.probs isa AbstractVector

scc_alg = SCCNonlinearSolve.SCCAlg(
    nlalg = NewtonRaphson(; autodiff = AutoFiniteDiff()), linalg = nothing
)

scc_sol = solve(sccprob, scc_alg)
@test SciMLBase.successful_retcode(scc_sol)
@test scc_sol.original === nothing

scc_alg_debug = SCCNonlinearSolve.SCCAlg(
    nlalg = NewtonRaphson(; autodiff = AutoFiniteDiff()),
    linalg = nothing,
    store_original = Val(true),
)
scc_sol_debug = solve(sccprob, scc_alg_debug)
@test SciMLBase.successful_retcode(scc_sol_debug)
@test scc_sol_debug.original !== nothing
@test eltype(scc_sol_debug.original) !== Any

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

if VERSION >= v"1.11"
    stats = @timed solve(sccprob, scc_alg)
    @test stats.compile_time == 0.0
end

# Mixed container: first explicitfun is a plain no-op, second is a legacy
# FunctionWrapper. The buffer eltype must still come from the wrapper.
mixed_explicitfuns = Any[explicitfun1_raw, ef2_wrapped]
mixed_prob = SciMLBase.SCCNonlinearProblem(probs, mixed_explicitfuns)
mixed_sol = solve(mixed_prob, scc_alg)
@test SciMLBase.successful_retcode(mixed_sol)
@test mixed_sol.u ≈ ref_sol.u atol = 1.0e-10
