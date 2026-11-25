using SCCNonlinearSolve
include("setup_corerootfindtesting.jl")

using NonlinearSolveFirstOrder
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

_probe_sol = solve(prob1, NewtonRaphson(; autodiff = AutoFiniteDiff()))
ConcreteSol = typeof(
    SciMLBase.strip_solution(
        SciMLBase.build_solution(
            prob1, nothing, _probe_sol.u, _probe_sol.resid; _probe_sol.retcode
        )
    )
)

uType = Vector{Float64}
PartialSol = SciMLBase.NonlinearSolution{
    Float64, 1, uType, uType,
    NamedTuple{(:p,), Tuple{Nothing}}, Nothing, Nothing, Nothing, Nothing, Nothing,
}

function wrapped_transfer(f, ::Type{S}) where {S}
    V = SubArray{S, 1, Vector{S}, Tuple{UnitRange{Int}}, true}
    return FunctionWrapper{Nothing, Tuple{Vector{Float64}, V}}(
        (p, sols) -> begin
            @test eltype(sols) === S
            return f(p, sols)
        end
    )
end

# Use AutoFiniteDiff since FunctionWrapper enforces strict Float64 signatures
# and cannot accept ForwardDiff.Dual arguments
scc_alg = SCCNonlinearSolve.SCCAlg(
    nlalg = NewtonRaphson(; autodiff = AutoFiniteDiff()), linalg = nothing
)

scc_alg_debug = SCCNonlinearSolve.SCCAlg(
    nlalg = NewtonRaphson(; autodiff = AutoFiniteDiff()),
    linalg = nothing,
    store_original = Val(true),
)
function f_full(du, u, p)
    du[1] = cos(u[2]) - u[1]
    du[2] = sin(u[1] + u[2]) + u[2]
    du[3] = 2u[4] + u[3] + u[1]
    du[4] = u[5]^2 + u[4]
    du[5] = u[3]^2 + u[5]
    return
end
ref_prob = NonlinearProblem(f_full, zeros(5))
ref_sol = solve(ref_prob, NewtonRaphson())

decorate(f) = (p, sols) -> f(p, sols)

@testset "Solution history with $S" for S in (ConcreteSol, PartialSol)
    direct = [wrapped_transfer(f, S) for f in (explicitfun1_raw, explicitfun2_raw)]
    @test typeof(direct[1]) === typeof(direct[2])
    adapted = map(decorate, direct)
    @testset "$label" for (label, callbacks, stored) in (
            ("direct", direct, S), ("decorated", adapted, PartialSol),
            ("plain and direct", Any[explicitfun1_raw, direct[2]], S),
            ("plain and decorated", Any[explicitfun1_raw, adapted[2]], PartialSol),
            ("mixed decorated", Any[wrapped_transfer(explicitfun1_raw, PartialSol), adapted[2]], PartialSol),
            ("mixed direct", Any[wrapped_transfer(explicitfun1_raw, ConcreteSol), direct[2]], ConcreteSol),
        )
        sccprob = SciMLBase.SCCNonlinearProblem(probs, callbacks)
        @test sccprob.probs isa AbstractVector
        scc_sol = solve(sccprob, scc_alg)
        @test SciMLBase.successful_retcode(scc_sol)
        @test scc_sol.u ≈ ref_sol.u atol = 1.0e-10
        @test scc_sol.original === nothing
        scc_sol_debug = solve(sccprob, scc_alg_debug)
        @test SciMLBase.successful_retcode(scc_sol_debug)
        @test scc_sol_debug.u ≈ ref_sol.u atol = 1.0e-10
        @test eltype(scc_sol_debug.original) === stored
        if callbacks === direct
            if S === ConcreteSol
                @test isconcretetype(eltype(scc_sol_debug.original))
            end
            if VERSION >= v"1.11"
                stats = @timed solve(sccprob, scc_alg)
                @test stats.compile_time == 0.0
            end
        end
    end
end

@testset "Decorated callback with despecialized parameters" begin
    blocks = map(probs) do prob
        f = NonlinearFunction{true, SciMLBase.AutoDespecialize}(prob.f.f)
        return NonlinearProblem(f, prob.u0, prob.p)
    end
    callbacks = map((explicitfun1_raw, explicitfun2_raw)) do transfer
        wrapped = wrapped_transfer(transfer, ConcreteSol)
        return (p::Vector{Float64}, history) -> wrapped(p, history)
    end
    sol = solve(SciMLBase.SCCNonlinearProblem(blocks, collect(callbacks)), scc_alg)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ ref_sol.u atol = 1.0e-10
end

@testset "Ordinary transfer callbacks receive a view" begin
    transfer(p, history::SubArray) = explicitfun2_raw(p, history)
    for callback in (transfer, decorate(transfer))
        blocks = [remake(prob; p = zero(prob.p)) for prob in probs]
        sol = solve(SciMLBase.SCCNonlinearProblem(blocks, [explicitfun1_raw, callback]), scc_alg)
        @test SciMLBase.successful_retcode(sol)
        @test blocks[2].p[1] == sol.u[1]
        @test sol.u ≈ ref_sol.u atol = 1.0e-10
    end
end
