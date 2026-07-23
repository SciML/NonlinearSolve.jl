using NonlinearSolve
using CommonSolve
using SciMLBase
using Test

alg = KantorovichHomotopy()
@test alg.predictor === :constant
@test alg.predictor_order == 1
@test alg.strict
@test alg.qmin == 1 // 5
@test alg.qmax == 5
@test KantorovichHomotopy(; predictor = :secant).predictor_order == 2
@test KantorovichHomotopy(; predictor = :secant, predictor_order = 3).predictor_order == 3
@test NonlinearSolveBase._kantorovich_step_factor(alg, NaN, true, Float64) == 0.2

@test_throws ArgumentError KantorovichHomotopy(; nsteps = 0)
@test_throws ArgumentError KantorovichHomotopy(; initial_step_factor = 0)
@test_throws ArgumentError KantorovichHomotopy(; min_dλ = 0)
@test_throws ArgumentError KantorovichHomotopy(; max_step_factor = 2)
@test_throws ArgumentError KantorovichHomotopy(; qmin = 1)
@test_throws ArgumentError KantorovichHomotopy(; qmax = 0.9)
@test_throws ArgumentError KantorovichHomotopy(; Θmin = 0.6, Θbar = 0.5)
@test_throws ArgumentError KantorovichHomotopy(; Θbar = 0.96)
@test_throws ArgumentError KantorovichHomotopy(; γ = 1)
@test_throws ArgumentError KantorovichHomotopy(; predictor = :quadratic)
@test_throws ArgumentError KantorovichHomotopy(; predictor_order = 0)
@test_throws ArgumentError KantorovichHomotopy(; expand_quality = 0)
@test_throws ArgumentError KantorovichHomotopy(; tracking_maxiters = 0)
@test_throws ArgumentError KantorovichHomotopy(; tracking_abstol = 0)
@test_throws ArgumentError KantorovichHomotopy(; maxsteps = 0)

H(u, p, λ) = [u[1]^3 - (1 + λ)]
prob = HomotopyProblem(H, [1.0]; λspan = (0.0, 1.0))
sol = solve(prob, KantorovichHomotopy(; inner = NewtonRaphson()))
@test SciMLBase.successful_retcode(sol)
@test sol.u[1] ≈ cbrt(2) atol = 1.0e-9

soldefault = solve(prob, KantorovichHomotopy())
@test SciMLBase.successful_retcode(soldefault)
@test soldefault.u[1] ≈ cbrt(2) atol = 1.0e-9

solnocache = solve(prob, KantorovichHomotopy(; inner = SimpleNewtonRaphson()))
@test SciMLBase.successful_retcode(solnocache)
@test solnocache.u[1] ≈ cbrt(2) atol = 1.0e-9

function Hiip!(du, u, p, λ)
    du[1] = u[1]^2 - (1 + λ)
    return nothing
end
probiip = HomotopyProblem{true}(Hiip!, Float32[1]; λspan = (0.0f0, 1.0f0))
soliip = solve(probiip, KantorovichHomotopy(; inner = NewtonRaphson()))
@test SciMLBase.successful_retcode(soliip)
@test eltype(soliip.u) == Float32
@test soliip.u[1] ≈ sqrt(2.0f0) atol = 2.0f-5

Hdecreasing(u, p, λ) = [u[1] - (1 + λ)]
probdecreasing = HomotopyProblem(Hdecreasing, [2.0]; λspan = (1.0, 0.0))
soldecreasing = solve(
    probdecreasing, KantorovichHomotopy(; inner = NewtonRaphson())
)
@test SciMLBase.successful_retcode(soldecreasing)
@test soldecreasing.u[1] ≈ 1.0 atol = 1.0e-10

λcalls = Float64[]
function Htracked(u, p, λ)
    push!(λcalls, λ)
    return [u[1] - λ]
end
probtracked = HomotopyProblem(Htracked, [0.0]; λspan = (0.0, 1.0))
soltracked = solve(
    probtracked,
    KantorovichHomotopy(;
        inner = NewtonRaphson(), initial_step_factor = 0.1, max_step_factor = 1.0
    )
)
@test SciMLBase.successful_retcode(soltracked)
λattempts = [λcalls[i] for i in eachindex(λcalls) if i == 1 || λcalls[i] != λcalls[i - 1]]
q_expected = 0.95 *
    ((sqrt(1 + 4 * 0.5) - 1) / (sqrt(1 + 4 * (1 / 8)) - 1))
@test λattempts[1:3] ≈ [0.0, 0.1, 0.1 * (1 + q_expected)] atol = 1.0e-12

struct SlowContractionAlgorithm <: SciMLBase.AbstractNonlinearAlgorithm end

mutable struct SlowContractionCache{U, P, A, S} <:
    NonlinearSolveBase.AbstractNonlinearSolveCache
    u::U
    fu::U
    prob::P
    alg::A
    stats::S
    nsteps::Int
    maxiters::Int
    force_stop::Bool
    retcode::SciMLBase.ReturnCode.T
end

function SciMLBase.__init(
        prob::SciMLBase.AbstractNonlinearProblem, alg::SlowContractionAlgorithm, args...;
        maxiters = 3, kwargs...
    )
    u = copy(prob.u0)
    fu = prob.f(u, prob.p)
    solved = iszero(NonlinearSolveBase.L2_NORM(fu))
    return SlowContractionCache(
        u, fu, prob, alg, nothing, 0, maxiters, solved,
        solved ? ReturnCode.Success : ReturnCode.Default
    )
end

function SciMLBase.reinit!(cache::SlowContractionCache, u0; kwargs...)
    copyto!(cache.u, u0)
    cache.fu = cache.prob.f(cache.u, cache.prob.p)
    cache.nsteps = 0
    cache.force_stop = iszero(NonlinearSolveBase.L2_NORM(cache.fu))
    cache.retcode = cache.force_stop ? ReturnCode.Success : ReturnCode.Default
    return cache
end

function CommonSolve.step!(cache::SlowContractionCache; kwargs...)
    cache.nsteps += 1
    cache.u .-= 0.01 .* cache.fu
    cache.fu = cache.prob.f(cache.u, cache.prob.p)
    if cache.nsteps == cache.maxiters
        cache.u .-= cache.fu
        cache.fu = cache.prob.f(cache.u, cache.prob.p)
        cache.retcode = ReturnCode.Success
        cache.force_stop = true
    end
    return cache
end

function CommonSolve.solve!(cache::SlowContractionCache)
    while NonlinearSolveBase.not_terminated(cache)
        CommonSolve.step!(cache)
    end
    cache.retcode == ReturnCode.Default && (cache.retcode = ReturnCode.MaxIters)
    return SciMLBase.build_solution(
        cache.prob, cache.alg, copy(cache.u), copy(cache.fu); retcode = cache.retcode
    )
end

corrector_prob = NonlinearProblem((u, p) -> [u[1] - 1], [0.0])
corrector_cache = init(corrector_prob, SlowContractionAlgorithm(); maxiters = 2)
corrector_sol, first_Θ, rejected_Θ, has_Θ, contraction_rejected =
    NonlinearSolveBase._homotopy_corrector!(corrector_cache, alg, Float64)
@test SciMLBase.successful_retcode(corrector_sol)
@test has_Θ
@test first_Θ ≈ 0.99
@test rejected_Θ ≈ 0.99
@test contraction_rejected

slowprob = HomotopyProblem((u, p, λ) -> [u[1] - λ], [0.0]; λspan = (0.0, 0.1))
strictsol = solve(
    slowprob,
    KantorovichHomotopy(;
        inner = SlowContractionAlgorithm(), initial_step_factor = 1.0,
        min_dλ = 1.0e-3, strict = true
    )
)
@test strictsol.retcode == ReturnCode.ConvergenceFailure
@test strictsol.u == [0.0]

nonstrictsol = solve(
    slowprob,
    KantorovichHomotopy(;
        inner = SlowContractionAlgorithm(), initial_step_factor = 1.0,
        min_dλ = 1.0e-3, strict = false
    )
)
@test SciMLBase.successful_retcode(nonstrictsol)
@test nonstrictsol.u ≈ [0.1] atol = 1.0e-12

foldtarget = 2.1038034
Hfold(u, p, λ) = [u[1]^3 - 3 * u[1] - (-3 + 6λ)]
foldprob = HomotopyProblem(Hfold, [-foldtarget]; λspan = (0.0, 1.0))
foldstage = KantorovichHomotopy(;
    inner = NewtonRaphson(), min_dλ = 1.0e-2
)
foldstage_sol = solve(foldprob, foldstage; maxiters = 10)
@test !SciMLBase.successful_retcode(foldstage_sol)

foldpoly_sol = solve(
    foldprob,
    HomotopyPolyAlgorithm(
        (foldstage, ArcLengthContinuation(; inner = SimpleNewtonRaphson()));
        store_original = Val(true)
    );
    maxiters = 10
)
@test SciMLBase.successful_retcode(foldpoly_sol)
@test foldpoly_sol.u[1] ≈ foldtarget atol = 1.0e-4
@test foldpoly_sol.original !== nothing
@test first(foldpoly_sol.original.prob.λspan) > first(foldprob.λspan)
