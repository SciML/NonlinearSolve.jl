using NonlinearSolve, SciMLBase, SymbolicIndexingInterface, Test
using ADTypes: AutoFiniteDiff
using NonlinearSolveBase: solve_cache!

struct UnsupportedCache <: NonlinearSolveBase.AbstractNonlinearSolveCache end

mutable struct StepObserver
    iterations::Int
    residual_norm::Float64
end

function (observer::StepObserver)(u, fu, iteration)
    observer.iterations = iteration
    observer.residual_norm = sum(abs2, fu)
    return nothing
end

function cache_solve!(cache, observer)
    observer.iterations = 0
    observer.residual_norm = Inf
    return solve_cache!(cache; step_observer = observer)
end

function cache_solve_allocations(cache, observer)
    reinit!(cache, [1.0])
    GC.gc()
    return @allocated cache_solve!(cache, observer)
end

function cache_problem!(resid, u, p)
    resid[1] = u[1]^2 - p
    return nothing
end

prob = NonlinearProblem(cache_problem!, [1.0], 2.0)
cache = init(prob, NewtonRaphson(; autodiff = AutoFiniteDiff()))
observer = StepObserver(0, Inf)

retcode = cache_solve!(cache, observer)
@test SciMLBase.successful_retcode(retcode)
@test only(state_values(cache)) ≈ √2
@test observer.iterations > 0
@test observer.residual_norm ≤ 1.0e-12

reinit!(cache, [1.0])
@test SciMLBase.successful_retcode(cache_solve!(cache, observer))
allocations = cache_solve_allocations(cache, observer)
@test SciMLBase.successful_retcode(cache.retcode)
VERSION ≥ v"1.11" && @test allocations == 0

@test_throws ArgumentError solve_cache!(UnsupportedCache())
