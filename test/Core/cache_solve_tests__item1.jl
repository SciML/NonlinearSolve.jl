using NonlinearSolve, SciMLBase, SymbolicIndexingInterface, Test
using ADTypes: AutoFiniteDiff
using NonlinearSolveBase: get_fu, get_nsteps, get_u, refresh_residual!, solve_cache!,
    supports_deferred_residual

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

function solve_any_cache(prob, alg)
    cache = init(prob, alg)
    if cache isa NonlinearSolveBase.NonlinearSolveNoInitCache
        return solve!(cache)
    end
    solve_cache!(cache)
    return solve!(cache)
end

function cache_problem!(resid, u, p)
    resid[1] = u[1]^2 - p
    return nothing
end

prob = NonlinearProblem(cache_problem!, [1.0], 2.0)
cache = init(prob, NewtonRaphson(; autodiff = AutoFiniteDiff()))
observer = StepObserver(0, Inf)

interface_cache = init(prob, NewtonRaphson(; autodiff = AutoFiniteDiff()))
@test get_u(interface_cache) == [1.0]
@test get_fu(interface_cache) == [-1.0]
@test get_nsteps(interface_cache) == 0
@test !supports_deferred_residual(interface_cache)
@test refresh_residual!(interface_cache) === nothing
step!(interface_cache)
@test get_nsteps(interface_cache) == 1
@test get_u(interface_cache) != [1.0]
@test get_fu(interface_cache) != [-1.0]
@test all(isfinite, get_u(interface_cache))
@test all(isfinite, get_fu(interface_cache))

generic_simple_sol = solve_any_cache(
    prob, SimpleNewtonRaphson(; autodiff = AutoFiniteDiff())
)
@test SciMLBase.successful_retcode(generic_simple_sol)
@test only(generic_simple_sol.u) ≈ √2

generic_stepping_sol = solve_any_cache(
    prob, NewtonRaphson(; autodiff = AutoFiniteDiff())
)
@test SciMLBase.successful_retcode(generic_stepping_sol)
@test only(generic_stepping_sol.u) ≈ √2

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
