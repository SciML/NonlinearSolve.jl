using NonlinearSolveFirstOrder
using ForwardDiff: value
using SciMLBase
using StaticArrays
using LineSearch: BackTracking

# The residual throws outside the box, so a solve that returns never evaluated it there.
struct OutsideBox <: Exception end

function box_residual(u, p, lb, ub)
    x = value.(u)
    if any(x .< lb) || any(x .> ub)
        throw(OutsideBox())
    end
    return u .^ 2 .- p
end
box_residual!(resid, u, p, lb, ub) = (resid .= box_residual(u, p, lb, ub); nothing)

lb, ub = [0.5, 0.5], [2.0, 5.0]
p = [4.0, 9.0]    # the first root, 2, is exactly on the upper bound
u0 = [1.0, 1.0]

algs = (
    "NewtonRaphson" => NewtonRaphson(; bounds_handling = BoundsProjection()),
    "NewtonRaphson, BackTracking" => NewtonRaphson(;
        bounds_handling = BoundsProjection(), linesearch = BackTracking()
    ),
    "NewtonRaphson, jacobian_reuse" => NewtonRaphson(;
        bounds_handling = BoundsProjection(), jacobian_reuse = true
    ),
    "TrustRegion" => TrustRegion(; bounds_handling = BoundsProjection()),
    "TrustRegionDogleg" => TrustRegionDogleg(; bounds_handling = BoundsProjection()),
)

@testset "the transform stays the default" begin
    @test !SciMLBase.allowsbounds(NewtonRaphson())
    @test !SciMLBase.allowsbounds(TrustRegion())
    @test SciMLBase.allowsbounds(NewtonRaphson(; bounds_handling = BoundsProjection()))
    @test SciMLBase.allowsbounds(TrustRegion(; bounds_handling = BoundsProjection()))

    interior = NonlinearProblem(
        (u, p) -> u .^ 2 .- p, [1.5, 1.5], [1.0, 4.0]; lb, ub
    )
    sol = solve(interior, NewtonRaphson())
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ [1.0, 2.0]
end

@testset "$(name)" for (name, alg) in algs
    @testset "$(iip ? "in-place" : "out-of-place")" for iip in (false, true)
        f = iip ? (resid, u, p) -> box_residual!(resid, u, p, lb, ub) :
            (u, p) -> box_residual(u, p, lb, ub)
        sol = solve(NonlinearProblem{iip}(f, u0, p; lb, ub), alg; abstol = 1.0e-10)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u[1] ≈ 2.0
        @test sol.u[2] ≈ 3.0
        @test all(lb .<= sol.u .<= ub)
    end

    @testset "an infeasible initial guess is clamped" begin
        f = (u, p) -> box_residual(u, p, lb, ub)
        prob = NonlinearProblem(f, [-1.0, 10.0], p; lb, ub)
        sol = solve(prob, alg; abstol = 1.0e-10)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [2.0, 3.0]
        @test prob.u0 == [-1.0, 10.0]
    end

    @testset "scalar state and scalar bounds" begin
        f = (u, p) -> box_residual(u, p, 0.5, 2.0)[1]
        sol = solve(NonlinearProblem(f, 1.0, 4.0; lb = 0.5, ub = 2.0), alg; abstol = 1.0e-10)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ 2.0
    end

    @testset "static arrays" begin
        f = (u, p) -> SVector{2}(box_residual(u, p, lb, ub))
        prob = NonlinearProblem(f, SVector{2}(u0), SVector{2}(p); lb, ub)
        sol = solve(prob, alg; abstol = 1.0e-10)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u isa SVector{2}
        @test sol.u ≈ [2.0, 3.0]
    end

    @testset "one-sided and fixed coordinates" begin
        lower, upper = [0.5, 3.0], [Inf, 3.0]
        f = (u, p) -> box_residual(u, p, lower, upper)
        sol = solve(NonlinearProblem(f, [1.0, 3.0], p; lb = lower, ub = upper), alg; abstol = 1.0e-10)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [2.0, 3.0]
    end
end

@testset "reinit! with a carried Jacobian keeps the iterate in the box" begin
    f = (u, p) -> box_residual(u, p, lb, ub)
    alg = NewtonRaphson(; bounds_handling = BoundsProjection(), jacobian_reuse = true)
    cache = init(NonlinearProblem(f, u0, [1.0, 4.0]; lb, ub), alg; abstol = 1.0e-10)
    solve!(cache)
    for p_next in ([2.0, 6.0], [4.0, 9.0], [3.0, 4.0])
        reinit!(cache, [3.0, -1.0]; p = p_next, reuse_jacobian = true)
        sol = solve!(cache)
        reference = solve(NonlinearProblem(f, [2.0, 0.5], p_next; lb, ub), alg; abstol = 1.0e-10)
        @test sol.u ≈ reference.u
        @test all(lb .<= sol.u .<= ub)
    end
end

@testset "a clamped trust-region step is judged on the clamped step" begin
    # With a radius that does not limit the step, the Newton step from [1.5, 2] overshoots the
    # upper bound of the first coordinate, whose root lies on that bound. The clamped step
    # reduces the residual, so it is accepted and does not shrink the radius.
    f = (u, p) -> box_residual(u, p, lb, ub)
    alg = TrustRegion(;
        bounds_handling = BoundsProjection(), initial_trust_radius = 10.0,
        max_trust_radius = 100.0
    )
    cache = init(NonlinearProblem(f, [1.5, 2.0], p; lb, ub), alg; abstol = 1.0e-10)
    scheme = cache.trustregion_cache
    step!(cache)
    @test cache.u[1] == ub[1]
    @test scheme.last_step_accepted
    @test scheme.ρ > scheme.step_threshold
    @test scheme.shrink_counter == 0
end
