using ADTypes: AutoForwardDiff
using NonlinearSolveQuasiNewton
using ForwardDiff: value
using LineSearch: BackTracking
using SciMLBase
using StaticArrays
using Test

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
    "Broyden" => Broyden(; project_bounds = true),
    "Broyden, BackTracking" => Broyden(;
        project_bounds = true, linesearch = BackTracking(; autodiff = AutoForwardDiff())
    ),
    "Broyden (true jacobian)" => Broyden(;
        project_bounds = true, init_jacobian = Val(:true_jacobian)
    ),
    "Klement" => Klement(; project_bounds = true),
    "LimitedMemoryBroyden" => LimitedMemoryBroyden(; project_bounds = true),
)

@testset "the transform stays the default" begin
    @test !SciMLBase.allowsbounds(Broyden())
    @test !SciMLBase.allowsbounds(Klement())
    @test !SciMLBase.allowsbounds(LimitedMemoryBroyden())
    @test SciMLBase.allowsbounds(Broyden(; project_bounds = true))
    @test SciMLBase.allowsbounds(Klement(; project_bounds = true))
    @test SciMLBase.allowsbounds(LimitedMemoryBroyden(; project_bounds = true))
end

@testset "$(name)" for (name, alg) in algs
    @testset "$(iip ? "in-place" : "out-of-place")" for iip in (false, true)
        f = iip ? (resid, u, p) -> box_residual!(resid, u, p, lb, ub) :
            (u, p) -> box_residual(u, p, lb, ub)
        sol = solve(
            NonlinearProblem{iip}(f, u0, p; lb, ub), alg;
            abstol = 1.0e-10, maxiters = 1000
        )
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [2.0, 3.0]
        @test all(lb .<= sol.u .<= ub)
    end

    @testset "an infeasible initial guess is clamped" begin
        prob = NonlinearProblem((u, p) -> box_residual(u, p, lb, ub), [-1.0, 10.0], p; lb, ub)
        sol = solve(prob, alg; abstol = 1.0e-10, maxiters = 1000)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [2.0, 3.0]
        @test prob.u0 == [-1.0, 10.0]
    end

    @testset "scalar bounds and a fixed coordinate" begin
        f = (u, p) -> box_residual(u, p, [0.5, 3.0], [Inf, 3.0])
        prob = NonlinearProblem(f, [1.0, 3.0], p; lb = [0.5, 3.0], ub = [Inf, 3.0])
        sol = solve(prob, alg; abstol = 1.0e-10, maxiters = 1000)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [2.0, 3.0]

        scalar = NonlinearProblem((u, p) -> box_residual(u, p, 0.5, 2.0)[1], 1.0, 4.0; lb = 0.5, ub = 2.0)
        sol = solve(scalar, alg; abstol = 1.0e-10, maxiters = 1000)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ 2.0
    end
end

@testset "reinit! with a carried Jacobian keeps the iterate in the box" begin
    f = (u, p) -> box_residual(u, p, lb, ub)
    alg = Broyden(; project_bounds = true, init_jacobian = Val(:true_jacobian))
    cache = init(
        NonlinearProblem(f, u0, [1.0, 4.0]; lb, ub), alg; abstol = 1.0e-10, maxiters = 1000
    )
    solve!(cache)
    for p_next in ([2.0, 6.0], [4.0, 9.0], [3.0, 4.0])
        reinit!(cache, [3.0, -1.0]; p = p_next, reuse_jacobian = true)
        sol = solve!(cache)
        reference = solve(
            NonlinearProblem(f, [2.0, 0.5], p_next; lb, ub), alg;
            abstol = 1.0e-10, maxiters = 1000
        )
        @test sol.u ≈ reference.u
        @test all(lb .<= sol.u .<= ub)
    end
end
