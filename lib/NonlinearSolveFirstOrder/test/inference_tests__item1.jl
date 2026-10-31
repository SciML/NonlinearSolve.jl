using NonlinearSolveFirstOrder

using ADTypes, NonlinearSolveBase, NonlinearSolveFirstOrder, SciMLBase, Test

f_iip = NonlinearFunction{true}((du, u, p) -> du .= u .^ 2 .- p)
f_oop = NonlinearFunction{false}((u, p) -> u .^ 2 .- p)

u0 = [1.0, 2.0]
p = [1.0, 4.0]
fu = similar(u0)
f_iip(fu, u0, p)

prob_iip = NonlinearProblem(f_iip, u0, p)
prob_oop = NonlinearProblem(f_oop, u0, p)

stats = NonlinearSolveBase.NLStats(0, 0, 0, 0, 0)
alg = TrustRegion(autodiff = AutoFiniteDiff())
autodiff = AutoFiniteDiff()

@testset "In-place, no analytic Jacobian" begin
    @inferred NonlinearSolveBase.construct_jacobian_cache(
        prob_iip, alg, prob_iip.f, fu, u0, p; stats, autodiff
    )
end

@testset "Out-of-place, no analytic Jacobian" begin
    @inferred NonlinearSolveBase.construct_jacobian_cache(
        prob_oop, alg, prob_oop.f, fu, u0, p; stats, autodiff
    )
end

@testset "With analytic Jacobian" begin
    f_jac = NonlinearFunction{true}(
        (du, u, p) -> du .= u .^ 2 .- p;
        jac = (J, u, p) -> (
            J[1, 1] = 2u[1]; J[2, 2] = 2u[2];
            J[1, 2] = 0; J[2, 1] = 0
        ),
    )
    prob_jac = NonlinearProblem(f_jac, u0, p)
    @inferred NonlinearSolveBase.construct_jacobian_cache(
        prob_jac, alg, prob_jac.f, fu, u0, p; stats, autodiff
    )
end

@testset "Full init: $name, $(nameof(alg_ctor))" for (name, prob) in (
            "in-place" => prob_iip, "out-of-place" => prob_oop,
        ), alg_ctor in (TrustRegion, NewtonRaphson)

    @inferred SciMLBase.init(prob, alg_ctor(autodiff = AutoFiniteDiff()))
end

# A view `u0` (e.g. the algebraic unknowns that `BrownFullBasicInit` hands over): the
# alias flag is not a constant inside `__init`, so the cache and solution types must not
# depend on it, and the alias specifier must still be honored.
f_iip_full = NonlinearFunction{true, SciMLBase.FullSpecialize}((du, u, p) -> du .= u .^ 2 .- p)
@testset "View u0: $(nameof(alg_ctor)), alias_u0 = $alias_u0" for alg_ctor in (
            TrustRegion, NewtonRaphson,
        ), alias_u0 in (nothing, false, true)

    parent_u0 = [3.0, 0.0, 3.0]
    u0_view = view(parent_u0, [1, 3])
    prob_view = NonlinearProblem(f_iip_full, u0_view, p)
    alg = alg_ctor(autodiff = AutoFiniteDiff())
    kwargs = alias_u0 === nothing ? (;) :
        (; alias = SciMLBase.NonlinearAliasSpecifier(; alias_u0))
    @inferred SciMLBase.init(prob_view, alg; kwargs...)
    sol = @inferred solve(prob_view, alg; kwargs...)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ sqrt.(p)
    @test sol.u isa typeof(u0_view)
    if alias_u0 === true
        @test sol.u === u0_view
        @test parent_u0[[1, 3]] ≈ sqrt.(p)
    else
        @test parent_u0 == [3.0, 0.0, 3.0]
    end
end

# An out-of-place solve assigns new arrays to `u`, so its iterate stays an `Array`:
# rolling back to the best iterate and `reinit!` with a `Vector` must keep working.
@testset "View u0, out-of-place and other views: $(nameof(alg_ctor))" for alg_ctor in (
        TrustRegion, NewtonRaphson,
    )
    alg = alg_ctor(autodiff = AutoFiniteDiff())
    f_oop_full = NonlinearFunction{false, SciMLBase.FullSpecialize}((u, p) -> u .^ 2 .- p)
    cache = init(NonlinearProblem(f_oop_full, view([3.0, 0.0, 3.0], [1, 3]), p), alg)
    @test NonlinearSolveBase.get_u(cache) isa Vector{Float64}
    reinit!(cache, [2.0, 2.0])
    @test solve!(cache).u ≈ sqrt.(p)

    f_cubic = NonlinearFunction{false, SciMLBase.FullSpecialize}(
        (u, p) -> u .^ 3 .- 2 .* u .+ 2
    )
    sol = solve(NonlinearProblem(f_cubic, view([1.0, 9.0], [1]), nothing), alg; maxiters = 1)
    @test sol.u == [1.0]

    # indices `Base.unaliascopy` cannot copy to the same type: copied as before
    for idx in (PermutedDimsArray([1, 3], (1,)), UInt8[1, 3])
        sol = solve(NonlinearProblem(f_iip_full, view([3.0, 0.0, 3.0], idx), p), alg)
        @test sol.u ≈ sqrt.(p)
    end
    # a matrix column: copied without the rest of the matrix
    A = fill(3.0, 2, 3)
    cache = init(NonlinearProblem(f_iip_full, view(A, :, 2), p), alg)
    @test NonlinearSolveBase.get_u(cache) isa typeof(view(A, :, 2))
    @test length(parent(NonlinearSolveBase.get_u(cache))) == 2
    @test solve!(cache).u ≈ sqrt.(p)
    @test A == fill(3.0, 2, 3)
end

# `set_u!` with a `Vector` on an in-place cache whose iterate is a view: the values are
# copied into the iterate
@testset "View u0, set_u!: $(nameof(alg_ctor)), alias_u0 = $alias_u0" for alg_ctor in (
            TrustRegion, NewtonRaphson,
        ), alias_u0 in (false, true)

    parent_u0 = [3.0, 0.0, 3.0]
    cache = init(
        NonlinearProblem(f_iip_full, view(parent_u0, [1, 3]), p),
        alg_ctor(autodiff = AutoFiniteDiff());
        alias = SciMLBase.NonlinearAliasSpecifier(; alias_u0)
    )
    SciMLBase.set_u!(cache, [2.0, 2.0])
    @test NonlinearSolveBase.get_u(cache) == [2.0, 2.0]
    sol = solve!(cache)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ sqrt.(p)
    @test parent_u0[[1, 3]] ≈ (alias_u0 ? sqrt.(p) : [3.0, 3.0])
    @test parent_u0[2] == 0.0
end

# In-place best-iterate rollback on a view: one Newton step from 1 on u^3 - 2u + 2 goes to
# 0, where the residual (2) is larger than at 1 (1), so the solution is the initial value
@testset "View u0, in-place rollback: $(nameof(alg_ctor))" for alg_ctor in (
        NewtonRaphson,
    )
    f_cubic! = NonlinearFunction{true, SciMLBase.FullSpecialize}(
        (du, u, p) -> (du .= u .^ 3 .- 2 .* u .+ 2)
    )
    parent_u0 = [1.0, 9.0]
    sol = solve(
        NonlinearProblem(f_cubic!, view(parent_u0, [1]), nothing),
        alg_ctor(autodiff = AutoFiniteDiff()); maxiters = 1
    )
    @test sol.u == [1.0]
    @test parent_u0 == [1.0, 9.0]
end
