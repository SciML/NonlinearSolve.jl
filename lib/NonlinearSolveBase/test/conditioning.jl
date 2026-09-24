module ConditioningTests

using Test
using NonlinearSolveBase
using SciMLBase
using NonlinearSolveBase: NonlinearVerbosity
using SciMLLogging: SciMLLogging
# Load ForwardDiff so PreallocationTools' `FixedSizeDiffCache` dual-cache extension is
# active: the bounds transform builds one for the default/ForwardDiff autodiff path.
import ForwardDiff
using NonlinearSolveBase: needs_conditioning, transform_conditioned_problem,
    PreconditionWrapper, apply_postcondition!!, get_precondition, get_postcondition,
    PostconditionSpecifier, PostconditionSpace, postcondition_space,
    transform_bounded_problem, _to_unbounded, _from_unbounded

struct UnsupportedAlg <: NonlinearSolveBase.AbstractNonlinearSolveAlgorithm end
struct SupportedAlg <: NonlinearSolveBase.AbstractNonlinearSolveAlgorithm end
NonlinearSolveBase.supports_postcondition(::SupportedAlg) = true

# minimal cache stand-in exposing the fields the corrector path reads
struct FakeCache{P, K}
    prob::P
    kwargs::K
end

const F = (u, p) -> u .^ 2 .- p
const FIIP = (du, u, p) -> (du .= u .^ 2 .- p; nothing)

@testset "options are read from solve keywords, then problem keywords" begin
    G = (fu, u, p) -> asinh.(fu)
    G2 = (fu, u, p) -> fu
    prob = NonlinearProblem(F, [1.0], 2.0)
    @test get_precondition(prob, (;)) === nothing
    @test get_precondition(prob, (; precondition = G)) === G

    probk = NonlinearProblem(F, [1.0], 2.0; precondition = G)
    @test get_precondition(probk, (;)) === G
    # solve-time wins over the problem's stored option, matching `alias`
    @test get_precondition(probk, (; precondition = G2)) === G2

    H = (up, uprev, p, cache) -> up
    @test get_postcondition(NonlinearProblem(F, [1.0], 2.0; postcondition = H), (;)) === H
end

@testset "needs_conditioning" begin
    prob = NonlinearProblem(F, [1.0], 2.0)
    @test !needs_conditioning(prob, (;))
    @test needs_conditioning(prob, (; precondition = (fu, u, p) -> fu))
    @test needs_conditioning(prob, (; postcondition = (up, uprev, p, cache) -> up))
    @test needs_conditioning(NonlinearProblem(F, [1.0], 2.0; postcondition = (a, b, c, d) -> a), (;))
end

@testset "precondition composes into the residual (oop and iip)" begin
    G = (fu, u, p) -> asinh.(fu)
    kw = (; precondition = G)
    tprob = transform_conditioned_problem(NonlinearProblem(F, [2.0], 3.0), nothing, kw)
    @test tprob.f.f isa PreconditionWrapper
    @test tprob.f.f([2.0], 3.0) ≈ asinh.(F([2.0], 3.0))

    Giip = (fu, u, p) -> (fu .= asinh.(fu); nothing)
    tprobi = transform_conditioned_problem(
        NonlinearProblem(FIIP, [2.0], 3.0), nothing, (; precondition = Giip)
    )
    du = zeros(1)
    tprobi.f.f(du, [2.0], 3.0)
    @test du ≈ asinh.([2.0^2 - 3.0])
end

@testset "the transform is idempotent across nested entry points" begin
    G = (fu, u, p) -> asinh.(fu)
    kw = (; precondition = G)
    prob = NonlinearProblem(F, [2.0], 3.0)
    tprob = transform_conditioned_problem(prob, nothing, kw)
    # the composed problem still carries the option in `kwargs`, so the guard has to be
    # the wrapper itself: the funnels each check before transforming again
    @test !needs_conditioning(tprob, kw)
    tprob2 = transform_conditioned_problem(tprob, nothing, kw)
    @test !(tprob2.f.f.f isa PreconditionWrapper)
end

@testset "postcondition corrects the initial guess" begin
    H = (up, uprev, p, cache) -> clamp.(up, 0.5, 1.0)
    tprob = transform_conditioned_problem(
        NonlinearProblem(F, [2.0], 3.0), SupportedAlg(), (; postcondition = H)
    )
    @test tprob.u0 ≈ [1.0]

    Hiip = (up, uprev, p, cache) -> (up .= clamp.(up, 0.5, 1.0); nothing)
    probi = NonlinearProblem(FIIP, [2.0], 3.0)
    tprobi = transform_conditioned_problem(probi, SupportedAlg(), (; postcondition = Hiip))
    @test tprobi.u0 ≈ [1.0]
    @test probi.u0 ≈ [2.0]   # the original problem is untouched
end

@testset "apply_postcondition!! follows the in-place convention and passes the cache" begin
    prob = NonlinearProblem(F, [1.0], 2.0)
    probi = NonlinearProblem(FIIP, [1.0], 2.0)

    H = (up, uprev, p, cache) -> up .+ uprev
    @test apply_postcondition!!([3.0], [1.0], FakeCache(prob, (; postcondition = H))) ≈ [4.0]

    Hiip = (up, uprev, p, cache) -> (up .+= uprev; nothing)
    u = [3.0]
    @test apply_postcondition!!(u, [1.0], FakeCache(probi, (; postcondition = Hiip))) === u
    @test u ≈ [4.0]

    u2 = [3.0]
    @test apply_postcondition!!(u2, [1.0], FakeCache(prob, (;))) === u2
    @test u2 ≈ [3.0]

    # four-argument correctors receive the cache and are preferred when both exist
    seen = Ref{Any}(:unset)
    H4 = (up, uprev, p, cache) -> (seen[] = cache; up)
    fc = FakeCache(prob, (; postcondition = H4))
    apply_postcondition!!([3.0], [1.0], fc)
    @test seen[] === fc
end

@testset "unsupported algorithms and bounds are reported, not rejected" begin
    H = (up, uprev, p, cache) -> up
    kw = (; postcondition = H)
    prob = NonlinearProblem(F, [1.0], 2.0)

    # an algorithm that cannot apply the corrector reports it at ErrorLevel, which
    # SciMLLogging raises after logging — so it still stops the solve by default
    @test_throws ErrorException transform_conditioned_problem(prob, UnsupportedAlg(), kw)
    # ... but unlike a bare `throw` it is a verbosity toggle, so it can be turned down
    silent = (; postcondition = H, verbose = NonlinearVerbosity(SciMLLogging.None()))
    @test transform_conditioned_problem(prob, UnsupportedAlg(), silent) isa
        SciMLBase.AbstractNonlinearProblem
    @test transform_conditioned_problem(prob, SupportedAlg(), kw).u0 ≈ [1.0]

    # bounds compose with the corrector; the initial guess is still in the original
    # coordinates here, so an Original-space corrector (the default) corrects it and a
    # Transformed one is skipped
    Hc = (up, uprev, p, cache) -> clamp.(up, 0.5, 1.0)
    prob_bounds = NonlinearProblem(F, [2.0], 3.0; lb = [0.0], ub = [4.0])
    tprob = transform_conditioned_problem(
        prob_bounds, SupportedAlg(), (; postcondition = Hc)
    )
    @test tprob.u0 ≈ [1.0]
    @test tprob.lb == prob_bounds.lb

    kw_transformed = (;
        postcondition = PostconditionSpecifier(
            Hc; space = PostconditionSpace.Transformed
        ),
    )
    @test transform_conditioned_problem(
        prob_bounds, SupportedAlg(), kw_transformed
    ).u0 ≈ [2.0]
end

@testset "PostconditionSpecifier declares the corrector's coordinates" begin
    H = (up, uprev, p, cache) -> up .+ 1

    @test postcondition_space(H) === PostconditionSpace.Original
    @test postcondition_space(PostconditionSpecifier(H)) === PostconditionSpace.Original
    @test postcondition_space(
        PostconditionSpecifier(H; space = PostconditionSpace.Transformed)
    ) === PostconditionSpace.Transformed
    @test_throws TypeError PostconditionSpecifier(H; space = :bounded)

    # the wrapper is transparent: it forwards the corrector call unchanged
    @test PostconditionSpecifier(H)([1.0], [0.0], nothing, nothing) ≈ [2.0]
end

@testset "bounded problems: the corrector acts in the space it declares" begin
    lb, ub = [0.0], [4.0]
    Hc = (up, uprev, p, cache) -> clamp.(up, 0.5, 1.0)
    Hc_iip = (up, uprev, p, cache) -> (up .= clamp.(up, 0.5, 1.0); nothing)

    tprob = transform_bounded_problem(
        NonlinearProblem(F, [2.0], 3.0; lb, ub), SupportedAlg()
    )
    tprob_iip = transform_bounded_problem(
        NonlinearProblem(FIIP, [2.0], 3.0; lb, ub), SupportedAlg()
    )
    # the solver iterates on the unconstrained variable, so the commit-point iterates
    # are the transforms of the physical values 3.0 (proposed) and 2.0 (previous)
    u = _to_unbounded.([3.0], lb, ub)
    u_prev = _to_unbounded.([2.0], lb, ub)

    # Original: the clamp lands on the physical value it names
    corrected = apply_postcondition!!(
        copy(u), u_prev, FakeCache(tprob, (; postcondition = Hc))
    )
    @test _from_unbounded.(corrected, lb, ub) ≈ [1.0]

    # Transformed: the same clamp applies to the unconstrained coordinate, which is a
    # different physical correction
    spec_t = PostconditionSpecifier(Hc; space = PostconditionSpace.Transformed)
    corrected_t = apply_postcondition!!(
        copy(u), u_prev, FakeCache(tprob, (; postcondition = spec_t))
    )
    @test corrected_t ≈ clamp.(u, 0.5, 1.0)
    @test !isapprox(_from_unbounded.(corrected_t, lb, ub), [1.0])

    # the mapped path is dispatched on the in-place convention rather than branching on
    # it, so the solver's `cache.u = apply_postcondition!!(...)` sees a concrete type
    @test (
        @inferred apply_postcondition!!(
            copy(u), u_prev, FakeCache(tprob, (; postcondition = Hc))
        )
    ) isa Vector{Float64}

    # in-place: the transformed iterate is still mutated in place and returned
    u_iip = copy(u)
    @test apply_postcondition!!(
        u_iip, u_prev, FakeCache(tprob_iip, (; postcondition = Hc_iip))
    ) === u_iip
    @test _from_unbounded.(u_iip, lb, ub) ≈ [1.0]

    # IIP original-space path must reuse the BoundedWrapper temps, not allocate the
    # two mapped-back buffers on every commit
    u_iip2 = copy(u)
    fc_iip = FakeCache(tprob_iip, (; postcondition = Hc_iip))
    apply_postcondition!!(u_iip2, u_prev, fc_iip)  # warm up
    u_iip2 .= u
    allocs = @allocated apply_postcondition!!(u_iip2, u_prev, fc_iip)
    @test allocs == 0

    # the previous iterate reaches the corrector in the original variable too
    seen = Ref(NaN)
    Hprev = (up, uprev, p, cache) -> (seen[] = uprev[1]; up)
    apply_postcondition!!(copy(u), u_prev, FakeCache(tprob, (; postcondition = Hprev)))
    @test seen[] ≈ 2.0

    # a correction landing exactly *on* a bound is at infinity in the transformed
    # variable, so it has to be nudged into the interior before the inverse map
    Hbound = (up, uprev, p, cache) -> [ub[1]]
    at_bound = apply_postcondition!!(
        copy(u), u_prev, FakeCache(tprob, (; postcondition = Hbound))
    )
    @test all(isfinite, at_bound)
    @test _from_unbounded.(at_bound, lb, ub) ≈ ub

    # without bounds there is no coordinate change and the declaration is inert
    prob = NonlinearProblem(F, [1.0], 2.0)
    spec = PostconditionSpecifier(Hc; space = PostconditionSpace.Transformed)
    @test apply_postcondition!!([3.0], [2.0], FakeCache(prob, (; postcondition = spec))) ≈
        [1.0]
end

end
