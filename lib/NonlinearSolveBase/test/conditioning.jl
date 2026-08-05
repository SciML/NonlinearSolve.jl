module ConditioningTests

using Test
using NonlinearSolveBase
using SciMLBase
using NonlinearSolveBase: needs_conditioning, transform_conditioned_problem,
    PreconditionWrapper, apply_postcondition!!, get_precondition, get_postcondition

struct UnsupportedAlg <: NonlinearSolveBase.AbstractNonlinearSolveAlgorithm end
struct SupportedAlg <: NonlinearSolveBase.AbstractNonlinearSolveAlgorithm end
NonlinearSolveBase.supports_postcondition(::SupportedAlg) = true

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

    H = (up, uprev, p) -> up
    @test get_postcondition(NonlinearProblem(F, [1.0], 2.0; postcondition = H), (;)) === H
end

@testset "needs_conditioning" begin
    prob = NonlinearProblem(F, [1.0], 2.0)
    @test !needs_conditioning(prob, (;))
    @test needs_conditioning(prob, (; precondition = (fu, u, p) -> fu))
    @test needs_conditioning(prob, (; postcondition = (up, uprev, p) -> up))
    @test needs_conditioning(NonlinearProblem(F, [1.0], 2.0; postcondition = (a, b, c) -> a), (;))
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
    H = (up, uprev, p) -> clamp.(up, 0.5, 1.0)
    tprob = transform_conditioned_problem(
        NonlinearProblem(F, [2.0], 3.0), SupportedAlg(), (; postcondition = H)
    )
    @test tprob.u0 ≈ [1.0]

    Hiip = (up, uprev, p) -> (up .= clamp.(up, 0.5, 1.0); nothing)
    probi = NonlinearProblem(FIIP, [2.0], 3.0)
    tprobi = transform_conditioned_problem(probi, SupportedAlg(), (; postcondition = Hiip))
    @test tprobi.u0 ≈ [1.0]
    @test probi.u0 ≈ [2.0]   # the original problem is untouched
end

@testset "apply_postcondition!! follows the in-place convention and passes the cache" begin
    # minimal cache stand-in exposing the two fields the helper reads
    struct FakeCache{P, K}
        prob::P
        kwargs::K
    end
    prob = NonlinearProblem(F, [1.0], 2.0)
    probi = NonlinearProblem(FIIP, [1.0], 2.0)

    H = (up, uprev, p) -> up .+ uprev
    @test apply_postcondition!!([3.0], [1.0], FakeCache(prob, (; postcondition = H))) ≈ [4.0]

    Hiip = (up, uprev, p) -> (up .+= uprev; nothing)
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

@testset "guards: unsupported algorithm and bounds" begin
    H = (up, uprev, p) -> up
    kw = (; postcondition = H)
    prob = NonlinearProblem(F, [1.0], 2.0)
    @test_throws ArgumentError transform_conditioned_problem(prob, UnsupportedAlg(), kw)
    @test transform_conditioned_problem(prob, SupportedAlg(), kw).u0 ≈ [1.0]

    prob_bounds = NonlinearProblem(F, [1.0], 2.0; lb = [0.0], ub = [2.0])
    @test_throws ArgumentError transform_conditioned_problem(prob_bounds, SupportedAlg(), kw)
end

end
