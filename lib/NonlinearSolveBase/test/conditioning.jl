module ConditioningTests

using Test
using NonlinearSolveBase
using SciMLBase
using NonlinearSolveBase: needs_conditioning, transform_conditioned_problem,
    PreconditionWrapper, ComposedPrecondition, apply_postcondition!!,
    has_precondition, has_postcondition

struct UnsupportedAlg <: NonlinearSolveBase.AbstractNonlinearSolveAlgorithm end
struct SupportedAlg <: NonlinearSolveBase.AbstractNonlinearSolveAlgorithm end
NonlinearSolveBase.supports_postcondition(::SupportedAlg) = true

@testset "needs_conditioning" begin
    f = (u, p) -> u .^ 2 .- p
    @test !needs_conditioning(NonlinearProblem(f, [1.0], 2.0))
    G = (fu, u, p) -> asinh.(fu)
    @test needs_conditioning(NonlinearProblem(NonlinearFunction(f; precondition = G), [1.0], 2.0))
    H = (up, uprev, p) -> up
    @test needs_conditioning(NonlinearProblem(NonlinearFunction(f; postcondition = H), [1.0], 2.0))
end

@testset "precondition composes into the residual (oop and iip)" begin
    f = (u, p) -> u .^ 2 .- p
    G = (fu, u, p) -> asinh.(fu)
    prob = NonlinearProblem(NonlinearFunction(f; precondition = G), [2.0], 3.0)
    tprob = transform_conditioned_problem(prob, nothing)
    @test tprob.f.f isa PreconditionWrapper
    @test tprob.f.precondition isa ComposedPrecondition
    @test tprob.f.f([2.0], 3.0) ≈ asinh.(f([2.0], 3.0))

    fiip = (du, u, p) -> (du .= u .^ 2 .- p; nothing)
    Giip = (fu, u, p) -> (fu .= asinh.(fu); nothing)
    probi = NonlinearProblem(NonlinearFunction(fiip; precondition = Giip), [2.0], 3.0)
    tprobi = transform_conditioned_problem(probi, nothing)
    du = zeros(1)
    tprobi.f.f(du, [2.0], 3.0)
    @test du ≈ asinh.([2.0^2 - 3.0])
end

@testset "transform is idempotent through remake field merging" begin
    f = (u, p) -> u .^ 2 .- p
    G = (fu, u, p) -> asinh.(fu)
    prob = NonlinearProblem(NonlinearFunction(f; precondition = G), [2.0], 3.0)
    tprob = transform_conditioned_problem(prob, nothing)
    # `remake(prob; f = new_f)` lets `nothing` fields of the new function fall back to
    # the old function's values, so the cleared hook must survive as a non-`nothing`
    # marker or the next pass would re-wrap.
    @test !has_precondition(tprob)
    @test !needs_conditioning(tprob)
    tprob2 = transform_conditioned_problem(tprob, nothing)
    @test !(tprob2.f.f isa PreconditionWrapper && tprob2.f.f.f isa PreconditionWrapper)
end

@testset "postcondition corrects the initial guess" begin
    f = (u, p) -> u .^ 2 .- p
    H = (up, uprev, p) -> clamp.(up, 0.5, 1.0)
    prob = NonlinearProblem(NonlinearFunction(f; postcondition = H), [2.0], 3.0)
    tprob = transform_conditioned_problem(prob, SupportedAlg())
    @test tprob.u0 ≈ [1.0]
    @test has_postcondition(tprob)

    Hiip = (up, uprev, p) -> (up .= clamp.(up, 0.5, 1.0); nothing)
    fiip = (du, u, p) -> (du .= u .^ 2 .- p; nothing)
    probi = NonlinearProblem(NonlinearFunction(fiip; postcondition = Hiip), [2.0], 3.0)
    tprobi = transform_conditioned_problem(probi, SupportedAlg())
    @test tprobi.u0 ≈ [1.0]
    @test probi.u0 ≈ [2.0]
end

@testset "apply_postcondition!! follows the in-place convention" begin
    f = (u, p) -> u .^ 2 .- p
    H = (up, uprev, p) -> up .+ uprev
    prob = NonlinearProblem(NonlinearFunction(f; postcondition = H), [1.0], 2.0)
    @test apply_postcondition!!([3.0], [1.0], prob) ≈ [4.0]

    fiip = (du, u, p) -> (du .= u .^ 2 .- p; nothing)
    Hiip = (up, uprev, p) -> (up .+= uprev; nothing)
    probi = NonlinearProblem(NonlinearFunction(fiip; postcondition = Hiip), [1.0], 2.0)
    u = [3.0]
    @test apply_postcondition!!(u, [1.0], probi) === u
    @test u ≈ [4.0]

    plain = NonlinearProblem(f, [1.0], 2.0)
    u2 = [3.0]
    @test apply_postcondition!!(u2, [1.0], plain) === u2
    @test u2 ≈ [3.0]
end

@testset "guards: unsupported algorithm and bounds" begin
    f = (u, p) -> u .^ 2 .- p
    H = (up, uprev, p) -> up
    fn = NonlinearFunction(f; postcondition = H)
    prob = NonlinearProblem(fn, [1.0], 2.0)
    @test_throws ArgumentError transform_conditioned_problem(prob, UnsupportedAlg())
    tprob = transform_conditioned_problem(prob, SupportedAlg())
    @test tprob.u0 ≈ [1.0]

    prob_bounds = NonlinearProblem(fn, [1.0], 2.0; lb = [0.0], ub = [2.0])
    @test_throws ArgumentError transform_conditioned_problem(prob_bounds, SupportedAlg())
end

end
