module BoundsTransformTests

using Test
using NonlinearSolveBase
using SciMLBase
# Load ForwardDiff so PreallocationTools' `FixedSizeDiffCache` dual-cache extension is
# active (the transform builds one for the default/ForwardDiff autodiff path).
import ForwardDiff
using NonlinearSolveBase: transform_bounded_problem, BoundedWrapper, _to_unbounded

# An algorithm type without an `autodiff` field, mirroring `QuasiNewtonAlgorithm`
# (which the default polyalgorithm reaches). `transform_bounded_problem` must not
# assume every algorithm exposes `autodiff`.
struct NoAutodiffAlg end

@testset "bounds transform handles algorithms without an `autodiff` field" begin
    f(u, p) = u .^ 2 .- p
    prob = NonlinearProblem(f, [1.5, 1.5], [2.0, 2.0]; lb = [0.0, 0.0], ub = [3.0, 3.0])

    for alg in (nothing, NoAutodiffAlg())
        tprob = transform_bounded_problem(prob, alg)
        @test tprob.f.f isa BoundedWrapper
        # bounds are removed from the transformed (unconstrained) problem
        @test tprob.lb === nothing
        @test tprob.ub === nothing
        # u0 is mapped into unbounded space
        @test tprob.u0 ≈ _to_unbounded.(prob.u0, prob.lb, prob.ub)
    end
end

end
