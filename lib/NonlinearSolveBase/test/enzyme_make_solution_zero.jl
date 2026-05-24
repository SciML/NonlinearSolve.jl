module EnzymeMakeSolutionZeroTests

using Test
using NonlinearSolveBase
import ChainRulesCore, Enzyme  # triggers NonlinearSolveBaseEnzymeExt
using SciMLBase

const EXT = Base.get_extension(NonlinearSolveBase, :NonlinearSolveBaseEnzymeExt)

@testset "EnzymeExt._make_solution_zero preserves prob.p / prob.u0 aliasing" begin
    # The reverse rule builds the return-value shadow via `make_zero(sol)`.
    # A plain `Enzyme.make_zero` recursively zeros every mutable field of
    # `sol`, including `sol.prob.p` and `sol.prob.u0`, which the outer caller
    # has already registered as active shadows for the `p` / `u0` arguments.
    # Severing that aliasing means any cotangent written into the returned
    # `sol.prob.p` (or `.u0`) by a downstream consumer lands in a dangling
    # buffer instead of the one the outer Enzyme tape is tracking, silently
    # dropping that gradient contribution.
    #
    # `_make_solution_zero` pre-seeds the `make_zero` seen-set with identity
    # entries for `prob.p` and `prob.u0` so the recursion short-circuits and
    # the original buffers are reused verbatim in the shadow.

    f(u, p) = u .^ 2 .- p
    u0 = [1.0, 1.0]
    p = [2.0, 4.0]
    prob = NonlinearProblem{false}(f, u0, p)
    sol = SciMLBase.build_solution(prob, nothing, [1.5, 2.0], zeros(2))

    # Naive `Enzyme.make_zero` allocates fresh buffers for prob.p / prob.u0.
    dsol_naive = Enzyme.make_zero(sol)
    @test objectid(dsol_naive.prob.p) != objectid(sol.prob.p)
    @test objectid(dsol_naive.prob.u0) != objectid(sol.prob.u0)

    # The extension helper keeps them aliased to the primal.
    dsol = EXT._make_solution_zero(sol)
    @test objectid(dsol.prob.p) == objectid(sol.prob.p)
    @test objectid(dsol.prob.u0) == objectid(sol.prob.u0)
    @test dsol.prob.p === sol.prob.p
    @test dsol.prob.u0 === sol.prob.u0
    # The actual derivative-carrying field (u) is still a fresh zero buffer.
    @test objectid(dsol.u) != objectid(sol.u)
    @test all(iszero, dsol.u)

    # Guard the `nothing`/non-mutable path: a problem with `nothing` u0 or p
    # must not crash the pre-seed helper.
    prob_nop = NonlinearProblem{false}(f, u0, nothing)
    sol_nop = SciMLBase.build_solution(prob_nop, nothing, [1.5, 2.0], zeros(2))
    dsol_nop = EXT._make_solution_zero(sol_nop)
    @test dsol_nop.prob.p === nothing
    @test dsol_nop.prob.u0 === sol_nop.prob.u0
end

end
