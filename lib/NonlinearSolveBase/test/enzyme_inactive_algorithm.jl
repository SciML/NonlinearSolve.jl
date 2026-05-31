module EnzymeInactiveAlgorithmTests

using Test
using NonlinearSolveBase
# `NonlinearSolveBaseEnzymeExt` (which carries the `inactive_type` declaration)
# is triggered by `["ChainRulesCore", "Enzyme"]`; both must be loaded for the
# extension method to be active.
using ChainRulesCore
using Enzyme

# Nonlinear-solve algorithms are pure solver configuration (Vals, Rationals,
# Nothing/Missing sentinels, nested immutable sub-algorithms, and a ForwardDiff
# tag type parameter) with no differentiable floating-point data. When a
# `NonlinearProblem` is solved under `Enzyme.set_runtime_activity(Reverse)` — e.g.
# the ModelingToolkit DAE-initialization solve reached through `solve_up` with the
# default `NonlinearSolvePolyAlgorithm` as a positional `args...` — Enzyme can
# otherwise promote that configuration argument to `Duplicated`, tripping the
# `roots_activep != activep` assertion in `enzyme_custom_setup_args` (or leaving a
# spurious algorithm shadow that corrupts activity bookkeeping for the genuinely
# active inputs). `NonlinearSolveBaseEnzymeExt` declares the whole
# `AbstractNonlinearSolveAlgorithm` hierarchy inactive so Enzyme treats it as
# `Const`. This is a regression test for that declaration.
@testset "Enzyme inactive_type covers the nonlinear-solve algorithm hierarchy" begin
    @test Enzyme.EnzymeRules.inactive_type(NonlinearSolveBase.AbstractNonlinearSolveAlgorithm)

    # The abstract-supertype declaration must cover the default polyalgorithm
    # wrapper (the argument actually promoted in the DAE-init solve) and any
    # concrete sub-algorithm.
    poly = NonlinearSolveBase.NonlinearSolvePolyAlgorithm((nothing,))
    @test poly isa NonlinearSolveBase.AbstractNonlinearSolveAlgorithm
    @test Enzyme.EnzymeRules.inactive_type(typeof(poly))

    # Sanity: a genuinely differentiable type is not affected by this declaration.
    @test !Enzyme.EnzymeRules.inactive_type(Vector{Float64})
end

end
