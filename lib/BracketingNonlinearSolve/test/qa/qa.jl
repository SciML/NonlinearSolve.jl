using SciMLTesting, BracketingNonlinearSolve, Test
import ForwardDiff

run_qa(
    BracketingNonlinearSolve;
    explicit_imports = true,
    aqua_kwargs = (;
        stale_deps = (; ignore = [:SciMLJacobianOperators]),
        deps_compat = (; ignore = [:SciMLJacobianOperators]),
        piracies = (; treat_as_own = [IntervalNonlinearProblem]),
        ambiguities = (; recursive = false),
    ),
    ei_kwargs = (;
        # @SciMLMessage / AbstractVerbosityPreset are owned by SciMLLogging and
        # re-exported through NonlinearSolveBase (where they are imported from).
        all_explicit_imports_via_owners = (;
            ignore = (Symbol("@SciMLMessage"), :AbstractVerbosityPreset),
        ),
        # Still non-public in their owning packages after the make-public round:
        #   SciMLBase: __solve;  ForwardDiff (ChainRulesCore/ForwardDiff ext): partials, value
        all_qualified_accesses_are_public = (;
            ignore = (:__solve, :partials, :value),
        ),
        # Still non-public where they are imported from (@SciMLMessage /
        # AbstractVerbosityPreset are SciMLLogging-owned re-exports flagged here too):
        #   NonlinearSolveBase: @SciMLMessage, AbstractNonlinearSolveAlgorithm,
        #     AbstractVerbosityPreset
        #   ForwardDiff (ChainRulesCore/ForwardDiff ext): Dual, Partials
        all_explicit_imports_are_public = (;
            ignore = (
                Symbol("@SciMLMessage"), :AbstractNonlinearSolveAlgorithm,
                :AbstractVerbosityPreset, :Dual, :Partials,
            ),
        ),
    ),
)
