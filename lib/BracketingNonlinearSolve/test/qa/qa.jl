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
        # Still non-public in their owning packages. __solve dropped: now public in
        # SciMLBase.
        #   ForwardDiff (ChainRulesCore/ForwardDiff ext): partials, value
        all_qualified_accesses_are_public = (;
            ignore = (:partials, :value),
        ),
        # Still non-public in their owning packages. @SciMLMessage / AbstractVerbosityPreset
        # dropped: now imported directly from their owner SciMLLogging (public there).
        #   NonlinearSolveBase: AbstractNonlinearSolveAlgorithm
        #   ForwardDiff (ChainRulesCore/ForwardDiff ext): Dual, Partials
        all_explicit_imports_are_public = (;
            ignore = (:AbstractNonlinearSolveAlgorithm, :Dual, :Partials),
        ),
    ),
)
