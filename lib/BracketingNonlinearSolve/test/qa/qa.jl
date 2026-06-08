using BracketingNonlinearSolve

using Aqua, BracketingNonlinearSolve

Aqua.test_all(
    BracketingNonlinearSolve;
    piracies = false, ambiguities = false, stale_deps = false, deps_compat = false
)
Aqua.test_stale_deps(BracketingNonlinearSolve; ignore = [:SciMLJacobianOperators])
Aqua.test_deps_compat(BracketingNonlinearSolve; ignore = [:SciMLJacobianOperators])
Aqua.test_piracies(BracketingNonlinearSolve; treat_as_own = [IntervalNonlinearProblem])
Aqua.test_ambiguities(BracketingNonlinearSolve; recursive = false)
