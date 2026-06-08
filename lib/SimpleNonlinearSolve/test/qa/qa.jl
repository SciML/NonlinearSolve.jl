using SimpleNonlinearSolve

using Aqua, SimpleNonlinearSolve

Aqua.test_all(
    SimpleNonlinearSolve;
    piracies = false, ambiguities = false, stale_deps = false, deps_compat = false
)
Aqua.test_stale_deps(SimpleNonlinearSolve; ignore = [:SciMLJacobianOperators])
Aqua.test_deps_compat(SimpleNonlinearSolve; ignore = [:SciMLJacobianOperators])
Aqua.test_piracies(
    SimpleNonlinearSolve;
    treat_as_own = [
        NonlinearProblem, NonlinearLeastSquaresProblem, IntervalNonlinearProblem,
    ]
)
Aqua.test_ambiguities(SimpleNonlinearSolve; recursive = false)
