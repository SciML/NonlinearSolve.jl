using NonlinearSolve

using NonlinearSolve, SimpleNonlinearSolve, Aqua

Aqua.test_all(
    NonlinearSolve; ambiguities = false, piracies = false,
    stale_deps = false, deps_compat = false, persistent_tasks = false
)
Aqua.test_ambiguities(NonlinearSolve; recursive = false)
Aqua.test_stale_deps(SimpleNonlinearSolve; ignore = [:SciMLJacobianOperators])
Aqua.test_deps_compat(SimpleNonlinearSolve; ignore = [:SciMLJacobianOperators])
Aqua.test_piracies(
    NonlinearSolve,
    treat_as_own = [
        NonlinearProblem, NonlinearLeastSquaresProblem, SciMLBase.AbstractNonlinearProblem,
        SimpleNonlinearSolve.AbstractSimpleNonlinearSolveAlgorithm,
    ]
)
