using SCCNonlinearSolve

using Aqua, SCCNonlinearSolve

Aqua.test_all(
    SCCNonlinearSolve;
    piracies = false, ambiguities = false, stale_deps = false, deps_compat = false
)
Aqua.test_stale_deps(
    SCCNonlinearSolve; ignore = [:SciMLJacobianOperators, :NonlinearSolveBase]
)
Aqua.test_deps_compat(
    SCCNonlinearSolve; ignore = [:SciMLJacobianOperators, :NonlinearSolveBase]
)
Aqua.test_piracies(
    SCCNonlinearSolve; treat_as_own = [SCCNonlinearSolve.SciMLBase.solve]
)
Aqua.test_ambiguities(SCCNonlinearSolve; recursive = false)
