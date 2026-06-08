using NonlinearSolveQuasiNewton

using Aqua, NonlinearSolveQuasiNewton

Aqua.test_all(
    NonlinearSolveQuasiNewton;
    piracies = false, ambiguities = false, stale_deps = false, deps_compat = false
)
Aqua.test_stale_deps(NonlinearSolveQuasiNewton; ignore = [:SciMLJacobianOperators])
Aqua.test_deps_compat(NonlinearSolveQuasiNewton; ignore = [:SciMLJacobianOperators])
Aqua.test_piracies(NonlinearSolveQuasiNewton)
Aqua.test_ambiguities(NonlinearSolveQuasiNewton; recursive = false)
