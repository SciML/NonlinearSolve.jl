@testitem "Aqua" tags=[:misc] begin
    using NonlinearSolve, SimpleNonlinearSolve, Aqua

    Aqua.find_persistent_tasks_deps(NonlinearSolve)
    Aqua.test_ambiguities(NonlinearSolve; recursive = false)
    Aqua.test_deps_compat(NonlinearSolve)
    Aqua.test_piracies(NonlinearSolve,
        treat_as_own = [NonlinearProblem, NonlinearLeastSquaresProblem,
            SimpleNonlinearSolve.AbstractSimpleNonlinearSolveAlgorithm])
    Aqua.test_project_extras(NonlinearSolve)
    # Timer Outputs needs to be enabled via Preferences
    Aqua.test_stale_deps(NonlinearSolve; ignore = [:TimerOutputs])
    Aqua.test_unbound_args(NonlinearSolve)
    Aqua.test_undefined_exports(NonlinearSolve)
end
