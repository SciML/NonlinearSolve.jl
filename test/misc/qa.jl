using NonlinearSolve, Aqua

@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(NonlinearSolve)
    Aqua.test_ambiguities(NonlinearSolve; recursive = false)
    Aqua.test_deps_compat(NonlinearSolve)
    Aqua.test_piracies(NonlinearSolve,
        treat_as_own = [NonlinearProblem, NonlinearLeastSquaresProblem])
    Aqua.test_project_extras(NonlinearSolve)
    Aqua.test_stale_deps(NonlinearSolve)
    Aqua.test_unbound_args(NonlinearSolve)
    Aqua.test_undefined_exports(NonlinearSolve)
end
