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

@testitem "Explicit Imports" tags=[:misc] begin
    using NonlinearSolve, ADTypes, SimpleNonlinearSolve, SciMLBase
    import BandedMatrices, FastLevenbergMarquardt, FixedPointAcceleration,
           LeastSquaresOptim, MINPACK, NLsolve, NLSolvers, SIAMFANLEquations, SpeedMapping,
           Symbolics, Zygote

    using ExplicitImports

    @test check_no_implicit_imports(NonlinearSolve;
        skip = (NonlinearSolve, Base, Core, SimpleNonlinearSolve, SciMLBase)) === nothing
    @test check_no_stale_explicit_imports(NonlinearSolve) === nothing
    @test check_all_qualified_accesses_via_owners(NonlinearSolve) === nothing
end
