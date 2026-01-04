@testitem "Aqua" tags = [:misc] begin
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
end

@testitem "Explicit Imports" tags = [:misc] begin
    using NonlinearSolve, ADTypes, SimpleNonlinearSolve, SciMLBase
    import FastLevenbergMarquardt, FixedPointAcceleration, LeastSquaresOptim, MINPACK,
        NLsolve, NLSolvers, SIAMFANLEquations, SpeedMapping

    using ExplicitImports

    @test check_no_implicit_imports(NonlinearSolve; skip = (Base, Core)) === nothing
    @test check_no_stale_explicit_imports(NonlinearSolve) === nothing
    @test check_all_qualified_accesses_via_owners(NonlinearSolve) === nothing
end
