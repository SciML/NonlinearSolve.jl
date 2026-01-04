@testitem "Aqua" tags = [:core] begin
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
end

@testitem "Explicit Imports" tags = [:core] begin
    import ReverseDiff, Tracker, StaticArrays, Zygote
    using ExplicitImports, SimpleNonlinearSolve

    @test check_no_implicit_imports(SimpleNonlinearSolve; skip = (Base, Core)) === nothing
    @test check_no_stale_explicit_imports(SimpleNonlinearSolve) === nothing
    @test check_all_qualified_accesses_via_owners(SimpleNonlinearSolve) === nothing
end
