@testitem "Aqua" tags=[:core] begin
    using Aqua, BracketingNonlinearSolve

    Aqua.test_all(BracketingNonlinearSolve; piracies = false, ambiguities = false)
    Aqua.test_piracies(BracketingNonlinearSolve; treat_as_own = [IntervalNonlinearProblem])
    Aqua.test_ambiguities(BracketingNonlinearSolve; recursive = false)
end

@testitem "Explicit Imports" tags=[:core] begin
    import ForwardDiff
    using ExplicitImports, BracketingNonlinearSolve

    @test check_no_implicit_imports(BracketingNonlinearSolve; skip = (Base, Core)) === nothing
    @test check_no_stale_explicit_imports(BracketingNonlinearSolve) === nothing
    @test check_all_qualified_accesses_via_owners(BracketingNonlinearSolve) === nothing
end
