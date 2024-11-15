@testitem "Aqua" tags=[:core] begin
    using Aqua, SCCNonlinearSolve

    Aqua.test_all(
        SCCNonlinearSolve;
        piracies = false, ambiguities = false, stale_deps = false, deps_compat = false
    )
    Aqua.test_stale_deps(SCCNonlinearSolve; ignore = [:SciMLJacobianOperators])
    Aqua.test_deps_compat(SCCNonlinearSolve; ignore = [:SciMLJacobianOperators])
    Aqua.test_piracies(SCCNonlinearSolve; treat_as_own = [SCCNonlinearSolve.SciMLBase.solve])
    Aqua.test_ambiguities(SCCNonlinearSolve; recursive = false)
end

@testitem "Explicit Imports" tags=[:core] begin
    using ExplicitImports, SciMLBase, SCCNonlinearSolve

    @test check_no_implicit_imports(
        SCCNonlinearSolve; skip = (Base, Core, SciMLBase)
    ) === nothing
    @test check_no_stale_explicit_imports(SCCNonlinearSolve) === nothing
    @test check_all_qualified_accesses_via_owners(SCCNonlinearSolve) === nothing
end
