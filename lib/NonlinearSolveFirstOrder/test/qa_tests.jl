@testitem "Aqua" tags=[:core] begin
    using Aqua, NonlinearSolveFirstOrder

    Aqua.test_all(
        NonlinearSolveFirstOrder;
        piracies = false, ambiguities = false
    )
    Aqua.test_piracies(NonlinearSolveFirstOrder)
    Aqua.test_ambiguities(NonlinearSolveFirstOrder; recursive = false)
end

@testitem "Explicit Imports" tags=[:core] begin
    using ExplicitImports, NonlinearSolveFirstOrder

    @test check_no_implicit_imports(
        NonlinearSolveFirstOrder; skip = (Base, Core, SciMLBase)
    ) === nothing
    @test check_no_stale_explicit_imports(NonlinearSolveFirstOrder) === nothing
    @test check_all_qualified_accesses_via_owners(NonlinearSolveFirstOrder) === nothing
end
