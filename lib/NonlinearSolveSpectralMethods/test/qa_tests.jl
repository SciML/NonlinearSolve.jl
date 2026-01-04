@testitem "Aqua" tags = [:core] begin
    using Aqua, NonlinearSolveSpectralMethods

    Aqua.test_all(
        NonlinearSolveSpectralMethods;
        piracies = false, ambiguities = false, stale_deps = false, deps_compat = false
    )
    Aqua.test_stale_deps(NonlinearSolveSpectralMethods; ignore = [:SciMLJacobianOperators])
    Aqua.test_deps_compat(NonlinearSolveSpectralMethods; ignore = [:SciMLJacobianOperators])
    Aqua.test_piracies(NonlinearSolveSpectralMethods)
    Aqua.test_ambiguities(NonlinearSolveSpectralMethods; recursive = false)
end

@testitem "Explicit Imports" tags = [:core] begin
    using ExplicitImports, NonlinearSolveSpectralMethods

    @test check_no_implicit_imports(
        NonlinearSolveSpectralMethods; skip = (Base, Core, SciMLBase)
    ) === nothing
    @test check_no_stale_explicit_imports(NonlinearSolveSpectralMethods) === nothing
    @test check_all_qualified_accesses_via_owners(NonlinearSolveSpectralMethods) === nothing
end
