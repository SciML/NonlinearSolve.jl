@testitem "Aqua" tags = [:core] begin
    using Aqua, NonlinearSolveQuasiNewton

    Aqua.test_all(
        NonlinearSolveQuasiNewton;
        piracies = false, ambiguities = false, stale_deps = false, deps_compat = false
    )
    Aqua.test_stale_deps(NonlinearSolveQuasiNewton; ignore = [:SciMLJacobianOperators])
    Aqua.test_deps_compat(NonlinearSolveQuasiNewton; ignore = [:SciMLJacobianOperators])
    Aqua.test_piracies(NonlinearSolveQuasiNewton)
    Aqua.test_ambiguities(NonlinearSolveQuasiNewton; recursive = false)
end

@testitem "Explicit Imports" tags = [:core] begin
    using ExplicitImports, NonlinearSolveQuasiNewton

    @test check_no_implicit_imports(
        NonlinearSolveQuasiNewton; skip = (Base, Core, SciMLBase)
    ) === nothing
    @test check_no_stale_explicit_imports(NonlinearSolveQuasiNewton) === nothing
    @test check_all_qualified_accesses_via_owners(NonlinearSolveQuasiNewton) === nothing
end
