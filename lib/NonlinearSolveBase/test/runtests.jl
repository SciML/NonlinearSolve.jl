using InteractiveUtils, Test

@info sprint(InteractiveUtils.versioninfo)

# Changing any code here triggers all the other tests to be run. So we intentionally
# keep the tests here minimal.
@testset "NonlinearSolveBase.jl" begin
    @testset "Aqua" begin
        using Aqua, NonlinearSolveBase

        Aqua.test_all(
            NonlinearSolveBase; piracies = false, ambiguities = false, stale_deps = false
        )
        Aqua.test_stale_deps(NonlinearSolveBase; ignore = [:TimerOutputs])
        Aqua.test_piracies(NonlinearSolveBase)
        Aqua.test_ambiguities(NonlinearSolveBase; recursive = false)
    end

    @testset "Explicit Imports" begin
        import ForwardDiff, SparseArrays, DiffEqBase
        using ExplicitImports, NonlinearSolveBase

        @test check_no_implicit_imports(NonlinearSolveBase; skip = (Base, Core)) === nothing
        @test check_no_stale_explicit_imports(NonlinearSolveBase) === nothing
        @test check_all_qualified_accesses_via_owners(NonlinearSolveBase) === nothing
    end

    @testset "Banded Matrix vcat" begin
        using BandedMatrices, LinearAlgebra, SparseArrays

        b = BandedMatrix(Ones(5, 5), (1, 1))
        d = Diagonal(ones(5, 5))

        @test NonlinearSolveBase.Utils.faster_vcat(b, d) == vcat(sparse(b), d)
    end

    @testset "Termination Conditions" begin
        using NonlinearSolveBase, SciMLBase
        @testset "reinit! with AbsTerminationMode" begin
            mode = NonlinearSolveBase.AbsTerminationMode()
            u_unaliased = nothing
            T = Float64
            cache = NonlinearSolveBase.NonlinearTerminationModeCache(
                u_unaliased, SciMLBase.ReturnCode.Default, 1e-8, 1e-8, Inf, mode,
                nothing, nothing, 0, nothing, nothing, nothing, nothing, nothing, false
            )
            du = [1.0, 1.0]
            u = [1.1, 1.1]
            @test_nowarn SciMLBase.reinit!(cache, du, u)
        end
    end
end
