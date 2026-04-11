using InteractiveUtils, Test

@info sprint(InteractiveUtils.versioninfo)

# Changing any code here triggers all the other tests to be run. So we intentionally
# keep the tests here minimal.
@testset "NonlinearSolveBase.jl" begin
    @testset "Aqua" begin
        using Aqua, NonlinearSolveBase
        using NonlinearSolveBase: AbstractNonlinearProblem, NonlinearProblem

        Aqua.test_all(
            NonlinearSolveBase; piracies = false, ambiguities = false, stale_deps = false
        )
        Aqua.test_stale_deps(NonlinearSolveBase; ignore = [:TimerOutputs])
        Aqua.test_piracies(NonlinearSolveBase, treat_as_own = [AbstractNonlinearProblem, NonlinearProblem])
        Aqua.test_ambiguities(NonlinearSolveBase; recursive = false)
    end

    @testset "Explicit Imports" begin
        import ForwardDiff, SparseArrays
        using ExplicitImports, NonlinearSolveBase

        @test check_no_implicit_imports(NonlinearSolveBase; skip = (Base, Core)) === nothing
        # Ignore SciMLLogging types used by @verbosity_specifier macro (ExplicitImports can't track macro usage)
        @test check_no_stale_explicit_imports(
            NonlinearSolveBase;
            ignore = (:AbstractMessageLevel, :AbstractVerbositySpecifier, :All, :Detailed, :Minimal, :None, :Standard, :SciMLLogging)
        ) === nothing
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
                u_unaliased, SciMLBase.ReturnCode.Default, 1.0e-8, 1.0e-8, Inf, mode,
                nothing, nothing, 0, nothing, nothing, nothing, nothing, nothing, false
            )
            du = [1.0, 1.0]
            u = [1.1, 1.1]
            @test_nowarn SciMLBase.reinit!(cache, du, u)
        end
    end

    @testset "standardize_forwarddiff_tag leaves unwrapped problems alone (#3381)" begin
        # Regression for SciML/OrdinaryDiffEq.jl#3381: under FullSpecialize (or
        # any path where the user function was not wrapped via AutoSpecialize),
        # `standardize_forwarddiff_tag` must return the AD backend unchanged
        # and NOT substitute in a canonical `Tag{NonlinearSolveTag, Float64}`.
        # Substituting the pre-baked canonical tag used to drag in ForwardDiff's
        # precompile-time `@generated tagcount` literal for that exact type and
        # `≺`-reverse against nested tags created later inside an inner ODE
        # solve, which crashed `setindex!(du, ...)` in the user body with a
        # `Float64(::nested_dual)` MethodError.
        using NonlinearSolveBase, SciMLBase, ADTypes, ForwardDiff

        # FullSpecialize nonlinear function with Vector{Float64} u0.
        resid!(du, u, p) = (du .= u .- p; nothing)
        f = NonlinearFunction{true, SciMLBase.FullSpecialize}(
            resid!, resid_prototype = zeros(2)
        )
        prob = NonlinearLeastSquaresProblem(f, [1.0, 2.0])

        ad = AutoForwardDiff()
        out = NonlinearSolveBase.standardize_forwarddiff_tag(ad, prob)
        @test out === ad

        # AutoPolyesterForwardDiff path must also leave `ad` alone when the
        # function is not wrapped.
        adp = AutoPolyesterForwardDiff()
        outp = NonlinearSolveBase.standardize_forwarddiff_tag(adp, prob)
        @test outp === adp
    end
end
