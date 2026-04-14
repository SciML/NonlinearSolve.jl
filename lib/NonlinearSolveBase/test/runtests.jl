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

    @testset "maybe_wrap_nonlinear_f wraps IIP array problems regardless of eltype" begin
        # Regression: wrapping must not be gated on `Vector{Float64}`. The
        # `ForwardDiff`-aware `wrapfun_iip` builds Dual-eltype signatures over
        # `similar(u0, ::DualT)`, so it works for any `AbstractArray` state —
        # including Dual-eltype u0 (e.g. `Vector{Dual}` after `promote_u0`
        # promotes an outer-AD nested-ForwardDiff NLLS problem) and
        # multi-dimensional arrays (e.g. `Array{Float64, 3}` for a Brusselator
        # 2D residual). Gating on `Vector{Float64}` was unnecessary and was
        # removed together with its sibling gate in `standardize_forwarddiff_tag`.
        using NonlinearSolveBase, SciMLBase, ForwardDiff

        resid!(du, u, p) = (du .= vec(u) .- (p isa Tuple ? collect(p)[1:length(u)] : p); nothing)
        f = NonlinearFunction{true, SciMLBase.AutoSpecialize}(
            resid!, resid_prototype = zeros(2)
        )

        # Vector{Float64} u0 — wraps.
        prob_f64 = NonlinearProblem(f, [1.0, 2.0], [0.5, 0.25])
        @test NonlinearSolveBase.is_fw_wrapped(
            NonlinearSolveBase.maybe_wrap_nonlinear_f(prob_f64)
        )

        # Vector{Dual} u0 — also wraps (ForwardDiff ext `wrapfun_iip` builds
        # a nested-Dual signature from `similar(u0, dualgen(eltype(u0)))`).
        DualF = ForwardDiff.Dual{ForwardDiff.Tag{typeof(identity), Float64}, Float64, 2}
        u0_dual = DualF[DualF(1.0), DualF(2.0)]
        p_dual = DualF[DualF(0.5), DualF(0.25)]
        prob_dual = NonlinearProblem(f, u0_dual, p_dual)
        @test NonlinearSolveBase.is_fw_wrapped(
            NonlinearSolveBase.maybe_wrap_nonlinear_f(prob_dual)
        )

        # Array{Float64, 3} u0 — wraps (VdT derived via `similar` respects
        # the user's concrete array kind and ndims).
        f3 = NonlinearFunction{true, SciMLBase.AutoSpecialize}(resid!)
        u3d = zeros(2, 2, 2)
        p_tup = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        prob_3d = NonlinearProblem(f3, u3d, p_tup)
        @test NonlinearSolveBase.is_fw_wrapped(
            NonlinearSolveBase.maybe_wrap_nonlinear_f(prob_3d)
        )
    end
end
