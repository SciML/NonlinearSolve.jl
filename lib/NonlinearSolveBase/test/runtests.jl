using SafeTestsets, Test, InteractiveUtils
using SciMLTesting

@info sprint(InteractiveUtils.versioninfo)

# SublibraryCI sets NONLINEARSOLVE_TEST_GROUP; fall back to GROUP for local runs.
if !haskey(ENV, "NONLINEARSOLVE_TEST_GROUP") && haskey(ENV, "GROUP")
    ENV["NONLINEARSOLVE_TEST_GROUP"] = ENV["GROUP"]
end

run_tests(;
    env = "NONLINEARSOLVE_TEST_GROUP",
    # All NonlinearSolveBase tests run under the default Core group.
    core = function ()
        @safetestset "Banded Matrix vcat" begin
            using NonlinearSolveBase
            using BandedMatrices, LinearAlgebra, SparseArrays

            b = BandedMatrix(Ones(5, 5), (1, 1))
            d = Diagonal(ones(5, 5))

            @test NonlinearSolveBase.Utils.faster_vcat(b, d) == vcat(sparse(b), d)
        end

        @safetestset "Termination Conditions" begin
            using NonlinearSolveBase, SciMLBase
            using StaticArrays: SA

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

            @testset "termination_condition_result public contract" begin
                using NonlinearSolveBase: AbsNormSafeBestTerminationMode, AbsNormTerminationMode,
                    termination_condition_result
                using SciMLBase: NonlinearProblem, ReturnCode, init

                # `public` (and `Base.ispublic`) only exist on Julia >= 1.11; on the
                # 1.10 LTS `@compat public` expands to nothing, so there is no
                # publicness marker to inspect.
                @static if VERSION ≥ v"1.11"
                    @test Base.ispublic(NonlinearSolveBase, :termination_condition_result)
                end

                prob = NonlinearProblem((u, p) -> u, [1.0])
                internalnorm = x -> maximum(abs, x)
                standard = init(
                    prob, AbsNormTerminationMode(internalnorm), [1.0], [1.0];
                    abstol = 1.0e-8, reltol = 1.0e-8
                )
                @test termination_condition_result(
                    standard, [2.0], 2.0, ReturnCode.Terminated
                ) == ([2.0], 2.0, ReturnCode.Success)

                safe_best = init(
                    prob, AbsNormSafeBestTerminationMode(internalnorm), [1.0], [1.0], 0.0;
                    abstol = 1.0e-8, reltol = 1.0e-8
                )
                @test safe_best([0.0], [0.5], [1.0], 3.0)
                @test termination_condition_result(
                    safe_best, [2.0], 2.0, ReturnCode.Terminated
                ) == ([0.5], 3.0, ReturnCode.Success)
            end

            # `max_stalled_steps` is a documented constructor option on every safe mode,
            # but the non-`Best` ones retain no iterate, so the step scratch cannot be
            # sized from one.
            @testset "max_stalled_steps: $modename, u0::$(typeof(u0))" for (
                        modename, mode,
                    ) in (
                        (
                            "AbsNormSafe", NonlinearSolveBase.AbsNormSafeTerminationMode(
                                Base.Fix1(maximum, abs); max_stalled_steps = 3
                            ),
                        ),
                        (
                            "RelNormSafe", NonlinearSolveBase.RelNormSafeTerminationMode(
                                Base.Fix1(maximum, abs); max_stalled_steps = 3
                            ),
                        ),
                        (
                            "AbsNormSafeBest", NonlinearSolveBase.AbsNormSafeBestTerminationMode(
                                Base.Fix1(maximum, abs); max_stalled_steps = 3
                            ),
                        ),
                        (
                            "RelNormSafeBest", NonlinearSolveBase.RelNormSafeBestTerminationMode(
                                Base.Fix1(maximum, abs); max_stalled_steps = 3
                            ),
                        ),
                    ),
                    u0 in ([1.0, 1.0], 1.0, SA[1.0, 1.0])

                du = u0 isa Number ? 1.0 : (u0 .* 0 .+ 1.0)
                prob = SciMLBase.NonlinearProblem((u, p) -> du, u0)
                cache = SciMLBase.init(
                    prob, mode, du, u0; abstol = 1.0e-8, reltol = 1.0e-8
                )
                # The iterate stops moving while the residual stays far above `abstol`:
                # the stall safeguard is exactly what must fire.
                terminated = false
                for _ in 1:5
                    terminated = cache(du, u0, u0)
                end
                @test terminated
                @test cache.retcode == SciMLBase.ReturnCode.Stalled
            end

            @testset "protective_threshold measures against the initial residual" begin
                mode = NonlinearSolveBase.AbsNormSafeTerminationMode(
                    Base.Fix1(maximum, abs); protective_threshold = 1.0
                )
                prob = SciMLBase.NonlinearProblem((u, p) -> [100.0], [1.0])
                cache = SciMLBase.init(
                    prob, mode, [100.0], [1.0]; abstol = 1.0e-10, reltol = 1.0e-10
                )
                @test cache.initial_objective == 100.0

                @test !cache([1.0e-3], [1.1], [1.0])
                @test cache.initial_objective == 100.0

                # 5.0 is twenty times *below* the initial residual, so nothing about it
                # is divergence.
                @test !cache([5.0], [1.2], [1.1])
                @test cache.retcode != SciMLBase.ReturnCode.Unstable
            end

            @testset "deferred residual helper contracts" begin
                using NonlinearSolveBase: AbstractNonlinearTerminationMode,
                    AbsTerminationMode, NonlinearSolveTrace, RelTerminationMode, TraceMinimal,
                    residual_only_termination_mode, trace_is_active

                struct ResidualOnlyTestMode <: AbstractNonlinearTerminationMode end
                NonlinearSolveBase.residual_only_termination_mode(::ResidualOnlyTestMode) = true

                @static if VERSION ≥ v"1.11"
                    @test Base.ispublic(
                        NonlinearSolveBase, :residual_only_termination_mode
                    )
                    @test Base.ispublic(NonlinearSolveBase, :trace_is_active)
                end

                @test residual_only_termination_mode(AbsTerminationMode())
                @test !residual_only_termination_mode(RelTerminationMode())
                @test residual_only_termination_mode(ResidualOnlyTestMode())
                @test !trace_is_active(nothing)
                @test !trace_is_active(missing)
                @test !trace_is_active(
                    NonlinearSolveTrace(Val(false), Val(false), nothing, TraceMinimal(), nothing)
                )
                @test trace_is_active(
                    NonlinearSolveTrace(Val(true), Val(false), nothing, TraceMinimal(), nothing)
                )
                @test trace_is_active(
                    NonlinearSolveTrace(Val(false), Val(true), nothing, TraceMinimal(), nothing)
                )
            end
        end

        @safetestset "Abstract interface trait contracts" begin
            using NonlinearSolveBase, Test

            mutable struct TraitDescentCache <: NonlinearSolveBase.AbstractDescentCache
                δu::Vector{Float64}
                δus::Vector{Vector{Float64}}
                preinverted_jacobian::Val{false}
                normal_form::Val{false}
                last_step_accepted::Bool
            end

            struct TraitStructure <: NonlinearSolveBase.AbstractApproximateJacobianStructure end
            struct TraitFullStructure <: NonlinearSolveBase.AbstractApproximateJacobianStructure end
            struct TraitInitialization <: NonlinearSolveBase.AbstractJacobianInitialization end
            struct TraitUpdateRule <: NonlinearSolveBase.AbstractApproximateJacobianUpdateRule
                store_inverse_jacobian::Bool
            end
            struct TraitUpdateCache <: NonlinearSolveBase.AbstractApproximateJacobianUpdateRuleCache
                rule::TraitUpdateRule
            end
            struct TraitDamping <: NonlinearSolveBase.AbstractDampingFunction end

            NonlinearSolveBase.requires_normal_form_jacobian(::TraitDamping) = true
            NonlinearSolveBase.requires_normal_form_rhs(::TraitDamping) = false
            NonlinearSolveBase.stores_full_jacobian(::TraitFullStructure) = true

            cache = TraitDescentCache(
                [1.0], [[1.0], [2.0]], Val(false), Val(false), false
            )
            @test NonlinearSolveBase.last_step_accepted(cache) === false
            @test NonlinearSolveBase.preinverted_jacobian(cache) === false
            @test NonlinearSolveBase.normal_form(cache) === false

            @test NonlinearSolveBase.last_step_accepted(
                TraitDescentCache([1.0], [[1.0]], Val(false), Val(false), true)
            )

            @test NonlinearSolveBase.stores_full_jacobian(TraitStructure()) === false
            @test_throws ErrorException NonlinearSolveBase.get_full_jacobian(
                nothing, TraitStructure(), [1.0]
            )
            J = [1.0 0.0; 0.0 1.0]
            @test NonlinearSolveBase.get_full_jacobian(
                nothing, TraitFullStructure(), J
            ) === J
            @test NonlinearSolveBase.jacobian_initialized_preinverted(TraitInitialization()) ===
                false

            rule = TraitUpdateRule(true)
            @test NonlinearSolveBase.store_inverse_jacobian(rule) === true
            @test NonlinearSolveBase.store_inverse_jacobian(TraitUpdateCache(rule)) === true
            @test NonlinearSolveBase.returns_norm_form_damping(TraitDamping()) === true

            @static if VERSION ≥ v"1.11"
                for name in (
                        :last_step_accepted, :preinverted_jacobian, :normal_form,
                        :requires_normal_form_jacobian, :requires_normal_form_rhs,
                        :returns_norm_form_damping, :stores_full_jacobian, :get_full_jacobian,
                        :jacobian_initialized_preinverted, :store_inverse_jacobian,
                    )
                    @test Base.ispublic(NonlinearSolveBase, name)
                end
            end
        end

        @safetestset "standardize_forwarddiff_tag leaves unwrapped problems alone (#3381)" begin
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

        @safetestset "AutoDePSpecialize opaque-p" include("autodepspecialize.jl")

        @safetestset "AutoDespecialize dynamic-p" include("autodespecialize.jl")

        @safetestset "maybe_wrap_nonlinear_f wraps non-dual IIP array problems of any eltype or ndims" begin
            # Wrapping is keyed off `eltype(u0)`: the ForwardDiff-aware `wrapfun_iip`
            # builds Dual-eltype signatures via `similar(u0, ::DualT)`, so it works
            # for any `AbstractArray` state with a non-dual eltype — `Vector{Float64}`,
            # `Array{Float64, 3}` (Brusselator 2D residual), etc. It must NOT wrap
            # when `u0` already carries a `Dual` eltype (which happens whenever
            # `promote_u0` upgrades `u0` against outer-AD Dual parameters, e.g. a
            # nested-ForwardDiff NLLS over the `#445` Hessian case or a
            # `ForwardDiff.derivative(solve, p)` pass with Dual `p`). If wrapping
            # fired in that case, the stored signatures would be keyed off the
            # outer Dual tag and miss the value-typed inner dispatch produced by
            # the forward-diff extension.
            using NonlinearSolveBase, SciMLBase, ForwardDiff

            resid!(du, u, p) = (du .= vec(u); nothing)
            f = NonlinearFunction{true, SciMLBase.AutoSpecialize}(
                resid!, resid_prototype = zeros(2)
            )

            # Vector{Float64} u0 — wraps.
            prob_f64 = NonlinearProblem(f, [1.0, 2.0], [0.5, 0.25])
            @test NonlinearSolveBase.is_fw_wrapped(
                NonlinearSolveBase.maybe_wrap_nonlinear_f(prob_f64)
            )

            # Vector{Dual} u0 — must NOT wrap.
            DualF = ForwardDiff.Dual{ForwardDiff.Tag{typeof(identity), Float64}, Float64, 2}
            u0_dual = DualF[DualF(1.0), DualF(2.0)]
            p_dual = DualF[DualF(0.5), DualF(0.25)]
            prob_dual = NonlinearProblem(f, u0_dual, p_dual)
            @test NonlinearSolveBase.maybe_wrap_nonlinear_f(prob_dual) === f.f
            @test !NonlinearSolveBase.is_fw_wrapped(
                NonlinearSolveBase.maybe_wrap_nonlinear_f(prob_dual)
            )

            # Array{Float64, 3} u0 — wraps (VdT derived via `similar` respects the
            # user's concrete array kind and ndims).
            f3 = NonlinearFunction{true, SciMLBase.AutoSpecialize}(resid!)
            u3d = zeros(2, 2, 2)
            p_tup = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
            prob_3d = NonlinearProblem(f3, u3d, p_tup)
            @test NonlinearSolveBase.is_fw_wrapped(
                NonlinearSolveBase.maybe_wrap_nonlinear_f(prob_3d)
            )
        end

        @safetestset "EnzymeExt _accum_tangent! caches accumulation (#935)" include("enzyme_accum_tangent.jl")

        @safetestset "PolyAlgorithm solution type is concrete (#878)" include("polyalg_solution_type.jl")

        @safetestset "EnzymeExt algorithms are inactive_type" include("enzyme_inactive_algorithm.jl")

        @safetestset "Bounds transform (#955)" include("bounds_transform.jl")

        @safetestset "Nonlinear preconditioning options (#351)" include("conditioning.jl")

        @safetestset "linsolve_identity!! workspace (#1020)" include("linsolve_workspace.jl")

        @safetestset "Linear solver routing" include("linear_solver_routing.jl")

        @safetestset "Descent buffers start defined" include("descent_buffer_init.jl")

        @safetestset "Jacobian and restructure allocation fast paths" include(
            "allocation_fastpaths.jl"
        )

        @safetestset "Operator Jacobian cache dispatch" begin
            using NonlinearSolveBase, SciMLBase, SciMLOperators

            f! = (du, u, p) -> (du .= u; nothing)
            f = NonlinearFunction(f!; jac_prototype = MatrixOperator(ones(1, 1)))
            prob = NonlinearProblem(f, [1.0])
            stats = SciMLBase.NLStats(0, 0, 0, 0, 0)
            cache = NonlinearSolveBase.construct_jacobian_cache(
                prob, nothing, f, [1.0]; stats
            )

            @test_throws ErrorException cache(nothing)
        end

        @safetestset "dampen_jacobian!! touches only the diagonal" include(
            "dampen_jacobian.jl"
        )
        @safetestset "safe_similar hands out zeroed buffers" include("safe_similar.jl")
        @safetestset "Non-finite objective protective break" include(
            "nonfinite_objective.jl"
        )

        @safetestset "enable_timer_outputs precompiles and times (#1224)" include(
            "timer_outputs_enabled.jl"
        )

        return @safetestset "Dense LU refactorization allocations" include(
            "lu_refactorization_allocs.jl"
        )
    end,
    # QA (Aqua/ExplicitImports via SciMLTesting.run_qa) is a dep-adding group: it runs
    # in its own isolated sub-env under test/qa (excluded from the base/Core/All run).
    qa = (;
        env = joinpath(@__DIR__, "qa"),
        body = joinpath(@__DIR__, "qa", "qa.jl"),
    ),
)
