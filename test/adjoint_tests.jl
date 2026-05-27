@testitem "Adjoint Tests" tags = [:nopre] begin
    # Skip adjoint tests on Julia 1.12+ due to Enzyme/SciMLSensitivity compatibility issues
    # To re-enable: change condition to `false` or `VERSION >= v"1.13"`
    @static if VERSION < v"1.12"
        using ForwardDiff, ReverseDiff, SciMLSensitivity, Tracker, Zygote, Enzyme, Mooncake

        ff(u, p) = u .^ 2 .- p

        function solve_nlprob(p)
            prob = NonlinearProblem{false}(ff, [1.0, 2.0], p)
            sol = solve(prob, NewtonRaphson())
            res = sol isa AbstractArray ? sol : sol.u
            return sum(abs2, res)
        end

        p = [3.0, 2.0]

        ∂p_zygote = only(Zygote.gradient(solve_nlprob, p))
        ∂p_forwarddiff = ForwardDiff.gradient(solve_nlprob, p)
        ∂p_tracker = Tracker.data(only(Tracker.gradient(solve_nlprob, p)))
        ∂p_reversediff = ReverseDiff.gradient(solve_nlprob, p)
        ∂p_enzyme = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), solve_nlprob, p)[1]

        cache = Mooncake.prepare_gradient_cache(solve_nlprob, p)
        ∂p_mooncake = Mooncake.value_and_gradient!!(cache, solve_nlprob, p)[2][2]

        @test ∂p_zygote ≈ ∂p_tracker ≈ ∂p_reversediff ≈ ∂p_enzyme
        @test ∂p_zygote ≈ ∂p_forwarddiff ≈ ∂p_tracker ≈ ∂p_reversediff ≈ ∂p_enzyme
        @test ∂p_forwarddiff ≈ ∂p_mooncake
    else
        @info "Skipping adjoint tests on Julia $(VERSION) - Enzyme/SciMLSensitivity not compatible with 1.12+"
    end
end

@testitem "Enzyme reverse-mode over IIP NonlinearProblem (#939)" tags = [:nopre] begin
    # Regression for SciML/NonlinearSolve.jl#939: differentiating through
    # `solve(::NonlinearProblem, ...)` with `Enzyme.Reverse` (no
    # `set_runtime_activity`) used to fail with `EnzymeRuntimeActivityError`
    # inside `maybe_wrap_nonlinear_f` — the `FunctionWrappersWrapper` built for
    # AutoSpecialize IIP problems emits `ptrtoint`/`store` patterns that defeat
    # Enzyme's static activity analysis, and `set_runtime_activity` was not
    # sufficient to recover correctness. The fix short-circuits the wrap on the
    # outer-AD path via `EnzymeCore.within_autodiff()`. Verified on both
    # Julia 1.11 and 1.12; no version gate is needed (the sibling adjoint test's
    # `VERSION < v"1.12"` gate is about Tracker/Zygote/Mooncake combined, not
    # Enzyme+SciMLSensitivity alone, which works on 1.12).
    using SciMLSensitivity, Enzyme

    function simple_loss(p)
        prob = NonlinearProblem((du, u, p) -> du[1] = u[1] - p[1] + p[2], [0.0], p)
        sol = solve(prob, NewtonRaphson())
        return sum(sol.u)
    end

    p = [2.0, 1.0]
    dp = Enzyme.make_zero(p)
    Enzyme.autodiff(
        Enzyme.Reverse, simple_loss, Enzyme.Active,
        Enzyme.Duplicated(p, dp)
    )
    @test dp ≈ [1.0, -1.0]
end
