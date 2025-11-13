@testitem "Adjoint Tests" tags = [:nopre] begin
    using ForwardDiff, ReverseDiff, SciMLSensitivity, Tracker, Zygote, Mooncake

    # Disable Enzyme on Julia 1.12+ due to compatibility issues
    # To re-enable: change condition to `true` or `VERSION < v"1.13"`
    const ENZYME_ENABLED = VERSION < v"1.12"

    if ENZYME_ENABLED
        using Enzyme
    end

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

    if ENZYME_ENABLED
        ∂p_enzyme = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), solve_nlprob, p)[1]
        @test ∂p_zygote ≈ ∂p_tracker ≈ ∂p_reversediff ≈ ∂p_enzyme
        @test ∂p_zygote ≈ ∂p_forwarddiff ≈ ∂p_tracker ≈ ∂p_reversediff ≈ ∂p_enzyme
    else
        @test ∂p_zygote ≈ ∂p_tracker ≈ ∂p_reversediff
        @test ∂p_zygote ≈ ∂p_forwarddiff ≈ ∂p_tracker ≈ ∂p_reversediff
    end

    cache = Mooncake.prepare_gradient_cache(solve_nlprob, p)
    ∂p_mooncake = Mooncake.value_and_gradient!!(cache, solve_nlprob, p)[2][2]
    @test_broken ∂p_forwarddiff ≈ ∂p_mooncake
end
