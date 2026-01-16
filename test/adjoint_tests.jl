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

@testitem "Reactant AD Tests" tags = [:nopre] begin
    using Reactant, Enzyme, ForwardDiff

    # Test basic nonlinear solve with Reactant arrays
    ff(u, p) = u .^ 2 .- p

    function solve_nlprob_simple(p)
        # Use SimpleNewtonRaphson for better Reactant compatibility
        prob = NonlinearProblem{false}(ff, [1.0, 2.0], p)
        sol = solve(prob, SimpleNewtonRaphson())
        res = sol isa AbstractArray ? sol : sol.u
        return sum(abs2, res)
    end

    p = [3.0, 2.0]

    # Test that the solver works with regular arrays first
    result = solve_nlprob_simple(p)
    @test result ≈ 5.0  # sum(abs2, [sqrt(3), sqrt(2)])

    # Reference gradient using ForwardDiff
    ∂p_forwarddiff = ForwardDiff.gradient(solve_nlprob_simple, p)

    # Test forward mode gradient with Reactant
    p_rarray = Reactant.to_rarray(Float64[3.0, 2.0])

    function reactant_forward_grad(p)
        return Enzyme.gradient(Enzyme.Forward, solve_nlprob_simple, p)
    end

    # Compile and run with Reactant
    ∂p_reactant_fwd = @jit reactant_forward_grad(p_rarray)
    @test ∂p_reactant_fwd isa AbstractArray
    @test Array(∂p_reactant_fwd) ≈ ∂p_forwarddiff atol = 1e-6

    # Test reverse mode gradient with Reactant
    function reactant_reverse_grad(p)
        return Enzyme.gradient(Enzyme.Reverse, solve_nlprob_simple, p)
    end

    ∂p_reactant_rev = @jit reactant_reverse_grad(p_rarray)
    @test ∂p_reactant_rev isa AbstractArray
    @test Array(∂p_reactant_rev) ≈ ∂p_forwarddiff atol = 1e-6
end

@testitem "Reactant AD with SimpleBroyden" tags = [:nopre] begin
    using Reactant, Enzyme, ForwardDiff

    ff(u, p) = u .^ 2 .- p

    function solve_nlprob_broyden(p)
        prob = NonlinearProblem{false}(ff, [1.0, 2.0], p)
        sol = solve(prob, SimpleBroyden())
        res = sol isa AbstractArray ? sol : sol.u
        return sum(abs2, res)
    end

    p = [3.0, 2.0]

    # Test that solver works correctly
    result_broyden = solve_nlprob_broyden(p)
    @test result_broyden ≈ 5.0 atol = 1e-6

    # Reference gradient
    ∂p_forwarddiff = ForwardDiff.gradient(solve_nlprob_broyden, p)

    # Test with Reactant
    p_rarray = Reactant.to_rarray(Float64[3.0, 2.0])

    function reactant_broyden_grad(p)
        return Enzyme.gradient(Enzyme.Reverse, solve_nlprob_broyden, p)
    end

    ∂p_broyden = @jit reactant_broyden_grad(p_rarray)
    @test Array(∂p_broyden) ≈ ∂p_forwarddiff atol = 1e-5
end

@testitem "Reactant AD with NewtonRaphson" tags = [:nopre] begin
    using Reactant, Enzyme, ForwardDiff

    # Test with full NewtonRaphson solver
    ff(u, p) = u .^ 2 .- p

    function solve_nlprob_newton(p)
        prob = NonlinearProblem{false}(ff, [1.0, 2.0], p)
        sol = solve(prob, NewtonRaphson())
        res = sol isa AbstractArray ? sol : sol.u
        return sum(abs2, res)
    end

    p = [3.0, 2.0]
    p_rarray = Reactant.to_rarray(Float64[3.0, 2.0])

    # Reference gradient
    ∂p_forwarddiff = ForwardDiff.gradient(solve_nlprob_newton, p)

    # Test forward mode
    function reactant_newton_fwd(p)
        return Enzyme.gradient(Enzyme.Forward, solve_nlprob_newton, p)
    end

    ∂p_newton_fwd = @jit reactant_newton_fwd(p_rarray)
    @test Array(∂p_newton_fwd) ≈ ∂p_forwarddiff atol = 1e-6

    # Test reverse mode
    function reactant_newton_rev(p)
        return Enzyme.gradient(Enzyme.Reverse, solve_nlprob_newton, p)
    end

    ∂p_newton_rev = @jit reactant_newton_rev(p_rarray)
    @test Array(∂p_newton_rev) ≈ ∂p_forwarddiff atol = 1e-6
end
