@testitem "Adjoint Tests" tags = [:adjoint] begin
    using ForwardDiff, ReverseDiff, SciMLSensitivity, Tracker, Zygote, NonlinearSolve, Test

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
    ∂p_enzyme = Enzyme.gradient(Enzyme.Reverse, solve_nlprob, p)[1]
    @test ∂p_zygote ≈ ∂p_tracker ≈ ∂p_reversediff ≈ ∂p_enzyme
    @test ∂p_zygote ≈ ∂p_forwarddiff ≈ ∂p_tracker ≈ ∂p_reversediff ≈ ∂p_enzyme
end

@testitem "Simple Adjoint Test" tags=[:adjoint] begin
    using ForwardDiff, Zygote, BracketingNonlinearSolve

    ff(u, p) = u^2 .- p[1]

    function solve_nlprob(p)
        prob = IntervalNonlinearProblem{false}(ff, (1.0, 3.0), p)
        sol = solve(prob, Bisection())
        res = sol isa AbstractArray ? sol : sol.u
        return sum(abs2, res)
    end

    p = [2.0, 2.0]

    ∂p_zygote = only(Zygote.gradient(solve_nlprob, p))
    ∂p_forwarddiff = ForwardDiff.gradient(solve_nlprob, p)
    @test ∂p_zygote ≈ ∂p_forwarddiff
end
