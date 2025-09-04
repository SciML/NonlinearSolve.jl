@testitem "Adjoint Tests" tags = [:adjoint] begin
    using ForwardDiff, ReverseDiff, SciMLSensitivity, Tracker, Zygote, Enzyme

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
