@testitem "Simple Adjoint Test" begin
    using ForwardDiff, SciMLSensitivity, Zygote

    ff(u, p) = u .^ 2 .- p

    function solve_nlprob(p)
        prob = NonlinearProblem{false}(ff, [1.0, 2.0], p)
        return sum(abs2, solve(prob, SimpleNewtonRaphson()).u)
    end

    p = [3.0, 2.0]

    @test only(Zygote.gradient(solve_nlprob, p)) ≈ ForwardDiff.gradient(solve_nlprob, p)
end
