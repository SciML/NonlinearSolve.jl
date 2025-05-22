@testitem "Simple Adjoint Test" tags=[:adjoint] begin
    using ForwardDiff, Zygote, DiffEqBase

    ff(u, p) = u^2 .- p[1]

    function solve_nlprob(p)
        prob = IntervalNonlinearProblem{false}(ff, (1.0, 3.0), p)
        sol = solve(prob, Broyden())
        res = sol isa AbstractArray ? sol : sol.u
        return sum(abs2, res)
    end

    p = [2.0, 2.0]

    ∂p_zygote = Zygote.gradient(solve_nlprob, p)
    ∂p_forwarddiff = ForwardDiff.gradient(solve_nlprob, p)
    @test ∂p_zygote ≈ ∂p_forwarddiff
end
