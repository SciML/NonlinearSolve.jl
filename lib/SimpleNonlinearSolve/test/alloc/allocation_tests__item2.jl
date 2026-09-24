using SimpleNonlinearSolve, StaticArrays, AllocCheck
using SciMLBase

# The whole continuation sweep — anchor, adaptive predictor-corrector loop, and the
# returned solution — must be allocation-free for StaticArray and scalar states. Every
# return path builds the same concrete NonlinearSolution type, which is what makes the
# escaping return stack-allocatable.
homotopy_f(u, p, λ) = @. (1 - λ) * (u - p) + λ * (u^2 - p)

@check_allocs hsweep(prob, alg) = SciMLBase.solve(prob, alg)

hprob_scalar = HomotopyProblem(homotopy_f, 4.0, 4.0)
hprob_sa = HomotopyProblem(homotopy_f, @SVector[4.0], 4.0)

@testset "SimpleHomotopySweep" begin
    for prob in (hprob_scalar, hprob_sa)
        try
            sol = hsweep(prob, SimpleHomotopySweep())
            @test SciMLBase.successful_retcode(sol)
        catch e
            @error e
            @test false
        end
    end
end
