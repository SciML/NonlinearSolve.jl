using NonlinearSolveSciPy

using SciMLBase, NonlinearSolveSciPy
success = false
try
    import PythonCall
    PythonCall.pyimport("scipy.optimize")
    success = true
catch
end
if success
    function fvec(u, p)
        return [2 - 2u[1]; u[1] - 4u[2]]
    end
    prob_vec = NonlinearProblem(fvec, zeros(2))
    sol_vec = solve(prob_vec, SciPyRoot())
    @test SciMLBase.successful_retcode(sol_vec)
    @test maximum(abs, sol_vec.resid) < 1.0e-6

    fscalar(x, p) = x^2 - 2
    prob_interval = IntervalNonlinearProblem(fscalar, (1.0, 2.0))
    sol_scalar = solve(prob_interval, SciPyRootScalar())
    @test SciMLBase.successful_retcode(sol_scalar)
    @test abs(sol_scalar.u - sqrt(2)) < 1.0e-6
else
    @test true
end
