@testitem "SciPyLeastSquares" tags=[:wrappers] begin
    using SciMLBase, NonlinearSolveSciPy
    success = false
    try
        import PythonCall
        PythonCall.pyimport("scipy.optimize")
        success = true
    catch
    end
    if success
        xdata = collect(0:0.1:1)
        ydata = 2.0 .* xdata .+ 1.0
        function residuals(params, p = nothing)
            a, b = params
            return ydata .- (a .* xdata .+ b)
        end
        x0_ls = [1.0, 0.0]
        prob = NonlinearLeastSquaresProblem(residuals, x0_ls)
        sol = solve(prob, SciPyLeastSquaresTRF())
        @test SciMLBase.successful_retcode(sol)
        prob_bounded = NonlinearLeastSquaresProblem(
            residuals, x0_ls; lb = [0.0, -2.0], ub = [
                5.0, 3.0])
        sol2 = solve(prob_bounded, SciPyLeastSquares(method = "trf"))
        @test SciMLBase.successful_retcode(sol2)
    else
        @test true
    end
end

@testitem "SciPyRoot + SciPyRootScalar" tags=[:wrappers] begin
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
        @test maximum(abs, sol_vec.resid) < 1e-6

        fscalar(x, p) = x^2 - 2
        prob_interval = IntervalNonlinearProblem(fscalar, (1.0, 2.0))
        sol_scalar = solve(prob_interval, SciPyRootScalar())
        @test SciMLBase.successful_retcode(sol_scalar)
        @test abs(sol_scalar.u - sqrt(2)) < 1e-6
    else
        @test true
    end
end
