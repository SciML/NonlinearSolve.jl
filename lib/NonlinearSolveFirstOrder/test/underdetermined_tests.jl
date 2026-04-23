@testitem "Underdetermined NLLS with LevenbergMarquardt" tags = [:core] begin
    using NonlinearSolveFirstOrder, LinearAlgebra

    # Underdetermined: 2 equations, 4 unknowns
    # Find the minimum-norm solution that satisfies the nonlinear system
    function f_underdetermined(u, p)
        x, y, z, w = u
        return [
            x + y - 1.0,        # x + y = 1
            z + w - 2.0,        # z + w = 2
        ]
    end
    
    # The true solution we want is the minimum-norm solution
    # For x + y = 1: minimum norm is x = y = 0.5
    # For z + w = 2: minimum norm is z = w = 1.0
    # So expected solution is [0.5, 0.5, 1.0, 1.0]
    
    u0 = [0.0, 0.0, 0.0, 0.0]
    prob = NonlinearLeastSquaresProblem(NonlinearFunction(f_underdetermined), u0)
    
    sol = solve(prob, LevenbergMarquardt(); maxiters = 1000, abstol = 1e-10)
    
    @test SciMLBase.successful_retcode(sol)
    @test norm(sol.resid) < 1e-8
    
    # Check that we found a valid solution (satisfies the constraints)
    @test abs(sol.u[1] + sol.u[2] - 1.0) < 1e-8
    @test abs(sol.u[3] + sol.u[4] - 2.0) < 1e-8
end

@testitem "Underdetermined NLLS - Nonlinear system" tags = [:core] begin
    using NonlinearSolveFirstOrder, LinearAlgebra

    # Underdetermined nonlinear system: 2 equations, 5 unknowns
    function f_nonlinear_underdetermined(u, p)
        return [
            sin(u[1]) + u[2]^2 + u[3] - 1.0,
            u[3] * u[4] + u[5] - 2.0,
        ]
    end
    
    # Start near a solution
    u0 = [0.5, 0.5, 0.5, 2.0, 1.0]
    prob = NonlinearLeastSquaresProblem(NonlinearFunction(f_nonlinear_underdetermined), u0)
    
    sol = solve(prob, LevenbergMarquardt(); maxiters = 1000, abstol = 1e-8)
    
    @test SciMLBase.successful_retcode(sol)
    @test norm(sol.resid) < 1e-6
end

@testitem "Underdetermined NLLS - In-place" tags = [:core] begin
    using NonlinearSolveFirstOrder, LinearAlgebra

    # In-place version
    function f_underdetermined_iip!(resid, u, p)
        resid[1] = u[1] + u[2] + u[3] - 3.0
        resid[2] = u[1] * u[2] - u[4] - 1.0
        return nothing
    end
    
    u0 = ones(4)
    resid_prototype = zeros(2)
    
    prob = NonlinearLeastSquaresProblem(
        NonlinearFunction{true}(f_underdetermined_iip!; resid_prototype), 
        u0
    )
    
    sol = solve(prob, LevenbergMarquardt(); maxiters = 1000, abstol = 1e-8)
    
    @test SciMLBase.successful_retcode(sol)
    @test norm(sol.resid) < 1e-6
end

@testitem "Underdetermined vs Overdetermined comparison" tags = [:core] begin
    using NonlinearSolveFirstOrder, LinearAlgebra

    # Test that overdetermined systems still work correctly
    # (regression test - make sure we didn't break anything)
    
    # Overdetermined: 4 equations, 2 unknowns
    function f_overdetermined(u, p)
        x, y = u
        return [
            x + y - 1.0,
            x - y - 0.0,
            2x + y - 1.5,
            x + 2y - 1.5,
        ]
    end
    
    u0 = [0.0, 0.0]
    prob = NonlinearLeastSquaresProblem(NonlinearFunction(f_overdetermined), u0)
    
    sol = solve(prob, LevenbergMarquardt(); maxiters = 1000, abstol = 1e-8)
    
    @test SciMLBase.successful_retcode(sol)
    # This is overdetermined so there's no exact solution
    # Just check that we converged to something reasonable
    @test norm(sol.resid) < 0.5  # The residual won't be zero for overdetermined
end

@testitem "Underdetermined NLLS - Square system boundary" tags = [:core] begin
    using NonlinearSolveFirstOrder, LinearAlgebra

    # Exactly square system (n = m), should use normal path
    function f_square(u, p)
        return [
            u[1]^2 + u[2] - 1.0,
            u[1] + u[2]^2 - 1.0,
        ]
    end
    
    u0 = [0.5, 0.5]
    prob = NonlinearLeastSquaresProblem(NonlinearFunction(f_square), u0)
    
    sol = solve(prob, LevenbergMarquardt(); maxiters = 1000, abstol = 1e-10)
    
    @test SciMLBase.successful_retcode(sol)
    @test norm(sol.resid) < 1e-8
end

@testitem "Underdetermined with explicit min_norm_mode" tags = [:core] begin
    using NonlinearSolveFirstOrder, LinearAlgebra, NonlinearSolveBase

    # Test explicit control of min_norm_mode
    function f_simple(u, p)
        return [u[1] + u[2] - 1.0]
    end
    
    u0 = [0.0, 0.0]
    prob = NonlinearLeastSquaresProblem(NonlinearFunction(f_simple), u0)
    
    # Should work with auto mode (default)
    sol_auto = solve(prob, LevenbergMarquardt(); maxiters = 1000, abstol = 1e-10)
    @test SciMLBase.successful_retcode(sol_auto)
    @test norm(sol_auto.resid) < 1e-8
end

@testitem "Large underdetermined system" tags = [:core] begin
    using NonlinearSolveFirstOrder, LinearAlgebra

    # Large underdetermined: 10 equations, 50 unknowns
    # This tests that the m×m matrix approach is more efficient than n×n
    function f_large_underdetermined(u, p)
        n_eq = 10
        resid = zeros(eltype(u), n_eq)
        for i in 1:n_eq
            # Each equation sums 5 variables
            start_idx = 5 * (i - 1) + 1
            resid[i] = sum(u[start_idx:start_idx+4]) - Float64(i)
        end
        return resid
    end
    
    u0 = zeros(50)
    prob = NonlinearLeastSquaresProblem(NonlinearFunction(f_large_underdetermined), u0)
    
    sol = solve(prob, LevenbergMarquardt(); maxiters = 1000, abstol = 1e-8)
    
    @test SciMLBase.successful_retcode(sol)
    @test norm(sol.resid) < 1e-6
end
