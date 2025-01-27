using NonlinearSolve
using NonlinearSolveHomotopyContinuation
using SciMLBase: NonlinearSolution

alg = HomotopyContinuationJL{true}(; threading = false)

@testset "scalar u" begin
    rhs = function (u, p)
        return u * u - p[1] * u + p[2]
    end
    jac = function (u, p)
        return 2u - p[1]
    end
    @testset "`NonlinearProblem` - $name" for (jac, name) in [(nothing, "no jac"), (jac, "jac")]
        fn = NonlinearFunction(rhs; jac)
        prob = NonlinearProblem(fn, 1.0, [5.0, 6.0])
        sol = solve(prob, alg)

        @test sol isa EnsembleSolution
        @test sol.converged
        sort!(sol.u; by = x -> x.u)
        @test sol.u[1] isa NonlinearSolution
        @test SciMLBase.successful_retcode(sol.u[1])
        @test sol.u[1].u ≈ 2.0 atol = 1e-10
        @test sol.u[2] isa NonlinearSolution
        @test SciMLBase.successful_retcode(sol.u[2])
        @test sol.u[2].u ≈ 3.0 atol = 1e-10

        @testset "no real solutions" begin
            prob = NonlinearProblem(1.0, 0.5) do u, p
                return u * u - 2p * u  + p
            end
            sol = solve(prob, alg)
            @test length(sol) == 1
            @test sol.u[1].retcode == SciMLBase.ReturnCode.ConvergenceFailure
            @test !sol.converged
        end
    end

    @testset "`HomotopyContinuationFunction`" begin
        denominator = function (u, p)
            return [u - 0.7]
        end
        polynomialize = function (u, p)
            return sin(u)
        end
        unpolynomialize = function (u, p)
            return [asin(u)]
        end
        fn = HomotopyNonlinearFunction(; denominator, polynomialize, unpolynomialize) do u, p
            return (u - p[1]) * (u - p[2])
        end
        prob = NonlinearProblem(fn, 0.0, [0.5, 0.7])

        sol = solve(prob, alg)
        @test length(sol) == 1
        @test sin(sol.u[1][1]) ≈ 0.5

        @testset "no valid solutions" begin
            prob2 = remake(prob; p = [0.7, 0.7])
            sol2 = solve(prob2, alg)
            @test !sol2.converged
            @test length(sol2) == 1
            @test sol2.u[1].retcode == SciMLBase.ReturnCode.Infeasible
        end

        @testset "multiple solutions" begin
            prob3 = remake(prob; p = [0.5, 0.6])
            sol3 = solve(prob3, alg)
            @test length(sol3) == 2
            @test sort(sin.([sol3.u[1][1], sol3.u[2][1]])) ≈ [0.5, 0.6]
        end
    end
end

f! = function (du, u, p)
    du[1] = u[1] * u[1] - p[1] * u[2] + u[2] ^ 3 + 1
    du[2] = u[2] ^ 3 + 2 * p[2] * u[1] * u[2] + u[2]
end

f = function (u, p)
    [u[1] * u[1] - p[1] * u[2] + u[2] ^ 3 + 1, u[2] ^ 3 + 2 * p[2] * u[1] * u[2] + u[2]]
end

jac! = function (j, u, p)
    j[1, 1] = 2u[1]
    j[1, 2] = -p[1] + 3 * u[2]^2
    j[2, 1] = 2 * p[2] * u[2]
    j[2, 2] = 3 * u[2]^2 + 2 * p[2] * u[1] + 1
end
jac = function (u, p)
    [2u[1] -p[1] + 3 * u[2]^2;
     2 * p[2] * u[2] 3 * u[2]^2 + 2 * p[2] * u[1] + 1]
end

@testset "vector u - $name" for (rhs, jac, name) in [(f, nothing, "oop"), (f, jac, "oop + jac"), (f!, nothing, "iip"), (f!, jac!, "iip + jac")]
    sol = nothing
    @testset "`NonlinearProblem`" begin
        fn = NonlinearFunction(rhs; jac)
        prob = NonlinearProblem(fn, [1.0, 2.0], [2.0, 3.0])
        sol = solve(prob, alg)
        @test sol isa EnsembleSolution
        @test sol.converged
        for nlsol in sol.u
            @test SciMLBase.successful_retcode(nlsol)
            @test f(nlsol.u, prob.p) ≈ [0.0, 0.0] atol = 1e-10
        end

        @testset "no real solutions" begin
            _prob = remake(prob; p = zeros(2))
            _sol = solve(_prob, alg)
            @test !_sol.converged
            @test length(_sol) == 1
            @test !SciMLBase.successful_retcode(_sol.u[1])
        end
    end

    @testset "`HomotopyNonlinearFunction`" begin
        denominator = function (u, p)
            return [u[1] - p[3], u[2] - p[4]]
        end
        unpolynomialize = function (u, p)
            return [[cbrt(u[1]), sin(u[2] / 40)]]
        end
        polynomialize = function (u, p)
            return [u[1] ^ 3, 40asin(u[2])]
        end
        nlfn = NonlinearFunction(rhs; jac)
        fn = HomotopyNonlinearFunction(nlfn; denominator, polynomialize, unpolynomialize)
        prob = NonlinearProblem(fn, [1.0, 2.0], [2.0, 3.0, 4.0, 5.0])
        sol2 = solve(prob, alg)
        @test sol2 isa EnsembleSolution
        @test sol2.converged
        @test length(sol.u) == length(sol2.u)
        for nlsol2 in sol2.u
            @test any(nlsol -> isapprox(polynomialize(nlsol2.u, prob.p), nlsol.u; rtol = 1e-8), sol.u)
        end

        @testset "some invalid solutions" begin
            prob3 = remake(prob; p = [2.0, 3.0, polynomialize(sol2.u[1].u, prob.p)...])
            sol3 = solve(prob3, alg)
            @test length(sol3.u) == length(sol2.u) - 1
        end
    end
end
