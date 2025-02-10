using NonlinearSolve
using NonlinearSolveHomotopyContinuation
using SciMLBase: NonlinearSolution
import NaNMath

alg = HomotopyContinuationJL{false}(; threading = false)

@testset "scalar u" begin
    rhs = function (u, p)
        return (u - 3.0) * (u - p)
    end
    jac = function (u, p)
        return 2u - (p + 3)
    end
    @testset "`NonlinearProblem` - $name" for (jac_or_autodiff, name) in [
        (AutoForwardDiff(), "no jac - forwarddiff"), (AutoEnzyme(), "no jac - enzyme"), (
            jac, "jac")]
        if jac_or_autodiff isa Function
            jac = jac_or_autodiff
            autodiff = nothing
        else
            jac = nothing
            autodiff = jac_or_autodiff
        end
        fn = NonlinearFunction(rhs; jac)
        prob = NonlinearProblem(fn, 1.0, 2.0)
        _alg = HomotopyContinuationJL(alg; autodiff)
        sol = solve(prob, _alg)

        @test sol isa NonlinearSolution
        @test sol.u≈2.0 atol=1e-10

        @testset "no real solutions" begin
            prob = NonlinearProblem(1.0, 1.0) do u, p
                return u * u - 2p * u + p
            end
            sol = solve(prob, alg)
            @test sol.retcode == SciMLBase.ReturnCode.ConvergenceFailure
        end
    end

    @testset "`HomotopyContinuationFunction`" begin
        denominator = function (u, p)
            return [u - 0.7, u - 0.9]
        end
        polynomialize = function (u, p)
            return sin(u)
        end
        unpolynomialize = function (u, p)
            return [asin(u)]
        end
        fn = HomotopyNonlinearFunction(;
            denominator, polynomialize, unpolynomialize) do u, p
            return (u - p[1]) * (u - p[2])
        end
        prob = NonlinearProblem(fn, 0.0, [0.5, 0.7])

        sol = solve(prob, alg)
        @test sin(sol.u[1])≈0.5 atol=1e-10

        @testset "no valid solutions" begin
            prob2 = remake(prob; p = [0.7, 0.9])
            sol2 = solve(prob2, alg)
            @test sol2.retcode == SciMLBase.ReturnCode.Infeasible
        end

        @testset "closest root" begin
            prob3 = remake(prob; p = [0.5, 0.6], u0 = asin(0.4))
            sol3 = solve(prob3, alg)
            @test sin(sol3.u)≈0.5 atol=1e-10
            prob4 = remake(prob3; u0 = asin(0.7))
            sol4 = solve(prob4, alg)
            @test sin(sol4.u)≈0.6 atol=1e-10
        end
    end
end

f! = function (du, u, p)
    du[1] = u[1] * u[1] - p[1] * u[2] + u[2]^3 + 1
    du[2] = u[2]^3 + 2 * p[2] * u[1] * u[2] + u[2]
end

f = function (u, p)
    [u[1] * u[1] - p[1] * u[2] + u[2]^3 + 1, u[2]^3 + 2 * p[2] * u[1] * u[2] + u[2]]
end

jac! = function (j, u, p)
    j[1, 1] = 2u[1]
    j[1, 2] = -p[1] + 3 * u[2]^2
    j[2, 1] = 2 * p[2] * u[2]
    j[2, 2] = 3 * u[2]^2 + 2 * p[2] * u[1] + 1
end

jac = function (u, p)
    [2u[1] -p[1]+3 * u[2]^2;
     2*p[2]*u[2] 3*u[2]^2+2*p[2]*u[1]+1]
end

@testset "vector u - $name" for (rhs, jac_or_autodiff, name) in [
    (f, AutoForwardDiff(), "oop + forwarddiff"), (f, AutoEnzyme(), "oop + enzyme"), (
        f, jac, "oop + jac"),
    (f!, AutoForwardDiff(), "iip + forwarddiff"), (f!, AutoEnzyme(), "iip + enzyme"), (
        f!, jac!, "iip + jac")]
    if jac_or_autodiff isa Function
        jac = jac_or_autodiff
        autodiff = nothing
    else
        jac = nothing
        autodiff = jac_or_autodiff
    end
    _alg = HomotopyContinuationJL(alg; autodiff)
    @testset "`NonlinearProblem`" begin
        fn = NonlinearFunction(rhs; jac)
        prob = NonlinearProblem(fn, [1.0, 2.0], [2.0, 3.0])
        sol = solve(prob, _alg)
        @test SciMLBase.successful_retcode(sol)
        @test f(sol.u, prob.p)≈[0.0, 0.0] atol=1e-10

        @testset "no real solutions" begin
            prob2 = remake(prob; p = zeros(2))
            sol2 = solve(prob2, _alg)
            @test !SciMLBase.successful_retcode(sol2)
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
            return [u[1]^3, 40asin(u[2])]
        end
        nlfn = NonlinearFunction(rhs; jac)
        fn = HomotopyNonlinearFunction(nlfn; denominator, polynomialize, unpolynomialize)
        prob = NonlinearProblem(fn, [1.0, 1.0], [2.0, 3.0, 4.0, 5.0])
        sol = solve(prob, _alg)
        @test SciMLBase.successful_retcode(sol)
        @test f(polynomialize(sol.u, prob.p), prob.p)≈zeros(2) atol=1e-10

        @testset "some invalid solutions" begin
            prob2 = remake(prob; p = [2.0, 3.0, polynomialize(sol.u, prob.p)...])
            sol2 = solve(prob2, _alg)
            @test !SciMLBase.successful_retcode(sol2)
        end
    end
end

@testset "`NaN` unpolynomialize" begin
    polynomialize = function (u, p)
        return sin(u^2)
    end
    unpolynomialize = function (u, p)
        return NaN
    end
    rhs = function (u, p)
        return u^2 + u - 1
    end
    prob = NonlinearProblem(
        HomotopyNonlinearFunction(rhs; polynomialize, unpolynomialize), 1.0)
    sol = solve(prob, alg)
    @test !SciMLBase.successful_retcode(sol)
end
