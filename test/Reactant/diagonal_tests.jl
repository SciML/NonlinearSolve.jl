using LinearAlgebra: Diagonal, diag, diagm, dot
using NonlinearSolve
using Reactant
using Test

@testset "Compiled diagonal quasi-Newton" begin
    @testset "$name" for (name, alg, nf, u0, p, maxiters, expected, retcode) in (
            (
                "analytic diagonal", Klement(; init_jacobian = Val(:true_jacobian_diagonal)),
                NonlinearFunction((u, p) -> u .^ 2 .- p; jac = (u, p) -> diagm(2 .* u)),
                Float32[1, 1], Float32[2], 1, Float32[1.5, 1.5], ReturnCode.MaxIters,
            ),
            (
                "zero inverse scaling", Broyden(; update_rule = Val(:diagonal), alpha = 0.0f0),
                NonlinearFunction((u, p) -> u .- p),
                Float32[2, 3], Float32[1], 1, Float32[2, 3], ReturnCode.MaxIters,
            ),
            (
                "singular diagonal exhausts the reset budget", Klement(; alpha = 1.0f0, max_resets = 1),
                NonlinearFunction((u, p) -> one.(u)),
                Float32[0, 0], Float32[1], 2, Float32[-1, -1], ReturnCode.ConvergenceFailure,
            ),
            (
                "nonzero ill-conditioned diagonal", Klement(; alpha = 1.0f0, max_resets = 1),
                NonlinearFunction((u, p) -> p .* u .- 1),
                Float32[0, 0], Float32[1, 1.0f-5], 12, Float32[1, 100000], ReturnCode.Success,
            ),
        )
        function run_diagonal(u, p)
            return solve(
                NonlinearProblem(nf, u, p), alg; maxiters, abstol = 1.0f-5,
                termination_condition = AbsNormTerminationMode(Base.Fix1(maximum, abs))
            )
        end
        host = run_diagonal(u0, p)
        @test host.retcode == retcode
        @test host.u ≈ expected
        compiled = Reactant.@jit run_diagonal(Reactant.to_rarray(u0), Reactant.to_rarray(p))
        @test compiled.retcode == retcode
        @test Array(compiled.u) ≈ expected
        @test Array(compiled.resid) ≈ host.resid atol = 1.0f-5
    end
end

@testset "Broyden update with regular and zero denominators" begin
    function update(dfu, du, rule, ::Val{diagonal}) where {diagonal}
        J = diagonal ? Diagonal(one.(du)) : diagm(one.(du))
        api = NonlinearSolveBase.InternalAPI
        cache = api.init(NonlinearProblem((x, p) -> x, du), rule, J, zero.(dfu), du, du)
        updated = api.solve!(cache, J, dfu, du, du)
        return diagonal ? diag(updated) : updated
    end
    @testset for good in (true, false), diagonal in (true, false), singular in (true, false)
        rule = Broyden(; update_rule = Val(good ? :good_broyden : :bad_broyden)).update_rule
        dfu = good || !singular ? Float32[1, 0] : Float32[0, 0]
        du = good && singular ? Float32[0, -4] : Float32[2, -4]
        H = diagm(one.(du))
        denom = good ? dot(du, H * dfu) : dot(dfu, dfu)
        right = good ? du' * H : dfu'
        expected = H + (du - H * dfu) * right / (iszero(denom) ? 1.0f-5 : denom)
        host = update(dfu, du, rule, Val(diagonal))
        compiled = Reactant.@jit update(Reactant.to_rarray(dfu), Reactant.to_rarray(du), rule, Val(diagonal))
        for result in (host, compiled)
            @test Array(result) == (diagonal ? diag(expected) : expected)
        end
    end
    @testset "One-dimensional inverse secant" for (dfu, du) in (
                (Float32[1], Float32[2]), (ComplexF32[im], ComplexF32[2im]),
            ), diagonal in (true, false), compiled in (false, true)
        rule = Broyden().update_rule
        result = if compiled
            Reactant.@jit update(Reactant.to_rarray(dfu), Reactant.to_rarray(du), rule, Val(diagonal))
        else
            update(dfu, du, rule, Val(diagonal))
        end
        @test only(Array(result)) * only(dfu) == only(du)
    end
end

@testset "Two-step Broyden solve" begin
    function two_steps(u, target, rule)
        prob = NonlinearProblem((x, p) -> x ./ 2 .- p, u, target)
        return solve(prob, Broyden(; update_rule = rule, alpha = 1.0f0); maxiters = 2)
    end
    @testset for (rule, target) in (
                (Val(:diagonal), Float32[2, 0]), (Val(:good_broyden), Float32[2, 0]),
                (Val(:good_broyden), Float32[2]), (Val(:diagonal), ComplexF32[2im]),
            ), compiled in (false, true)
        u = zero(target)
        sol = if compiled
            Reactant.@jit two_steps(Reactant.to_rarray(u), Reactant.to_rarray(target), rule)
        else
            two_steps(u, target, rule)
        end
        @test Array(sol.u) == 2 .* target
        @test Array(sol.u) ./ 2 .- target == zero(target)
        @test Array(sol.resid) == zero(target)
        @test sol.retcode == ReturnCode.Success
    end
end
