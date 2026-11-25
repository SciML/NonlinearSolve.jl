using LinearAlgebra: diagm
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
        return diagonal ? diag(updated) : cache.du_cache
    end
    @testset for good in (true, false), diagonal in (true, false), singular in (true, false)
        rule = Broyden(; update_rule = Val(good ? :good_broyden : :bad_broyden)).update_rule
        dfu = good || !singular ? Float32[1, 0] : Float32[0, 0]
        du = good && singular ? Float32[0, -4] : Float32[2, -4]
        denom = singular ? 1.0f-5 : good ? 2.0f0 : 1.0f0
        expected = if diagonal && good
            1 .+ (du .- dfu .* du) .* du ./ denom
        elseif diagonal
            1 .+ (du .- dfu) .* dfu ./ denom
        else
            (du .- dfu) ./ denom
        end
        result = Reactant.@jit update(Reactant.to_rarray(dfu), Reactant.to_rarray(du), rule, Val(diagonal))
        @test Array(result) == expected
    end
end
