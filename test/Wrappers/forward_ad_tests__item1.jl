using NonlinearSolve
include("setup_forwardadtesting.jl")

@testset for alg in (
        NewtonRaphson(),
        TrustRegion(),
        LevenbergMarquardt(),
        PseudoTransient(; alpha_initial = 10.0),
        Broyden(),
        Klement(),
        DFSane(),
        FastShortcutNonlinearPolyalg(),
        nothing,
        NLsolveJL(),
        CMINPACK(),
        KINSOL(; globalization_strategy = :LineSearch),
    )
    us = (2.0, @SVector[1.0, 1.0], [1.0, 1.0], ones(2, 2), @SArray ones(2, 2))

    alg isa CMINPACK && Sys.isapple() && continue

    @testset "Scalar AD" begin
        for p in 1.0:0.1:100.0, u0 in us, mode in (:iip, :oop, :iip_cache, :oop_cache)
            compatible(u0, alg) || continue
            compatible(u0, Val(mode)) || continue
            compatible(alg, Val(mode)) || continue

            sol = solve(NonlinearProblem(test_f, u0, p), alg)
            if SciMLBase.successful_retcode(sol)
                gs = abs.(ForwardDiff.derivative(solve_with(Val{mode}(), u0, alg), p))
                gs_true = abs.(jacobian_f(u0, p))
                if !(isapprox(gs, gs_true, atol = 1.0e-5))
                    @error "ForwardDiff Failed for u0=$(u0) and p=$(p) with $(alg)" forwardiff_gradient = gs true_gradient = gs_true
                else
                    @test abs.(gs) ≈ abs.(gs_true) atol = 1.0e-5
                end
            end
        end
    end

    @testset "Jacobian" begin
        for u0 in us,
                p in ([2.0, 1.0], [2.0 1.0; 3.0 4.0]),
                mode in (:iip, :oop, :iip_cache, :oop_cache)
            compatible(u0, p) || continue
            compatible(u0, alg) || continue
            compatible(u0, Val(mode)) || continue
            compatible(alg, Val(mode)) || continue

            sol = solve(NonlinearProblem(test_f, u0, p), alg)
            if SciMLBase.successful_retcode(sol)
                gs = abs.(ForwardDiff.jacobian(solve_with(Val{mode}(), u0, alg), p))
                gs_true = abs.(jacobian_f(u0, p))
                if !(isapprox(gs, gs_true, atol = 1.0e-5))
                    @show sol.retcode, sol.u
                    @error "ForwardDiff Failed for u0=$(u0) and p=$(p) with $(alg)" forwardiff_jacobian = gs true_jacobian = gs_true
                else
                    @test abs.(gs) ≈ abs.(gs_true) atol = 1.0e-5
                end
            end
        end
    end
end
