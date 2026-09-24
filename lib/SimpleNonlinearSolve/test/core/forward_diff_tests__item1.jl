using SimpleNonlinearSolve

using ArrayInterface
using ForwardDiff, FiniteDiff, SimpleNonlinearSolve, StaticArrays, LinearAlgebra,
    Zygote, ReverseDiff, SciMLBase
using DifferentiationInterface

const DI = DifferentiationInterface

test_f!(du, u, p) = (@. du = u^2 - p)
test_f(u, p) = u .^ 2 .- p

jacobian_f(::Number, p) = 1 / (2 * √p)
jacobian_f(::Number, p::Number) = 1 / (2 * √p)
jacobian_f(u, p::Number) = one.(u) .* (1 / (2 * √p))
jacobian_f(u, p::AbstractArray) = diagm(vec(@. 1 / (2 * √p)))

@testset "$(nameof(typeof(alg)))" for alg in (
        SimpleNewtonRaphson(),
        SimpleTrustRegion(),
        SimpleTrustRegion(; nlsolve_update_rule = Val(true)),
        SimpleHalley(),
        SimpleBroyden(),
        SimpleKlement(),
        SimpleDFSane(),
    )
    us = (
        2.0,
        @SVector([1.0, 1.0]),
        [1.0, 1.0],
        ones(2, 2),
        @SArray(ones(2, 2)),
    )

    @testset "Scalar AD" begin
        for p in 1.0:0.1:100.0, u0 in us

            sol = solve(NonlinearProblem{false}(test_f, u0, p), alg)
            if SciMLBase.successful_retcode(sol)
                gs = abs.(
                    ForwardDiff.derivative(p) do pᵢ
                        solve(NonlinearProblem{false}(test_f, u0, pᵢ), alg).u
                    end
                )
                gs_true = abs.(jacobian_f(u0, p))

                if !(isapprox(gs, gs_true, atol = 1.0e-5))
                    @show sol.retcode, sol.u
                    @error "ForwardDiff Failed for u0=$(u0) and p=$(p) with $(alg)" forwardiff_gradient = gs true_gradient = gs_true
                else
                    @test abs.(gs) ≈ abs.(gs_true) atol = 1.0e-5
                end
            end
        end
    end

    @testset "Jacobian" begin
        @testset "$(typeof(u0))" for u0 in us[2:end],
                p in ([2.0, 1.0], [2.0 1.0; 3.0 4.0])

            if u0 isa AbstractArray && p isa AbstractArray
                size(u0) != size(p) && continue
            end

            @testset for (iip, fn) in ((false, test_f), (true, test_f!))
                iip && (u0 isa Number || !ArrayInterface.can_setindex(u0)) && continue

                sol = solve(NonlinearProblem{iip}(fn, u0, p), alg)
                if SciMLBase.successful_retcode(sol)
                    gs = abs.(
                        ForwardDiff.jacobian(p) do pᵢ
                            solve(NonlinearProblem{iip}(fn, u0, pᵢ), alg).u
                        end
                    )
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
end
