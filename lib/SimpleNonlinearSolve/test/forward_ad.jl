using ForwardDiff, SimpleNonlinearSolve, StaticArrays, XUnit, LinearAlgebra
import SimpleNonlinearSolve: AbstractSimpleNonlinearSolveAlgorithm

test_f!(du, u, p) = (@. du = u^2 - p)
test_f(u, p) = (@. u^2 - p)

jacobian_f(::Number, p) = 1 / (2 * √p)
jacobian_f(::Number, p::Number) = 1 / (2 * √p)
jacobian_f(u, p::Number) = one.(u) .* (1 / (2 * √p))
jacobian_f(u, p::AbstractArray) = diagm(vec(@. 1 / (2 * √p)))

function solve_with(::Val{mode}, u, alg) where {mode}
    f = if mode === :iip
        solve_iip(p) = solve(NonlinearProblem(test_f!, u, p), alg).u
    elseif mode === :oop
        solve_oop(p) = solve(NonlinearProblem(test_f, u, p), alg).u
    end
    return f
end

__compatible(::Any, ::Val{:oop}) = true
__compatible(::Number, ::Val{:iip}) = false
__compatible(::AbstractArray, ::Val{:iip}) = true
__compatible(::StaticArray, ::Val{:iip}) = false

__compatible(::Any, ::Number) = true
__compatible(::Number, ::AbstractArray) = false
__compatible(u::AbstractArray, p::AbstractArray) = size(u) == size(p)

__compatible(u::Number, ::AbstractSimpleNonlinearSolveAlgorithm) = true
__compatible(u::AbstractArray, ::AbstractSimpleNonlinearSolveAlgorithm) = true
__compatible(u::StaticArray, ::AbstractSimpleNonlinearSolveAlgorithm) = true

__compatible(::AbstractSimpleNonlinearSolveAlgorithm, ::Val{:iip}) = true
__compatible(::AbstractSimpleNonlinearSolveAlgorithm, ::Val{:oop}) = true
__compatible(::SimpleHalley, ::Val{:iip}) = false

@testcase "ForwardDiff.jl Integration: $(alg)" for alg in (SimpleNewtonRaphson(),
    SimpleTrustRegion(), SimpleHalley(), SimpleBroyden(), SimpleKlement(), SimpleDFSane())
    us = (2.0, @SVector[1.0, 1.0], [1.0, 1.0], ones(2, 2), @SArray ones(2, 2))

    @testset "Scalar AD" begin
        for p in 1.0:0.1:100.0, u0 in us, mode in (:iip, :oop)
            __compatible(u0, alg) || continue
            __compatible(u0, Val(mode)) || continue
            __compatible(alg, Val(mode)) || continue

            sol = solve(NonlinearProblem(test_f, u0, p), alg)
            if SciMLBase.successful_retcode(sol)
                gs = abs.(ForwardDiff.derivative(solve_with(Val{mode}(), u0, alg), p))
                gs_true = abs.(jacobian_f(u0, p))
                if !(isapprox(gs, gs_true, atol = 1e-5))
                    @show sol.retcode, sol.u
                    @error "ForwardDiff Failed for u0=$(u0) and p=$(p) with $(alg)" forwardiff_gradient=gs true_gradient=gs_true
                else
                    @test abs.(gs)≈abs.(gs_true) atol=1e-5
                end
            end
        end
    end

    @testset "Jacobian" begin
        for u0 in us, p in ([2.0, 1.0], [2.0 1.0; 3.0 4.0]), mode in (:iip, :oop)
            __compatible(u0, p) || continue
            __compatible(u0, alg) || continue
            __compatible(u0, Val(mode)) || continue
            __compatible(alg, Val(mode)) || continue

            sol = solve(NonlinearProblem(test_f, u0, p), alg)
            if SciMLBase.successful_retcode(sol)
                gs = abs.(ForwardDiff.jacobian(solve_with(Val{mode}(), u0, alg), p))
                gs_true = abs.(jacobian_f(u0, p))
                if !(isapprox(gs, gs_true, atol = 1e-5))
                    @show sol.retcode, sol.u
                    @error "ForwardDiff Failed for u0=$(u0) and p=$(p) with $(alg)" forwardiff_jacobian=gs true_jacobian=gs_true
                else
                    @test abs.(gs)≈abs.(gs_true) atol=1e-5
                end
            end
        end
    end
end
