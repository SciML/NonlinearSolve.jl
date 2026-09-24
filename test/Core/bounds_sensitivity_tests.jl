using NonlinearSolve, SciMLBase, ForwardDiff, LinearAlgebra, Test

@testset "Bounded least-squares implicit sensitivities" begin
    # Test strictly active or interior solutions, where the active set is locally constant.
    for inplace in (false, true), derivative in (:autodiff, :jac, :vjp)
        f(u, p) = u .- p
        f!(r, u, p) = (r .= u .- p; nothing)
        jac(u, p) = Matrix{eltype(u)}(I, length(u), length(u))
        jac!(J, u, p) = (J .= jac(u, p); nothing)
        vjp(v, u, p) = v
        vjp!(out, v, u, p) = (out .= v; nothing)
        options = derivative === :jac ? (; jac = inplace ? jac! : jac) :
            derivative === :vjp ? (; vjp = inplace ? vjp! : vjp) : (;)
        nf = NonlinearFunction{inplace}(inplace ? f! : f; options...)
        function solution(p; cached = false)
            prob = NonlinearLeastSquaresProblem(
                nf, [0.5, 0.5, 0.5, 0.5], p;
                lb = [0.0, 0.0, 0.0, 0.5], ub = [1.0, 1.0, 1.0, 0.5]
            )
            if cached
                cache = init(prob, BoundedTrustRegion(); abstol = 1.0e-10, reltol = 1.0e-10)
                solve!(cache)
                reinit!(cache; u0 = prob.u0, p = p .+ [3.0, -3.0, 0.0, 0.0])
                return solve!(cache).u
            end
            return solve(prob, BoundedTrustRegion(); abstol = 1.0e-10, reltol = 1.0e-10).u
        end
        p = [-1.0, 2.0, 0.3, 2.0]
        expected = diagm([0.0, 0.0, 1.0, 0.0])
        @test solution(p) ≈ [0.0, 1.0, 0.3, 0.5] atol = 1.0e-8
        @test ForwardDiff.jacobian(solution, p) ≈ expected atol = 1.0e-10
        @test solution(p; cached = true) ≈ [1.0, 0.0, 0.3, 0.5] atol = 1.0e-8
        @test ForwardDiff.jacobian(p -> solution(p; cached = true), p) ≈ expected atol = 1.0e-10
    end
    for (lb, ub, p, expected) in (
            (nothing, 1.0, 2.0, 0.0), (0.0, nothing, -1.0, 0.0),
            (0.0, 1.0, 0.3, 1.0), (0.5, 0.5, 2.0, 0.0),
        )
        scalar_parameter(p) = only(
            solve(
                NonlinearLeastSquaresProblem(
                    (u, p) -> u .- p,
                    [0.5], p; lb, ub
                ), BoundedTrustRegion(); abstol = 1.0e-10, reltol = 1.0e-10
            ).u
        )
        @test ForwardDiff.derivative(scalar_parameter, p) ≈ expected atol = 1.0e-10
    end
    for derivative in (:autodiff, :jac, :vjp), (p, expected) in ((2.0, 0.0), (0.3, 1.0))
        options = derivative === :jac ? (; jac = (u, p) -> one(u)) :
            derivative === :vjp ? (; vjp = (v, u, p) -> v) : (;)
        nf = NonlinearFunction((u, p) -> u - p; options...)
        scalar_state(p) = solve(
            NonlinearLeastSquaresProblem(
                nf, 0.5, p; lb = 0.0,
                ub = 1.0
            ), BoundedTrustRegion(); abstol = 1.0e-10, reltol = 1.0e-10
        ).u
        @test ForwardDiff.derivative(scalar_state, p) ≈ expected atol = 1.0e-10
        function scalar_cached(p)
            prob = NonlinearLeastSquaresProblem(nf, 0.5, p; lb = 0.0, ub = 1.0)
            cache = init(prob, BoundedTrustRegion(); abstol = 1.0e-10, reltol = 1.0e-10)
            solve!(cache)
            reinit!(cache; u0 = 0.5, p)
            return solve!(cache).u
        end
        @test ForwardDiff.derivative(scalar_cached, p) ≈ expected atol = 1.0e-10
    end
    A = [1.0 1.0; 0.0 1.0]
    coupled(p) = solve(
        NonlinearLeastSquaresProblem(
            (u, p) -> A * u - p,
            [0.5, 0.5], p; lb = [0.0, -Inf], ub = [1.0, Inf]
        ), BoundedTrustRegion();
        abstol = 1.0e-10, reltol = 1.0e-10
    ).u
    @test coupled([3.0, 0.0]) ≈ [1.0, 1.0] atol = 1.0e-8
    @test ForwardDiff.jacobian(coupled, [3.0, 0.0]) ≈ [0.0 0.0; 0.5 0.5] atol = 1.0e-10
end
