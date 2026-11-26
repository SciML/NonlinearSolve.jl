using LinearAlgebra: dot, norm
using SimpleNonlinearSolve, SciMLBase, StaticArrays, Test

@testset "Scale-safe curvature-corrected dogleg" begin
    for rule in (Val(false), Val(true))
        for (T, exponent) in ((Float32, -32), (Float64, -300))
            scale = ldexp(one(T), exponent)
            abstol = scale / 1024
            for x0 in (T[0], SVector(zero(T)), zero(T))
                trials = typeof(x0)[]
                f = function (x, p)
                    push!(trials, copy(x))
                    return scale .* (x .- one(T))
                end
                # Supply the Jacobian so the recorded residual calls are actual iterates.
                jac = x0 isa Number ? scale :
                    x0 isa StaticArray ? SMatrix{1, 1}(scale) : fill(scale, 1, 1)
                prob = NonlinearProblem(NonlinearFunction{false}(f; jac = (x, p) -> jac), x0)
                alg = SimpleTrustRegion(;
                    initial_trust_radius = 0.5, nlsolve_update_rule = rule,
                    curvature_scaling = Val(true)
                )
                sol = solve(prob, alg; abstol, reltol = 0)
                @test SciMLBase.successful_retcode(sol)
                @test all(x -> all(isfinite, x), trials)
                @test sol.u ≈ one.(x0)
                @test norm(sol.resid, Inf) ≤ abstol
            end
        end

        # J = diag(2, 1) and f(0) = [-2, -2] before residual scaling.
        # The Cauchy point is [20/17, 10/17] and Newton's step is [1, 2].
        cauchy = [20 / 17, 10 / 17]
        segment = [1, 2] - cauchy
        radius = Float32(1.6)
        a = dot(segment, segment)
        b = dot(cauchy, segment)
        t = (-b + sqrt(b^2 + a * (Float64(radius)^2 - dot(cauchy, cauchy)))) / a
        expected = cauchy + t * segment
        for scale in (1.0f0, ldexp(1.0f0, 22))
            for x0 in (zeros(Float32, 2), SVector(0.0f0, 0.0f0))
                prob = NonlinearProblem(x0) do x, p
                    return scale .* oftype(x, [2 * x[1] - 2, x[2] - 2])
                end
                alg = SimpleTrustRegion(;
                    initial_trust_radius = radius, nlsolve_update_rule = rule,
                    curvature_scaling = Val(true)
                )
                sol = solve(prob, alg; maxiters = 1, abstol = 0, reltol = 0)
                @test all(isfinite, sol.u)
                @test norm(sol.u - expected) ≤ 8eps(Float32) * radius
            end
        end
    end
end
