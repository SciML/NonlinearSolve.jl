using NonlinearSolveFirstOrder, LinearAlgebra, Test

rosenbrock(u, p) = [10 * (u[2] - u[1]^2), 1 - u[1]]

@testset "LevenbergMarquardt step acceptance" begin
    @testset "b_uphill = 0 never accepts an uphill step" begin
        prob = NonlinearLeastSquaresProblem(rosenbrock, [-1.2, 1.0])
        alg = LevenbergMarquardt(;
            b_uphill = 0.0, disable_geodesic = Val(true), damping_initial = 1.0e-8
        )
        cache = init(prob, alg; abstol = 1.0e-12)
        for _ in 1:30
            cost = norm(cache.fu)
            step!(cache)
            @test norm(cache.fu) ≤ cost
        end
    end

    # A trial step that lowers the cost is accepted, and one that raises it while
    # turning back on the last accepted step (β ≤ 0) is rejected
    @testset "b_uphill = 1, damping_initial = $λ" for λ in (1.0, 1.0e-8)
        prob = NonlinearLeastSquaresProblem(rosenbrock, [-2.0, 3.0])
        alg = LevenbergMarquardt(; disable_geodesic = Val(true), damping_initial = λ)
        cache = init(prob, alg; abstol = 1.0e-12)
        tr = cache.trustregion_cache
        for _ in 1:50
            cost, u, v_old, norm_v_old = norm(cache.fu), copy(cache.u), copy(tr.v_cache),
                tr.norm_v_old
            nf = cache.stats.nf
            step!(cache)
            cache.stats.nf == nf && continue
            trial = norm(tr.fu_cache)
            δu = tr.u_cache - u
            β = dot(δu, v_old) / (norm(δu) * norm_v_old)
            trial < cost && @test tr.last_step_accepted
            trial > cost && β ≤ 0 && @test !tr.last_step_accepted
        end
    end
end
