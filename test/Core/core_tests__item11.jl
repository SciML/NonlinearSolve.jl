using NonlinearSolve

no_ad_fast = FastShortcutNonlinearPolyalg(autodiff = AutoFiniteDiff())
no_ad_robust = RobustMultiNewton(autodiff = AutoFiniteDiff())
no_ad_algs = Set([no_ad_fast, no_ad_robust, no_ad_fast.algs..., no_ad_robust.algs...])

@testset "Inplace" begin
    f_iip = Base.Experimental.@opaque (du, u, p) -> du .= u .* u .- p
    u0 = [0.5]
    prob = NonlinearProblem(f_iip, u0, 1.0)
    for alg in no_ad_algs
        sol = solve(prob, alg)
        @test isapprox(only(sol.u), 1.0)
        @test SciMLBase.successful_retcode(sol.retcode)
    end
end

@testset "Out of Place" begin
    f_oop = Base.Experimental.@opaque (u, p) -> u .* u .- p
    u0 = [0.5]
    prob = NonlinearProblem{false}(f_oop, u0, 1.0)
    for alg in no_ad_algs
        sol = solve(prob, alg)
        @test isapprox(only(sol.u), 1.0)
        @test SciMLBase.successful_retcode(sol.retcode)
    end
end
