using NonlinearSolveQuasiNewton, NonlinearSolveBase, SciMLBase, Test

# An in-place problem on a view `u0`. u^3 + u - q is increasing, so its only real root is
# r with r^3 + r = q: q = 2 gives 1, q = 10 gives 2 and q = 30 gives 3.
f_view! = NonlinearFunction{true, SciMLBase.FullSpecialize}(
    (du, u, q) -> (du .= u .^ 3 .+ u .- q)
)

@testset "View u0: $(nameof(typeof(alg))), alias_u0 = $alias_u0" for alg in (
            Broyden(), Klement(), LimitedMemoryBroyden(),
        ), alias_u0 in (false, true)

    parent_u0 = [3.0, 0.0, 3.0]
    u0 = view(parent_u0, [1, 3])
    alias = SciMLBase.NonlinearAliasSpecifier(; alias_u0)
    prob = NonlinearProblem(f_view!, u0, [2.0, 10.0])

    sol = solve(prob, alg; alias, abstol = 1.0e-10)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ [1.0, 2.0]
    @test sol.u isa typeof(u0)
    @test parent_u0[[1, 3]] ≈ (alias_u0 ? [1.0, 2.0] : [3.0, 3.0])
    @test parent_u0[2] == 0.0

    parent_u0 .= [3.0, 0.0, 3.0]
    cache = init(prob, alg; alias, abstol = 1.0e-10)
    reinit!(cache, [2.0, 2.0]; p = [10.0, 30.0])
    @test solve!(cache).u ≈ [2.0, 3.0]

    parent_u0 .= [3.0, 0.0, 3.0]
    cache = init(prob, alg; alias, abstol = 1.0e-10)
    SciMLBase.set_u!(cache, [2.0, 2.0])
    @test NonlinearSolveBase.get_u(cache) == [2.0, 2.0]
    @test solve!(cache).u ≈ [1.0, 2.0]
end
