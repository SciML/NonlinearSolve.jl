using NonlinearSolveQuasiNewton, SciMLBase, Test

cubic(u, p) = u.^3 .+ u .- p
cubic!(du, u, p) = (du .= u.^3 .+ u .- p; nothing)

algs = (
    "Broyden" => Broyden(),
    "Broyden (bad)" => Broyden(; update_rule = Val(:bad_broyden)),
    "Broyden (true jacobian)" => Broyden(; init_jacobian = Val(:true_jacobian)),
    "Klement" => Klement(),
    "Klement (true jacobian diagonal)" => Klement(; init_jacobian = Val(
        :true_jacobian_diagonal,
    )),
    "LimitedMemoryBroyden" => LimitedMemoryBroyden(),
)
true_jacobian_algs = ("Broyden (true jacobian)", "Klement (true jacobian diagonal)")
kw = (; abstol = 1.0e-10, reltol = 1.0e-10)

@testset "$(algname)" for (algname, alg) in algs
    @testset "$(name)" for (name, f, iip) in (("oop", cubic, false), ("iip", cubic!, true))
        carried = init(NonlinearProblem{iip}(f, fill(1.5, 4), 2.0), alg; kw...)
        solve!(carried)
        for p in (2.1, 2.2)
            reinit!(carried, copy(carried.u); p, reuse_jacobian = true)
            sol = solve!(carried)

            plain = init(NonlinearProblem{iip}(f, fill(1.5, 4), p), alg; kw...)
            plain_sol = solve!(plain)

            @test SciMLBase.successful_retcode(sol)
            @test sol.u ≈ plain_sol.u rtol = 1.0e-8
            if algname in true_jacobian_algs
                @test carried.stats.njacs == 0
                @test plain.stats.njacs ≥ 1
            end
        end
    end
end

@testset "a forced refresh still reinitializes a carried Jacobian" begin
    alg = Broyden(; init_jacobian = Val(:true_jacobian))
    cache = init(NonlinearProblem(cubic, fill(1.5, 4), 2.0), alg; kw...)
    solve!(cache)
    reinit!(cache, copy(cache.u); p = 2.1, reuse_jacobian = true)
    step!(cache; recompute_jacobian = true)
    @test cache.stats.njacs == 1
    @test SciMLBase.successful_retcode(solve!(cache))
end

@testset "reuse_jacobian before any step has no Jacobian to carry" begin
    alg = Broyden(; init_jacobian = Val(:true_jacobian))
    cache = init(NonlinearProblem(cubic, fill(1.5, 4), 2.0), alg; kw...)
    reinit!(cache, fill(1.5, 4); p = 2.1, reuse_jacobian = true)
    @test SciMLBase.successful_retcode(solve!(cache))
    @test cache.stats.njacs ≥ 1
end
