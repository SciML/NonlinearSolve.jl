using NonlinearSolveQuasiNewton, ForwardDiff, SciMLBase, Test

# The update rules difference the current residual against the previous iterate's, which
# they hold across iterations. `reinit!` has to reseed it, or the first secant update of
# the reused cache differences against the residual at the previous problem's root.
quadratic_f(u, p) = u .^ 2 .- p
quadratic_f!(du, u, p) = (du .= u .^ 2 .- p; nothing)

algs = (
    "Broyden" => Broyden(),
    "Broyden (bad)" => Broyden(; update_rule = Val(:bad_broyden)),
    "Broyden (diagonal)" => Broyden(; update_rule = Val(:diagonal)),
    "Broyden (true jacobian)" => Broyden(; init_jacobian = Val(:true_jacobian)),
    "Klement" => Klement(),
    "Klement (true jacobian diagonal)" => Klement(;
        init_jacobian = Val(:true_jacobian_diagonal)
    ),
    "LimitedMemoryBroyden" => LimitedMemoryBroyden(),
)

problems = (
    "oop" => (quadratic_f, [1.5, 1.5], [2.0, 3.0], [1.0, 1.0], [9.0, 16.0], false),
    "iip" => (quadratic_f!, [1.5, 1.5], [2.0, 3.0], [1.0, 1.0], [9.0, 16.0], true),
    "scalar" => (quadratic_f, 1.5, 2.0, 1.0, 9.0, false),
)

@testset "$(algname)" for (algname, alg) in algs
    @testset "$(probname)" for (probname, (f, u0A, pA, u0B, pB, iip)) in problems
        probA = NonlinearProblem{iip}(f, u0A, pA)
        probB = NonlinearProblem{iip}(f, u0B, pB)

        fresh = init(probB, alg)
        solve!(fresh)

        reused = init(probA, alg)
        solve!(reused)
        SciMLBase.reinit!(reused, u0B; p = pB)
        solve!(reused)

        @test reused.u == fresh.u
        @test reused.nsteps == fresh.nsteps
        @test reused.stats.nf == fresh.stats.nf
    end
end
