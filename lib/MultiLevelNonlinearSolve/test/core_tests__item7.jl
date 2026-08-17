using MultiLevelNonlinearSolve, Test
include("setup_barmodel.jl")

using LinearSolve: KrylovJL_GMRES
using LinearAlgebra, SparseArrays

# T5 — the linear-preconditioning slot. `precs` belongs to the linear solver inside the
# global solver; it is a different knob from the local forcing, the linear forcing and the
# nonlinear preconditioning, and `MultiLevelNewton` exposes none of them itself.
#
# What is checked: the preconditioner is built from the Schur matrix itself, it can see the
# current `ū`, and it is rebuilt exactly as often as `S` is assembled. That last property is
# what makes `jacobian_reuse = :chord` worth anything when `precs` is an ILU or AMG setup
# whose cost dominates the step.
@testset "T5 precs is built from S and rebuilt with it — $(label)" for (
        label, reuse, chord_after,
    ) in
                                                                      (
        ("always", :always, 0), ("chord", :chord, 1),
    )
    same_A, same_u, nbuilds = Bool[], Bool[], Ref(0)
    expected = Ref{Any}(nothing)      # filled with the cache once it exists

    function spy_precs(A, p)
        nbuilds[] += 1
        cache = expected[]
        if cache !== nothing
            push!(same_A, A === cache.global_cache.jac_cache.J)
            push!(same_u, p.u === NonlinearSolveBase.get_u(cache.global_cache))
        end
        return (Diagonal(diag(A)), I)     # a real (Jacobi) preconditioner, not a no-op
    end

    prob, model = bar_problem()
    alg = MultiLevelNewton(;
        jacobian_reuse = reuse,
        global_solver = NewtonRaphson(;
            linsolve = KrylovJL_GMRES(; precs = spy_precs), concrete_jac = true
        )
    )
    cache = init(prob, alg; abstol = 1.0e-12, maxiters = 100)
    expected[] = cache
    residual_history(cache; chord_after)

    @test !isempty(same_A)
    @test all(same_A)          # the preconditioner is built from the Schur matrix itself
    @test all(same_u)          # and `p.u` aliases the current condensed iterate

    # One build at `init`, then one per assembly.
    @test nbuilds[] == model.counters.nassembly + 1
    if reuse === :chord
        @test model.counters.nassembly == 1
        @test nbuilds[] == 2
    else
        @test SciMLBase.successful_retcode(cache.retcode)
        @test model.counters.nassembly > 1
    end
end

# T17 — Eisenstat–Walker linear forcing must actually adapt across global iterations. It is
# driven by the global solver's own iteration counter, which `MultiLevelNewton` has to
# advance itself because it steps the condensed cache through the internal entry point that
# bypasses the counter. The failure mode is silent: η stays pinned at η₀ for the whole solve
# and every linear solve runs at reltol 0.5.
@testset "T17 Eisenstat-Walker forcing adapts" begin
    prob, _ = bar_problem()
    alg = MultiLevelNewton(;
        global_solver = NewtonRaphson(;
            linsolve = KrylovJL_GMRES(), concrete_jac = true,
            forcing = EisenstatWalkerForcing2()
        )
    )
    cache = init(prob, alg; abstol = 1.0e-10, maxiters = 50)
    forcing = cache.global_cache.forcing_cache
    @test forcing !== nothing

    ηs = Float64[]
    run_steps!(cache; maxsteps = 50, each = _ -> push!(ηs, forcing.η))
    @test SciMLBase.successful_retcode(cache.retcode)
    @test length(unique(ηs)) > 1
    @test any(!=(0.5), ηs)     # η₀ = 0.5 frozen for the whole solve is the failure mode
    @test cache.global_cache.nsteps == cache.nsteps
end

# A multi-level problem cannot be solved matrix-free: `S` is assembled, not differentiated.
# A Krylov linear solver without `concrete_jac` makes the global solver skip the assembly
# entirely and drive the solve through the Jacobian-vector product, which then has no `S`.
# That has to say so rather than fail somewhere inside the Krylov workspace.
@testset "a matrix-free global solver is rejected with an explanation" begin
    prob, _ = bar_problem()
    alg = MultiLevelNewton(; global_solver = NewtonRaphson(; linsolve = KrylovJL_GMRES()))
    cache = init(prob, alg; abstol = 1.0e-12)
    @test_throws ArgumentError step!(cache)
end
