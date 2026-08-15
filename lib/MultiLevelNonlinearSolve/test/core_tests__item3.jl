using MultiLevelNonlinearSolve, Test
include("setup_barmodel.jl")

using LineSearch: BackTracking
using LinearAlgebra, SparseArrays

# T8 — the reuse signal reaches the Schur assembler. `njacs` alone cannot show this: it only
# counts assemblies the Jacobian cache makes, and the whole point of F1 is that other parts
# of the solver can call `assemble_S!` behind its back. The fixture's own counter is checked
# alongside it.
@testset "T8 recompute_jacobian = false suppresses the assembly" begin
    prob, model = bar_problem()
    cache = init(prob, MultiLevelNewton(); abstol = 1.0e-12)

    step!(cache)
    njacs, nassembly = cache.stats.njacs, model.counters.nassembly
    @test njacs == 1
    @test nassembly == 1

    step!(cache; recompute_jacobian = false)
    @test cache.stats.njacs == njacs
    @test model.counters.nassembly == nassembly

    step!(cache; recompute_jacobian = true)
    @test cache.stats.njacs == njacs + 1
    @test model.counters.nassembly == nassembly + 1
end

# T14 — every assembly happens at an iterate whose committed internal state already solves
# the local problems there, so the assembler may read the committed tangents instead of
# re-solving. Checked directly: at each assembly, the committed `q` is a converged root of
# the local problem at that `ū`.
@testset "T14 S is assembled at committed iterates — $(name)" for (name, alg) in
                                                                 (
        ("NewtonRaphson", MultiLevelNewton()),
        ("TrustRegion", MultiLevelNewton(; global_solver = TrustRegion())),
    )
    worst = Float64[]
    function checking_assembly!(S, ū, p)
        model = user_parameters(p)
        for i in 1:model.n
            ε = strain(ū, model, i)
            q = committed_state(model.buffer, i)
            r = q .- GAMMA .* tanh.(elem_c(model, i) .* ε .+ D_MAT * q)
            push!(worst, norm(r, Inf))
        end
        return assemble_S!(S, ū, p)
    end

    prob, model = bar_problem(; jac = checking_assembly!)
    sol = solve(prob, alg; abstol = 1.0e-12)
    @test SciMLBase.successful_retcode(sol)
    @test !isempty(worst)
    @test maximum(worst) ≤ model.local_tol
end

# T16 — a line search must not assemble `S` itself. Without an analytic `jvp` on the condensed
# function, the directional derivative falls back to allocating a dense `n̄ × n̄` matrix and
# calling the assembler into it once per step, which is invisible to `njacs`, fatal at FEM
# sizes, and not even type-correct for a sparse assembler. The auto-wired `jvp` removes it.
@testset "T16 a line search triggers no extra assembly" begin
    prob, model = bar_problem()
    alg = MultiLevelNewton(;
        global_solver = NewtonRaphson(; linesearch = BackTracking())
    )
    sol = solve(prob, alg; abstol = 1.0e-12)
    @test SciMLBase.successful_retcode(sol)
    @test model.counters.nassembly == sol.stats.njacs
    # The assembler only ever sees its own sparse storage — never a dense fallback buffer.
    @test all(T -> T <: SparseMatrixCSC, model.counters.assembly_types)

    # The `jvp` is what does it: with one supplied by the user it is left alone, and with
    # neither the fallback would reappear, so check the wiring is actually in place.
    cache = init(prob, alg; abstol = 1.0e-12)
    @test SciMLBase.has_jvp(cache.global_cache.prob.f)
end
