using NonlinearSolve, NonlinearSolveBase, SciMLBase, Test

# A symbolic system whose `remake` hook is not inferable, as happens for
# ModelingToolkit problems whose parameters are `DespecializedParameters`.
# With `alias_u0 = true` the generated polyalgorithm solve must still infer a
# concrete subproblem type and solution.
struct UninferableRemakeSys end

@static if isdefined(SciMLBase, :LateBindingUpdateU0PContext)
    function SciMLBase.late_binding_update_u0_p(
            prob, ::UninferableRemakeSys, u0, p, t0, newu0, newp,
            ::SciMLBase.LateBindingUpdateU0PContext
        )
        return Base.inferencebarrier(newu0), newp
    end
else
    function SciMLBase.late_binding_update_u0_p(
            prob, ::UninferableRemakeSys, u0, p, t0, newu0, newp
        )
        return Base.inferencebarrier(newu0), newp
    end
end

f!(du, u, p) = (du .= u .* u .- p; nothing)
nlf = NonlinearFunction{true}(f!; sys = UninferableRemakeSys())
prob = NonlinearProblem(nlf, [1.0, 1.0], [2.0, 3.0])
alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = true)

@test !isconcretetype(
    Base.infer_return_type(
        (prob, u0) -> SciMLBase.remake(prob; u0), (typeof(prob), Vector{Float64})
    )
)

for alg in (
        NonlinearSolvePolyAlgorithm((Broyden(), NewtonRaphson())),
        FastShortcutNonlinearPolyalg(),
    )
    sol = @inferred NonlinearSolveBase.__generated_polysolve(prob, alg; alias)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ sqrt.(prob.p)
end
