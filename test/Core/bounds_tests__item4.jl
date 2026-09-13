using NonlinearSolve

using SciMLBase
using NonlinearSolveBase

# The quasi-Newton `autodiff`-access fix lives in NonlinearSolveBase, so only
# exercise the full solve when a NonlinearSolveBase new enough to contain it is
# actually loaded. On Julia < 1.11 the umbrella resolves the *registered*
# NonlinearSolveBase (which predates the fix) because `[sources]` path redirects
# are ignored there; the fix itself is unit-tested in NonlinearSolveBase's own
# suite, which always runs against the in-repo code. See SciML/NonlinearSolve.jl#955.
if pkgversion(NonlinearSolveBase) >= v"2.30.3"
    # `x^2 - 4x + 3` has roots at 1 and 3; bounds select which root is reachable.
    f(u, p) = u .^ 2 .- 4 .* u .+ 3

    # The default polyalgorithm tries quasi-Newton methods (which have no `autodiff`
    # field) as part of its sequence; the bounds transform must not error on them.
    for alg in (nothing, FastShortcutNonlinearPolyalg(), Broyden(), Klement(), NewtonRaphson())
        prob = NonlinearProblem(f, [1.5], nothing; lb = [0.0], ub = [2.0])
        sol = alg === nothing ? solve(prob) : solve(prob, alg)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u[1] ≈ 1.0 atol = 1.0e-6
        @test 0.0 <= sol.u[1] <= 2.0
    end
end

@testset "transformed analytic Jacobians and products" begin
    using LinearAlgebra, LinearSolve, SparseArrays

    for iip in (false, true), representation in (:dense, :sparse, :operator)
        f(u, p) = u .^ 2 .- p
        jac(u, p) = Diagonal(2 .* u)
        jvp(v, u, p) = 2 .* u .* v
        f!(r, u, p) = (r .= f(u, p))
        jac!(J, u, p) = (J .= jac(u, p))
        jvp!(Jv, v, u, p) = (Jv .= jvp(v, u, p))
        prototype = representation === :sparse ? sparse(Diagonal(ones(2))) : nothing
        nf = if iip
            NonlinearFunction{true}(
                f!; jac = representation === :operator ? nothing : jac!,
                jvp = jvp!, vjp = jvp!, jac_prototype = prototype
            )
        else
            NonlinearFunction{false}(
                f; jac = representation === :operator ? nothing : jac,
                jvp, vjp = jvp, jac_prototype = prototype
            )
        end
        linsolve = representation === :operator ? KrylovJL_GMRES() : nothing
        prob = NonlinearProblem(nf, [1.0, 1.5], [4.0, 9.0]; lb = 0.0, ub = 10.0)
        cache = init(prob, NewtonRaphson(; linsolve); abstol = 1.0e-10, reltol = 1.0e-10)
        J = cache.jac_cache(cache.u)
        representation === :sparse && @test issparse(J)
        representation === :operator && @test !(J isa AbstractMatrix)
        sol = solve!(cache)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [2.0, 3.0] atol = 1.0e-8
    end

    prob = NonlinearProblem(
        NonlinearFunction((u, p) -> u^2 - p; jac = (u, p) -> 2u),
        1.0, 4.0; lb = 0.0, ub = 10.0
    )
    sol = solve(prob, NewtonRaphson(); abstol = 1.0e-10, reltol = 1.0e-10)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ 2.0 atol = 1.0e-8
end
