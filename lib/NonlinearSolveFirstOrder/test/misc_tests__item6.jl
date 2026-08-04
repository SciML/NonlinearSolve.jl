using NonlinearSolveFirstOrder, LinearSolve, SciMLBase

f(u, p) = u .* u .- p
prob = NonlinearProblem(f, [1.0, 1.0], 2.0)
cache = init(prob, NewtonRaphson(; linsolve = KrylovJL_GMRES(), forcing = EisenstatWalkerForcing2()))

fc = cache.forcing_cache
sol = solve!(cache)
@test SciMLBase.successful_retcode(sol)
@test fc.η != fc.p.η₀

# reinit! with a new p must not throw and must reset the forcing state
reinit!(cache; p = 3.0)
@test fc.η == fc.p.η₀

# the re-solve must refresh the residual norms and solve the new problem
@test solve!(cache).u ≈ [sqrt(3.0), sqrt(3.0)]
