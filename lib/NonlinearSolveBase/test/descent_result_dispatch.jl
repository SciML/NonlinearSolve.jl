using NonlinearSolveBase, Test

# Historical public parameter prefix is `{DU,U,Extras}`; traced Bool carriers append after.
r = NonlinearSolveBase.DescentResult(; δu = [1.0])
read_extras(d::NonlinearSolveBase.DescentResult{D, U, E}) where {D, U, E <: NamedTuple} = d.extras
@test read_extras(r) == NamedTuple()
@test r.success === true
@test r.linsolve_success === true
