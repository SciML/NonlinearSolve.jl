# Solving Nonlinear Problems with Bound Constraints

Box constraints specify lower and upper bounds on each unknown. Supply `lb` and `ub`
when constructing a `NonlinearProblem` or `NonlinearLeastSquaresProblem`.
For algorithm choices and performance guidance, see [Bounded Solvers](@ref bounded-solvers).

## Finding a root in a box

```@example bounds
import NonlinearSolve as NLS
import SciMLBase

f(u, p) = u .* u .- p
prob = NLS.NonlinearProblem(
    f, [1.0, 1.0], [2.0, 3.0]; lb = [0.0, 0.0], ub = [10.0, 10.0]
)
sol = NLS.solve(prob)
@assert SciMLBase.successful_retcode(sol)
@assert all(0.0 .<= sol.u .<= 10.0)
sol.u
```

A root problem requires a small residual. If no root exists inside the box, a
constrained least-squares stationary point is not a successful root solve.

## Fitting bounded parameters

Fit the amplitude and decay rate of an exponential curve while restricting their
physical ranges:

```@example bounds
x = collect(range(0.0, 3.0; length = 20))
y = 2.0 .* exp.(-0.5 .* x)
model(u, p) = u[1] .* exp.(u[2] .* p.x) .- p.y
nf = NLS.NonlinearFunction(model; resid_prototype = zeros(length(x)))
lb, ub = [0.0, -2.0], [5.0, 0.0]
u0 = [1.0, -1.0]
fit = NLS.NonlinearLeastSquaresProblem(nf, u0, (; x, y); lb, ub)
sol_fit = NLS.solve(fit)
@assert SciMLBase.successful_retcode(sol_fit)
sol_fit.u
```

Bounds can actively constrain the optimum. Start inside the new box when changing
its bounds:

```@example bounds
lb_tight = [3.0, -2.0]
fit_tight = NLS.NonlinearLeastSquaresProblem(
    nf, clamp.(u0, lb_tight, ub), (; x, y); lb = lb_tight, ub
)
sol_tight = NLS.solve(fit_tight)
@assert all(lb_tight .<= sol_tight.u .<= ub)
sol_tight.u
```

A least-squares solve can succeed with a nonzero residual when the projected merit
gradient is small. This means it has found a constrained stationary point; local
methods do not guarantee a global minimum.

## Other bound layouts

Pass only `lb` or only `ub` for one-sided bounds. Individual `-Inf` and `Inf` entries
leave coordinates unbounded. Equal lower and upper bounds fix a coordinate.
Scalar bounds apply to every coordinate; array bounds must match the state shape.

In-place residual functions use the same problem keywords:

```@example bounds
function residual!(r, u, p)
    r .= u .- p
    return nothing
end
prob_ip = NLS.NonlinearLeastSquaresProblem(
    residual!, [0.0, 0.5], [2.0, 0.5]; lb = [0.0, 0.5], ub = [1.0, 0.5]
)
sol_ip = NLS.solve(prob_ip)
@assert sol_ip.u ≈ [1.0, 0.5]
sol_ip.u
```
