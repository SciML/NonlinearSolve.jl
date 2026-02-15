# Solving Nonlinear Problems with Bound Constraints

Many real-world problems have physical constraints on the parameters — concentrations must
be positive, probabilities must lie in $$[0, 1]$$, and so on. NonlinearSolve.jl supports
**box constraints** (lower and upper bounds on each variable) for both `NonlinearProblem`
and `NonlinearLeastSquaresProblem`.

## How It Works

When you pass `lb` and/or `ub` to a problem, NonlinearSolve checks whether the chosen
algorithm natively supports bounds. If it does not, the solver automatically applies a
**variable transformation** that maps the bounded variables into unconstrained space using
the logistic/logit functions. After solving, the solution is mapped back to the original
bounded space. This means you can use any algorithm — `NewtonRaphson`, `TrustRegion`,
`LevenbergMarquardt`, etc. — with bounds, without changing anything about your solver setup.

## Basic Example: Nonlinear System with Bounds

Let's solve a simple nonlinear system $$f(u) = u^2 - p$$ where we constrain the solution
to be positive:

```@example bounds
import NonlinearSolve as NLS

f(u, p) = u .* u .- p
u0 = [1.0, 1.0]
p = [2.0, 3.0]

prob = NLS.NonlinearProblem(f, u0, p; lb = [0.0, 0.0], ub = [10.0, 10.0])
sol = NLS.solve(prob, NLS.NewtonRaphson())
```

We can verify the solution satisfies the bounds:

```@example bounds
all(0.0 .<= sol.u .<= 10.0)
```

## Curve Fitting with Bounded Parameters

A common use case is nonlinear least squares curve fitting where parameters have physical
meaning and known ranges. Here we fit the model $$y = a \cdot e^{b x}$$ to data,
constraining the amplitude $$a > 0$$ and the decay rate $$b < 0$$:

```@example bounds
import NonlinearSolve as NLS

true_a, true_b = 2.0, -0.5
x = collect(range(0.0, 3.0; length = 20))
y = true_a .* exp.(true_b .* x)

model(u, p) = u[1] .* exp.(u[2] .* p) .- y
nf = NLS.NonlinearFunction(model; resid_prototype = zeros(20))

u0 = [1.0, -1.0]
prob = NLS.NonlinearLeastSquaresProblem(nf, u0, x; lb = [0.0, -2.0], ub = [5.0, 0.0])
sol = NLS.solve(prob, NLS.LevenbergMarquardt())
```

The fitted parameters should be close to the true values:

```@example bounds
sol.u
```

And they respect the bounds:

```@example bounds
all(sol.u .>= [0.0, -2.0]) && all(sol.u .<= [5.0, 0.0])
```

### When Bounds Actively Constrain the Solution

If the unconstrained optimum lies outside the feasible region, the solver finds the best
solution within the bounds:

```@example bounds
prob_tight = NLS.NonlinearLeastSquaresProblem(
    nf, u0, x; lb = [3.0, -2.0], ub = [5.0, -0.1]
)
sol_tight = NLS.solve(prob_tight, NLS.LevenbergMarquardt())
sol_tight.u
```

The true amplitude is 2.0, but the lower bound forces it to 3.0 or above:

```@example bounds
sol_tight.u[1] >= 3.0
```

## One-Sided Bounds

You don't need to specify both a lower and an upper bound. Use `-Inf` or `Inf` entries to
leave a direction unconstrained, or pass only `lb` or only `ub`:

```@example bounds
import NonlinearSolve as NLS

f(u, p) = u .- p
nf = NLS.NonlinearFunction(f; resid_prototype = zeros(2))

u0 = [5.0, 5.0]
p = [1.0, 2.0]

# Lower bound only — keep parameters positive
prob_lb = NLS.NonlinearLeastSquaresProblem(nf, u0, p; lb = [0.0, 0.0])
sol_lb = NLS.solve(prob_lb, NLS.LevenbergMarquardt())
sol_lb.u
```

```@example bounds
# Per-variable: first variable unbounded below, second bounded below at 0
prob_mixed = NLS.NonlinearLeastSquaresProblem(nf, u0, p; lb = [-Inf, 0.0])
sol_mixed = NLS.solve(prob_mixed, NLS.LevenbergMarquardt())
sol_mixed.u
```

## In-Place Formulation

Bounds work the same way with in-place problem formulations:

```@example bounds
import NonlinearSolve as NLS

function f_ip!(resid, u, p)
    resid .= u .- p
    return nothing
end

nf = NLS.NonlinearFunction(f_ip!)
prob = NLS.NonlinearLeastSquaresProblem(nf, [5.0, 5.0], [1.0, 2.0];
    lb = [0.0, 0.0], ub = [10.0, 10.0])

sol = NLS.solve(prob, NLS.LevenbergMarquardt())
sol.u
```

The original problem (with bounds) is preserved in the solution object:

```@example bounds
sol.prob.lb, sol.prob.ub
```

## Notes

  - **Any algorithm works.** Algorithms that natively support bounds use their own handling;
    algorithms that don't get automatic variable transformation.
  - **The transformation.** For two-sided bounds $$[\ell, u]$$, the transform is
    $$t = \text{logit}\!\left(\frac{x - \ell}{u - \ell}\right)$$ with inverse
    $$x = \ell + (u - \ell)\,\sigma(t)$$ where $$\sigma$$ is the logistic function. For
    one-sided bounds, a simple $$\log$$/$$\exp$$ transform is used.
  - **Initial guess.** If `u0` is exactly on a bound, it is automatically nudged into the
    strict interior to avoid numerical issues.
