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
bounded space. This means you can use algorithms such as `NewtonRaphson`, `TrustRegion`,
and `LevenbergMarquardt` with bounds without changing the solver setup.

[`BoundedTrustRegion`](@ref) instead handles bounds directly. It keeps every iterate in the
original coordinates and can converge to a least-squares optimum exactly on a bound.

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

## Native Bound Handling

Use [`BoundedTrustRegion`](@ref) when the solver should operate directly in the bounded
coordinates:

```@example bounds
import NonlinearSolve as NLS

residual(u, p) = [u[1] - 2.0, 1.0]
prob_native = NLS.NonlinearLeastSquaresProblem(
    residual, [0.0], nothing; lb = [-1.0], ub = [1.0]
)
sol_native = NLS.solve(prob_native, NLS.BoundedTrustRegion())
sol_native.u
```

The unconstrained minimum is outside the box, so the constrained optimum is exactly on the
upper bound:

```@example bounds
sol_native.u == [1.0]
```

`BoundedTrustRegion` is a local solver. Like other local least-squares methods, bounds do
not guarantee that it will select a different basin containing a lower minimum or a root.

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
prob = NLS.NonlinearLeastSquaresProblem(
    nf, [5.0, 5.0], [1.0, 2.0];
    lb = [0.0, 0.0], ub = [10.0, 10.0]
)

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
    strict interior before a coordinate transformation. `BoundedTrustRegion` accepts an
    initial guess exactly on a bound, but rejects an initial guess outside the feasible box.

## Reflective trust regions

[`TrustRegionReflective`](@ref) uses distance-to-bound scaling, a diagonal correction
to the quadratic model, and reflected search directions. It moves exact-bound initial
guesses into the strict interior and keeps fixed variables fixed. Its scaled linear
subproblem uses the same Jacobian and LinearSolve caches as `GaussNewton`.

```@example bounds
prob_reflective = NLS.NonlinearLeastSquaresProblem(
    (u, p) -> [u[1] - 2, u[2] - 0.5, 1.0], [0.0, 0.0]; lb = 0.0, ub = 1.0
)
sol_reflective = NLS.solve(prob_reflective, NLS.TrustRegionReflective())
sol_reflective.u
```

An analytic Jacobian or an AD backend can be used. With `AutoFiniteDiff()`, finite
difference stencils are restricted to the box and fixed coordinates are not perturbed.
Least-squares convergence includes projected-gradient stationarity; a root problem
still requires a small residual for success. The existing default is unchanged.

Sparse prototypes and Krylov solvers follow the usual solver interface:

```@example bounds
using LinearSolve, SparseArrays
f_sparse! = (r, u, p) -> (r .= u .- p)
f_sparse = NLS.NonlinearFunction(f_sparse!; jac_prototype = spdiagm(0 => ones(20)))
prob_sparse = NLS.NonlinearLeastSquaresProblem(
    f_sparse, fill(0.5, 20), collect(range(-0.5, 1.5; length = 20)); lb = 0.0, ub = 1.0
)
sol_sparse = NLS.solve(prob_sparse, NLS.TrustRegionReflective())
sol_operator = NLS.solve(prob_sparse, NLS.TrustRegionReflective(; linsolve = KrylovJL_LSMR()))
maximum(abs, sol_sparse.u - sol_operator.u)
```

`concrete_jac = true` forces a concrete Jacobian even with a Krylov solver.
`jvp_autodiff`, `vjp_autodiff`, or analytic `jvp`/`vjp` callbacks control operator
products. Linear solvers that require square systems use normal equations;
`KrylovJL_LSMR()` works on the augmented rectangular least-squares operator.
The reflective trust-region
radius is handled in a small gradient/Gauss–Newton subspace, without materializing
an operator as a matrix.

## Bound-constrained Levenberg–Marquardt

[`BoundedLevenbergMarquardt`](@ref) minimizes a damped linear least-squares model
subject to the original bounds. It permits exact-bound iterates and handles fixed
variables, rank deficiency, and underdetermined residuals. A feasible line search
globalizes the model step, with a projected-gradient fallback.

```@example bounds
sol_bounded_lm = NLS.solve(prob_reflective, NLS.BoundedLevenbergMarquardt())
sol_bounded_lm.u
```

The shared linear-solver cache preserves sparse Jacobians and matrix-free operators.
`damping` sets the initial
regularization scale and `max_backtracks` limits each line search.

## Rectangular trust regions

[`Dogbox`](@ref) uses an infinity-norm trust region intersected with the original
bounds. It solves on free variables and follows a rectangular dogleg path toward
the Gauss–Newton step.

```@example bounds
sol_dogbox = NLS.solve(prob_reflective, NLS.Dogbox())
sol_dogbox.u
```

Choose a linear solver that supports rank deficiency when it is expected. A
minimum-norm step does not ensure fast nonlinear convergence. For
example, residuals `[u[1]^2, u[1]^2]` with a second unused variable give a rank-deficient
Jacobian: the dogleg repeatedly halves `u[1]` near the solution. A small iteration
budget can therefore return `MaxIters`. Prefer [`BoundedLevenbergMarquardt`](@ref)
when rank deficiency is expected.

## Active-set Gauss–Newton

[`BoundedGaussNewton`](@ref) holds fixed variables and outward-gradient active-bound
variables fixed while solving the reduced linear least-squares problem. Choose
`globalization = :linesearch`, `:trustregion`, or `:trustregion_linesearch`; the last
option tries a projected line search when the trust-region trial is rejected.

```@example bounds
sol_bounded_gn = NLS.solve(
    prob_reflective, NLS.BoundedGaussNewton(; globalization = :trustregion_linesearch)
)
sol_bounded_gn.u
```

The reduced systems preserve the selected Jacobian representation and linear solver.
For square `NonlinearProblem`s, this
provides a feasible reduced Newton path, retaining residual-based success. It does
not solve complementarity conditions or use an exact Hessian of the merit function.
