# [Symbolic Nonlinear System Definition and Acceleration via ModelingToolkit](@id modelingtoolkit)

[ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/dev/) is a symbolic-numeric
modeling system for the Julia SciML ecosystem. It adds a high-level interactive interface
for the numerical solvers which can make it easy to symbolically modify and generate
equations to be solved. The basic form of using ModelingToolkit looks as follows:

```@example mtk
using ModelingToolkit, NonlinearSolve

@variables x y z
@parameters σ ρ β

# Define a nonlinear system
eqs = [0 ~ σ * (y - x),
    0 ~ x * (ρ - z) - y,
    0 ~ x * y - β * z]
@named ns = NonlinearSystem(eqs, [x, y, z], [σ, ρ, β])

u0 = [x => 1.0,
    y => 0.0,
    z => 0.0]

ps = [σ => 10.0
      ρ => 26.0
      β => 8 / 3]

prob = NonlinearProblem(ns, u0, ps)
sol = solve(prob, NewtonRaphson())
```

## Symbolic Derivations of Extra Functions

As a symbolic system, ModelingToolkit can be used to represent the equations and derive new
forms. For example, let's look at the equations:

```@example mtk
equations(ns)
```

We can ask it what the Jacobian of our system is via `calculate_jacobian`:

```@example mtk
calculate_jacobian(ns)
```

We can tell MTK to generate a computable form of this analytical Jacobian via `jac = true`
to help the solver use efficient forms:

```@example mtk
prob = NonlinearProblem(ns, u0, ps, jac = true)
sol = solve(prob, NewtonRaphson())
```

## Symbolic Simplification of Nonlinear Systems via Tearing

One of the major reasons for using ModelingToolkit is to allow structural simplification of
the systems. It's very easy to write down a mathematical model that, in theory, could be
solved more simply. Let's take a look at a quick system:

```@example mtk
@variables u1 u2 u3 u4 u5
eqs = [
    0 ~ u1 - sin(u5),
    0 ~ u2 - cos(u1),
    0 ~ u3 - hypot(u1, u2),
    0 ~ u4 - hypot(u2, u3),
    0 ~ u5 - hypot(u4, u1)
]
@named sys = NonlinearSystem(eqs, [u1, u2, u3, u4, u5], [])
```

If we run structural simplification, we receive the following form:

```@example mtk
sys = structural_simplify(sys)
```

```@example mtk
equations(sys)
```

How did it do this? Let's look at the `observed` to see the relationships that it found:

```@example mtk
observed(sys)
```

Using ModelingToolkit, we can build and solve the simplified system:

```@example mtk
u0 = [u5 .=> 1.0]
prob = NonlinearProblem(sys, u0)
sol = solve(prob, NewtonRaphson())
```

We can then use symbolic indexing to retrieve any variable:

```@example mtk
sol[u1]
```

```@example mtk
sol[u2]
```

```@example mtk
sol[u3]
```

```@example mtk
sol[u4]
```

```@example mtk
sol[u5]
```

## Component-Based and Acausal Modeling

If you're interested in building models in a component or block based form, such as seen in
systems like Simulink or Modelica, take a deeper look at
[ModelingToolkit.jl's documentation](https://docs.sciml.ai/ModelingToolkit/stable/) which
goes into detail on such features.
