# [Optimizing a Parameterized ODE](@id optimizing-parameterized-ode)

Let us fit a parameterized ODE to some data. We will use the Lotka-Volterra model as an
example. We will use Single Shooting to fit the parameters.

```@example parameterized_ode
using OrdinaryDiffEq, NonlinearSolve, Plots
```

Let us simulate some real data from the Lotka-Volterra model.

```@example parameterized_ode
function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α * x - β * x * y
    du[2] = dy = -δ * y + γ * x * y
end

# Initial condition
u0 = [1.0, 1.0]

# Simulation interval and intermediary points
tspan = (0.0, 2.0)
tsteps = 0.0:0.1:10.0

# LV equation parameter. p = [α, β, δ, γ]
p = [1.5, 1.0, 3.0, 1.0]

# Setup the ODE problem, then solve
prob = ODEProblem(lotka_volterra!, u0, tspan, p)
sol = solve(prob, Tsit5(); saveat = tsteps)

# Plot the solution
using Plots
plot(sol; linewidth = 3)
```

Let us now formulate the parameter estimation as a Nonlinear Least Squares Problem.

```@example parameterized_ode
function loss_function(ode_param, data)
    sol = solve(prob, Tsit5(); p = ode_param, saveat = tsteps)
    return vec(reduce(hcat, sol.u)) .- data
end

p_init = zeros(4)

nlls_prob = NonlinearLeastSquaresProblem(loss_function, p_init, vec(reduce(hcat, sol.u)))
```

Now, we can use any NLLS solver to solve this problem.

```@example parameterized_ode
res = solve(nlls_prob, LevenbergMarquardt(); maxiters = 1000, show_trace = Val(true),
    trace_level = TraceWithJacobianConditionNumber(25))
nothing # hide
```

```@example parameterized_ode
res
```

We can also use Trust Region methods.

```@example parameterized_ode
res = solve(nlls_prob, TrustRegion(); maxiters = 1000, show_trace = Val(true),
    trace_level = TraceWithJacobianConditionNumber(25))
nothing # hide
```

```@example parameterized_ode
res
```

Let's plot the solution.

```@example parameterized_ode
prob2 = remake(prob; tspan = (0.0, 10.0))
sol_fit = solve(prob2, Tsit5(); p = res.u)
sol_true = solve(prob2, Tsit5(); p = p)
plot(sol_true; linewidth = 3)
plot!(sol_fit; linewidth = 3, linestyle = :dash)
```
