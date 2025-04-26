# BracketingNonlinearSolve.jl

These methods can be used independently of the rest of NonlinearSolve.jl

```@index
Pages = ["bracketingnonlinearsolve.md"]
```

## Interval Methods

These methods are suited for interval (scalar) root-finding problems,
i.e. [`IntervalNonlinearProblem`](@ref).

```@docs
Alefeld
Bisection
Brent
Falsi
ITP
Muller
Ridder
```
