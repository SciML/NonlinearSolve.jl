# Frequently Asked Questions

## How is the performance of Julia's NonlinearSolve.jl vs MATLAB's fzero?

This is addressed in a [Twitter thread with the author of the improved fzero](https://twitter.com/ChrisRackauckas/status/1544743542094020615).
On the test example:

```@example
using NonlinearSolve, BenchmarkTools

const N = 100_000;
levels = 1.5 .* rand(N);
out = zeros(N);
myfun(x, lv) = x * sin(x) - lv

function f(out, levels, u0)
    for i in 1:N
        out[i] = solve(
            IntervalNonlinearProblem{false}(
                IntervalNonlinearFunction{false}(myfun), u0, levels[i]),
            Falsi()).u
    end
end

function f2(out, levels, u0)
    for i in 1:N
        out[i] = solve(
            NonlinearProblem{false}(NonlinearFunction{false}(myfun), u0, levels[i]),
            SimpleNewtonRaphson()).u
    end
end

@btime f(out, levels, (0.0, 2.0))
@btime f2(out, levels, 1.0)
```

MATLAB 2022a achieves 1.66s. Try this code yourself: we receive 0.009 seconds, or a 184x
speedup.

For more information on performance of SciML, see the [SciMLBenchmarks](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/).

## The solver tried to set a Dual Number in my Vector of Floats. How do I fix that?

This is a common problem that occurs if the code was not written to be generic based on the
input types. For example, consider this example taken from
[this issue](https://github.com/SciML/NonlinearSolve.jl/issues/298)

```@example dual_error_faq
using NonlinearSolve, Random

function fff_incorrect(var, p)
    v_true = [1.0, 0.1, 2.0, 0.5]
    xx = [1.0, 2.0, 3.0, 4.0]
    xx[1] = var[1] - v_true[1]
    return var - v_true
end

v_true = [1.0, 0.1, 2.0, 0.5]
v_init = v_true .+ randn!(similar(v_true)) * 0.1

prob_oop = NonlinearLeastSquaresProblem{false}(fff_incorrect, v_init)
try
    sol = solve(prob_oop, LevenbergMarquardt(); maxiters = 10000, abstol = 1e-8)
catch e
    @error e
end
```

Essentially what happened was, NonlinearSolve checked that we can use ForwardDiff.jl to
differentiate the function based on the input types. However, this function has
`xx = [1.0, 2.0, 3.0, 4.0]` followed by a `xx[1] = var[1] - v_true[1]` where `var` might
be a Dual number. This causes the error. To fix it:

1. Specify the `autodiff` to be `AutoFiniteDiff`

   ```@example dual_error_faq
   sol = solve(prob_oop, LevenbergMarquardt(; autodiff = AutoFiniteDiff());
       maxiters = 10000, abstol = 1e-8)
   ```

   This worked but, Finite Differencing is not the recommended approach in any scenario.

2. Rewrite the function to use
   [PreallocationTools.jl](https://github.com/SciML/PreallocationTools.jl) or write it as

   ```@example dual_error_faq
   function fff_correct(var, p)
       v_true = [1.0, 0.1, 2.0, 0.5]
       xx = eltype(var)[1.0, 2.0, 3.0, 4.0]
       xx[1] = var[1] - v_true[1]
       return xx - v_true
   end

   prob_oop = NonlinearLeastSquaresProblem{false}(fff_correct, v_init)
   sol = solve(prob_oop, LevenbergMarquardt(); maxiters = 10000, abstol = 1e-8)
   ```

## I thought NonlinearSolve.jl was type-stable and fast. But it isn't, why?

It is hard to say why your code is not fast. Take a look at the
[Diagnostics API](@ref diagnostics_api) to pin-point the problem. One common issue is that
there is type instability.

If you are using the defaults for the autodiff and your problem is not a scalar or using
static arrays, ForwardDiff will create type unstable code and lead to dynamic dispatch
internally. See this simple example:

```@example type_unstable
using NonlinearSolve, InteractiveUtils

f(u, p) = @. u^2 - p

prob = NonlinearProblem{false}(f, 1.0, 2.0)

@code_warntype solve(prob, NewtonRaphson())
nothing # hide
```

Notice that this was type-stable, since it is a scalar problem. Now what happens for static
arrays

```@example type_unstable
using StaticArrays

prob = NonlinearProblem{false}(f, @SVector([1.0, 2.0]), 2.0)

@code_warntype solve(prob, NewtonRaphson())
nothing # hide
```

Again Type-Stable! Now let's try using a regular array:

```@example type_unstable
prob = NonlinearProblem(f, [1.0, 2.0], 2.0)

@code_warntype solve(prob, NewtonRaphson())
nothing # hide
```

Ah it is still type stable. But internally since the chunksize is not statically inferred,
it will be dynamic and lead to dynamic dispatch. To fix this, we directly specify the
chunksize:

```@example type_unstable
@code_warntype solve(
    prob,
    NewtonRaphson(;
        autodiff = AutoForwardDiff(; chunksize = NonlinearSolve.pickchunksize(prob.u0))
    )
)
nothing # hide
```

And boom! Type stable again. We always recommend picking the chunksize via
[`NonlinearSolveBase.pickchunksize`](@ref), however, if you manually specify the chunksize, it
must be `≤ length of input`. However, a very large chunksize can lead to excessive
compilation times and slowdown.

```@docs
NonlinearSolveBase.pickchunksize
```
