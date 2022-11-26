# Frequently Asked Questions

## How is the performance of Julia's NonlinearSolve.jl vs MATLAB's fzero?

This is addressed in a [Twitter thread with the author of the improved fzero](https://twitter.com/ChrisRackauckas/status/1544743542094020615).
On the test example:

```@example
using NonlinearSolve, BenchmarkTools

N = 100_000;
levels = 1.5 .* rand(N);
out = zeros(N);
myfun(x, lv) = x * sin(x) - lv

function f(out, levels, u0)
    for i in 1:N
        out[i] = solve(IntervalNonlinearProblem{false}(IntervalNonlinearFunction{false}(myfun),
                u0, levels[i]), Falsi()).u
    end
end

function f2(out, levels, u0)
    for i in 1:N
        out[i] = solve(NonlinearProblem{false}(NonlinearFunction{false}(myfun),
                u0, levels[i]), SimpleNewtonRaphson()).u
    end
end

@btime f(out, levels, (0.0, 2.0))
@btime f2(out, levels, 1.0)
```

MATLAB 2022a achieves 1.66s. Try this code yourself: we receive 0.06 seconds, or a 28x speedup.
This example is still not optimized in the Julia code and we expect an improvement in a near
future version.

For more information on performance of SciML, see the [SciMLBenchmarks](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/)
