### dense 50×50 generator (rhs costs more than the interpolation), Tsit5 with abstol = reltol = 1.0e-6
condition (interpolation + g): 90 ns; rhs mat-vec: 294 ns; interpolant derivative: 34 ns
| method | f evals per event | jac or Dual evals per event | interpolations per event | μs per event | max abs(t - t_ModAB) |
|---|---|---|---|---|---|
| ModAB | 7.28 | 0.00 | 5.22 | 0.48 | 0.0e+00 |
| PR NewtonBisection, rhs | 7.12 | 6.22 | 5.12 | 1.86 | 1.8e-15 |
| NewtonBisection, rhs | 7.30 | 4.70 | 5.33 | 1.52 | 1.8e-15 |
| ModABNewton, rhs | 8.22 | 4.62 | 6.22 | 1.69 | 1.6e-15 |
| PR NewtonBisection, interp | 6.55 | 5.65 | 4.55 | 0.70 | 1.2e-15 |
| NewtonBisection, interp | 7.12 | 3.92 | 5.12 | 0.73 | 1.6e-15 |
| ModABNewton, interp | 7.70 | 3.95 | 5.70 | 0.75 | 2.2e-15 |
| ModAB, out of place | 7.33 | 0.00 | 0.00 | 0.84 | 1.2e-15 |
| NewtonBisection, AD | 0.00 | 7.85 | 0.00 | 1.77 | 4.4e-15 |
| ModABNewton, AD | 3.00 | 7.25 | 0.00 | 2.43 | 4.4e-15 |
### tridiagonal 50×50 generator (rhs cheap), Tsit5 with abstol = reltol = 1.0e-6
condition (interpolation + g): 88 ns; rhs mat-vec: 23 ns; interpolant derivative: 26 ns
| method | f evals per event | jac or Dual evals per event | interpolations per event | μs per event | max abs(t - t_ModAB) |
|---|---|---|---|---|---|
| ModAB | 7.78 | 0.00 | 5.72 | 0.51 | 0.0e+00 |
| PR NewtonBisection, rhs | 6.78 | 5.85 | 4.78 | 0.75 | 3.6e-15 |
| NewtonBisection, rhs | 6.85 | 4.42 | 4.85 | 0.69 | 3.6e-15 |
| ModABNewton, rhs | 7.92 | 4.40 | 5.92 | 0.60 | 3.6e-15 |
| PR NewtonBisection, interp | 6.35 | 5.47 | 4.35 | 0.69 | 4.4e-15 |
| NewtonBisection, interp | 6.55 | 3.85 | 4.55 | 0.65 | 4.4e-15 |
| ModABNewton, interp | 7.72 | 3.95 | 5.72 | 0.72 | 4.4e-15 |
| ModAB, out of place | 7.80 | 0.00 | 0.00 | 0.88 | 2.2e-15 |
| NewtonBisection, AD | 0.00 | 8.45 | 0.00 | 2.25 | 4.8e-15 |
| ModABNewton, AD | 3.00 | 6.62 | 0.00 | 2.30 | 4.0e-15 |
### tridiagonal 1000×1000 generator (rhs cheap), Tsit5 with abstol = reltol = 1.0e-6
condition (interpolation + g): 697 ns; rhs mat-vec: 303 ns; interpolant derivative: 392 ns
| method | f evals per event | jac or Dual evals per event | interpolations per event | μs per event | max abs(t - t_ModAB) |
|---|---|---|---|---|---|
| ModAB | 8.00 | 0.00 | 5.95 | 6.97 | 0.0e+00 |
| PR NewtonBisection, rhs | 8.50 | 7.78 | 6.50 | 11.03 | 2.7e-15 |
| NewtonBisection, rhs | 8.55 | 5.95 | 6.55 | 10.14 | 2.7e-15 |
| ModABNewton, rhs | 8.93 | 5.40 | 6.92 | 9.95 | 2.7e-15 |
| PR NewtonBisection, interp | 6.45 | 5.60 | 4.45 | 8.91 | 2.7e-15 |
| NewtonBisection, interp | 6.67 | 4.12 | 4.67 | 8.13 | 2.7e-15 |
| ModABNewton, interp | 7.33 | 4.08 | 5.33 | 8.48 | 2.7e-15 |
| ModAB, out of place | 8.28 | 0.00 | 0.00 | 12.17 | 1.8e-15 |
| NewtonBisection, AD | 0.00 | 10.68 | 0.00 | 40.71 | 1.9e-14 |
| ModABNewton, AD | 3.00 | 9.78 | 0.00 | 55.58 | 1.9e-14 |
### tridiagonal 50×50 generator (rhs cheap), Tsit5 with abstol = reltol = 1.0e-8
condition (interpolation + g): 90 ns; rhs mat-vec: 25 ns; interpolant derivative: 27 ns
| method | f evals per event | jac or Dual evals per event | interpolations per event | μs per event | max abs(t - t_ModAB) |
|---|---|---|---|---|---|
| ModAB | 7.35 | 0.00 | 5.33 | 0.47 | 0.0e+00 |
| PR NewtonBisection, rhs | 6.38 | 5.45 | 4.38 | 0.69 | 2.0e-15 |
| NewtonBisection, rhs | 6.62 | 3.92 | 4.62 | 0.67 | 1.9e-15 |
| ModABNewton, rhs | 8.03 | 3.92 | 6.03 | 0.77 | 2.0e-15 |
| PR NewtonBisection, interp | 6.67 | 5.75 | 4.67 | 0.74 | 2.2e-15 |
| NewtonBisection, interp | 7.05 | 3.88 | 5.05 | 0.72 | 2.2e-15 |
| ModABNewton, interp | 7.72 | 3.90 | 5.72 | 0.74 | 2.2e-15 |
| ModAB, out of place | 7.45 | 0.00 | 0.00 | 0.86 | 6.7e-16 |
| NewtonBisection, AD | 0.00 | 8.70 | 0.00 | 2.40 | 3.8e-15 |
| ModABNewton, AD | 3.00 | 7.20 | 0.00 | 2.51 | 3.6e-15 |
In-place condition with AD: MethodError: no method matching Float64(::ForwardDiff.Dual{ForwardDiff.Tag{IntervalNonlinearFunction{false, SciMLBase.AutoSpecialize, var"#inplace_dual_check##0#inplace_dual_check##1"{Cond{OrdinaryDif
