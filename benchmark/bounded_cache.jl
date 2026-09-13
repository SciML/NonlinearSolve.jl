include("bounded_solvers.jl")

function benchmark_bounded_cache(output; samples = 17)
    BLAS.set_num_threads(1)
    names = ("rosenbrock_100_true_1", "linear_4_8_0.9", "exponential_0.2_3", "diffusion_256_true_operator_0.1")
    cases = filter(c -> c.name in names, bounded_benchmark_cases())
    return open(output, "w") do io
        println(io, "case,algorithm,retain_best,passed,time_ns,memory,allocs,samples,mean_time_ns,max_time_ns")
        for case in cases, (name, alg) in bounded_benchmark_algorithms(case)
            name in ("bounded_default", "bounded_trust_region", "gn_linesearch") || continue
            for retain_best in (false, true)
                cache = init(case.prob, alg; abstol = 1.0e-9, reltol = 1.0e-9, maxiters = 1000)
                sol = solve!(cache)
                metrics = bounded_benchmark_metrics(case, sol)
                u0, p = case.prob.u0, case.prob.p
                trial = @benchmark begin
                    reinit!($cache; u0 = $u0, p = $p, retain_best = $retain_best)
                    solve!($cache)
                end samples = samples evals = 1 seconds = 60
                estimate = median(trial)
                println(
                    io, join(
                        csv_field.(
                            (
                                case.name, name, retain_best, metrics.passed,
                                estimate.time, estimate.memory, estimate.allocs, length(trial.times), mean(trial.times), maximum(trial.times),
                            )
                        ), ','
                    )
                )
                flush(io)
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("Usage: bounded_cache.jl OUTPUT.csv")
    benchmark_bounded_cache(only(ARGS))
end
