include("bounded_solvers.jl")

function benchmark_bounded_pairwise(output; samples = 15)
    BLAS.set_num_threads(1)
    cases = vcat(bounded_benchmark_cases(), bounded_validation_cases())
    return open(output, "w") do io
        println(io, "case,family,representation,kind,n,m,algorithm,passed,retcode,time_ns,memory,allocs,cost,residual,stationarity,feasible,error,samples")
        for (index, case) in enumerate(cases)
            prob = case.prob
            pair = filter(p -> first(p) in ("bounded_trust_region", "gn_linesearch"), bounded_benchmark_algorithms(case))
            results = map(pair) do (_, alg)
                sol = solve(prob, alg; abstol = 1.0e-9, reltol = 1.0e-9, maxiters = 1000)
                (sol, bounded_benchmark_metrics(case, sol))
            end
            for k in (isodd(index) ? (1, 2) : (2, 1))
                name, alg = pair[k]
                sol, metrics = results[k]
                trial = @benchmark solve($prob, $alg; abstol = 1.0e-9, reltol = 1.0e-9, maxiters = 1000) samples = samples evals = 1 seconds = 60
                estimate = median(trial)
                row = (
                    case.name, case.family, case.representation,
                    prob isa NonlinearProblem ? "root" : "least_squares", length(prob.u0), length(sol.resid),
                    name, metrics.passed, sol.retcode, estimate.time, estimate.memory, estimate.allocs,
                    metrics.cost, metrics.residual, metrics.stationarity, metrics.feasible, "", length(trial.times),
                )
                println(io, join(csv_field.(row), ','))
                flush(io)
            end
            println(case.name)
            flush(stdout)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("Usage: bounded_pairwise.jl OUTPUT.csv")
    benchmark_bounded_pairwise(only(ARGS))
end
