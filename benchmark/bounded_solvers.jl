using BenchmarkTools, LinearSolve, ADTypes, Statistics
include("bounded_problems.jl")
include("bounded_validation_problems.jl")

function bounded_benchmark_algorithms(case)
    linsolve = case.representation == :operator ? KrylovJL_LSMR() : nothing
    kwargs = (; linsolve, autodiff = AutoForwardDiff())
    legacy = case.prob isa NonlinearProblem ?
        FastShortcutNonlinearPolyalg(; kwargs..., must_use_jacobian = Val(SciMLBase.has_jac(case.prob.f)), u0_len = length(case.prob.u0)) :
        FastShortcutNLLSPolyalg(; kwargs...)
    return [
        "bounded_default" => FastShortcutBoundedPolyalg(; kwargs...),
        "bounded_trust_region" => BoundedTrustRegion(; kwargs...),
        "reflective" => TrustRegionReflective(; kwargs...),
        "bounded_lm" => BoundedLevenbergMarquardt(; kwargs...),
        "dogbox" => Dogbox(; kwargs...),
        "gn_linesearch" => BoundedGaussNewton(; kwargs...),
        "gn_trustregion" => BoundedGaussNewton(; kwargs..., globalization = :trustregion),
        "gn_hybrid" => BoundedGaussNewton(; kwargs..., globalization = :trustregion_linesearch),
        "transformed_lm" => LevenbergMarquardt(; kwargs...),
        "transformed_trust_region" => TrustRegion(; kwargs...),
        "legacy_default" => legacy,
    ]
end

csv_field(x) = "\"" * replace(string(x), "\"" => "\"\"") * "\""

function benchmark_bounded_solvers(output; pattern = "", algorithm_pattern = "", samples = 5, maxiters = 1000, validation = false)
    BLAS.set_num_threads(1)
    mkpath(dirname(abspath(output)))
    cases = filter(c -> occursin(pattern, c.name), validation ? bounded_validation_cases() : bounded_benchmark_cases())
    open(output, "w") do io
        println(io, "case,family,representation,kind,n,m,algorithm,passed,retcode,time_ns,memory,allocs,cost,residual,stationarity,feasible,error,samples")
        for case in cases, (name, alg) in bounded_benchmark_algorithms(case)
            selected = if algorithm_pattern == "native"
                name in (
                    "bounded_trust_region", "reflective", "bounded_lm", "dogbox",
                    "gn_linesearch", "gn_trustregion", "gn_hybrid",
                )
            elseif algorithm_pattern == "baseline"
                name in ("transformed_lm", "transformed_trust_region", "legacy_default")
            else
                occursin(algorithm_pattern, name)
            end
            selected || continue
            prob = case.prob
            prefix = (
                case.name, case.family, case.representation,
                prob isa NonlinearProblem ? "root" : "least_squares", length(prob.u0),
                prob.f.resid_prototype === nothing ? length(prob.u0) : length(prob.f.resid_prototype), name,
            )
            try
                sol = solve(prob, alg; abstol = 1.0e-9, reltol = 1.0e-9, maxiters)
                metrics = bounded_benchmark_metrics(case, sol)
                trial = @benchmark solve($prob, $alg; abstol = 1.0e-9, reltol = 1.0e-9, maxiters = $maxiters) samples = samples evals = 1 seconds = 60
                estimate = median(trial)
                row = (
                    prefix..., metrics.passed, sol.retcode, estimate.time,
                    estimate.memory, estimate.allocs, metrics.cost, metrics.residual,
                    metrics.stationarity, metrics.feasible, "", length(trial.times),
                )
                println(io, join(csv_field.(row), ','))
                println(case.name, " / ", name, ": ", metrics.passed ? "pass" : "FAIL", " (", sol.retcode, ")")
            catch err
                row = (prefix..., false, "Exception", Inf, 0, 0, Inf, Inf, Inf, false, sprint(showerror, err), 0)
                println(io, join(csv_field.(row), ','))
                println(case.name, " / ", name, ": ERROR ", nameof(typeof(err)))
            end
            flush(io)
            flush(stdout)
        end
    end
    return output
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("Usage: bounded_solvers.jl OUTPUT.csv [NAME_FILTER] [SAMPLES] [ALGORITHM_FILTER] [validation]")
    benchmark_bounded_solvers(
        first(ARGS);
        pattern = length(ARGS) >= 2 ? ARGS[2] : "",
        samples = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 5,
        algorithm_pattern = length(ARGS) >= 4 ? ARGS[4] : "",
        validation = length(ARGS) >= 5 && ARGS[5] == "validation"
    )
end
