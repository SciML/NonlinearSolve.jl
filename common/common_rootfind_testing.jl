using NonlinearSolveBase, SciMLBase

const TERMINATION_CONDITIONS = [
    NormTerminationMode(Base.Fix1(maximum, abs)),
    RelTerminationMode(),
    RelNormTerminationMode(Base.Fix1(maximum, abs)),
    RelNormSafeTerminationMode(Base.Fix1(maximum, abs)),
    RelNormSafeBestTerminationMode(Base.Fix1(maximum, abs)),
    AbsTerminationMode(),
    AbsNormTerminationMode(Base.Fix1(maximum, abs)),
    AbsNormSafeTerminationMode(Base.Fix1(maximum, abs)),
    AbsNormSafeBestTerminationMode(Base.Fix1(maximum, abs)),
]

quadratic_f(u, p) = u .* u .- p
quadratic_f!(du, u, p) = (du .= u .* u .- p)
quadratic_f2(u, p) = @. p[1] * u * u - p[2]

function newton_fails(u, p)
    return 0.010000000000000002 .+
        10.000000000000002 ./ (
        1 .+
            (
            0.21640425613334457 .+
                216.40425613334457 ./ (
                1 .+
                    (
                    0.21640425613334457 .+
                        216.40425613334457 ./ (1 .+ 0.0006250000000000001(u .^ 2.0))
                ) .^ 2.0
            )
        ) .^
            2.0
    ) .- 0.0011552453009332421u .- p
end

function solve_oop(f, u0, p = 2.0; solver, kwargs...)
    prob = NonlinearProblem{false}(f, u0, p)
    return solve(prob, solver; abstol = 1.0e-9, kwargs...)
end

function solve_iip(f, u0, p = 2.0; solver, kwargs...)
    prob = NonlinearProblem{true}(f, u0, p)
    return solve(prob, solver; abstol = 1.0e-9, kwargs...)
end

function nlprob_iterator_interface(f, p_range, isinplace, solver)
    probN = NonlinearProblem{isinplace}(f, isinplace ? [0.5] : 0.5, p_range[begin])
    cache = init(probN, solver; maxiters = 100, abstol = 1.0e-10)
    sols = zeros(length(p_range))
    for (i, p) in enumerate(p_range)
        reinit!(cache, isinplace ? [cache.u[1]] : cache.u; p = p)
        sol = solve!(cache)
        sols[i] = isinplace ? sol.u[1] : sol.u
    end
    return sols
end

export TERMINATION_CONDITIONS
export quadratic_f, quadratic_f!, quadratic_f2, newton_fails
export solve_oop, solve_iip
export nlprob_iterator_interface
