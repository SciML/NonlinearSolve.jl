using NonlinearSolve

using ForwardDiff

function multiple_solves(ps::Vector)
    res = similar(ps, 4, length(ps))
    for (i, p) in enumerate(ps)
        prob = NonlinearProblem{false}((u, p) -> u .* u .- p, rand(4), ps[i])
        sol = solve(prob)
        res[:, i] .= sol.u
    end
    return sum(abs2, res)
end

function multiple_solves_cached(ps::Vector)
    res = similar(ps, 4, length(ps))
    prob = NonlinearProblem{false}((u, p) -> u .* u .- p, rand(4), ps[1])
    cache = init(prob, NewtonRaphson())
    for (i, p) in enumerate(ps)
        reinit!(cache; p)
        sol = solve!(cache)
        res[:, i] .= sol.u
    end
    return sum(abs2, res)
end

ps = collect(1.0:5.0)

@test ForwardDiff.gradient(multiple_solves, ps) ≈
    ForwardDiff.gradient(multiple_solves_cached, ps)
