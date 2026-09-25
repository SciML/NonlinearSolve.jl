# Load time and invalidations of BracketingNonlinearSolve and friends, each in a fresh
# process, for environments given on the command line (one per variant, each with
# BracketingNonlinearSolve, NonlinearSolveBase, SimpleNonlinearSolve, ForwardDiff,
# SnoopCompileCore; see REPORT.md for how they were made).
#   julia study/loadtime.jl name1=path/to/env1 name2=path/to/env2
using Printf, Statistics

const CASES = [
    "using BracketingNonlinearSolve",
    "using ForwardDiff",
    "using ForwardDiff; using BracketingNonlinearSolve",
    "using NonlinearSolveBase",
    "using SimpleNonlinearSolve",
]

function elapsed(env, code; reps = 7)
    ts = map(1:reps) do _
        out = read(`$(Base.julia_cmd()) --startup-file=no --project=$env -e "t = @elapsed ($code); print(t)"`, String)
        parse(Float64, out)
    end
    return median(ts)
end

function invalidations(env, code)
    # length of the raw invalidation logs (method-table and edge logs); not the number of
    # distinct invalidated MethodInstances, which needs SnoopCompile to count
    script = "using SnoopCompileCore; invs = @snoop_invalidations ($code); print(length(invs.logmeths) + length(invs.logedges))"
    return parse(Int, read(`$(Base.julia_cmd()) --startup-file=no --project=$env -e $script`, String))
end

envs = [Pair(split(a, "=")...) for a in ARGS]
println("| code | ", join(first.(envs), " (s) | "), " (s) | ", join(first.(envs), " invalidation log length | "), " invalidation log length |")
println("|", repeat("---|", 1 + 2length(envs)))
for code in CASES
    ts = [elapsed(env, code) for (_, env) in envs]
    ns = [invalidations(env, code) for (_, env) in envs]
    println("| `", code, "` | ", join([@sprintf("%.3f", t) for t in ts], " | "), " | ", join(ns, " | "), " |")
end
