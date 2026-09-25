# Evaluation counts of the package methods on the problems of problems.jl.
#   julia --project=study study/bench_counts.jl
# Columns "f/jac" count calls of f with a Float argument and calls of the user `jac`; for the
# AD variants (no `jac`), "f/dual" counts calls of f with a Float argument and with a
# ForwardDiff.Dual argument (one Dual call gives f and f').
using BracketingNonlinearSolve, ForwardDiff, Printf
using ADTypes: AutoForwardDiff
include("problems.jl")
include("pr1330_newton_bisection.jl")

const COUNTS = zeros(Int, 3)   # f with a Float argument, f with a Dual argument, jac
counted_f(f) = (u, p) -> (u isa ForwardDiff.Dual ? (COUNTS[2] += 1) : (COUNTS[1] += 1); f(u))
counted_jac(df) = (u, p) -> (COUNTS[3] += 1; df(u))

const METHODS = [
    ("ModAB", ModAB(), :nojac),
    ("ITP", ITP(), :nojac),
    ("Brent", Brent(), :nojac),
    ("PR NewtonBisection", PRNewtonBisection(), :jac),
    ("NewtonBisection", NewtonBisection(), :jac),
    ("ModABNewton", ModABNewton(), :jac),
    ("NewtonBisection AD", NewtonBisection(; autodiff = AutoForwardDiff()), :ad),
    ("ModABNewton AD", ModABNewton(; autodiff = AutoForwardDiff()), :ad),
]

function count_run(alg, mode, f, df, (a, b); abstol)
    fill!(COUNTS, 0)
    fun = mode === :jac ? IntervalNonlinearFunction(counted_f(f); jac = counted_jac(df)) :
        counted_f(f)
    sol = solve(IntervalNonlinearProblem(fun, (a, b)), alg; abstol)
    l, r = minmax(sol.left, sol.right)
    ok = SciMLBase.successful_retcode(sol) &&
        (iszero(f(sol.u)) || (sign(f(l)) != sign(f(r)) && (r - l) / 2 <= max(abstol, eps(r))))
    return (nf = COUNTS[1], ndual = COUNTS[2], nj = COUNTS[3], ok, retcode = sol.retcode)
end

cell(c, mode) = string(c.nf, "/", mode === :ad ? c.ndual : c.nj, c.ok ? "" : " FAIL")

function table(set, abstol)
    println("\n| problem | bracket | ", join(first.(METHODS), " | "), " |")
    println("|", repeat("---|", length(METHODS) + 2))
    tot = zeros(Int, length(METHODS), 2)
    for (name, f, df, _, br) in set
        cells = String[]
        for (i, (_, alg, mode)) in enumerate(METHODS)
            c = count_run(alg, mode, f, df, br; abstol)
            push!(cells, cell(c, mode))
            tot[i, 1] += c.nf
            tot[i, 2] += mode === :ad ? c.ndual : c.nj
        end
        println("| ", name, " | ", br, " | ", join(cells, " | "), " |")
    end
    println("| **total** | | ", join([string(tot[i, 1], "/", tot[i, 2]) for i in axes(tot, 1)], " | "), " |")
    return tot
end

for abstol in (0.0, 1.0e-8)
    println("\n### Standard functions, abstol = $abstol")
    table(STANDARD, abstol)
    println("\n### Hard cases, abstol = $abstol")
    table(HARD, abstol)
end

# Float32 and BigFloat on the standard set, abstol = 0
for T in (Float32, BigFloat)
    println("\n### Standard functions in $T, abstol = 0 (totals)")
    tot = zeros(Int, length(METHODS), 2)
    for (name, f, df, _, (a, b)) in STANDARD
        for (i, (_, alg, mode)) in enumerate(METHODS)
            c = count_run(alg, mode, u -> f(u), u -> df(u), (T(a), T(b)); abstol = zero(T))
            c.ok || println("  FAIL: ", name, " ", first(METHODS[i]), " ", c.retcode)
            tot[i, 1] += c.nf
            tot[i, 2] += mode === :ad ? c.ndual : c.nj
        end
    end
    for (i, (n, _, _)) in enumerate(METHODS)
        println("  ", rpad(n, 22), tot[i, 1], "/", tot[i, 2])
    end
end
