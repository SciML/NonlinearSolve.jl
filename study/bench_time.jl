# Wall time of the package methods.
#   julia --project=study study/bench_time.jl
# Part A: the standard functions as they are (f and f' cost a few ns, so the time is mostly
#         the solver's own work); median time per solve, summed over the table.
# Part B: the standard functions made expensive: f does `N` extra dependent multiply-adds
#         and f' does `r N`; ~1 μs per f, so the time is mostly evaluations.
# Part C: ForwardDiff cost of one Dual evaluation relative to f, per standard function.
using BracketingNonlinearSolve, ForwardDiff, BenchmarkTools, Printf
using ADTypes: AutoForwardDiff
include("problems.jl")
include("pr1330_newton_bisection.jl")

const ABSTOL = 0.0

function work(u, n)
    w = u
    for _ in 1:n
        w = muladd(w, 0.9999999, 1.0e-9)
    end
    Base.donotdelete(w)
    return nothing
end

heavy(g, n) = (u, p) -> (work(u, n); g(u))

methods_for(f, df, br) = [
    ("ModAB", IntervalNonlinearProblem((u, p) -> f(u), br), ModAB()),
    ("ITP", IntervalNonlinearProblem((u, p) -> f(u), br), ITP()),
    ("Brent", IntervalNonlinearProblem((u, p) -> f(u), br), Brent()),
    ("PR NewtonBisection", IntervalNonlinearProblem(IntervalNonlinearFunction((u, p) -> f(u); jac = (u, p) -> df(u)), br), PRNewtonBisection()),
    ("NewtonBisection", IntervalNonlinearProblem(IntervalNonlinearFunction((u, p) -> f(u); jac = (u, p) -> df(u)), br), NewtonBisection()),
    ("ModABNewton", IntervalNonlinearProblem(IntervalNonlinearFunction((u, p) -> f(u); jac = (u, p) -> df(u)), br), ModABNewton()),
    ("NewtonBisection AD", IntervalNonlinearProblem((u, p) -> f(u), br), NewtonBisection(; autodiff = AutoForwardDiff())),
    ("ModABNewton AD", IntervalNonlinearProblem((u, p) -> f(u), br), ModABNewton(; autodiff = AutoForwardDiff())),
]

time_solve(prob, alg) = @belapsed solve($prob, $alg; abstol = $ABSTOL) samples = 200 evals = 5

function part_a()
    println("\n### A. Standard functions as they are, abstol = 0: median time per solve (ns)\n")
    names = first.(methods_for(identity, identity, (0.0, 1.0)))
    println("| problem | ", join(names, " | "), " |")
    println("|", repeat("---|", length(names) + 1))
    tot = zeros(length(names))
    for (name, f, df, _, br) in STANDARD
        ts = [time_solve(prob, alg) * 1.0e9 for (_, prob, alg) in methods_for(f, df, br)]
        tot .+= ts
        println("| ", name, " | ", join([@sprintf("%.0f", t) for t in ts], " | "), " |")
    end
    return println("| **total** | ", join([@sprintf("%.0f", t) for t in tot], " | "), " |")
end

function part_b(; n = 1000)
    t_f = @belapsed work(1.2345, $n)
    @printf("\n### B. Expensive f (%d multiply-adds, %.2f μs per f), f' costs r times f, abstol = 0: total μs over the standard functions\n\n", n, t_f * 1.0e6)
    names = first.(methods_for(identity, identity, (0.0, 1.0)))
    names = filter(!endswith("AD"), names)
    println("| r | ", join(names, " | "), " |")
    println("|", repeat("---|", length(names) + 1))
    for r in (0.0, 0.1, 0.25, 0.5, 1.0)
        tot = zeros(length(names))
        for (_, f, df, _, br) in STANDARD
            fh = heavy(f, n)
            dfh = heavy(df, round(Int, r * n))
            for (i, (nm, _, alg)) in enumerate(filter(m -> !endswith(m[1], "AD"), methods_for(f, df, br)))
                prob = nm in ("ModAB", "ITP", "Brent") ? IntervalNonlinearProblem(fh, br) :
                    IntervalNonlinearProblem(IntervalNonlinearFunction(fh; jac = dfh), br)
                tot[i] += (@belapsed solve($prob, $alg; abstol = $ABSTOL) samples = 30 evals = 1) * 1.0e6
            end
        end
        println("| ", r, " | ", join([@sprintf("%.1f", t) for t in tot], " | "), " |")
    end
    return
end

function part_c()
    println("\n### C. Cost of one ForwardDiff evaluation relative to one f evaluation\n")
    println("Mean over 1000 points of the bracket, summed in a loop (so the plain f may use SIMD).")
    println("f' uses a Dual; f' and f'' (for Halley) use a Dual of Duals.\n")
    println("| problem | f (ns) | f, f' (ns) | ratio | f, f', f'' (ns) | ratio |")
    println("|---|---|---|---|---|---|")
    for (name, f, _, _, (a, b)) in STANDARD
        xs = collect(range(a, b; length = 1000))
        tf = @belapsed sum($f, $xs)
        T1 = typeof(ForwardDiff.Tag(f, Float64))
        ds = [ForwardDiff.Dual{T1}(x, 1.0) for x in xs]
        td = @belapsed sum(ForwardDiff.value ∘ $f, $ds)
        T2 = typeof(ForwardDiff.Tag(sin, ForwardDiff.Dual{T1, Float64, 1}))
        dds = [ForwardDiff.Dual{T2}(ForwardDiff.Dual{T1}(x, 1.0), ForwardDiff.Dual{T1}(1.0, 0.0)) for x in xs]
        tdd = @belapsed sum(x -> ForwardDiff.value(ForwardDiff.value($f(x))), $dds)
        @printf(
            "| %s | %.2f | %.2f | %.2f | %.2f | %.2f |\n", name, tf * 1.0e6, td * 1.0e6, td / tf,
            tdd * 1.0e6, tdd / tf
        )
    end
    return
end

part_a()
part_b()
part_c()
