# Evaluation counts of the prototypes in proto.jl. Prints, per problem and method,
# "f-evaluations/f'-evaluations" (an evaluation of f and f' together counts once in each,
# f'' is not counted) and flags wrong answers.
include("problems.jl")
include("proto.jl")

const VARIANTS = [
    ("ModAB(ref)", (o, a, b, tol) -> modab_ref(o, a, b, tol)),
    ("Newton", (o, a, b, tol) -> hybrid(o, a, b, tol)),
    ("Newton+mult", (o, a, b, tol) -> hybrid(o, a, b, tol; multiplicity = true)),
    ("NModAB", (o, a, b, tol) -> newton_modab(o, a, b, tol)),
    ("lazy", (o, a, b, tol) -> newton_modab(o, a, b, tol; base = :smaller_f, lazy = true, modab = false)),
    ("lazy+mult", (o, a, b, tol) -> newton_modab(o, a, b, tol; base = :smaller_f, lazy = true, multiplicity = true, modab = false)),
    ("lazy+mult+push", (o, a, b, tol) -> newton_modab(o, a, b, tol; base = :smaller_f, lazy = true, multiplicity = true, push = true, modab = false)),
    ("eager+mult", (o, a, b, tol) -> newton_modab(o, a, b, tol; multiplicity = true, modab = false)),
    ("eager+mult+push", (o, a, b, tol) -> newton_modab(o, a, b, tol; multiplicity = true, push = true, modab = false)),
    ("Hermite", (o, a, b, tol) -> hybrid(o, a, b, tol; model = :hermite)),
    ("Halley", (o, a, b, tol) -> hybrid(o, a, b, tol; model = :halley)),
    ("ModAB→Newton", (o, a, b, tol) -> modab_polish(o, a, b, tol)),
    ("ModAB→Newton+push", (o, a, b, tol) -> modab_polish(o, a, b, tol; push = true)),
    ("ModAB→Hermite", (o, a, b, tol) -> modab_polish(o, a, b, tol; model = :hermite)),
    ("ModAB→lazy", (o, a, b, tol) -> modab_polish(o, a, b, tol; kernel = :lazy)),
    ("ModAB→lazy+mult", (o, a, b, tol) -> modab_polish(o, a, b, tol; kernel = :lazy, multiplicity = true)),
    ("ModAB→lazy+push", (o, a, b, tol) -> modab_polish(o, a, b, tol; kernel = :lazy, push = true)),
]

function explore(tol = 0.0; set = vcat(STANDARD, HARD), quiet = false)
    println("abstol = ", tol)
    quiet || println(rpad("problem", 30), join(rpad.(first.(VARIANTS), 18)))
    tot = zeros(Int, length(VARIANTS), 2)
    for (name, f, df, d2f, (a, b)) in set
        quiet || print(rpad(name, 30))
        for (i, (vn, m)) in enumerate(VARIANTS)
            o = Oracle(f, df, x -> (f(x), df(x)), x -> (f(x), df(x), d2f(x)))
            u, l, r, code = m(o, a, b, tol)
            ok = code === :exact || (sign(f(l)) != sign(f(r)) && (r - l) / 2 < max(tol, eps(r)))
            nf = o.nf + o.nfd + o.nfdd
            nj = o.nj + o.nfd + o.nfdd
            tot[i, 1] += nf
            tot[i, 2] += nj
            quiet || print(rpad(string(nf, "/", nj, ok ? "" : "!", code === :maxiters ? "M" : ""), 18))
        end
        quiet || println()
    end
    println("totals: f / f' and f + r f' for r = 0.25, 0.5, 1")
    for (i, (vn, _)) in enumerate(VARIANTS)
        nf, nj = tot[i, :]
        println("  ", rpad(vn, 20), rpad(string(nf, " / ", nj), 12), join(rpad.(nf .+ (0.25, 0.5, 1) .* nj, 8)))
    end
    return
end
if abspath(PROGRAM_FILE) == @__FILE__
    for tol in (0.0, 1.0e-8)
        explore(tol)
        println("standard set only")
        explore(tol; set = STANDARD, quiet = true)
        println("hard set only")
        explore(tol; set = HARD, quiet = true)
    end
end
