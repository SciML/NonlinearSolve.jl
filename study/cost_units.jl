# Cost in units of one f evaluation of the AD variants on the standard functions at
# abstol = 0: a Dual evaluation costs the measured ratio c of part C of results/time.md.
#   julia study/cost_units.jl   (after run_all.sh)
using Printf

cells(l) = strip.(split(strip(l, '|'), '|'))
rows(section) = [cells(l) for l in split(section, '\n') if startswith(l, "| ") && !startswith(l, "| problem") && !occursin("total", l)]

counts = rows(split(read(joinpath(@__DIR__, "results", "counts.md"), String), "### ")[2])
ratio = Dict(c[1] => parse(Float64, c[4]) for c in rows(split(read(joinpath(@__DIR__, "results", "time.md"), String), "### C.")[2]))

println("| problem | Dual/f cost ratio c | ModAB (f) | NewtonBisection AD (c × Dual) | ModABNewton AD (f + c × Dual) |")
println("|---|---|---|---|---|")
tot = zeros(3)
for c in counts
    r = ratio[c[1]]
    modab = parse(Int, first(split(c[3], "/")))
    nb = r * parse(Int, last(split(c[9], "/")))
    mf, md = parse.(Int, split(c[10], "/"))
    mn = mf + r * md
    tot .+= (modab, nb, mn)
    @printf("| %s | %.2f | %d | %.1f | %.1f |\n", c[1], r, modab, nb, mn)
end
@printf("| **total** | | %d | %.1f | %.1f |\n", tot...)
