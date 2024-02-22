using NonlinearSolve

f(u, p) = u .* u .- 2

u0 = [1.0, 1.0]

prob = NonlinearProblem(f, u0)

nlcache = init(prob);

for i in 1:10
    step!(nlcache)
    @show nlcache.retcode
end
