using NonlinearSolve
using CommonSolve
using SciMLBase
using Test

function Hcache!(du, u, p, λ)
    du[1] = (1 - λ) * (u[1] - p[1]) + λ * (u[1]^2 - p[1])
    return nothing
end

prob = HomotopyProblem(Hcache!, [4.0], [4.0])
algs = (
    HomotopySweep(; inner = NewtonRaphson(), adaptive = false, nsteps = 10),
    KantorovichHomotopy(; inner = NewtonRaphson(), nsteps = 10, strict = false),
    HomotopySweep(; adaptive = false, nsteps = 10),
)

for alg in algs
    cache = init(prob, alg; abstol = 0.0)
    @test reinit!(cache, prob.u0; p = prob.p, abstol = 1.0e-10) === cache
    sol = solve!(cache)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u[1] ≈ 2.0 atol = 1.0e-10

    u0 = [9.0]
    p = [9.0]
    @test reinit!(cache, u0; p, abstol = 1.0e-10) === cache
    sol = solve!(cache)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u[1] ≈ 3.0 atol = 1.0e-10
    @test sol.prob.u0 === u0
    @test sol.prob.p === p
end
