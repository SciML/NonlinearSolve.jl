import NonlinearSolve as NLS
import Reactant
import SciMLBase
using LinearAlgebra: diagm
using Test

# Verbatim tutorial example from docs/src/tutorials/nonlinear_solve_gpus.md
# ("Whole-solve compilation with Reactant.jl").
@testset "docs/tutorials/nonlinear_solve_gpus.md Reactant NewtonRaphson example" begin
    f(u, p) = u .* u .- p
    jac(u, p) = diagm(0 => 2 .* u)

    function reactant_solve(u0, p)
        prob = NLS.NonlinearProblem(NLS.NonlinearFunction(f; jac), u0, p)
        return NLS.solve(prob, NLS.NewtonRaphson())
    end

    u0 = Reactant.to_rarray(Float32[1, 1])
    p = Reactant.to_rarray(Float32[2])
    hlo = Reactant.@code_hlo optimize = false reactant_solve(u0, p)
    has_host_callback = occursin("reactant_julia_callback", repr(hlo))
    @test !has_host_callback
    sol = Reactant.@jit reactant_solve(u0, p)
    @test sol.retcode == SciMLBase.ReturnCode.Success
    @test Array(sol.u) ≈ fill(sqrt(2.0f0), 2)
    @test maximum(abs, Array(sol.resid)) ≤ 1.0f-5
end
