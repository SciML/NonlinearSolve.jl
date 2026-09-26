using NonlinearSolve, PETSc, SparseArrays, Test

function residual!(out, x, p)
    GC.gc(true)
    out .= x .^ 2 .- 2
    return nothing
end

function jacobian!(J, x, p)
    fill!(J, 0)
    for i in eachindex(x)
        J[i, i] = 2x[i]
    end
    return nothing
end

@testset "PETSc buffers survive callback collection" begin
    for sparse in (false, true)
        prototype = sparse ? spdiagm(0 => ones(128)) : zeros(128, 128)
        f = NonlinearFunction(residual!; jac = jacobian!, jac_prototype = prototype)
        sol = solve(NonlinearProblem(f, fill(0.5, 128)), PETScSNES(); abstol = 1.0e-8)
        @test maximum(abs, sol.resid) < 1.0e-8
    end
end
