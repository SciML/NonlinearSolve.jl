include("setup_corerootfindtesting.jl")

using NonlinearSolveFirstOrder
using SCCNonlinearSolve

@testset "Vector{Any} SCC container eltype is preserved" begin

    function anyvec_scc_residual!(resid, u, p)
        resid[1] = u[1]^2 - p[1]
        return nothing
    end

    _hp = [2.0]
    _hq1 = NonlinearProblem(anyvec_scc_residual!, [1.0], _hp)
    _hq2 = NonlinearProblem(anyvec_scc_residual!, [1.0], _hp)
    _sccprob = SciMLBase.SCCNonlinearProblem(
        Any[_hq1, _hq2], Any[Returns(nothing), Returns(nothing)], _hp, true
    )
    @test _sccprob.probs isa Vector{Any}

    _scc_alg = SCCNonlinearSolve.SCCAlg(; nlalg = NewtonRaphson(), linalg = nothing)
    _sol = solve(_sccprob, _scc_alg)
    @test SciMLBase.successful_retcode(_sol)
    @test _sol.u ≈ [sqrt(2.0), sqrt(2.0)]
    @test _sol.prob.probs isa Vector{Any}

    # Mixed specialization levels and Union containers keep solving.
    _fad = NonlinearFunction{true, SciMLBase.AutoDespecialize}(anyvec_scc_residual!)
    _nad = NonlinearProblem(_fad, [1.0], _hp)
    _nfs = NonlinearProblem(anyvec_scc_residual!, [1.0], _hp)
    _efs = Any[Returns(nothing), Returns(nothing)]
    for _probs in ([_nfs, _nad], Union{typeof(_nfs), typeof(_nad)}[_nfs, _nad])
        _sp = SciMLBase.SCCNonlinearProblem(_probs, _efs, _hp, true)
        _s = solve(_sp, _scc_alg)
        @test SciMLBase.successful_retcode(_s)
        @test _s.u ≈ [sqrt(2.0), sqrt(2.0)]
    end

    # Homogeneous and heterogeneous vectors reach the same problem type
    # (explicit `u0` on the linear block so both problems have a `Vector` state).
    _lin = LinearProblem([2.0;;], [4.0]; u0 = [4.0])
    _shet = solve(
        SciMLBase.SCCNonlinearProblem(Any[_nad, _lin], _efs, _hp, true), _scc_alg
    )
    @test SciMLBase.successful_retcode(_shet)
    @test _shet.u ≈ [sqrt(2.0), 2.0]
    @test typeof(_sol.prob) === typeof(_shet.prob)

end
