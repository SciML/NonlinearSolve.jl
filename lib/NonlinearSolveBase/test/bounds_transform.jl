module BoundsTransformTests

using Test
using NonlinearSolveBase
using SciMLBase
# Load ForwardDiff so PreallocationTools' `FixedSizeDiffCache` dual-cache extension is
# active (the transform builds one for the default/ForwardDiff autodiff path).
import ForwardDiff
using NonlinearSolveBase: transform_bounded_problem, BoundedWrapper, _to_unbounded

# An algorithm type without an `autodiff` field, mirroring `QuasiNewtonAlgorithm`
# (which the default polyalgorithm reaches). `transform_bounded_problem` must not
# assume every algorithm exposes `autodiff`.
struct NoAutodiffAlg end

struct BoundsSolutionCache{P, U, F} <: NonlinearSolveBase.AbstractNonlinearSolveCache
    prob::P
    u::U
    fu::F
    alg::NoAutodiffAlg
    retcode::ReturnCode.T
    stats::SciMLBase.NLStats
    trace::Nothing
end

@testset "bounds transform handles algorithms without an `autodiff` field" begin
    f(u, p) = u .^ 2 .- p
    prob = NonlinearProblem(f, [1.5, 1.5], [2.0, 2.0]; lb = [0.0, 0.0], ub = [3.0, 3.0])

    for alg in (nothing, NoAutodiffAlg())
        tprob = transform_bounded_problem(prob, alg)
        @test tprob.f.f isa BoundedWrapper
        # bounds are removed from the transformed (unconstrained) problem
        @test tprob.lb === nothing
        @test tprob.ub === nothing
        # u0 is mapped into unbounded space
        @test tprob.u0 ≈ _to_unbounded.(prob.u0, prob.lb, prob.ub)
    end
end

@testset "analytic derivatives follow bounded coordinates" begin
    using LinearAlgebra, SparseArrays

    for iip in (false, true), sparse_jacobian in (false, true)
        A = [1.0 2.0 0.0 0.0; 0.0 1.0 3.0 0.0; 2.0 0.0 1.0 4.0]
        sparse_jacobian && (A = sparse(A))
        f(u, p) = p .* (A * u .^ 2)
        jac(u, p) = p .* A * Diagonal(2 .* u)
        jvp(v, u, p) = p .* (A * (2 .* u .* v))
        vjp(v, u, p) = 2p .* u .* (A' * v)
        paramjac(u, p) = A * u .^ 2
        f!(r, u, p) = (r .= f(u, p))
        jac!(J, u, p) = (J .= jac(u, p))
        jvp!(Jv, v, u, p) = (Jv .= jvp(v, u, p))
        vjp!(Jv, v, u, p) = (Jv .= vjp(v, u, p))
        paramjac!(Jp, u, p) = (Jp .= paramjac(u, p))
        prototype = copy(A)
        nf = if iip
            NonlinearFunction{true}(
                f!; jac = jac!, jvp = jvp!, vjp = vjp!, paramjac = paramjac!,
                jac_prototype = prototype, resid_prototype = zeros(3)
            )
        else
            NonlinearFunction{false}(
                f; jac, jvp, vjp, paramjac, jac_prototype = prototype
            )
        end
        prob = NonlinearLeastSquaresProblem(
            nf, [0.7, 1.3, -0.4, 2.0], 1.7;
            lb = [0.0, 0.0, -Inf, -Inf], ub = [2.0, Inf, 1.0, Inf]
        )
        transformed = transform_bounded_problem(prob, nothing)
        @test transformed.f.jac_prototype === prototype
        for t in (transformed.u0, transformed.u0 .+ 0.2), p in (1.7, 0.8)
            residual(t) = if iip
                r = similar(t, 3)
                transformed.f(r, t, p)
            else
                transformed.f(t, p)
            end
            expected = ForwardDiff.jacobian(residual, t)
            v = [0.2, -0.3, 0.5, 0.7]
            w = [0.3, -0.2, 0.6]
            expected_p = ForwardDiff.derivative(p) do q
                if iip
                    r = zeros(typeof(q), 3)
                    transformed.f(r, t, q)
                else
                    transformed.f(t, q)
                end
            end
            J, Jv, Jtw, Jp = if iip
                J, Jv, Jtw, Jp = copy(prototype), zeros(3), zeros(4), zeros(3)
                transformed.f.jac(J, t, p)
                transformed.f.jvp(Jv, v, t, p)
                transformed.f.vjp(Jtw, w, t, p)
                transformed.f.paramjac(Jp, t, p)
                J, Jv, Jtw, Jp
            else
                transformed.f.jac(t, p), transformed.f.jvp(v, t, p),
                    transformed.f.vjp(w, t, p), transformed.f.paramjac(t, p)
            end
            @test J ≈ expected
            @test Jv ≈ expected * v
            @test Jtw ≈ expected' * w
            @test Jp ≈ expected_p
            @test issparse(J) == sparse_jacobian
        end
    end
end

@testset "zero transform derivatives preserve sparse structure" begin
    using LinearAlgebra, SparseArrays

    A = sparse([2.0 1.0; 1.0 3.0])
    f!(r, u, p) = mul!(r, A, u)
    jac!(J, u, p) = (nonzeros(J) .= nonzeros(A))
    nf = NonlinearFunction{true}(f!; jac = jac!, jac_prototype = copy(A))
    prob = NonlinearProblem(nf, [0.5, 0.5]; lb = 0.0, ub = 1.0)
    transformed = transform_bounded_problem(prob, nothing)
    J = copy(A)
    for t in ([-1000.0, 0.0], [0.0, 1000.0], [1000.0, -1000.0], [0.0, 0.0])
        transformed.f.jac(J, t, prob.p)
        @test J.colptr == A.colptr
        @test rowvals(J) == rowvals(A)
        @test nnz(J) == nnz(A)
        expected = ForwardDiff.jacobian(t) do x
            r = similar(x)
            transformed.f(r, x, prob.p)
        end
        @test Matrix(J) ≈ expected
    end
end

@testset "scalar analytic derivatives follow bounded coordinates" begin
    for (lb, ub) in ((0.0, 3.0), (0.0, nothing), (nothing, 3.0))
        nf = NonlinearFunction{false}(
            (u, p) -> p * u^2;
            jac = (u, p) -> 2p * u,
            jvp = (v, u, p) -> 2p * u * v,
            vjp = (v, u, p) -> 2p * u * v,
            paramjac = (u, p) -> u^2
        )
        prob = NonlinearProblem(nf, 1.2, 0.7; lb, ub)
        transformed = transform_bounded_problem(prob, nothing)
        t, p = transformed.u0, prob.p
        @test t isa Number
        @test transformed.f(t, p) ≈ prob.f(prob.u0, p)
        expected = ForwardDiff.derivative(t -> transformed.f(t, p), t)
        @test transformed.f.jac(t, p) ≈ expected
        @test transformed.f.jvp(0.3, t, p) ≈ 0.3expected
        @test transformed.f.vjp(0.3, t, p) ≈ 0.3expected
        @test transformed.f.paramjac(t, p) ≈ prob.u0^2
        cache = BoundsSolutionCache(
            transformed, t, transformed.f(t, p), NoAutodiffAlg(),
            ReturnCode.Success, SciMLBase.NLStats(0, 0, 0, 0, 0), nothing
        )
        sol = NonlinearSolveBase._solution_from_cache(cache; transform_bounds = true)
        @test sol.u ≈ prob.u0
        @test sol.prob.u0 ≈ prob.u0
        @test sol.prob.lb == (isnothing(lb) ? -Inf : lb)
        @test sol.prob.ub == (isnothing(ub) ? Inf : ub)
    end
end

end
