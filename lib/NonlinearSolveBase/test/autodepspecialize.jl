using NonlinearSolveBase, SciMLBase, RespecializeParams, ForwardDiff, Test

# An `isbits` struct parameter. Under `AutoDePSpecialize`, `get_concrete_problem`
# should pack `p` into a `RespecializeParams.OpaqueParams` and wrap the residual
# in an `AutoDePSpecializeCallable` whose signature carries `OpaqueParams` in the
# `p` slot, so problems with different parameter struct types share one wrapper.

struct DePP
    a::Float64
    b::Float64
end
resid_dep!(res, u, p::DePP) = (res[1] = u[1]^2 - p.a; res[2] = u[2]^2 - p.b; nothing)

struct DePP2
    x::Float64
    y::Float64
end
resid_dep2!(res, u, q::DePP2) = (res[1] = u[1]^2 - q.x; res[2] = u[2]^2 - q.y; nothing)

struct VecQ
    c::Vector{Float64}
end

deprob(f, u0, p) = NonlinearProblem(
    NonlinearFunction{true, SciMLBase.AutoDePSpecialize}(f), u0, p
)
concretize(prob) = NonlinearSolveBase.get_concrete_problem(prob)

u0 = [1.0, 1.0]

@testset "AutoDePSpecialize opaque-p" begin
    @testset "isbits struct p is de-specialized" begin
        cp = concretize(deprob(resid_dep!, u0, DePP(2.0, 3.0)))
        @test cp.p isa RespecializeParams.OpaqueParams
        @test cp.f.f isa NonlinearSolveBase.AutoDePSpecializeCallable
        # the wrapped residual unpacks and evaluates correctly
        res = [0.0, 0.0]
        cp.f.f(res, [2.0, 3.0], cp.p)
        @test res ≈ [4.0 - 2.0, 9.0 - 3.0]
    end

    @testset "AutoSpecialize / FullSpecialize leave p untouched" begin
        for spec in (SciMLBase.AutoSpecialize, SciMLBase.FullSpecialize)
            prob = NonlinearProblem(
                NonlinearFunction{true, spec}(resid_dep!), u0, DePP(2.0, 3.0)
            )
            cp = NonlinearSolveBase.get_concrete_problem(prob)
            @test cp.p isa DePP
        end
    end

    @testset "NullParameters is not opaque-ified" begin
        fnull!(res, u, p) = (res[1] = u[1]^2 - 2.0; res[2] = u[2]^2 - 3.0; nothing)
        cp = concretize(
            NonlinearProblem(
                NonlinearFunction{true, SciMLBase.AutoDePSpecialize}(fnull!), u0
            )
        )
        @test cp.p isa SciMLBase.NullParameters
    end

    @testset "non-isbits p is de-specialized via OpaqueRef" begin
        # non-isbits struct (Vector field) → packed by reference into an OpaqueRef
        fvecp!(res, u, p::VecQ) = (res[1] = u[1]^2 - p.c[1]; nothing)
        cp = concretize(deprob(fvecp!, u0, VecQ([2.0])))
        @test cp.p isa RespecializeParams.OpaqueRef
        @test cp.f.f isa NonlinearSolveBase.AutoDePSpecializeCallable
        res = [0.0]
        cp.f.f(res, [3.0], cp.p)
        @test res[1] ≈ 9.0 - 2.0
        @test RespecializeParams.unpack(cp.p, VecQ).c == [2.0]

        # a bare Vector parameter is likewise de-specialized into an OpaqueRef
        fvec!(res, u, p::Vector{Float64}) = (res[1] = u[1]^2 - p[1]; nothing)
        cpv = concretize(deprob(fvec!, u0, [2.0]))
        @test cpv.p isa RespecializeParams.OpaqueRef
    end

    @testset "already-packed p is not re-wrapped (idempotent)" begin
        cp = concretize(deprob(resid_dep!, u0, DePP(2.0, 3.0)))
        cp2 = concretize(cp)
        @test cp2.p isa RespecializeParams.OpaqueParams
        @test typeof(cp2.f.f) === typeof(cp.f.f)
    end

    @testset "different struct p types share the wrapped-residual type" begin
        cp1 = concretize(deprob(resid_dep!, u0, DePP(2.0, 3.0)))
        cp2 = concretize(deprob(resid_dep2!, u0, DePP2(2.0, 3.0)))
        @test typeof(cp1.f.f) === typeof(cp2.f.f)
        @test typeof(cp1.p) === typeof(cp2.p) === RespecializeParams.OpaqueParams
    end

    @testset "unpack recovers the original p; wrapped call is type-stable" begin
        p = DePP(2.0, 3.0)
        cp = concretize(deprob(resid_dep!, u0, p))
        @test RespecializeParams.unsafe_unpack(cp.p, DePP) === p
        # the OpaqueVoid unpack-and-forward is inferred (returns nothing)
        raw = NonlinearSolveBase.get_raw_f(cp.f.f)
        @test raw isa RespecializeParams.OpaqueVoid
        @test (@inferred raw([0.0, 0.0], u0, cp.p)) === nothing
    end
end
