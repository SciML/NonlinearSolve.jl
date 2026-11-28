using BracketingNonlinearSolve
include("setup_rootfindingtestsnippet.jl")

@testset for alg in (
        Alefeld(), Bisection(), Brent(), Falsi(), ITP(), Muller(), Ridder(), ModAB(), nothing,
    )
    f1(u, p) = u * u - p
    f2(u, p) = p - u * u

    for p in 1:4
        sol1 = solve(IntervalNonlinearProblem(f1, (1.0, 2.0), p), alg)
        sol2 = solve(IntervalNonlinearProblem(f2, (1.0, 2.0), p), alg)
        sol3 = solve(IntervalNonlinearProblem(f1, (2.0, 1.0), p), alg)
        sol4 = solve(IntervalNonlinearProblem(f2, (2.0, 1.0), p), alg)
        sol5 = solve(IntervalNonlinearProblem(f1, [1.0, 2.0], p), alg)
        @test abs.(sol1.u) ≈ sqrt.(p)
        @test abs.(sol2.u) ≈ sqrt.(p)
        @test abs.(sol3.u) ≈ sqrt.(p)
        @test abs.(sol4.u) ≈ sqrt.(p)
        @test SciMLBase.successful_retcode(sol5)
        @test sol5.u === sol1.u
        # Test brackets consistency
        @test sol1.left ≤ sol1.right
        @test sol2.left ≤ sol2.right
        @test sol3.left ≥ sol3.right
        @test sol4.left ≥ sol4.right
    end
end

# Splatting a `Vector` bracket is a `Core._apply_iterate` call of unknown arity, which `--trim`
# rejects. The walk follows every call the compiler resolved, so it also finds an array splat in a helper.
argtype(x::Core.SSAValue, ci) = ci.ssavaluetypes[x.id]
argtype(x::Core.Argument, ci) = ci.slottypes[x.n]
argtype(x, ci) = typeof(x)
function array_splats!(found, seen, @nospecialize(sig))
    sig in seen && return found
    push!(seen, sig)
    for (ci, _) in Base.code_typed_by_type(sig), st in ci.code
        if Meta.isexpr(st, :invoke)
            mi = st.args[1]
            mi isa Core.MethodInstance || (mi = mi.def)  # a `CodeInstance` from Julia 1.12
            array_splats!(found, seen, mi.specTypes)
        elseif Meta.isexpr(st, :call) && st.args[1] == GlobalRef(Core, :_apply_iterate) &&
                any(x -> Core.Compiler.widenconst(argtype(x, ci)) <: AbstractArray, st.args[4:end])
            push!(found, sig => st)
        end
    end
    return found
end

@testset "Vector bracket, no splat: $alg" for alg in (ITP(), ModAB())
    prob = IntervalNonlinearProblem((u, p) -> u * u - p, [1.0, 2.0], 2.0)
    @test isempty(array_splats!([], Set{Any}(), Tuple{typeof(solve), typeof(prob), typeof(alg)}))
end

@testset "Bracket with a third entry: $alg" for alg in (ITP(), ModAB(), nothing)
    for tspan in ([0.0, 2.0, 4.0], (0.0, 2.0, 4.0))
        calls = Ref(0)
        prob = IntervalNonlinearProblem((u, p) -> (calls[] += 1; u - p), tspan, 1.0)
        @test_throws ArgumentError solve(prob, alg)
        @test calls[] == 0
    end
end
