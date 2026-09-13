function test_trust_region_float32(compile_solve)
    prob = convert(
        SciMLBase.ImmutableNonlinearProblem,
        NonlinearProblem{false}((u, p) -> u .- 1.0f0, SVector(0.0f0))
    )
    return @testset "Float32 trust-region arithmetic" begin
        @testset "$update" for update in (Val(false), Val(true))
            alg = SimpleTrustRegion(; nlsolve_update_rule = update)
            result, ir = compile_solve(prob, alg)
            @test result ≈ [1.0f0]
            @test isnothing(match(r"\b(?:fmul|fdiv|fsub|fadd|fcmp)\b[^\n]*\bdouble\b", ir))
        end
    end
end
