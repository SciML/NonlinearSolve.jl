using NonlinearSolveBase
using Test

# The Newton iterate λ = -1 leaves the bracket [0, 1], so the fallback `l + 0.01 (u - l)`
# is taken; under Reactant λ is loop-carried and must keep its type.
@testset "safeguarded λ update keeps eltype $T" for T in (Float32, Float64)
    λ, l, u = NonlinearSolveBase._more_update_λ(
        one(T), -one(T), one(T), one(T), T(0.5), zero(T), T(4)
    )
    @test λ isa T
    @test l isa T
    @test u isa T
    @test λ == T(0.01)
    @test (l, u) == (zero(T), one(T))
end
