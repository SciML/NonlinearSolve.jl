using NonlinearSolveBase
using Test

@testset "Safeguarded host damping update for $T" for T in (Float32, Float64)
    λ, l, u = NonlinearSolveBase._more_update_λ(
        one(T), -one(T), one(T), one(T), T(0.5), zero(T), T(4)
    )
    @test λ === 0.01
    @test l === zero(T)
    @test u === one(T)
end
