using NonlinearSolveBase, SciMLBase, Test
using NonlinearSolveBase: AbsNormTerminationMode, AbsTerminationMode
using SciMLBase: NonlinearProblem, init, reinit!

# A Float32-backed number type that refuses construction from Float64, standing in
# for strict number types that reject precision-narrowing conversions.
struct Strict32 <: AbstractFloat
    x::Float32
    Strict32(x::Float32) = new(x)
end
Strict32(x::Strict32) = x
Strict32(x::Integer) = Strict32(Float32(x))
Strict32(::Float64) = throw(ArgumentError("Strict32 forbids Float64 construction"))

Base.promote_rule(::Type{Strict32}, ::Type{<:Integer}) = Strict32
Base.abs(a::Strict32) = Strict32(abs(a.x))
Base.:-(a::Strict32) = Strict32(-a.x)
Base.:-(a::Strict32, b::Strict32) = Strict32(a.x - b.x)
Base.:+(a::Strict32, b::Strict32) = Strict32(a.x + b.x)
Base.:*(a::Strict32, b::Strict32) = Strict32(a.x * b.x)
Base.:<(a::Strict32, b::Strict32) = a.x < b.x
Base.:(<=)(a::Strict32, b::Strict32) = a.x <= b.x
Base.:(==)(a::Strict32, b::Strict32) = a.x == b.x
Base.isinf(a::Strict32) = isinf(a.x)
Base.isfinite(a::Strict32) = isfinite(a.x)
Base.isnan(a::Strict32) = isnan(a.x)
Base.zero(::Type{Strict32}) = Strict32(0.0f0)
Base.one(::Type{Strict32}) = Strict32(1.0f0)

is_positive_inf(x, ::Type{T}) where {T} = x isa T && isinf(x) && x > zero(T)

modes = (
    ("AbsTerminationMode", AbsTerminationMode()),
    ("AbsNormTerminationMode", AbsNormTerminationMode(Base.Fix1(maximum, abs))),
)

floats = (Float16, Float32, Float64, BigFloat)

@testset "non-safe best value is +Inf in the tolerance type: $T, $name" for T in floats,
        (name, mode) in modes

    u0 = T[1, 1]
    prob = NonlinearProblem((u, p) -> u, u0)
    tol = T(1) / T(1000)
    cache = init(prob, mode, u0, u0; abstol = tol, reltol = tol)
    @test is_positive_inf(cache.best_objective_value, T)

    cache.best_objective_value = zero(T)
    reinit!(cache, u0, u0)
    @test is_positive_inf(cache.best_objective_value, T)

    @test cache(T[0, 0], u0, u0)
    reinit!(cache, u0, u0)
    @test !cache(T[1, 1], u0, u0)
end

@testset "strict Float32 type: $name" for (name, mode) in modes
    T = Strict32
    @test_throws ArgumentError T(Inf)

    u0 = [T(1), T(1)]
    prob = NonlinearProblem((u, p) -> u, u0)
    tol = T(1.0f-3)
    cache = init(prob, mode, u0, u0; abstol = tol, reltol = tol)
    @test is_positive_inf(cache.best_objective_value, T)

    cache.best_objective_value = zero(T)
    reinit!(cache, u0, u0)
    @test is_positive_inf(cache.best_objective_value, T)
end

@testset "strict Float32 type, reinit! on its own: $name" for (name, mode) in modes
    T = Strict32
    u0 = [T(1), T(1)]
    tol = T(1.0f-3)
    # Built without `init`, so the `reinit!` reset is checked independently of it.
    direct = NonlinearSolveBase.NonlinearTerminationModeCache(
        nothing, SciMLBase.ReturnCode.Default, tol, tol, zero(T), mode,
        nothing, nothing, 0, nothing, nothing, nothing, nothing, nothing, false
    )
    reinit!(direct, u0, u0)
    @test is_positive_inf(direct.best_objective_value, T)
end
