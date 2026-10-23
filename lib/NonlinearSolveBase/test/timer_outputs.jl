using Test
using TimerOutputs: TimerOutput, ncalls, reset_timer!

module EnabledTimers
    # Exercise the compile-time preference without changing package preferences.
    macro load_preference(key, default)
        return true
    end

    macro set_preferences!(args...)
        return nothing
    end

    include(joinpath(@__DIR__, "..", "src", "timer_outputs.jl"))
end

using .EnabledTimers: @static_timeit

const function_timer = TimerOutput()

@static_timeit function_timer "long" function long_form(x::T; offset = 1) where {T <: Number}
    return x + offset
end

@static_timeit function_timer "short" short_form(x) = x * x

@testset "Function calls" begin
    @test !haskey(function_timer, "long")
    @test !haskey(function_timer, "short")
    @test long_form(2) == 3
    @test long_form(3; offset = 2) == 5
    @test short_form(3) == 9
    @test short_form(4) == 16
    @test ncalls(function_timer["long"]) == 2
    @test ncalls(function_timer["short"]) == 2

    reset_timer!(function_timer)
    @test !haskey(function_timer, "long")
    @test long_form(4) == 5
    @test ncalls(function_timer["long"]) == 1
    @test !haskey(function_timer, "short")
end

function block_form(to, name, x)
    result = @static_timeit to name begin
        assigned = x + 1
        @views sum([assigned, x][:])
    end
    return result, assigned
end

function return_form(to)
    @static_timeit to "return" begin
        return 7
    end
    return 0
end

@testset "Blocks and timer state" begin
    timer = TimerOutput()
    @test block_form(timer, "block", 3) == (7, 4)
    @test ncalls(timer["block"]) == 1
    @test return_form(timer) == 7
    @test ncalls(timer["return"]) == 1
    @test_throws ErrorException @static_timeit timer "error" error("timed failure")
    @test ncalls(timer["error"]) == 1
    @test block_form(timer, "recovery", 4) == (9, 5)
    @test ncalls(timer["recovery"]) == 1
    @test !haskey(timer["error"], "recovery")

    @test (
        @static_timeit timer "outer" begin
            @static_timeit timer "inner" 6 * 7
        end
    ) == 42
    @test ncalls(timer["outer"]) == 1
    @test ncalls(timer["outer"]["inner"]) == 1
    @test !haskey(timer, "inner")

    @test (@static_timeit timer "literal" 42) == 42
    @test ncalls(timer["literal"]) == 1
    value = 17
    @test (@static_timeit timer "variable" value) == 17
    @test ncalls(timer["variable"]) == 1
    @test (@static_timeit timer "quoted" :sentinel) === :sentinel
    @test ncalls(timer["quoted"]) == 1

    other = TimerOutput()
    @test block_form(other, "other", 1) == (3, 2)
    @test !haskey(timer, "other")
    reset_timer!(timer)
    @test !haskey(timer, "block")
    @test block_form(timer, "block", 5) == (11, 6)
    @test ncalls(timer["block"]) == 1
end
