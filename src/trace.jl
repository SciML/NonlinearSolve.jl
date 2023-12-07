abstract type AbstractNonlinearSolveTraceLevel end

"""
    TraceMinimal(freq)
    TraceMinimal(; print_frequency = 1, store_frequency::Int = 1)

Trace Minimal Information

 1. Iteration Number
 2. f(u) inf-norm
 3. Step 2-norm

## Arguments

  - `freq`: Sets both `print_frequency` and `store_frequency` to `freq`.

## Keyword Arguments

  - `print_frequency`: Print the trace every `print_frequency` iterations if
    `show_trace == Val(true)`.
  - `store_frequency`: Store the trace every `store_frequency` iterations if
    `store_trace == Val(true)`.
"""
@kwdef struct TraceMinimal <: AbstractNonlinearSolveTraceLevel
    print_frequency::Int = 1
    store_frequency::Int = 1
end

"""
    TraceWithJacobianConditionNumber(freq)
    TraceWithJacobianConditionNumber(; print_frequency = 1, store_frequency::Int = 1)

`TraceMinimal` + Print the Condition Number of the Jacobian.

## Arguments

  - `freq`: Sets both `print_frequency` and `store_frequency` to `freq`.

## Keyword Arguments

  - `print_frequency`: Print the trace every `print_frequency` iterations if
    `show_trace == Val(true)`.
  - `store_frequency`: Store the trace every `store_frequency` iterations if
    `store_trace == Val(true)`.
"""
@kwdef struct TraceWithJacobianConditionNumber <: AbstractNonlinearSolveTraceLevel
    print_frequency::Int = 1
    store_frequency::Int = 1
end

"""
    TraceAll(freq)
    TraceAll(; print_frequency = 1, store_frequency::Int = 1)

`TraceWithJacobianConditionNumber` + Store the Jacobian, u, f(u), and δu.

!!! warning

    This is very expensive and makes copyies of the Jacobian, u, f(u), and δu.

## Arguments

    - `freq`: Sets both `print_frequency` and `store_frequency` to `freq`.

## Keyword Arguments

    - `print_frequency`: Print the trace every `print_frequency` iterations if
    `show_trace == Val(true)`.
    - `store_frequency`: Store the trace every `store_frequency` iterations if
    `store_trace == Val(true)`.
"""
@kwdef struct TraceAll <: AbstractNonlinearSolveTraceLevel
    print_frequency::Int = 1
    store_frequency::Int = 1
end

for Tr in (:TraceMinimal, :TraceWithJacobianConditionNumber, :TraceAll)
    @eval begin
        $(Tr)(freq) = $(Tr)(; print_frequency = freq, store_frequency = freq)
    end
end

# NonlinearSolve Tracing Utilities
@concrete struct NonlinearSolveTraceEntry
    iteration::Int
    fnorm
    stepnorm
    condJ
    J
    u
    fu
    δu
end

function __show_top_level(io::IO, entry::NonlinearSolveTraceEntry)
    if entry.condJ === nothing
        @printf io "%-8s %-20s %-20s\n" "----" "-------------" "-----------"
        @printf io "%-8s %-20s %-20s\n" "Iter" "f(u) inf-norm" "Step 2-norm"
        @printf io "%-8s %-20s %-20s\n" "----" "-------------" "-----------"
    else
        @printf io "%-8s %-20s %-20s %-20s\n" "----" "-------------" "-----------" "-------"
        @printf io "%-8s %-20s %-20s %-20s\n" "Iter" "f(u) inf-norm" "Step 2-norm" "cond(J)"
        @printf io "%-8s %-20s %-20s %-20s\n" "----" "-------------" "-----------" "-------"
    end
end

function Base.show(io::IO, entry::NonlinearSolveTraceEntry)
    entry.iteration == 0 && __show_top_level(io, entry)
    if entry.iteration < 0
        # Special case for final entry
        @printf io "%-8s %-20.8e\n" "Final" entry.fnorm
        @printf io "%-28s\n" "----------------------"
    elseif entry.condJ === nothing
        @printf io "%-8d %-20.8e %-20.8e\n" entry.iteration entry.fnorm entry.stepnorm
    else
        @printf io "%-8d %-20.8e %-20.8e %-20.8e\n" entry.iteration entry.fnorm entry.stepnorm entry.condJ
    end
    return nothing
end

function NonlinearSolveTraceEntry(iteration, fu, δu)
    return NonlinearSolveTraceEntry(iteration, norm(fu, Inf), norm(δu, 2), nothing,
        nothing, nothing, nothing, nothing)
end

function NonlinearSolveTraceEntry(iteration, fu, δu, J)
    return NonlinearSolveTraceEntry(iteration, norm(fu, Inf), norm(δu, 2), __cond(J),
        nothing, nothing, nothing, nothing)
end

function NonlinearSolveTraceEntry(iteration, fu, δu, J, u)
    return NonlinearSolveTraceEntry(iteration, norm(fu, Inf), norm(δu, 2), __cond(J),
        __copy(J), __copy(u), __copy(fu), __copy(δu))
end

__cond(J::AbstractMatrix) = cond(J)
__cond(J::SVector) = __cond(Diagonal(MVector(J)))
__cond(J::AbstractVector) = __cond(Diagonal(J))
__cond(J::ApplyArray) = __cond(J.f(J.args...))
__cond(J) = -1  # Covers cases where `J` is a Operator, nothing, etc.

__copy(x::AbstractArray) = copy(x)
__copy(x::Number) = x
__copy(x) = x

@concrete struct NonlinearSolveTrace{show_trace, store_trace,
    Tr <: AbstractNonlinearSolveTraceLevel}
    history
    trace_level::Tr
end

function reset!(trace::NonlinearSolveTrace)
    (trace.history !== nothing && resize!(trace.history, 0))
end

function Base.show(io::IO, trace::NonlinearSolveTrace)
    if trace.history !== nothing
        foreach(entry -> show(io, entry), trace.history)
    else
        print(io, "Tracing Disabled")
    end
    return nothing
end

function init_nonlinearsolve_trace(alg, u, fu, J, δu; show_trace::Val = Val(false),
        trace_level::AbstractNonlinearSolveTraceLevel = TraceMinimal(),
        store_trace::Val = Val(false), uses_jac_inverse = Val(false), kwargs...)
    return init_nonlinearsolve_trace(alg, show_trace, trace_level, store_trace, u, fu, J,
        δu, uses_jac_inverse)
end

function init_nonlinearsolve_trace(alg, ::Val{show_trace},
        trace_level::AbstractNonlinearSolveTraceLevel, ::Val{store_trace}, u, fu, J,
        δu, ::Val{uses_jac_inverse}) where {show_trace, store_trace, uses_jac_inverse}
    if show_trace
        print("\nAlgorithm: ")
        Base.printstyled(alg, "\n\n"; color = :green, bold = true)
    end
    J_ = uses_jac_inverse ? (trace_level isa TraceMinimal ? J : __safe_inv(J)) : J
    history = __init_trace_history(Val{show_trace}(), trace_level, Val{store_trace}(), u,
        fu, J_, δu)
    return NonlinearSolveTrace{show_trace, store_trace}(history, trace_level)
end

function __init_trace_history(::Val{show_trace}, trace_level, ::Val{store_trace}, u, fu, J,
        δu) where {show_trace, store_trace}
    !store_trace && !show_trace && return nothing
    entry = __trace_entry(trace_level, 0, u, fu, J, δu)
    show_trace && show(entry)
    store_trace && return NonlinearSolveTraceEntry[entry]
    return nothing
end

function __trace_entry(::TraceMinimal, iter, u, fu, J, δu, α = 1)
    return NonlinearSolveTraceEntry(iter, fu, δu .* α)
end
function __trace_entry(::TraceWithJacobianConditionNumber, iter, u, fu, J, δu, α = 1)
    return NonlinearSolveTraceEntry(iter, fu, δu .* α, J)
end
function __trace_entry(::TraceAll, iter, u, fu, J, δu, α = 1)
    return NonlinearSolveTraceEntry(iter, fu, δu .* α, J, u)
end

function update_trace!(trace::NonlinearSolveTrace{ShT, StT}, iter, u, fu, J, δu,
        α = 1; last::Val{L} = Val(false)) where {ShT, StT, L}
    !StT && !ShT && return nothing

    if L
        entry = NonlinearSolveTraceEntry(-1, norm(fu, Inf), NaN32, nothing, nothing,
            nothing, nothing, nothing)
        ShT && show(entry)
        return trace
    end

    show_now = ShT && (mod1(iter, trace.trace_level.print_frequency) == 1)
    store_now = StT && (mod1(iter, trace.trace_level.store_frequency) == 1)
    (show_now || store_now) && (entry = __trace_entry(trace.trace_level, iter, u, fu, J,
        δu, α))
    store_now && push!(trace.history, entry)
    show_now && show(entry)
    return trace
end

function update_trace!(cache::AbstractNonlinearSolveCache, α = true)
    trace = __getproperty(cache, Val(:trace))
    trace === nothing && return nothing

    J = __getproperty(cache, Val(:J))
    if J === nothing
        J_inv = __getproperty(cache, Val(:J⁻¹))
        if J_inv === nothing
            update_trace!(trace, cache.stats.nsteps + 1, get_u(cache), get_fu(cache),
                nothing, cache.du, α)
        else
            update_trace!(trace, cache.stats.nsteps + 1, get_u(cache), get_fu(cache),
                ApplyArray(__safe_inv, J_inv), cache.du, α)
        end
    else
        update_trace!(trace, cache.stats.nsteps + 1, get_u(cache), get_fu(cache), J,
            cache.du, α)
    end
end
