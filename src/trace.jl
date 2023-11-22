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
        @printf io "%-6s %-20s %-20s\n" "Iter" "f(u) inf-norm" "Step 2-norm"
        @printf io "%-6s %-20s %-20s\n" "----" "-------------" "-----------"
    else
        @printf io "%-6s %-20s %-20s %-20s\n" "Iter" "f(u) inf-norm" "Step 2-norm" "cond(J)"
        @printf io "%-6s %-20s %-20s %-20s\n" "----" "-------------" "-----------" "-------"
    end
end

function Base.show(io::IO, entry::NonlinearSolveTraceEntry)
    entry.iteration == 0 && __show_top_level(io, entry)
    if entry.condJ === nothing
        @printf io "%-6d %-20.8e %-20.8e\n" entry.iteration entry.fnorm entry.stepnorm
    else
        @printf io "%-6d %-20.8e %-20.8e %-20.8e\n" entry.iteration entry.fnorm entry.stepnorm entry.condJ
    end
    return nothing
end

function NonlinearSolveTraceEntry(iteration, fu, δu)
    return NonlinearSolveTraceEntry(iteration, norm(fu, Inf), norm(δu, 2), nothing,
        nothing, nothing, nothing, nothing)
end

function NonlinearSolveTraceEntry(iteration, fu, δu, J)
    return NonlinearSolveTraceEntry(iteration, norm(fu, Inf), norm(δu, 2), __cond(J),
        nothing,
        nothing, nothing, nothing)
end

function NonlinearSolveTraceEntry(iteration, fu, δu, J, u)
    return NonlinearSolveTraceEntry(iteration, norm(fu, Inf), norm(δu, 2), __cond(J),
        copy(J), copy(u), copy(fu), copy(δu))
end

__cond(J::AbstractMatrix) = cond(J)
__cond(J) = NaN  # Covers cases where `J` is a Operator, nothing, etc.

@concrete struct NonlinearSolveTrace{show_trace, trace_level, store_trace}
    history
end

function Base.show(io::IO, trace::NonlinearSolveTrace)
    for entry in trace.history
        show(io, entry)
    end
    return nothing
end

function init_nonlinearsolve_trace(u, fu, J, δu; show_trace::Val = Val(false),
        trace_level::Val = Val(1), store_trace::Val = Val(false), kwargs...)
    return init_nonlinearsolve_trace(show_trace, trace_level, store_trace, u, fu, J, δu)
end

function init_nonlinearsolve_trace(::Val{show_trace}, ::Val{trace_level},
        ::Val{store_trace}, u, fu, J, δu) where {show_trace, trace_level, store_trace}
    history = __init_trace_history(Val{show_trace}(), Val{trace_level}(),
        Val{store_trace}(), u, fu, J, δu)
    return NonlinearSolveTrace{show_trace, trace_level, store_trace}(history)
end

function __init_trace_history(::Val{show_trace}, ::Val{trace_level}, ::Val{store_trace}, u,
        fu, J, δu) where {show_trace, trace_level, store_trace}
    !store_trace && !show_trace && return nothing
    entry = __trace_entry(Val{trace_level}(), 0, u, fu, J, δu)
    show_trace && show(entry)
    store_trace && return [entry]
    return nothing
end

function __trace_entry(::Val{1}, iter, u, fu, J, δu, α = 1)
    NonlinearSolveTraceEntry(iter, fu, δu .* α)
end
function __trace_entry(::Val{2}, iter, u, fu, J, δu, α = 1)
    NonlinearSolveTraceEntry(iter, fu, δu .* α, J)
end
function __trace_entry(::Val{3}, iter, u, fu, J, δu, α = 1)
    NonlinearSolveTraceEntry(iter, fu, δu .* α, J, u)
end
function __trace_entry(::Val{T}, iter, u, fu, J, δu, α = 1) where {T}
    throw(ArgumentError("::Val{trace_level} == ::Val{$(T)} is not supported. \
                        Possible values are `Val{1}()`/`Val{2}()`/`Val{3}()`."))
end

function update_trace!(trace::NonlinearSolveTrace{ShT, TrL, StT}, iter, u, fu, J,
        δu, α) where {ShT, TrL, StT}
    !StT && !ShT && return nothing
    entry = __trace_entry(Val{TrL}(), iter, u, fu, J, δu, α)
    StT && push!(trace.history, entry)
    ShT && show(entry)
    return trace
end

# Needed for Algorithms which directly use `inv(J)` instead of `J`
function update_trace_with_invJ!(trace::NonlinearSolveTrace{ShT, TrL, StT}, iter, u, fu, J,
        δu, α) where {ShT, TrL, StT}
    !StT && !ShT && return nothing
    if TrL == 1
        entry = __trace_entry(Val{1}(), iter, u, fu, J, δu, α)
    else
        entry = __trace_entry(Val{TrL}(), iter, u, fu, inv(J), δu, α)
    end
    StT && push!(trace.history, entry)
    ShT && show(entry)
    return trace
end
