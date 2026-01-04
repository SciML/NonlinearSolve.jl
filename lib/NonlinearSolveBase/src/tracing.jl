@concrete struct NonlinearSolveTracing
    trace_mode <: Union{Val{:minimal}, Val{:condition_number}, Val{:all}}
    print_frequency::Int
    store_frequency::Int
end

"""
    TraceMinimal(freq)
    TraceMinimal(; print_frequency = 1, store_frequency::Int = 1)

Trace Minimal Information

 1. Iteration Number
 2. f(u) inf-norm
 3. Step 2-norm

See also [`TraceWithJacobianConditionNumber`](@ref) and [`TraceAll`](@ref).
"""
function TraceMinimal(; print_frequency = 1, store_frequency::Int = 1)
    return NonlinearSolveTracing(Val(:minimal), print_frequency, store_frequency)
end

"""
    TraceWithJacobianConditionNumber(freq)
    TraceWithJacobianConditionNumber(; print_frequency = 1, store_frequency::Int = 1)

[`TraceMinimal`](@ref) + Print the Condition Number of the Jacobian.

See also [`TraceMinimal`](@ref) and [`TraceAll`](@ref).
"""
function TraceWithJacobianConditionNumber(; print_frequency = 1, store_frequency::Int = 1)
    return NonlinearSolveTracing(Val(:condition_number), print_frequency, store_frequency)
end

"""
    TraceAll(freq)
    TraceAll(; print_frequency = 1, store_frequency::Int = 1)

[`TraceWithJacobianConditionNumber`](@ref) + Store the Jacobian, u, f(u), and δu.

!!! warning

    This is very expensive and makes copies of the Jacobian, u, f(u), and δu.

See also [`TraceMinimal`](@ref) and [`TraceWithJacobianConditionNumber`](@ref).
"""
function TraceAll(; print_frequency = 1, store_frequency::Int = 1)
    return NonlinearSolveTracing(Val(:all), print_frequency, store_frequency)
end

for Tr in (:TraceMinimal, :TraceWithJacobianConditionNumber, :TraceAll)
    @eval $(Tr)(freq) = $(Tr)(; print_frequency = freq, store_frequency = freq)
end

# NonlinearSolve Tracing Utilities
@concrete struct NonlinearSolveTraceEntry
    iteration::Int
    fnorm
    stepnorm
    condJ
    storage
    norm_type::Symbol
end

function Base.getproperty(entry::NonlinearSolveTraceEntry, sym::Symbol)
    hasfield(typeof(entry), sym) && return getfield(entry, sym)
    return getproperty(entry.storage, sym)
end

function print_top_level(io::IO, entry::NonlinearSolveTraceEntry)
    return if entry.condJ === nothing
        @printf io "%-8s\t%-20s\t%-20s\n" "----" "-------------" "-----------"
        if entry.norm_type === :L2
            @printf io "%-8s\t%-20s\t%-20s\n" "Iter" "f(u) 2-norm" "Step 2-norm"
        else
            @printf io "%-8s\t%-20s\t%-20s\n" "Iter" "f(u) inf-norm" "Step 2-norm"
        end
        @printf io "%-8s\t%-20s\t%-20s\n" "----" "-------------" "-----------"
    else
        @printf io "%-8s\t%-20s\t%-20s\t%-20s\n" "----" "-------------" "-----------" "-------"
        if entry.norm_type === :L2
            @printf io "%-8s\t%-20s\t%-20s\t%-20s\n" "Iter" "f(u) 2-norm" "Step 2-norm" "cond(J)"
        else
            @printf io "%-8s\t%-20s\t%-20s\t%-20s\n" "Iter" "f(u) inf-norm" "Step 2-norm" "cond(J)"
        end
        @printf io "%-8s\t%-20s\t%-20s\t%-20s\n" "----" "-------------" "-----------" "-------"
    end
end

function Base.show(io::IO, ::MIME"text/plain", entry::NonlinearSolveTraceEntry)
    entry.iteration == 0 && print_top_level(io, entry)
    return if entry.iteration < 0 # Special case for final entry
        @printf io "%-8s\t%-20.8e\n" "Final" entry.fnorm
        @printf io "%-28s\n" "----------------------"
    elseif entry.condJ === nothing
        @printf io "%-8d\t%-20.8e\t%-20.8e\n" entry.iteration entry.fnorm entry.stepnorm
    else
        @printf io "%-8d\t%-20.8e\t%-20.8e\t%-20.8e\n" entry.iteration entry.fnorm entry.stepnorm entry.condJ
    end
end

function NonlinearSolveTraceEntry(prob::AbstractNonlinearProblem, iteration, fu, δu, J, u)
    norm_type = ifelse(prob isa NonlinearLeastSquaresProblem, :L2, :Inf)
    fnorm = prob isa NonlinearLeastSquaresProblem ? L2_NORM(fu) : Linf_NORM(fu)
    condJ = J !== missing ? Utils.condition_number(J) : nothing
    storage = if u === missing
        nothing
    else
        (;
            u = ArrayInterface.ismutable(u) ? copy(u) : u,
            fu = ArrayInterface.ismutable(fu) ? copy(fu) : fu,
            δu = ArrayInterface.ismutable(δu) ? copy(δu) : δu,
            J = ArrayInterface.ismutable(J) ? copy(J) : J,
        )
    end
    return NonlinearSolveTraceEntry(
        iteration, fnorm, L2_NORM(δu), condJ, storage, norm_type
    )
end

@concrete struct NonlinearSolveTrace
    show_trace <: Union{Val{false}, Val{true}}
    store_trace <: Union{Val{false}, Val{true}}
    history
    trace_level <: NonlinearSolveTracing
    prob
end

reset!(trace::NonlinearSolveTrace) = reset!(trace.history)
reset!(::Nothing) = nothing
reset!(history::Vector) = empty!(history)

function Base.show(io::IO, ::MIME"text/plain", trace::NonlinearSolveTrace)
    return if trace.history !== nothing
        foreach(trace.history) do entry
            show(io, MIME"text/plain"(), entry)
        end
    else
        print(io, "Tracing Disabled")
    end
end

function init_nonlinearsolve_trace(
        prob, alg, u, fu, J, δu; show_trace::Val = Val(false),
        trace_level::NonlinearSolveTracing = TraceMinimal(), store_trace::Val = Val(false),
        uses_jac_inverse = Val(false), kwargs...
    )
    return init_nonlinearsolve_trace(
        prob, alg, show_trace, trace_level, store_trace, u, fu, J, δu, uses_jac_inverse
    )
end

function init_nonlinearsolve_trace(
        prob::AbstractNonlinearProblem, alg, show_trace::Val,
        trace_level::NonlinearSolveTracing, store_trace::Val, u, fu, J, δu,
        uses_jac_inverse::Val
    )
    if show_trace isa Val{true}
        print("\nAlgorithm: ")
        str = Utils.clean_sprint_struct(alg, 0)
        Base.printstyled(str, "\n\n"; color = :green, bold = true)
    end
    J = uses_jac_inverse isa Val{true} ?
        (trace_level.trace_mode isa Val{:minimal} ? J : LinearAlgebra.pinv(J)) : J
    history = init_trace_history(prob, show_trace, trace_level, store_trace, u, fu, J, δu)
    return NonlinearSolveTrace(show_trace, store_trace, history, trace_level, prob)
end

function init_trace_history(
        prob::AbstractNonlinearProblem, show_trace::Val, trace_level,
        store_trace::Val, u, fu, J, δu
    )
    store_trace isa Val{false} && show_trace isa Val{false} && return nothing
    entry = if trace_level.trace_mode isa Val{:minimal}
        NonlinearSolveTraceEntry(prob, 0, fu, δu, missing, missing)
    elseif trace_level.trace_mode isa Val{:condition_number}
        NonlinearSolveTraceEntry(prob, 0, fu, δu, J, missing)
    else
        NonlinearSolveTraceEntry(prob, 0, fu, δu, J, u)
    end
    show_trace isa Val{true} && show(stdout, MIME"text/plain"(), entry)
    store_trace isa Val{true} && return NonlinearSolveTraceEntry[entry]
    return nothing
end

function update_trace!(
        trace::NonlinearSolveTrace, iter, u, fu, J, δu, α = true; last::Val = Val(false)
    )
    trace.store_trace isa Val{false} && trace.show_trace isa Val{false} && return nothing

    if last isa Val{true}
        norm_type = ifelse(trace.prob isa NonlinearLeastSquaresProblem, :L2, :Inf)
        fnorm = trace.prob isa NonlinearLeastSquaresProblem ? L2_NORM(fu) : Linf_NORM(fu)
        entry = NonlinearSolveTraceEntry(-1, fnorm, NaN32, nothing, nothing, norm_type)
        trace.show_trace isa Val{true} && show(stdout, MIME"text/plain"(), entry)
        return trace
    end

    show_now = trace.show_trace isa Val{true} &&
        (mod1(iter, trace.trace_level.print_frequency) == 1)
    store_now = trace.store_trace isa Val{true} &&
        (mod1(iter, trace.trace_level.store_frequency) == 1)
    if show_now || store_now
        entry = if trace.trace_level.trace_mode isa Val{:minimal}
            NonlinearSolveTraceEntry(trace.prob, iter, fu, δu .* α, missing, missing)
        else
            if !isnothing(J)
                J = convert(AbstractArray, J)
            end
            if trace.trace_level.trace_mode isa Val{:condition_number}
                NonlinearSolveTraceEntry(trace.prob, iter, fu, δu .* α, J, missing)
            else
                NonlinearSolveTraceEntry(trace.prob, iter, fu, δu .* α, J, u)
            end
        end
        show_now && show(stdout, MIME"text/plain"(), entry)
        store_now && push!(trace.history, entry)
    end
    return trace
end

function update_trace!(cache, α = true; uses_jac_inverse = Val(false))
    trace = Utils.safe_getproperty(cache, Val(:trace))
    trace === missing && return nothing

    J = Utils.safe_getproperty(cache, Val(:J))
    du = SciMLBase.get_du(cache)
    return if J === missing
        update_trace!(
            trace, cache.nsteps + 1, get_u(cache), get_fu(cache), nothing, du, α
        )
    else
        J = uses_jac_inverse isa Val{true} ? Utils.Pinv(cache.J) : cache.J
        update_trace!(trace, cache.nsteps + 1, get_u(cache), get_fu(cache), J, du, α)
    end
end
