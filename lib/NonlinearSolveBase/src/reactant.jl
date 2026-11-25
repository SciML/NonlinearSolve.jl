# Support for running the solver loops under `Reactant.@compile`/`@jit`. Outside a
# Reactant compilation `ReactantCore.within_compile()` is a compile-time `false`, so every
# helper here folds away on the ordinary Julia path.
#
# `within_compile()` must only be called from ordinary functions, never directly inside a
# `ReactantCore.@trace` body: the macro captures every symbol of the body, the module
# included, as a loop-carried variable, and the call then no longer goes through Reactant's
# overlay and returns `false` while tracing. The helpers below gate themselves, so loop
# bodies call them unconditionally.

"""
    maybe_traced(x)

Return `x` promoted to a traced scalar when called during a Reactant compilation and `x`
itself otherwise. Loop-carried scalars (`nsteps`, `force_stop`, `retcode`, ...) have to be
traced before a `ReactantCore.@trace while` loop for their final values to be visible after
it, and since solver caches are `@concrete`, at the point the cache is constructed.
"""
maybe_traced(x) = ReactantCore.within_compile() ? ReactantCore.promote_to_traced(x) : x

"""
    dealias_traced!(x)

Replace every traced array or scalar reachable from `x` with a fresh copy so that no two
places share a traced value. Reactant records only one path per traced object among the
values carried by a `@trace while` loop and requires the set of paths to be the same before
and after each iteration; solver caches alias freely (`u_cache`/`u`, `p` in several caches,
the problem inside the trace) and steps rebind fields, so every value is made distinct at
the loop boundary instead. Structs that need new immutable or const field bindings are rebuilt.
"""
function dealias_traced!(x)
    ReactantCore.within_compile() || return x
    # Traced arrays expose their traced scalar as eltype. Obtain the scalar family
    # through promotion so the traversal also recognizes new device element types.
    scalar = typejoin(typeof(maybe_traced(false)), typeof(maybe_traced(0)))
    rebuild = IdDict{Any, Bool}()
    _plan_const_copies!(rebuild, x, scalar, false)
    return _dealias_traced!(x, scalar, IdDict(), rebuild)
end

function _has_traced_leaf(x, scalar, seen)
    x isa scalar && return true
    x isa DenseArray && !(x isa Array) && return eltype(x) <: scalar
    (isbits(x) || x isa Union{Type, Module, AbstractString, Symbol}) && return false
    if ismutable(x)
        haskey(seen, x) && return false
        seen[x] = nothing
    end
    if x isa Array
        isbitstype(eltype(x)) && return false
        for i in eachindex(x)
            isassigned(x, i) && _has_traced_leaf(x[i], scalar, seen) && return true
        end
        return false
    end
    T = typeof(x)
    return any(fieldnames(T)) do name
        return isdefined(x, name) && _has_traced_leaf(getfield(x, name), scalar, seen)
    end
end

function _plan_const_copies!(rebuild, x, scalar, protect)
    if x isa Array
        isbitstype(eltype(x)) && return
        haskey(rebuild, x) && (!protect || rebuild[x]) && return
        rebuild[x] = protect
        for i in eachindex(x)
            isassigned(x, i) && _plan_const_copies!(rebuild, x[i], scalar, protect)
        end
        return
    end
    (
        x isa scalar || x isa DenseArray || isbits(x) ||
            x isa Union{Type, Module, AbstractString, Symbol}
    ) && return
    T = typeof(x)
    names = fieldnames(T)
    copy_children = if ismutable(x)
        haskey(rebuild, x) && (!protect || rebuild[x]) && return
        copy_object = any(names) do name
            return (protect || isconst(T, name)) && isdefined(x, name) &&
                _has_traced_leaf(getfield(x, name), scalar, IdDict())
        end
        rebuild[x] = copy_object
    else
        protect
    end
    for name in names
        isdefined(x, name) && _plan_const_copies!(rebuild, getfield(x, name), scalar, copy_children)
    end
    return
end

function _dealias_traced!(x, scalar, visited, rebuild)
    x isa scalar && return oftype(x, x + zero(x))
    if x isa Array
        isbitstype(eltype(x)) && return x
        haskey(visited, x) && return visited[x]::typeof(x)
        _has_traced_leaf(x, scalar, IdDict()) || return x
        result = copy(x)
        visited[x] = result
        for i in eachindex(x)
            isassigned(x, i) && (result[i] = _dealias_traced!(x[i], scalar, visited, rebuild))
        end
        delete!(visited, x)
        return result
    elseif x isa DenseArray
        eltype(x) <: scalar || return x
        return oftype(x, x .+ zero(eltype(x)))
    end
    (isbits(x) || x isa Union{Type, Module, AbstractString, Symbol}) && return x
    T = typeof(x)
    names = fieldnames(T)
    if ismutable(x)
        haskey(visited, x) && return visited[x]::T
        # Register the shell before visiting its fields so back-references use the copy.
        copy_object = rebuild[x]
        result = copy_object ? ccall(:jl_new_struct_uninit, Any, (Any,), T)::T : x
        visited[x] = result
        for (i, name) in pairs(names)
            isdefined(x, name) || continue
            field = getfield(x, name)
            value = _dealias_traced!(field, scalar, visited, rebuild)::typeof(field)
            if copy_object
                ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), result, i - 1, value)
            elseif value !== field
                setfield!(x, name, value)
            end
        end
        return result
    end
    fields = map(name -> getfield(x, name), names)
    values = map(field -> _dealias_traced!(field, scalar, visited, rebuild), fields)
    values === fields && return x
    x isa Tuple && return values
    x isa NamedTuple && return NamedTuple{names}(values)
    return ConstructionBase.setproperties(x, NamedTuple{names}(values))
end

"""
    build_nonlinear_solution(prob, alg, u, resid; retcode, stats = nothing, trace = nothing)

`SciMLBase.build_solution` for a nonlinear problem, except that during a Reactant
compilation the solution stores `nothing` as its problem: a problem carries its keyword
arguments as a `Base.Pairs`, which Reactant cannot rebuild when it returns the solution from
the compiled program.
"""
function build_nonlinear_solution(
        prob, alg, u, resid; retcode, stats = nothing, trace = nothing
    )
    stored_prob = ReactantCore.within_compile() ? nothing : prob
    return SciMLBase.NonlinearSolution(
        u, resid, stored_prob, alg, retcode, nothing, nothing, nothing, stats, trace
    )
end

"""
    select(cond, a, b)

`ifelse(cond, a, b)` that also works when `cond` is a traced scalar and `a`, `b` are
arrays, in which case the selection is elementwise.
"""
select(cond::Bool, a, b) = ifelse(cond, a, b)
# Two methods only: a condition of unknown type must not turn this into a dynamic dispatch
# that loses the result type, which is that of `a`/`b` either way.
select(cond, a, b) = a isa AbstractArray ? ifelse.(cond, a, b) : ifelse(cond, a, b)
