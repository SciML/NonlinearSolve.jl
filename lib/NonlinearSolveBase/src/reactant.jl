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
the loop boundary instead. Mutable structs are updated in place, immutable ones are rebuilt.
"""
function dealias_traced!(x)
    ReactantCore.within_compile() || return x
    return _dealias_traced!(x, Base.IdSet())
end

function _dealias_traced!(x, ancestors)
    (isbits(x) || x isa Union{Type, Module, Symbol, AbstractString, Array}) && return x
    x isa DenseArray && return copy(x)
    x isa Number && return copy(x)
    x isa Union{Tuple, NamedTuple} && return map(v -> _dealias_traced!(v, ancestors), x)
    names = fieldnames(typeof(x))
    if ismutable(x)
        x in ancestors && return x
        push!(ancestors, x)
        for name in names
            isdefined(x, name) || continue
            field = getfield(x, name)
            value = _dealias_traced!(field, ancestors)
            field === value || setfield!(x, name, value)
        end
        delete!(ancestors, x)
        return x
    end
    fields = map(name -> getfield(x, name), names)
    values = map(v -> _dealias_traced!(v, ancestors), fields)
    values === fields && return x
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
