module NonlinearSolveBaseEnzymeExt

using NonlinearSolveBase
import SciMLBase: SciMLBase, value
using Enzyme
import Enzyme: Const, MixedDuplicated
using ChainRulesCore
import SciMLStructures

# Accumulate a tangent `darg` into a shadow `dval`.
# When `dval` is a SciMLStructure (e.g. MTKParameters), `darg` may be:
#   - A tunable gradient vector (from SciMLSensitivity's EnzymeOriginator path)
#   - Another SciMLStructure
#   - A broadcastable array
# In all cases, accumulation goes through the SciMLStructures interface.
#
# `diff_tunables` mirrors the sensealg field of the same name and means
# "differentiate only the Tunable portion." When `true` (the default and
# the value carried by `SteadyStateAdjoint`/`Quadrature`/`Gauss` adjoints
# unless the user opted out) only the Tunable slice of a structured
# `darg` is accumulated. When `false`, `SciMLSensitivity.steadystatebackpass`
# returns a structured cotangent whose gradient contribution may live in
# non-Tunable fields such as `caches` (e.g. SCC sub-problem buffers feeding
# `explicitfuns!`), so those fields are walked in as well.
function _accum_tangent!(dval, darg; diff_tunables::Bool = true)
    if SciMLStructures.isscimlstructure(dval) && !(dval isa AbstractArray)
        if SciMLStructures.isscimlstructure(darg)
            shadow_tunables, _, _ = SciMLStructures.canonicalize(
                SciMLStructures.Tunable(), dval,
            )
            darg_tunables, _, _ = SciMLStructures.canonicalize(
                SciMLStructures.Tunable(), darg,
            )
            shadow_tunables .+= darg_tunables
            SciMLStructures.replace!(SciMLStructures.Tunable(), dval, shadow_tunables)
            if !diff_tunables
                for field in fieldnames(typeof(darg))
                    field === :tunable && continue
                    hasfield(typeof(dval), field) || continue
                    _accum_nested!(getfield(dval, field), getfield(darg, field))
                end
            end
        elseif darg isa AbstractVector
            shadow_tunables, _, _ = SciMLStructures.canonicalize(
                SciMLStructures.Tunable(), dval,
            )
            if length(darg) == length(shadow_tunables)
                shadow_tunables .+= darg
                SciMLStructures.replace!(
                    SciMLStructures.Tunable(), dval, shadow_tunables,
                )
            else
                dval .+= darg
            end
        elseif darg isa NamedTuple
            # Full parameter gradient from SteadyStateAdjoint (includes
            # caches and other non-tunable components). Accumulate each
            # matching field into the shadow.
            for field in fieldnames(typeof(darg))
                src = getfield(darg, field)
                src === nothing && continue
                if hasfield(typeof(dval), field)
                    dst = getfield(dval, field)
                    _accum_nested!(dst, src)
                end
            end
        else
            dval .+= darg
        end
    else
        dval .+= darg
    end
    return nothing
end

# Recursively accumulate nested containers (tuples of arrays, etc.)
function _accum_nested!(dst::AbstractArray, src::AbstractArray)
    dst .+= src
    return nothing
end
function _accum_nested!(dst::Tuple, src::Tuple)
    for (d, s) in zip(dst, src)
        _accum_nested!(d, s)
    end
    return nothing
end
_accum_nested!(::Any, ::Nothing) = nothing
_accum_nested!(::Nothing, ::Any) = nothing
_accum_nested!(::Nothing, ::Nothing) = nothing

# `solve_up`'s differentiable inputs (prob/u0/p) are positional; its keyword
# arguments are solver configuration (abstol/reltol/saveat/sensealg/...). Mark
# them inactive so the custom rule applies when this `solve_up` is reached under
# `set_runtime_activity` (e.g. the MTK DAE init NonlinearProblem solve), where a
# config kwarg may otherwise be promoted to active and raise
# NonConstantKeywordArgException. Mirrors the DiffEqBase `solve_up` declaration.
Enzyme.EnzymeRules.inactive_kwarg(::typeof(NonlinearSolveBase.solve_up), prob, sensealg::Union{Nothing, SciMLBase.AbstractSensitivityAlgorithm}, u0, p, args...; kwargs...) = nothing

# Nonlinear-solve algorithms are pure solver configuration (Vals, Rationals,
# Nothing/Missing sentinels, nested immutable sub-algorithms, and a ForwardDiff
# tag type parameter) with no differentiable floating-point data. When a
# `NonlinearProblem` is solved under `Enzyme.set_runtime_activity(Reverse)` — e.g.
# the MTK DAE-initialization solve reached through `solve_up` with the default
# `NonlinearSolvePolyAlgorithm` as a positional `args...` — Enzyme can otherwise
# promote that configuration argument to `Duplicated`. That trips the
# `roots_activep != activep` assertion in `enzyme_custom_setup_args`, or leaves a
# spurious algorithm shadow that corrupts activity bookkeeping for the genuinely
# active `u0`/`p` (silently dropping a cotangent component). Declaring the whole
# algorithm hierarchy inactive forces `Const`, so `activep == roots_activep`.
# Mirrors SciMLBase's `inactive_type(::Type{<:AbstractSensitivityAlgorithm})` and
# LinearSolve's `inactive_type(::Type{<:SciMLLinearSolveAlgorithm})`.
Enzyme.EnzymeRules.inactive_type(::Type{<:NonlinearSolveBase.AbstractNonlinearSolveAlgorithm}) = true

_solve_up_rt_valtype(::Type{<:Enzyme.Annotation{T}}) where {T} = T

function Enzyme.EnzymeRules.augmented_primal(
        config::Enzyme.EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(NonlinearSolveBase.solve_up)}, ::Type{RT}, prob,
        sensealg::Enzyme.Annotation{<:Union{Nothing, SciMLBase.AbstractSensitivityAlgorithm}},
        u0, p, args...; kwargs...
    ) where {RT <: Enzyme.Annotation}
    @inline function copy_or_reuse(val, idx)
        if Enzyme.EnzymeRules.overwritten(config)[idx] && ismutable(val)
            return deepcopy(val)
        else
            return val
        end
    end

    @inline function arg_copy(i)
        return copy_or_reuse(args[i].val, i + 5)
    end

    res = NonlinearSolveBase._solve_adjoint(
        copy_or_reuse(prob.val, 2), copy_or_reuse(sensealg.val, 3),
        copy_or_reuse(u0.val, 4), copy_or_reuse(p.val, 5),
        SciMLBase.EnzymeOriginator(), ntuple(arg_copy, Val(length(args)))...;
        kwargs...
    )

    mz = Enzyme.make_zero(res[1])
    # As in DiffEqBase's solve_up rule: when the return slot is abstract and the
    # solution's guessed activity is MixedDuplicated, the shadow must be a
    # `Ref`-wrapped value (Enzyme's by-reference MixedDuplicated representation),
    # otherwise `create_activity_wrapper` builds `MixedDuplicated(::Solution)` and
    # errors. The reverse rule dereferences it before handing it to the pullback.
    dres = if Base.isabstracttype(_solve_up_rt_valtype(RT)) &&
            (Enzyme.guess_activity(Core.Typeof(res[1]), Enzyme.Reverse) <: Enzyme.MixedDuplicated)
        Ref(mz)
    else
        mz
    end
    primal = EnzymeRules.needs_primal(config) ? res[1] : nothing
    shadow = EnzymeRules.needs_shadow(config) ? dres : nothing
    tup = (dres, res[2])
    RetType = Enzyme.EnzymeRules.augmented_rule_return_type(config, RT)
    return RetType(primal, shadow, tup::Any)
end

function Enzyme.EnzymeRules.reverse(
        config::Enzyme.EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(NonlinearSolveBase.solve_up)}, ::Type{RT}, tape, prob,
        sensealg::Enzyme.Annotation{<:Union{Nothing, SciMLBase.AbstractSensitivityAlgorithm}},
        u0, p, args...; kwargs...
    ) where {RT <: Enzyme.Annotation}
    dres, clos = tape
    # unwrap the Ref-wrapped MixedDuplicated shadow produced by augmented_primal
    dval = dres isa Base.RefValue ? dres[] : dres
    dargs = clos(dval)
    # Mirror the `diff_tunables` choice the inner adjoint will make. When the
    # user passes a concrete sensealg, honor its `diff_tunables` field. When
    # the outer sensealg is `nothing` (default), `_concrete_solve_adjoint`
    # delegates to `automatic_sensealg_choice`, which picks
    # `diff_tunables = Val(false)` whenever `prob.p` is a SciMLStructure with
    # a non-empty `caches` field (e.g. an MTKParameters tied to an
    # SCCNonlinearProblem's `explicitfuns!` coupling). Reproducing that
    # predicate here lets the accumulator walk every non-Tunable field of a
    # structured `darg` so the meaningful cotangent isn't dropped.
    diff_tunables = let s = sensealg.val, pv = p.val
        if s isa SciMLBase.AbstractSensitivityAlgorithm &&
                hasproperty(s, :diff_tunables)
            !(getproperty(s, :diff_tunables) isa Val{false})
        else
            !(
                SciMLStructures.isscimlstructure(pv) &&
                    !(pv isa AbstractArray) &&
                    hasfield(typeof(pv), :caches) &&
                    !isempty(pv.caches)
            )
        end
    end
    for (darg, ptr) in zip(dargs, (func, prob, sensealg, u0, p, args...))
        if ptr isa Enzyme.Const
            continue
        end
        # `sensealg` is inactive config; skip its slot whether it arrived as
        # Const or a runtime-activity-promoted Duplicated/MixedDuplicated.
        if ptr === sensealg
            continue
        end
        if darg == ChainRulesCore.NoTangent()
            continue
        end
        if ptr isa MixedDuplicated
            _accum_tangent!(ptr.dval[], darg; diff_tunables)
        else
            _accum_tangent!(ptr.dval, darg; diff_tunables)
        end
    end
    Enzyme.make_zero!(dval.u)
    # One return slot per (prob, sensealg, u0, p, args...). Active args (e.g. the
    # init polyalgorithm promoted to `Active` under runtime activity) require a
    # cotangent value, not `nothing`; these are inactive config, so return a
    # zeroed value. Const/Duplicated/MixedDuplicated slots return `nothing`.
    return map(_rev_arg_cotangent, (prob, sensealg, u0, p, args...))
end

@inline _rev_arg_cotangent(x) = x isa Enzyme.Active ? Enzyme.make_zero(x.val) : nothing

end
