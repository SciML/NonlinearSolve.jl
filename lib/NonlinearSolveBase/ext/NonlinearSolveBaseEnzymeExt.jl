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
function _accum_tangent!(dval, darg)
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
            # When the upstream rule returns a structured cotangent
            # (e.g. SciMLSensitivity's steadystatebackpass under
            # diff_tunables = Val(false)), the gradient contribution may
            # live in non-Tunable fields such as `caches` (SCC sub-problem
            # buffers feeding `explicitfuns!`). Accumulate every non-Tunable
            # field that both sides expose so those contributions are not
            # silently dropped.
            for field in fieldnames(typeof(darg))
                field === :tunable && continue
                hasfield(typeof(dval), field) || continue
                _accum_nested!(getfield(dval, field), getfield(darg, field))
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

function Enzyme.EnzymeRules.augmented_primal(
        config::Enzyme.EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(NonlinearSolveBase.solve_up)}, ::Type{RT}, prob,
        sensealg::Union{
            Const{Nothing}, Const{<:SciMLBase.AbstractSensitivityAlgorithm},
        },
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

    dres = Enzyme.make_zero(res[1])
    primal = EnzymeRules.needs_primal(config) ? res[1] : nothing
    shadow = EnzymeRules.needs_shadow(config) ? dres : nothing
    tup = (dres, res[2])
    RetType = Enzyme.EnzymeRules.augmented_rule_return_type(config, RT)
    return RetType(primal, shadow, tup::Any)
end

function Enzyme.EnzymeRules.reverse(
        config::Enzyme.EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(NonlinearSolveBase.solve_up)}, ::Type{RT}, tape, prob,
        sensealg::Union{
            Const{Nothing}, Const{<:SciMLBase.AbstractSensitivityAlgorithm},
        },
        u0, p, args...; kwargs...
    ) where {RT <: Enzyme.Annotation}
    dres, clos = tape
    dargs = clos(dres)
    for (darg, ptr) in zip(dargs, (func, prob, sensealg, u0, p, args...))
        if ptr isa Enzyme.Const
            continue
        end
        if darg == ChainRulesCore.NoTangent()
            continue
        end
        if ptr isa MixedDuplicated
            _accum_tangent!(ptr.dval[], darg)
        else
            _accum_tangent!(ptr.dval, darg)
        end
    end
    Enzyme.make_zero!(dres.u)
    return ntuple(_ -> nothing, Val(length(args) + 4))
end

end
