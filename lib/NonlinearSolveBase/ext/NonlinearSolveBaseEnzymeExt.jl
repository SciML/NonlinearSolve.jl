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
        shadow_tunables, _, _ = SciMLStructures.canonicalize(
            SciMLStructures.Tunable(), dval,
        )
        if SciMLStructures.isscimlstructure(darg)
            darg_tunables, _, _ = SciMLStructures.canonicalize(
                SciMLStructures.Tunable(), darg,
            )
            shadow_tunables .+= darg_tunables
        elseif darg isa AbstractVector && length(darg) == length(shadow_tunables)
            # Tunable gradient vector (returned by SciMLSensitivity for
            # EnzymeOriginator when p is a SciMLStructure)
            shadow_tunables .+= darg
        else
            # Fallback: try direct broadcast (may error for incompatible types)
            dval .+= darg
            return nothing
        end
        SciMLStructures.replace!(SciMLStructures.Tunable(), dval, shadow_tunables)
    else
        dval .+= darg
    end
    return nothing
end

function Enzyme.EnzymeRules.augmented_primal(
        config::Enzyme.EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(NonlinearSolveBase.solve_up)}, ::Union{Type{Duplicated{RT}}, Type{MixedDuplicated{RT}}}, prob,
        sensealg::Union{
            Const{Nothing}, Const{<:SciMLBase.AbstractSensitivityAlgorithm},
        },
        u0, p, args...; kwargs...
    ) where {RT}
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

    dres = Enzyme.make_zero(res[1])::RT
    tup = (dres, res[2])
    return Enzyme.EnzymeRules.AugmentedReturn{RT, RT, Any}(res[1], dres, tup::Any)
end

function Enzyme.EnzymeRules.reverse(
        config::Enzyme.EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(NonlinearSolveBase.solve_up)}, ::Union{Type{Duplicated{RT}}, Type{MixedDuplicated{RT}}}, tape, prob,
        sensealg::Union{
            Const{Nothing}, Const{<:SciMLBase.AbstractSensitivityAlgorithm},
        },
        u0, p, args...; kwargs...
    ) where {RT}
    dres, clos = tape
    dres = dres::RT
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
