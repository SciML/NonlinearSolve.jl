module NonlinearSolveBaseMooncakeExt

using NonlinearSolveBase, Mooncake
using ChainRulesCore: ChainRulesCore
using SciMLBase: SciMLBase
using Mooncake: rrule!!, CoDual, zero_fcodual, @is_primitive,
    @from_chainrules, @zero_adjoint, @mooncake_overlay, MinimalCtx,
    NoPullback

_accum_cr_tangent!(f, r, ::Nothing) = r
function _accum_cr_tangent!(f::NamedTuple, r::Mooncake.NoRData, t::NamedTuple)
    map(_accum_full_tangent!, f, t)
    return r
end
function _accum_cr_tangent!(f::Mooncake.NoFData, r::NamedTuple, t::NamedTuple)
    return map((ri, ti) -> _accum_cr_tangent!(f, ri, ti), r, t)
end
function _accum_cr_tangent!(f::NamedTuple, r::NamedTuple, t::NamedTuple)
    return map(_accum_cr_tangent!, f, r, t)
end
function _accum_cr_tangent!(
        f::Mooncake.FData, r::Mooncake.RData{R}, t::NamedTuple
    ) where {R}
    return Mooncake.RData{R}(map(_accum_cr_tangent!, f.data, r.data, t))
end
function _accum_cr_tangent!(
        f::Mooncake.MutableTangent, r::Mooncake.NoRData, t::NamedTuple
    )
    f.fields = map(_accum_full_tangent!, f.fields, t)
    return r
end
_accum_cr_tangent!(f, r, t) = Mooncake.increment_and_get_rdata!(f, r, t)

_accum_full_tangent!(f, ::Nothing) = f
function _accum_full_tangent!(f::Mooncake.MutableTangent, t::NamedTuple)
    f.fields = map(_accum_full_tangent!, f.fields, t)
    return f
end
function _accum_full_tangent!(f::Mooncake.Tangent, t::NamedTuple)
    return Mooncake.Tangent(map(_accum_full_tangent!, f.fields, t))
end
_accum_full_tangent!(f, t) = Mooncake.increment!!(f, t)

# `@from_chainrules` delegates unsupported cotangent conversions to this documented
# extension point; keep the otherwise-pirated signature specific to this wrapper.
function Mooncake.increment_and_get_rdata!(
        f::Mooncake.FData{F}, r::Mooncake.RData{R},
        t::ChainRulesCore.Tangent{SciMLBase.DespecializedParameters}
    ) where {
        F <: NamedTuple{(:params,)},
        R <: NamedTuple{(:params,)},
    }
    return _accum_cr_tangent!(f, r, getfield(t, :backing))
end

@from_chainrules MinimalCtx Tuple{
    typeof(NonlinearSolveBase.solve_up),
    SciMLBase.AbstractNonlinearProblem,
    Union{Nothing, SciMLBase.AbstractSensitivityAlgorithm},
    Any,
    Any,
    Any,
} true

# Dispatch for auto-alg
@from_chainrules MinimalCtx Tuple{
    typeof(NonlinearSolveBase.solve_up),
    SciMLBase.AbstractNonlinearProblem,
    Union{Nothing, SciMLBase.AbstractSensitivityAlgorithm},
    Any,
    Any,
} true

end
