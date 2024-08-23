module NonlinearSolveBaseForwardDiffExt

using ForwardDiff: ForwardDiff, Dual
using NonlinearSolveBase: Utils

Utils.value(::Type{Dual{T, V, N}}) where {T, V, N} = V
Utils.value(x::Dual) = Utils.value(ForwardDiff.value(x))

end