module NonlinearSolveBaseTaylorDiffExt
using SciMLBase: NonlinearFunction
using NonlinearSolveBase: HalleyDescentCache
import NonlinearSolveBase: evaluate_hvvp
using TaylorDiff: derivative, derivative!
using FastClosures: @closure

function evaluate_hvvp(
        hvvp, cache::HalleyDescentCache, f::NonlinearFunction{iip}, p, u, δu) where {iip}
    if iip
        binary_f = @closure (y, x) -> f(y, x, p)
        derivative!(hvvp, binary_f, cache.fu, u, δu, Val(2))
    else
        unary_f = Base.Fix2(f, p)
        hvvp = derivative(unary_f, u, δu, Val(2))
    end
    hvvp
end

end
