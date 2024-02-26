module SimpleNonlinearSolveZygoteExt

import SimpleNonlinearSolve, Zygote

SimpleNonlinearSolve.__is_extension_loaded(::Val{:Zygote}) = true

function SimpleNonlinearSolve.__zygote_compute_nlls_vjp(f::F, u, p) where {F}
    y, pb = Zygote.pullback(Base.Fix2(f, p), u)
    return 2 .* only(pb(y))
end

end
