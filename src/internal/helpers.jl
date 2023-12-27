function evaluate_f(prob::AbstractNonlinearProblem{uType, iip}, u) where {uType, iip}
    (; f, u0, p) = prob
    if iip
        fu = f.resid_prototype === nothing ? similar(u) :
             promote_type(eltype(u), eltype(f.resid_prototype)).(f.resid_prototype)
        f(fu, u, p)
    else
        fu = f(u, p)
    end
    return fu
end
