@concrete struct JacobianWrapper
    f
    p
end

(uf::JacobianWrapper)(u) = uf.f(u, uf.p)
(uf::JacobianWrapper)(res, u) = uf.f(res, u, uf.p)

# FIXME: This is a deviation from older versions. Previously if sparsity and colorvec were
#        provided we would use a sparse AD. Right now it requires an explicit specification
sparsity_detection_alg(f, ad) = NoSparsityDetection()
function sparsity_detection_alg(f, ad::AbstractSparseADType)
    if f.sparsity === nothing
        if f.jac_prototype === nothing
            return SymbolicsSparsityDetection()
        else
            jac_prototype = f.jac_prototype
        end
    else
        jac_prototype = f.sparsity
    end

    if SciMLBase.has_colorvec(f)
        return PrecomputedJacobianColorvec(; jac_prototype, f.colorvec,
            partition_by_rows = ad isa ADTypes.AbstractSparseReverseMode)
    else
        return JacPrototypeSparsityDetection(; jac_prototype)
    end
end

# NoOp for Jacobian if it is not a Abstract Array -- For eg, JacVec Operator
jacobian!!(J, _) = J
# `!!` notation is from BangBang.jl since J might be jacobian in case of oop `f.jac`
# and we don't want wasteful `copyto!`
function jacobian!!(J::Union{AbstractMatrix{<:Number}, Nothing}, cache)
    @unpack f, uf, u, p, jac_cache, alg, fu2 = cache
    iip = isinplace(cache)
    if iip
        has_jac(f) ? f.jac(J, u, p) : sparse_jacobian!(J, alg.ad, jac_cache, uf, fu2, u)
    else
        return has_jac(f) ? f.jac(u, p) : sparse_jacobian!(J, alg.ad, jac_cache, uf, u)
    end
    return nothing
end

# Build Jacobian Caches
function jacobian_caches(alg::AbstractNonlinearSolveAlgorithm, f, u, p,
    ::Val{iip}) where {iip}
    uf = JacobianWrapper(f, p)

    haslinsolve = hasfield(typeof(alg), :linsolve)

    has_analytic_jac = has_jac(f)
    linsolve_needs_jac = (concrete_jac(alg) === nothing &&
                          (!haslinsolve || (haslinsolve && (alg.linsolve === nothing ||
                             needs_concrete_A(alg.linsolve)))))
    alg_wants_jac = (concrete_jac(alg) === nothing && concrete_jac(alg))

    # NOTE: The deepcopy is needed here since we are using the resid_prototype elsewhere
    fu = f.resid_prototype === nothing ? (iip ? zero(u) : f(u, p)) :
         deepcopy(f.resid_prototype)
    if !has_analytic_jac && (linsolve_needs_jac || alg_wants_jac)
        sd = sparsity_detection_alg(f, alg.ad)
        jac_cache = iip ? sparse_jacobian_cache(alg.ad, sd, uf, fu, u) :
                    sparse_jacobian_cache(alg.ad, sd, uf, u; fx = fu)
    else
        jac_cache = nothing
    end

    J = if !linsolve_needs_jac
        # We don't need to construct the Jacobian
        JacVec(uf, u; autodiff = alg.ad)
    else
        if has_analytic_jac
            iip ? undefmatrix(u) : nothing
        else
            f.jac_prototype === nothing ? init_jacobian(jac_cache) : f.jac_prototype
        end
    end

    du = zero(u)
    linprob = LinearProblem(J, _vec(fu); u0 = _vec(du))

    weight = similar(u)
    recursivefill!(weight, true)

    Pl, Pr = wrapprecs(alg.precs(J, nothing, u, p, nothing, nothing, nothing, nothing,
            nothing)..., weight)
    linsolve = init(linprob, alg.linsolve; alias_A = true, alias_b = true, Pl, Pr)

    return uf, linsolve, J, fu, jac_cache, du
end
