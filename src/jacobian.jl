struct JacobianWrapper{fType, pType}
    f::fType
    p::pType
end

(uf::JacobianWrapper)(u) = uf.f(u, uf.p)
(uf::JacobianWrapper)(res, u) = uf.f(res, u, uf.p)

struct NonlinearSolveTag end

function sparsity_colorvec(f, x)
    sparsity = f.sparsity
    colorvec = DiffEqBase.has_colorvec(f) ? f.colorvec :
               (isnothing(sparsity) ? (1:length(x)) : matrix_colors(sparsity))
    sparsity, colorvec
end

function jacobian_finitediff_forward!(J, f, x, jac_config, forwardcache, cache)
    (FiniteDiff.finite_difference_jacobian!(J, f, x, jac_config, forwardcache);
     maximum(jac_config.colorvec))
end
function jacobian_finitediff!(J, f, x, jac_config, cache)
    (FiniteDiff.finite_difference_jacobian!(J, f, x, jac_config);
     2 * maximum(jac_config.colorvec))
end

function jacobian!(J::AbstractMatrix{<:Number}, cache)
    f = cache.f
    uf = cache.uf
    x = cache.u
    fx = cache.fu
    jac_config = cache.jac_config
    alg = cache.alg

    if SciMLBase.has_jac(f)
        f.jac(J, x, cache.p)
    elseif alg_autodiff(alg)
        forwarddiff_color_jacobian!(J, uf, x, jac_config)
        #cache.destats.nf += 1
    else
        isforward = alg_difftype(alg) === Val{:forward}
        if isforward
            uf(fx, x)
            #cache.destats.nf += 1
            tmp = jacobian_finitediff_forward!(J, uf, x, jac_config, fx,
                                               cache)
        else # not forward difference
            tmp = jacobian_finitediff!(J, uf, x, jac_config, cache)
        end
        #cache.destats.nf += tmp
    end
    nothing
end

function build_jac_config(alg, f::F1, uf::F2, du1, u, tmp, du2) where {F1, F2}
    haslinsolve = hasfield(typeof(alg), :linsolve)

    if !SciMLBase.has_jac(f) && # No Jacobian if has analytical solution
       ((concrete_jac(alg) === nothing && (!haslinsolve || (haslinsolve && # No Jacobian if linsolve doesn't want it
           (alg.linsolve === nothing || LinearSolve.needs_concrete_A(alg.linsolve))))) ||
        (concrete_jac(alg) !== nothing && concrete_jac(alg))) # Jacobian if explicitly asked for
        jac_prototype = f.jac_prototype
        sparsity, colorvec = sparsity_colorvec(f, u)

        if alg_autodiff(alg)
            _chunksize = get_chunksize(alg) === Val(0) ? nothing : get_chunksize(alg) # SparseDiffEq uses different convection...

            T = if standardtag(alg)
                typeof(ForwardDiff.Tag(NonlinearSolveTag(), eltype(u)))
            else
                typeof(ForwardDiff.Tag(uf, eltype(u)))
            end
            jac_config = ForwardColorJacCache(uf, u, _chunksize; colorvec = colorvec,
                                              sparsity = sparsity, tag = T)
        else
            if alg_difftype(alg) !== Val{:complex}
                jac_config = FiniteDiff.JacobianCache(tmp, du1, du2, alg_difftype(alg),
                                                      colorvec = colorvec,
                                                      sparsity = sparsity)
            else
                jac_config = FiniteDiff.JacobianCache(Complex{eltype(tmp)}.(tmp),
                                                      Complex{eltype(du1)}.(du1), nothing,
                                                      alg_difftype(alg), eltype(u),
                                                      colorvec = colorvec,
                                                      sparsity = sparsity)
            end
        end
    else
        jac_config = nothing
    end
    jac_config
end

function get_chunksize(jac_config::ForwardDiff.JacobianConfig{T, V, N, D}) where {T, V, N, D
                                                                                  }
    Val(N)
end # don't degrade compile time information to runtime information

function jacobian_finitediff(f, x, ::Type{diff_type}, dir, colorvec, sparsity,
                             jac_prototype) where {diff_type}
    (FiniteDiff.finite_difference_derivative(f, x, diff_type, eltype(x), dir = dir), 2)
end
function jacobian_finitediff(f, x::AbstractArray, ::Type{diff_type}, dir, colorvec,
                             sparsity, jac_prototype) where {diff_type}
    f_in = diff_type === Val{:forward} ? f(x) : similar(x)
    ret_eltype = eltype(f_in)
    J = FiniteDiff.finite_difference_jacobian(f, x, diff_type, ret_eltype, f_in,
                                              dir = dir, colorvec = colorvec,
                                              sparsity = sparsity,
                                              jac_prototype = jac_prototype)
    return J, _nfcount(maximum(colorvec), diff_type)
end
function jacobian(cache, f::F) where {F}
    x = cache.u
    alg = cache.alg
    uf = cache.uf
    local tmp

    if DiffEqBase.has_jac(cache.f)
        J = f.jac(cache.u, cache.p)
    elseif alg_autodiff(alg)
        J, tmp = jacobian_autodiff(uf, x, cache.f, alg)
    else
        jac_prototype = cache.f.jac_prototype
        sparsity, colorvec = sparsity_colorvec(cache.f, x)
        dir = true
        J, tmp = jacobian_finitediff(uf, x, alg_difftype(alg), dir, colorvec, sparsity,
                                     jac_prototype)
    end
    J
end

jacobian_autodiff(f, x, nonlinfun, alg) = (ForwardDiff.derivative(f, x), 1, alg)
function jacobian_autodiff(f, x::AbstractArray, nonlinfun, alg)
    jac_prototype = nonlinfun.jac_prototype
    sparsity, colorvec = sparsity_colorvec(nonlinfun, x)
    maxcolor = maximum(colorvec)
    chunk_size = get_chunksize(alg) === Val(0) ? nothing : get_chunksize(alg)
    num_of_chunks = chunk_size === nothing ?
                    Int(ceil(maxcolor /
                             SparseDiffTools.getsize(ForwardDiff.pickchunksize(maxcolor)))) :
                    Int(ceil(maxcolor / _unwrap_val(chunk_size)))
    (forwarddiff_color_jacobian(f, x, colorvec = colorvec, sparsity = sparsity,
                                jac_prototype = jac_prototype, chunksize = chunk_size),
     num_of_chunks)
end
