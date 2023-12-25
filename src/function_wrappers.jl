# NonlinearSolve can handle all NonlinearFunction specifications but that is not true for
# downstream packages. Make conversion to those easier.
function __construct_f(prob; alias_u0::Bool = false, can_handle_oop::Val{OOP} = Val(false),
        can_handle_scalar::Val{SCALAR} = Val(false), make_fixed_point::Val{FP} = Val(false),
        can_handle_arbitrary_dims::Val{DIMS} = Val(false),
        force_oop::Val{FOOP} = Val(false)) where {SCALAR, OOP, DIMS, FP, FOOP}
    if !OOP && SCALAR
        error("Incorrect Specification: OOP not supported but scalar supported.")
    end

    resid = evaluate_f(prob, prob.u0)

    if SCALAR || !(prob.u0 isa Number)
        u0 = __maybe_unaliased(prob.u0, alias_u0)
    else
        u0 = [prob.u0]
    end

    f = if FP
        if isinplace(prob)
            @closure (du, u, p) -> begin
                prob.f(du, u, p)
                @. du += u
            end
        else
            @closure (u, p) -> prob.f(u, p) .+ u
        end
    else
        prob.f
    end

    ff = if isinplace(prob)
        ninputs = 2
        if DIMS || u0 isa AbstractVector
            @closure (du, u) -> (f(du, u, prob.p); du)
        else
            u0_size = size(u0)
            du_size = size(resid)
            @closure (du, u) -> (f(reshape(du, du_size), reshape(u, u0_size), prob.p); du)
        end
    else
        if prob.u0 isa Number
            if SCALAR
                ninputs = 1
                @closure (u) -> f(u, prob.p)
            elseif OOP
                ninputs = 1
                @closure (u) -> [f(first(u), prob.p)]
            else
                ninputs = 2
                resid = [resid]
                @closure (du, u) -> (du[1] = f(first(u), prob.p); du)
            end
        else
            if OOP
                ninputs = 1
                if DIMS
                    @closure (u) -> f(u, prob.p)
                else
                    u0_size = size(u0)
                    @closure (u) -> _vec(f(reshape(u, u0_size), prob.p))
                end
            else
                ninputs = 2
                if DIMS
                    @closure (du, u) -> (copyto!(du, f(u, prob.p)); du)
                else
                    u0_size = size(u0)
                    @closure (du, u) -> begin
                        copyto!(vec(du), vec(f(reshape(u, u0_size), prob.p)))
                        return du
                    end
                end
            end
        end
    end

    f_final = if FOOP
        if ninputs == 1
            ff
        else
            du_ = DIMS ? similar(resid) : _vec(similar(resid))
            @closure (u) -> (ff(du_, u); du_)
        end
    else
        ff
    end

    return f_final, ifelse(DIMS, u0, _vec(u0))
end

function __construct_jac(prob, alg, u0; can_handle_oop::Val{OOP} = Val(false),
        can_handle_scalar::Val{SCALAR} = Val(false),
        can_handle_arbitrary_dims::Val{DIMS} = Val(false)) where {SCALAR, OOP, DIMS}
    if SciMLBase.has_jac(prob.f)
        jac = prob.f.jac

        jac_final = if isinplace(prob)
            if DIMS || u0 isa AbstractVector
                @closure (J, u) -> (jac(reshape(J, :, length(u)), u, prob.p); J)
            else
                u0_size = size(u0)
                @closure (J, u) -> (jac(reshape(J, :, length(u)), reshape(u, u0_size),
                    prob.p);
                J)
            end
        else
            if prob.u0 isa Number
                if SCALAR
                    @closure (u) -> jac(u, prob.p)
                elseif OOP
                    @closure (u) -> [jac(first(u), prob.p)]
                else
                    @closure (J, u) -> (J[1] = jac(first(u), prob.p); J)
                end
            else
                if OOP
                    if DIMS
                        @closure (u) -> jac(u, prob.p)
                    else
                        u0_size = size(u0)
                        @closure (u) -> jac(reshape(u, u0_size), prob.p)
                    end
                else
                    if DIMS
                        @closure (J, u) -> (copyto!(J, jac(u, prob.p)); J)
                    else
                        u0_size = size(u0)
                        @closure (J, u) -> begin
                            copyto!(J, jac(reshape(u, u0_size), prob.p))
                            return J
                        end
                    end
                end
            end
        end

        return jac_final
    end

    hasfield(typeof(alg), :ad) || return nothing

    uf, _, J, fu, jac_cache, _, _, _ = jacobian_caches(alg, prob.f, u0, prob.p,
        Val{isinplace(prob)}(); lininit = Val(false), linsolve_with_JᵀJ = Val(false))
    stats = SciMLBase.NLStats(0, 0, 0, 0, 0)
    return JacobianFunctionCache{isinplace(prob)}(J, prob.f, uf, u0, prob.p, jac_cache,
        alg, fu, stats)
end

# Currently used only in some of the extensions. Plan is to eventually use it in all the
# native algorithms and other extensions to provide better jacobian support
@concrete struct JacobianFunctionCache{iip, U, P} <: Function
    J
    f
    uf
    u::U
    p::P
    jac_cache
    alg
    fu_cache
    stats
end

SciMLBase.isinplace(::JacobianFunctionCache{iip}) where {iip} = iip

function (jac_cache::JacobianFunctionCache{iip, U, P})(J::AbstractMatrix, u::U,
        p::P = jac_cache.p) where {iip, U, P}
    jacobian!!(J, jac_cache; u, p)
    return J
end
function (jac_cache::JacobianFunctionCache{iip, U, P})(u::U, p::P) where {iip, U, P}
    return jacobian!!(cache.J, jac_cache; u, p)
end

@concrete struct InplaceFunction{iip} <: Function
    f
    p
end

(f::InplaceFunction{true})(du, u) = f.f(du, u, f.p)
(f::InplaceFunction{true})(du, u, p) = f.f(du, u, p)
(f::InplaceFunction{false})(du, u) = (du .= f.f(u, f.p))
(f::InplaceFunction{false})(du, u, p) = (du .= f.f(u, p))

struct __make_inplace{iip} end

@inline __make_inplace{iip}(f::F, p) where {iip, F} = InplaceFunction{iip}(f, p)
@inline __make_inplace{iip}(::Nothing, p) where {iip} = nothing
