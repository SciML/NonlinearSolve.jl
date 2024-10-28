# Extension Algorithm Helpers
function __test_termination_condition(termination_condition, alg)
    !(termination_condition isa AbsNormTerminationMode) &&
        termination_condition !== nothing &&
        error("`$(alg)` does not support termination conditions!")
end

function __construct_extension_f(prob::AbstractNonlinearProblem; alias_u0::Bool = false,
        can_handle_oop::Val = False, can_handle_scalar::Val = False,
        make_fixed_point::Val = False, force_oop::Val = False)
    if can_handle_oop === False && can_handle_scalar === True
        error("Incorrect Specification: OOP not supported but scalar supported.")
    end

    resid = evaluate_f(prob, prob.u0)
    u0 = can_handle_scalar === True || !(prob.u0 isa Number) ?
         __maybe_unaliased(prob.u0, alias_u0) : [prob.u0]

    fₚ = if make_fixed_point === True
        if isinplace(prob)
            @closure (du, u) -> (prob.f(du, u, prob.p); du .+= u)
        else
            @closure u -> prob.f(u, prob.p) .+ u
        end
    else
        if isinplace(prob)
            @closure (du, u) -> prob.f(du, u, prob.p)
        else
            @closure u -> prob.f(u, prob.p)
        end
    end

    𝐟 = if isinplace(prob)
        u0_size, du_size = size(u0), size(resid)
        @closure (du, u) -> (fₚ(reshape(du, du_size), reshape(u, u0_size)); du)
    else
        if prob.u0 isa Number
            if can_handle_scalar === True
                fₚ
            elseif can_handle_oop === True
                @closure u -> [fₚ(first(u))]
            else
                @closure (du, u) -> (du[1] = fₚ(first(u)); du)
            end
        else
            u0_size = size(u0)
            if can_handle_oop === True
                @closure u -> vec(fₚ(reshape(u, u0_size)))
            else
                @closure (du, u) -> (copyto!(du, fₚ(reshape(u, u0_size))); du)
            end
        end
    end

    𝐅 = if force_oop === True && applicable(𝐟, u0, u0)
        _resid = resid isa Number ? [resid] : _vec(resid)
        du = _vec(zero(_resid))
        @closure u -> begin
            𝐟(du, u)
            return du
        end
    else
        𝐟
    end

    return 𝐅, _vec(u0), (resid isa Number ? [resid] : _vec(resid))
end

function __construct_extension_jac(prob, alg, u0, fu; can_handle_oop::Val = False,
        can_handle_scalar::Val = False, autodiff = nothing, initial_jacobian = False,
        kwargs...)
    autodiff = select_jacobian_autodiff(prob, autodiff)

    Jₚ = construct_jacobian_cache(
        prob, alg, prob.f, fu, u0, prob.p; stats = empty_nlstats(), autodiff, kwargs...)

    𝓙 = (can_handle_scalar === False && prob.u0 isa Number) ? @closure(u->[Jₚ(u[1])]) : Jₚ

    𝐉 = (can_handle_oop === False && !isinplace(prob)) ?
        @closure((J, u)->copyto!(J, 𝓙(u))) : 𝓙

    initial_jacobian === False && return 𝐉

    return 𝐉, Jₚ(nothing)
end
