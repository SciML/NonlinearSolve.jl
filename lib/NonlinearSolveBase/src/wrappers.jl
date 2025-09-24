function assert_extension_supported_termination_condition(
        termination_condition, alg; abs_norm_supported = true
)
    no_termination_condition = termination_condition === nothing
    no_termination_condition && return nothing
    abs_norm_supported && termination_condition isa AbsNormTerminationMode && return nothing
    throw(AssertionError("`$(nameof(typeof(alg)))` does not support termination conditions!"))
end

function construct_extension_function_wrapper(
        prob::AbstractNonlinearProblem; alias_u0::Bool = false,
        can_handle_oop::Val = Val(false), can_handle_scalar::Val = Val(false),
        make_fixed_point::Val = Val(false), force_oop::Val = Val(false)
)
    if can_handle_oop isa Val{false} && can_handle_scalar isa Val{true}
        error("Incorrect Specification: OOP not supported but scalar supported.")
    end

    resid = Utils.evaluate_f(prob, prob.u0)
    u0 = can_handle_scalar isa Val{true} || !(prob.u0 isa Number) ?
         Utils.maybe_unaliased(prob.u0, alias_u0) : [prob.u0]

    fₚ = if make_fixed_point isa Val{true}
        if SciMLBase.isinplace(prob)
            @closure (du, u) -> begin
                prob.f(du, u, prob.p)
                du .+= u
                return du
            end
        else
            @closure u -> prob.f(u, prob.p) .+ u
        end
    else
        if SciMLBase.isinplace(prob)
            @closure (du, u) -> begin
                prob.f(du, u, prob.p)
                return du
            end
        else
            Base.Fix2(prob.f, prob.p)
        end
    end

    f_flat_structure = if SciMLBase.isinplace(prob)
        u0_size, du_size = size(u0), size(resid)
        @closure (du, u) -> begin
            fₚ(reshape(du, du_size), reshape(u, u0_size))
            return du
        end
    else
        if prob.u0 isa Number
            if can_handle_scalar isa Val{true}
                fₚ
            elseif can_handle_oop isa Val{true}
                @closure u -> [fₚ(first(u))]
            else
                @closure (du, u) -> begin
                    du[1] = fₚ(first(u))
                    return du
                end
            end
        else
            u0_size = size(u0)
            if can_handle_oop isa Val{true}
                @closure u -> vec(fₚ(reshape(u, u0_size)))
            else
                @closure (du, u) -> begin
                    copyto!(du, fₚ(reshape(u, u0_size)))
                    return du
                end
            end
        end
    end

    f_final = if force_oop isa Val{true} && applicable(f_flat_structure, u0, u0)
        resid = resid isa Number ? [resid] : Utils.safe_vec(resid)
        du = Utils.safe_vec(zero(resid))
        @closure u -> begin
            f_flat_structure(du, u)
            return du
        end
    else
        f_flat_structure
    end

    return f_final, Utils.safe_vec(u0), (resid isa Number ? [resid] : Utils.safe_vec(resid))
end

function construct_extension_jac(
        prob, alg, u0, fu;
        can_handle_oop::Val = Val(false), can_handle_scalar::Val = Val(false),
        autodiff = nothing, initial_jacobian = Val(false), kwargs...
)
    autodiff = select_jacobian_autodiff(prob, autodiff)

    Jₚ = construct_jacobian_cache(
        prob, alg, prob.f, fu, u0, prob.p;
        stats = NLStats(0, 0, 0, 0, 0), autodiff, kwargs...
    )

    J_no_scalar = can_handle_scalar isa Val{false} && prob.u0 isa Number ?
                  @closure(u->[Jₚ(u[1])]) : Jₚ

    J_final(J, u) = copyto!(J, J_no_scalar(u))
    J_final(u) = J_no_scalar(u)

    initial_jacobian isa Val{false} && return J_final

    return J_final, J = reused_jacobian(Jₚ, u0)
end
