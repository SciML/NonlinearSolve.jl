"""
    assert_extension_supported_termination_condition(
        termination_condition, alg; abs_norm_supported = true
    )

Validate that a wrapped external solver can support `termination_condition`.

Extension packages call this before dispatching to an external nonlinear solver whose
termination API is more limited than NonlinearSolveBase's native modes.

### Arguments

  - `termination_condition`: A nonlinear termination mode or `nothing`.
  - `alg`: The wrapper algorithm, used in the error message.

### Keyword Arguments

  - `abs_norm_supported`: Whether [`AbsNormTerminationMode`](@ref) is supported.

### Returns

`nothing` when the termination condition is supported; otherwise throws an
`AssertionError`.
"""
function assert_extension_supported_termination_condition(
        termination_condition, alg; abs_norm_supported = true
    )
    no_termination_condition = termination_condition === nothing
    no_termination_condition && return nothing
    abs_norm_supported && termination_condition isa AbsNormTerminationMode && return nothing
    throw(AssertionError(lazy"`$(nameof(typeof(alg)))` does not support termination conditions!"))
end

"""
    construct_extension_function_wrapper(
        prob; alias_u0 = false, can_handle_oop = Val(false),
        can_handle_scalar = Val(false), make_fixed_point = Val(false),
        force_oop = Val(false)
    )

Adapt a SciML nonlinear problem into the flattened residual function expected by an
external solver wrapper.

This developer API handles in-place vs out-of-place problems, scalar promotion, optional
fixed-point residual conversion, and vector reshaping.

### Arguments

  - `prob`: A SciML nonlinear problem.

### Keyword Arguments

  - `alias_u0`: Allow the returned initial vector to alias `prob.u0`.
  - `can_handle_oop`: Whether the external solver can call out-of-place residuals.
  - `can_handle_scalar`: Whether the external solver accepts scalar states directly.
  - `make_fixed_point`: Convert a fixed-point residual from `f(u)` to `f(u) + u`.
  - `force_oop`: Force an out-of-place wrapper even for in-place problems when possible.

### Returns

A tuple `(f, u0, resid)` containing the wrapped residual, flattened initial state, and
flattened residual prototype.

### Examples

```julia
using NonlinearSolveBase, SciMLBase

prob = NonlinearProblem((u, p) -> u^2 - 2, 1.0)
f, u0, resid = NonlinearSolveBase.construct_extension_function_wrapper(
    prob; can_handle_oop = Val(true), can_handle_scalar = Val(true)
)
```
"""
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

"""
    construct_extension_jac(
        prob, alg, u0, fu;
        can_handle_oop = Val(false), can_handle_scalar = Val(false),
        autodiff = nothing, initial_jacobian = Val(false), kwargs...
    )

Construct a Jacobian callback for an external solver wrapper.

The returned callback uses NonlinearSolveBase's AD selection, analytic Jacobian handling,
and scalar adaptation rules so wrapper packages do not need to duplicate Jacobian setup.

### Arguments

  - `prob`: A SciML nonlinear problem.
  - `alg`: The external solver wrapper algorithm.
  - `u0`: Flattened or solver-compatible initial state.
  - `fu`: Residual prototype at `u0`.

### Keyword Arguments

  - `can_handle_oop`: Whether the external solver can call out-of-place Jacobians.
  - `can_handle_scalar`: Whether scalar states should remain scalar.
  - `autodiff`: ADTypes backend or `nothing` for automatic selection.
  - `initial_jacobian`: Return the initial Jacobian together with the callback.
  - `kwargs...`: Additional keywords passed to [`construct_jacobian_cache`](@ref).

### Returns

`J` when `initial_jacobian = Val(false)`, or `(J, J0)` when
`initial_jacobian = Val(true)`.
"""
function construct_extension_jac(
        prob, alg, u0, fu;
        can_handle_oop::Val = Val(false), can_handle_scalar::Val = Val(false),
        autodiff = nothing, initial_jacobian = Val(false), kwargs...
    )
    autodiff = select_jacobian_autodiff(prob, autodiff)

    # Wrapper solvers flatten `u`; keep the caller's values, restore `prob.u0`'s shape.
    u_size = size(prob.u0)
    u_native = if prob.u0 isa Number
        u0 isa Number ? u0 : u0[1]
    else
        size(u0) == u_size ? u0 : reshape(u0, u_size)
    end
    # Residual shape is independent of state shape; honour `resid_prototype` when set.
    fu_native = if prob.u0 isa Number
        fu isa Number ? fu : fu[1]
    else
        rp = prob.f.resid_prototype
        if rp !== nothing
            size(fu) == size(rp) ? fu :
                (
                    length(fu) == length(rp) ? reshape(fu, size(rp)) :
                    Utils.evaluate_f(prob, u_native)
                )
        elseif size(fu) == u_size
            fu
        else
            Utils.evaluate_f(prob, u_native)
        end
    end
    Jₚ = construct_jacobian_cache(
        prob, alg, prob.f, fu_native, u_native, prob.p;
        stats = NLStats(0, 0, 0, 0, 0), autodiff, kwargs...
    )

    J_adapted = if prob.u0 isa Number
        if can_handle_scalar isa Val{true}
            Jₚ
        else
            @closure u -> [Jₚ(u isa Number ? u : u[1])]
        end
    else
        @closure u -> Jₚ(size(u) == u_size ? u : reshape(u, u_size))
    end

    J_final(J, u) = copyto!(J, J_adapted(u))
    J_final(u) = J_adapted(u)

    initial_jacobian isa Val{false} && return J_final

    return J_final, reused_jacobian(Jₚ, u_native)
end
