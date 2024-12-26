abstract type HomotopySystemVariant end

struct Inplace <: HomotopySystemVariant end
struct OutOfPlace <: HomotopySystemVariant end
struct Scalar <: HomotopySystemVariant end

@concrete struct HomotopySystemWrapper{variant <: HomotopySystemVariant} <: HC.AbstractSystem
    prob
    autodiff
    prep
    vars
end

Base.size(sys::HomotopySystemWrapper) = (length(sys.prob.u0), length(sys.prob.u0))
HC.ModelKit.variables(sys::HomotopySystemWrapper) = sys.vars

function HC.ModelKit.evaluate!(u, sys::HomotopySystemWrapper{Inplace}, x, p = nothing)
    sys.prob.f.f(u, x, parameter_values(sys.prob))
end

function HC.ModelKit.evaluate!(u, sys::HomotopySystemWrapper{OutOfPlace}, x, p = nothing)
    values = sys.prob.f.f(x, parameter_values(sys.prob))
    copyto!(u, values)
end

function HC.ModelKit.evaluate!(u, sys::HomotopySystemWrapper{Scalar}, x, p = nothing)
    u[1] = sys.prob.f.f(only(x), parameter_values(sys.prob))
end

function HC.ModelKit.evaluate_and_jacobian!(u, U, sys::HomotopySystemWrapper{Inplace}, x, p = nothing)
    f = sys.prob.f
    p = parameter_values(sys.prob)
    if SciMLBase.has_jac(f)
        f.f(u, x, p)
        f.jac(U, x, p)
        return
    end

    DI.value_and_jacobian!(f.f, u, U, sys.prep, sys.autodiff, x, DI.Constant(p))
end

function HC.ModelKit.evaluate_and_jacobian!(u, U, sys::HomotopySystemWrapper{OutOfPlace}, x, p = nothing)
    f = sys.prob.f
    p = parameter_values(sys.prob)
    if SciMLBase.has_jac(f)
        u_tmp = f.f(x, p)
        copyto!(u, u_tmp)
        j_tmp = f.jac(U, x, p)
        copyto!(U, j_tmp)
        return
    end
    u_tmp, _ = DI.value_and_jacobian(f.f, U, sys.prep, sys.autodiff, x, DI.Constant(p))
    copyto!(u, u_tmp)
end

function HC.ModelKit.evaluate_and_jacobian!(u, U, sys::HomotopySystemWrapper{Scalar}, x, p = nothing)
    f = sys.prob.f
    p = parameter_values(sys.prob)
    if SciMLBase.has_jac(f)
        HC.ModelKit.evaluate!(u, sys, x, p)
        U[1] = f.jac(only(x), p)
    else
        u[1], U[1] = DI.value_and_derivative(f.f, sys.prep, sys.autodiff, only(x), DI.Constant(p))
    end
    return nothing
end

@concrete struct GuessHomotopy <: HC.AbstractHomotopy
    sys <: HC.AbstractSystem
    fu0
end

Base.size(h::GuessHomotopy) = size(h.sys)

function HC.ModelKit.evaluate!(u, h::GuessHomotopy, x, t, p = nothing)
    HC.ModelKit.evaluate!(u, h.sys, x, p)
    @inbounds for i in eachindex(u)
        u[i] += (t + 1) * h.fu0[i]
    end
end

function HC.ModelKit.evaluate_and_jacobian!(u, U, h::GuessHomotopy, x, t, p = nothing)
    HC.ModelKit.evaluate_and_jacobian!(u, U, h.sys, x, p)
    @inbounds for i in eachindex(u)
        u[i] += (t + 1) * h.fu0[i]
    end
end
