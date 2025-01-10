abstract type HomotopySystemVariant end

struct Inplace <: HomotopySystemVariant end
struct OutOfPlace <: HomotopySystemVariant end
struct Scalar <: HomotopySystemVariant end

@concrete struct HomotopySystemWrapper{variant <: HomotopySystemVariant} <: HC.AbstractSystem
    prob
    autodiff
    prep
    vars
    taylorvars
    jacobian_buffers
end

Base.size(sys::HomotopySystemWrapper) = (length(sys.prob.u0), length(sys.prob.u0))
HC.ModelKit.variables(sys::HomotopySystemWrapper) = sys.vars

function HC.ModelKit.evaluate!(u, sys::HomotopySystemWrapper{Inplace}, x, p = nothing)
    sys.prob.f.f(u, x, parameter_values(sys.prob))
    return u
end

function HC.ModelKit.evaluate!(u, sys::HomotopySystemWrapper{OutOfPlace}, x, p = nothing)
    values = sys.prob.f.f(x, parameter_values(sys.prob))
    copyto!(u, values)
    return u
end

function HC.ModelKit.evaluate!(u, sys::HomotopySystemWrapper{Scalar}, x, p = nothing)
    u[1] = sys.prob.f.f(only(x), parameter_values(sys.prob))
    return u
end

function HC.ModelKit.evaluate_and_jacobian!(u, U, sys::HomotopySystemWrapper{Inplace}, x, p = nothing)
    f = sys.prob.f
    p = parameter_values(sys.prob)
    if SciMLBase.has_jac(f)
        f.f(u, x, p)
        f.jac(U, x, p)
        return
    end

    x_tmp, u_tmp, U_tmp = sys.jacobian_buffers
    copyto!(x_tmp, x)
    DI.value_and_jacobian!(f.f, u_tmp, U_tmp, sys.prep, sys.autodiff, x_tmp, DI.Constant(p))
    copyto!(u, u_tmp)
    copyto!(U, U_tmp)
    return u, U
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
    x_tmp, U_tmp = sys.jacobian_buffers
    copyto!(x_tmp, x)
    u_tmp, _ = DI.value_and_jacobian!(f.f, U_tmp, sys.prep, sys.autodiff, x_tmp, DI.Constant(p))
    copyto!(u, u_tmp)
    copyto!(U, U_tmp)
    return u, U
end

function HC.ModelKit.evaluate_and_jacobian!(u, U, sys::HomotopySystemWrapper{Scalar}, x, p = nothing)
    f = sys.prob.f
    p = parameter_values(sys.prob)
    if SciMLBase.has_jac(f)
        HC.ModelKit.evaluate!(u, sys, x, p)
        U[1] = f.jac(only(x), p)
    else
        x = real(first(x))
        u[1], U[1] = DI.value_and_derivative(f.f, sys.prep, sys.autodiff, x, DI.Constant(p))
    end
    return u, U
end

function HC.ModelKit.taylor!(u::AbstractVector, ::Val{N}, sys::HomotopySystemWrapper{Inplace}, x::HC.ModelKit.TaylorVector{M}, p = nothing) where {N, M}
    f = sys.prob.f
    p = parameter_values(sys.prob)
    buffer, vars = sys.taylorvars
    for i in eachindex(vars)
        for j in 0:M-1
            vars[i][j] = x[i, j + 1]
        end
        for j in M:4
            vars[i][j] = zero(vars[i][j])
        end
    end
    f.f(buffer, vars, p)
    if u isa Vector
        for i in eachindex(vars)
            u[i] = buffer[i][N]
        end
    else
        for i in eachindex(vars)
            u[i] = ntuple(j -> buffer[i][j - 1], Val(N + 1))
        end
    end
    return u
end

function HC.ModelKit.taylor!(u::AbstractVector, ::Val{N}, sys::HomotopySystemWrapper{OutOfPlace}, x::HC.ModelKit.TaylorVector{M}, p = nothing) where {N, M}
    f = sys.prob.f
    p = parameter_values(sys.prob)
    vars = sys.taylorvars
    for i in eachindex(vars)
        for j in 0:M
            vars[i][j] = x[i, j + 1]
        end
    end
    buffer = f.f(vars, p)
    for i in eachindex(vars)
        u[i] = ntuple(j -> buffer[i][j - 1], Val(N + 1))
    end
end

function HC.ModelKit.taylor!(u::AbstractVector, ::Val{N}, sys::HomotopySystemWrapper{Scalar}, x::HC.ModelKit.TaylorVector{M}, p = nothing) where {N, M}
    f = sys.prob.f
    p = parameter_values(sys.prob)
    var = sys.taylorvars
    for i in 0:M
        var[i] = x[1, i + 1]
    end
    taylor = f.f(var, p)
    val = ntuple(i -> taylor[i - 1], Val(N + 1))
    u[1] = val
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
