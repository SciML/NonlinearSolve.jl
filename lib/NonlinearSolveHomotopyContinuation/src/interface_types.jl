abstract type HomotopySystemVariant end

struct Inplace <: HomotopySystemVariant end
struct OutOfPlace <: HomotopySystemVariant end
struct Scalar <: HomotopySystemVariant end

"""
    $(TYPEDEF)

A simple struct that wraps a polynomial function which takes complex input and returns
complex output in a form that supports automatic differentiation. If the wrapped
function if ``f: \\mathbb{C}^n \\rightarrow \\mathbb{C}^n`` then it is assumed
the input arrays are real-valued and have length ``2n``. They are `reinterpret`ed
into complex arrays and passed into the function. This struct has an in-place signature
regardless of the signature of ``f``.
"""
@concrete struct ComplexJacobianWrapper{variant <: HomotopySystemVariant}
    f
end

function (cjw::ComplexJacobianWrapper{Inplace})(
        u::AbstractVector{T}, x::AbstractVector{T}, p) where {T}
    x = reinterpret(Complex{T}, x)
    u = reinterpret(Complex{T}, u)
    cjw.f(u, x, p)
    u = parent(u)
    return u
end

function (cjw::ComplexJacobianWrapper{OutOfPlace})(
        u::AbstractVector{T}, x::AbstractVector{T}, p) where {T}
    x = reinterpret(Complex{T}, x)
    u_tmp = cjw.f(x, p)
    u_tmp = reinterpret(T, u_tmp)
    copyto!(u, u_tmp)
    return u
end

function (cjw::ComplexJacobianWrapper{Scalar})(
        u::AbstractVector{T}, x::AbstractVector{T}, p) where {T}
    x = reinterpret(Complex{T}, x)
    u_tmp = cjw.f(x[1], p)
    u[1] = real(u_tmp)
    u[2] = imag(u_tmp)
    return u
end

"""
    $(TYPEDEF)

A struct which implements the `HomotopyContinuation.AbstractSystem` interface for
polynomial systems specified using `NonlinearProblem`.

# Fields

$(FIELDS)
"""
@concrete struct HomotopySystemWrapper{variant <: HomotopySystemVariant} <:
                 HC.AbstractSystem
    """
    The wrapped polynomial function.
    """
    f
    """
    The jacobian function, if provided to the `NonlinearProblem` being solved. Otherwise,
    a `ComplexJacobianWrapper` wrapping `f` used for automatic differentiation.
    """
    jac
    """
    The parameter object.
    """
    p
    """
    The ADType for automatic differentiation.
    """
    autodiff
    """
    The result from `DifferentiationInterface.prepare_jacobian`.
    """
    prep
    """
    HomotopyContinuation.jl's symbolic variables for the system.
    """
    vars
    """
    The `TaylorDiff.TaylorScalar` objects used to compute the taylor series of `f`.
    """
    taylorvars
    """
    Preallocated intermediate buffers used for calculating the jacobian.
    """
    jacobian_buffers
end

Base.size(sys::HomotopySystemWrapper) = (length(sys.vars), length(sys.vars))
HC.ModelKit.variables(sys::HomotopySystemWrapper) = sys.vars

function HC.ModelKit.evaluate!(u, sys::HomotopySystemWrapper{Inplace}, x, p = nothing)
    sys.f(u, x, sys.p)
    return u
end

function HC.ModelKit.evaluate!(u, sys::HomotopySystemWrapper{OutOfPlace}, x, p = nothing)
    values = sys.f(x, sys.p)
    copyto!(u, values)
    return u
end

function HC.ModelKit.evaluate!(u, sys::HomotopySystemWrapper{Scalar}, x, p = nothing)
    u[1] = sys.f(x[1], sys.p)
    return u
end

function HC.ModelKit.evaluate_and_jacobian!(
        u, U, sys::HomotopySystemWrapper{Inplace}, x, p = nothing)
    p = sys.p
    sys.f(u, x, p)
    sys.jac(U, x, p)
    return u, U
end

function HC.ModelKit.evaluate_and_jacobian!(
        u, U, sys::HomotopySystemWrapper{OutOfPlace}, x, p = nothing)
    p = sys.p
    u_tmp = sys.f(x, p)
    copyto!(u, u_tmp)
    j_tmp = sys.jac(x, p)
    copyto!(U, j_tmp)
    return u, U
end

function HC.ModelKit.evaluate_and_jacobian!(
        u, U, sys::HomotopySystemWrapper{Scalar}, x, p = nothing)
    p = sys.p
    u[1] = sys.f(x[1], p)
    U[1] = sys.jac(x[1], p)
    return u, U
end

for V in (Inplace, OutOfPlace, Scalar)
    @eval function HC.ModelKit.evaluate_and_jacobian!(
            u, U, sys::HomotopySystemWrapper{$V, F, J}, x,
            p = nothing) where {F, J <: ComplexJacobianWrapper}
        p = sys.p
        U_tmp = sys.jacobian_buffers
        x = reinterpret(Float64, x)
        u = reinterpret(Float64, u)
        DI.value_and_jacobian!(sys.jac, u, U_tmp, sys.prep, sys.autodiff, x, DI.Constant(p))
        U = reinterpret(Float64, U)
        @inbounds for j in axes(U, 2)
            jj = 2j - 1
            for i in axes(U, 1)
                U[i, j] = U_tmp[i, jj]
            end
        end
        u = parent(u)
        U = parent(U)

        return u, U
    end
end

function update_taylorvars_from_taylorvector!(
        vars, x::HC.ModelKit.TaylorVector)
    for i in eachindex(x)
        xvar = x[i]
        realx = ntuple(Val(4)) do j
            j <= length(xvar) ? real(xvar[j - 1]) : 0.0
        end
        imagx = ntuple(Val(4)) do j
            j <= length(xvar) ? imag(xvar[j - 1]) : 0.0
        end

        vars[2i - 1] = TaylorScalar(realx)
        vars[2i] = TaylorScalar(imagx)
    end
end

function update_taylorvars_from_taylorvector!(vars, x::AbstractVector)
    for i in eachindex(x)
        vars[2i - 1] = TaylorScalar(real(x[i]), ntuple(Returns(0.0), Val(3)))
        vars[2i] = TaylorScalar(imag(x[i]), ntuple(Returns(0.0), Val(3)))
    end
end

function check_taylor_equality(vars, x::HC.ModelKit.TaylorVector)
    for i in eachindex(x)
        TaylorDiff.flatten(vars[2i-1]) == map(real, x[i]) || return false
        TaylorDiff.flatten(vars[2i]) == map(imag, x[i]) || return false
    end
    return true
end
function check_taylor_equality(vars, x::AbstractVector)
    for i in eachindex(x)
        TaylorDiff.value(vars[2i-1]) != real(x[i]) && return false
        TaylorDiff.value(vars[2i]) != imag(x[i]) && return false
    end
    return true
end

function update_maybe_taylorvector_from_taylorvars!(
        u::Vector, vars, buffer, ::Val{N}) where {N}
    for i in eachindex(vars)
        rval = TaylorDiff.flatten(real(buffer[i]))
        ival = TaylorDiff.flatten(imag(buffer[i]))
        u[i] = rval[N] + im * ival[N]
    end
end

function update_maybe_taylorvector_from_taylorvars!(
        u::HC.ModelKit.TaylorVector{M}, vars, buffer, ::Val{N}) where {M, N}
    for i in eachindex(vars)
        rval = TaylorDiff.flatten(real(buffer[i]))
        ival = TaylorDiff.flatten(imag(buffer[i]))
        u[i] = ntuple(i -> rval[i]  + im * ival[i], Val(length(rval)))
    end
end

function HC.ModelKit.taylor!(u::AbstractVector, ::Val{N},
        sys::HomotopySystemWrapper{Inplace}, x, p = nothing) where {N}
    f = sys.f
    p = sys.p
    vars, buffer = sys.taylorvars
    if !check_taylor_equality(vars, x)
        update_taylorvars_from_taylorvector!(vars, x)
        vars = reinterpret(Complex{eltype(vars)}, vars)
        buffer = reinterpret(Complex{eltype(buffer)}, buffer)
        f(buffer, vars, p)
    else
        vars = reinterpret(Complex{eltype(vars)}, vars)
    end
    update_maybe_taylorvector_from_taylorvars!(u, vars, buffer, Val(N))
    return u
end

function HC.ModelKit.taylor!(u::AbstractVector, ::Val{N},
        sys::HomotopySystemWrapper{OutOfPlace}, x, p = nothing) where {N}
    f = sys.f
    p = sys.p
    vars = sys.taylorvars
    if !check_taylor_equality(vars, x)
        update_taylorvars_from_taylorvector!(vars, x)
        vars = reinterpret(Complex{eltype(vars)}, vars)
        buffer = f(vars, p)
        copyto!(vars, buffer)
    else
        vars = buffer = reinterpret(Complex{eltype(vars)}, vars)
    end
    update_maybe_taylorvector_from_taylorvars!(u, vars, buffer, Val(N))
    return u
end

function HC.ModelKit.taylor!(u::AbstractVector, ::Val{N},
        sys::HomotopySystemWrapper{Scalar}, x, p = nothing) where {N}
    f = sys.f
    p = sys.p
    var = sys.taylorvars
    if !check_taylor_equality(var, x)
        update_taylorvars_from_taylorvector!(var, x)
        var = reinterpret(Complex{eltype(var)}, var)
        buffer = f(var[1], p)
        var[1] = buffer
    else
        var = buffer = reinterpret(Complex{eltype(var)}, var)
    end
    update_maybe_taylorvector_from_taylorvars!(u, var, buffer, Val(N))
    return u
end

"""
    $(TYPEDEF)

A `HomotopyContinuation.AbstractHomotopy` which uses an inital guess ``x_0`` to construct
the start system for the homotopy. The homotopy is

```math
\\begin{aligned}
H(x, t) &= t * (F(x) - F(x_0)) + (1 - t) * F(x)
        &= F(x) - t * F(x_0)
\\end{aligned}
```

Where ``F: \\mathbb{R}^n \\rightarrow \\mathbb{R}^n`` is the polynomial system and
``x_0 \\in \\mathbb{R}^n`` is the guess provided to the `NonlinearProblem`.

# Fields

$(TYPEDFIELDS)
"""
@concrete struct GuessHomotopy <: HC.AbstractHomotopy
    """
    The `HomotopyContinuation.AbstractSystem` to solve.
    """
    sys <: HC.AbstractSystem
    """
    The residual of the initial guess, i.e. ``F(x_0)``.
    """
    fu0
    """
    A preallocated `TaylorVector` for efficient `taylor!` implementation.
    """
    taylorbuffer::HC.ModelKit.TaylorVector{5, ComplexF64}
end

function GuessHomotopy(sys, fu0)
    return GuessHomotopy(sys, fu0, HC.ModelKit.TaylorVector{5}(ComplexF64, length(fu0)))
end

Base.size(h::GuessHomotopy) = size(h.sys)

function HC.ModelKit.evaluate!(u, h::GuessHomotopy, x, t, p = nothing)
    HC.ModelKit.evaluate!(u, h.sys, x, p)
    @inbounds for i in eachindex(u)
        u[i] -= t * h.fu0[i]
    end
    return u
end

function HC.ModelKit.evaluate_and_jacobian!(u, U, h::GuessHomotopy, x, t, p = nothing)
    HC.ModelKit.evaluate_and_jacobian!(u, U, h.sys, x, p)
    @inbounds for i in eachindex(u)
        u[i] -= t * h.fu0[i]
    end
    return u, U
end

function HC.ModelKit.taylor!(
        u, v::Val{N}, H::GuessHomotopy, tx, t, incremental::Bool) where {N}
    HC.ModelKit.taylor!(u, v, H, tx, t)
end

function HC.ModelKit.taylor!(u, ::Val{N}, h::GuessHomotopy, x, t, p = nothing) where {N}
    HC.ModelKit.taylor!(h.taylorbuffer, Val(N), h.sys, x, p)
    @inbounds for i in eachindex(u)
        h.taylorbuffer[i, 1] -= t * h.fu0[i]
        h.taylorbuffer[i, 2] -= h.fu0[i]
        u[i] = h.taylorbuffer[i, N + 1]
    end
    return u
end
