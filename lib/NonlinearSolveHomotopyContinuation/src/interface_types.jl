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
    The `TaylorSeries.Taylor1` objects used to compute the taylor series of `f`.
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
        vars, x::HC.ModelKit.TaylorVector{M}) where {M}
    for i in eachindex(vars)
        for j in 0:(M - 1)
            vars[i][j] = x[i, j + 1]
        end
        for j in M:4
            vars[i][j] = zero(vars[i][j])
        end
    end
end

function update_taylorvars_from_taylorvector!(vars, x::AbstractVector)
    for i in eachindex(vars)
        vars[i][0] = x[i]
        for j in 1:4
            vars[i][j] = zero(vars[i][j])
        end
    end
end

function update_maybe_taylorvector_from_taylorvars!(
        u::Vector, vars, buffer, ::Val{N}) where {N}
    for i in eachindex(vars)
        u[i] = buffer[i][N]
    end
end

function update_maybe_taylorvector_from_taylorvars!(
        u::HC.ModelKit.TaylorVector, vars, buffer, ::Val{N}) where {N}
    for i in eachindex(vars)
        u[i] = ntuple(j -> buffer[i][j - 1], Val(N + 1))
    end
end

function HC.ModelKit.taylor!(u::AbstractVector, ::Val{N},
        sys::HomotopySystemWrapper{Inplace}, x, p = nothing) where {N}
    f = sys.f
    p = sys.p
    buffer, vars = sys.taylorvars
    update_taylorvars_from_taylorvector!(vars, x)
    f(buffer, vars, p)
    update_maybe_taylorvector_from_taylorvars!(u, vars, buffer, Val(N))
    return u
end

function HC.ModelKit.taylor!(u::AbstractVector, ::Val{N},
        sys::HomotopySystemWrapper{OutOfPlace}, x, p = nothing) where {N}
    f = sys.f
    p = sys.p
    vars = sys.taylorvars
    update_taylorvars_from_taylorvector!(vars, x)
    buffer = f(vars, p)
    update_maybe_taylorvector_from_taylorvars!(u, vars, buffer, Val(N))
    return u
end

function HC.ModelKit.taylor!(u::AbstractVector, ::Val{N},
        sys::HomotopySystemWrapper{Scalar}, x, p = nothing) where {N}
    f = sys.f
    p = sys.p
    var = sys.taylorvars
    update_taylorvars_from_taylorvector!((var,), x)
    buffer = f(var, p)
    update_maybe_taylorvector_from_taylorvars!(u, (var,), (buffer,), Val(N))
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
