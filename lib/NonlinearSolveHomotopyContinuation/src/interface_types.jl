abstract type HomotopySystemVariant end

struct Inplace <: HomotopySystemVariant end
struct OutOfPlace <: HomotopySystemVariant end
struct Scalar <: HomotopySystemVariant end

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
    A function which calculates both the polynomial and the jacobian. Must be a function
    of the form `f(u, U, x, p)` where `x` is the current unknowns and `p` is the parameter
    object, writing the value of the polynomial to `u` and the jacobian to `U`. Must be able
    to handle complex `x`.
    """
    jac
    """
    The parameter object.
    """
    p
    """
    HomotopyContinuation.jl's symbolic variables for the system.
    """
    vars
    """
    The `TaylorDiff.TaylorScalar` objects used to compute the taylor series of `f`.
    """
    taylorvars
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
        u, U, sys::HomotopySystemWrapper, x, p = nothing)
    sys.jac(u, U, x, sys.p)
    return u, U
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
        TaylorDiff.flatten(vars[2i - 1]) == map(real, x[i]) || return false
        TaylorDiff.flatten(vars[2i]) == map(imag, x[i]) || return false
    end
    return true
end
function check_taylor_equality(vars, x::AbstractVector)
    for i in eachindex(x)
        TaylorDiff.value(vars[2i - 1]) != real(x[i]) && return false
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
        u[i] = ntuple(i -> rval[i] + im * ival[i], Val(length(rval)))
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
