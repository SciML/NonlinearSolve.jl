"""
    $(TYPEDEF)

A simple struct that wraps a polynomial function which takes complex input and returns
complex output in a form that supports automatic differentiation. If the wrapped
function if ``f: \\mathbb{C}^n \\rightarrow \\mathbb{C}^n`` then it is assumed
the input arrays are real-valued and have length ``2n``. They are `reinterpret`ed
into complex arrays and passed into the function. This struct matches the signature
of ``f``, except if ``f`` is scalar (in which case it acts like an in-place function).
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

function (cjw::ComplexJacobianWrapper{OutOfPlace})(x::AbstractVector{T}, p) where {T}
    x = reinterpret(Complex{T}, x)
    u_tmp = cjw.f(x, p)
    u_tmp = reinterpret(T, u_tmp)
    return u_tmp
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

A callable struct which calculates complex jacobians using `ComplexJacobianWrapper` and
DifferentiationInterface.jl. Follows the signature required by `HomotopySystemWrapper`.

# Fields

$(TYPEDFIELDS)
"""
@concrete struct ComplexDIJacobian{variant}
    """
    The `ComplexJacobianWrapper`.
    """
    f
    """
    Preparation object from DifferentiationInterface.jl.
    """
    prep
    """
    Temporary buffer(s) required for improved AD performance.
    """
    buffers
    """
    Autodiff algorithm as an ADType.
    """
    autodiff
end

function (f::ComplexDIJacobian{Inplace})(u, U, x, p)
    U_tmp = f.buffers
    DI.value_and_jacobian!(f.f, reinterpret(Float64, u), reinterpret(Float64, U_tmp),
        f.prep, f.autodiff, reinterpret(Float64, x), DI.Constant(p))
    U = reinterpret(Float64, U)
    @inbounds for j in axes(U, 2)
        jj = 2j - 1
        for i in axes(U, 1)
            U[i, j] = U_tmp[i, jj]
        end
    end
end

function (f::ComplexDIJacobian{OutOfPlace})(u, U, x, p)
    U_tmp = f.buffers
    u_tmp, _ = DI.value_and_jacobian!(
        f.f, U_tmp, f.prep, f.autodiff, reinterpret(Float64, x), DI.Constant(p))
    copyto!(u, reinterpret(ComplexF64, u_tmp))
    U = reinterpret(Float64, U)
    @inbounds for j in axes(U, 2)
        jj = 2j - 1
        for i in axes(U, 1)
            U[i, j] = U_tmp[i, jj]
        end
    end
end

function (f::ComplexDIJacobian{Scalar})(u, U, x, p)
    U_tmp = f.buffers
    DI.value_and_jacobian!(f.f, reinterpret(Float64, u), U_tmp, f.prep,
        f.autodiff, reinterpret(Float64, x), DI.Constant(p))
    U[1] = U_tmp[1, 1] + im * U_tmp[2, 1]
    return nothing
end

"""
    $(TYPEDSIGNATURES)

Construct a jacobian function for the given function `f`, using `autodiff` as the AD
algorithm, `variant` being the `HomotopySystemVariant` of `f`, `u0` the state vector
and `p` the parameter object.

The returned function must have the signature required by `HomotopySystemWrapper`.
"""
function construct_jacobian(f, autodiff, variant, u0, p)
    if variant == Scalar
        tmp = reinterpret(Float64, Vector{ComplexF64}(undef, 1))
    else
        tmp = reinterpret(Float64, Vector{ComplexF64}(undef, length(u0)))
    end
    f = ComplexJacobianWrapper{variant}(f)
    if variant == OutOfPlace
        prep = DI.prepare_jacobian(f, autodiff, tmp, DI.Constant(p))
    else
        prep = DI.prepare_jacobian(f, tmp, autodiff, copy(tmp), DI.Constant(p))
    end

    if variant == Scalar
        U_tmp = Matrix{Float64}(undef, 2, 2)
    else
        U_tmp = Matrix{Float64}(undef, 2length(u0), 2length(u0))
    end

    return ComplexDIJacobian{variant}(f, prep, U_tmp, autodiff)
end

"""
    $(TYPEDEF)

Efficient version of `ComplexDIJacobian` which directly computes complex jacobians since
Enzyme supports complex differentiation.

# Fields

$(TYPEDFIELDS)
"""
@concrete struct EnzymeJacobian{variant}
    """
    The function to calculate the jacobian of.
    """
    f
    """
    Preparation object from DifferentiationInterface.
    """
    prep
    """
    AD algorithm specified as an ADType.
    """
    autodiff
end

function (f::EnzymeJacobian{Inplace})(u, U, x, p)
    DI.value_and_jacobian!(f.f, u, U, f.prep, f.autodiff, x, DI.Constant(p))
    return nothing
end

function (f::EnzymeJacobian{OutOfPlace})(u, U, x, p)
    u_tmp, _ = DI.value_and_jacobian!(f.f, U, f.prep, f.autodiff, x, DI.Constant(p))
    copyto!(u, u_tmp)
    return nothing
end

function (f::EnzymeJacobian{Scalar})(u, U, x, p)
    u_tmp, der_tmp = DI.value_and_derivative(f.f, f.prep, f.autodiff, x[1], DI.Constant(p))
    u[1] = u_tmp
    U[1] = der_tmp
    return nothing
end

"""
    $(TYPEDSIGNATURES)

Construct an `EnzymeJacobian` function.
"""
function construct_jacobian(f, autodiff::AutoEnzyme, variant, u0, p)
    if variant == Scalar
        prep = DI.prepare_derivative(f, autodiff, u0, DI.Constant(p))
    else
        tmp = Vector{ComplexF64}(undef, length(u0))
        if variant == Inplace
            prep = DI.prepare_jacobian(f, tmp, autodiff, copy(tmp), DI.Constant(p))
        else
            prep = DI.prepare_jacobian(f, autodiff, tmp, DI.Constant(p))
        end
    end
    return EnzymeJacobian{variant}(f, prep, autodiff)
end

"""
    $(TYPEDEF)

Jacobian function following the signature required by `HomotopySystemWrapper` using
a user-provided jacobian.

# Fields

$(TYPEDFIELDS)
"""
@concrete struct ExplicitJacobian{variant}
    """
    The RHS function.
    """
    f
    """
    The jacobian function.
    """
    jac
end

function (f::ExplicitJacobian{Inplace})(u, U, x, p)
    f.f(u, x, p)
    f.jac(U, x, p)
    return nothing
end

function (f::ExplicitJacobian{OutOfPlace})(u, U, x, p)
    u_tmp = f.f(x, p)
    copyto!(u, u_tmp)
    j_tmp = f.jac(x, p)
    copyto!(U, j_tmp)
    return nothing
end

function (f::ExplicitJacobian{Scalar})(u, U, x, p)
    u[1] = f.f(x[1], p)
    U[1] = f.jac(x[1], p)
    return nothing
end
