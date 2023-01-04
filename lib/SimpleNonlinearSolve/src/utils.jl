"""
  prevfloat_tdir(x, x0, x1)

Move `x` one floating point towards x0.
"""
function prevfloat_tdir(x, x0, x1)
    x1 > x0 ? prevfloat(x) : nextfloat(x)
end

function nextfloat_tdir(x, x0, x1)
    x1 > x0 ? nextfloat(x) : prevfloat(x)
end

function max_tdir(a, b, x0, x1)
    x1 > x0 ? max(a, b) : min(a, b)
end

alg_autodiff(alg::AbstractNewtonAlgorithm{CS, AD, FDT}) where {CS, AD, FDT} = AD
diff_type(alg::AbstractNewtonAlgorithm{CS, AD, FDT}) where {CS, AD, FDT} = FDT

"""
  value_derivative(f, x)

Compute `f(x), d/dx f(x)` in the most efficient way.
"""
function value_derivative(f::F, x::R) where {F, R}
    T = typeof(ForwardDiff.Tag(f, R))
    out = f(ForwardDiff.Dual{T}(x, one(x)))
    ForwardDiff.value(out), ForwardDiff.extract_derivative(T, out)
end
value_derivative(f::F, x::AbstractArray) where {F} = f(x), ForwardDiff.jacobian(f, x)

value(x) = x
value(x::Dual) = ForwardDiff.value(x)
value(x::AbstractArray{<:Dual}) = map(ForwardDiff.value, x)

function init_J(x)
    J = ArrayInterfaceCore.zeromatrix(x)
    if ismutable(x)
        J[diagind(J)] .= one(eltype(x))
    else
        J += I
    end
    return J
end
