"""
  prevfloat_tdir(x, x0, x1)

Move `x` one floating point towards x0.
"""
function prevfloat_tdir(x::T, x0::T, x1::T)::T where {T}
  x1 > x0 ? prevfloat(x) : nextfloat(x)
end
  
function nextfloat_tdir(x::T, x0::T, x1::T)::T where {T}
  x1 > x0 ? nextfloat(x) : prevfloat(x)
end

alg_autodiff(alg::AbstractNewtonAlgorithm{CS,AD}) where {CS,AD} = AD
alg_autodiff(alg) = false

"""
  value_derivative(f, x)

Compute `f(x), d/dx f(x)` in the most efficient way.
"""
function value_derivative(f::F, x::R) where {F,R}
    T = typeof(ForwardDiff.Tag(f, R))
    out = f(ForwardDiff.Dual{T}(x, one(x)))
    ForwardDiff.value(out), ForwardDiff.extract_derivative(T, out)
end

DiffEqBase.has_Wfact(f::Function) = false
DiffEqBase.has_Wfact_t(f::Function) = false
