"""
  prevfloat_tdir(x, x0, x1)

Move `x` one floating point towards x0.
"""
function prevfloat_tdir(x::T, x0::T, x1::T)::T where {T}
  x1 > x0 : prevfloat(x) : nextfloat(x)
end
  