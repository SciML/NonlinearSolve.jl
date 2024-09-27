# SciMLJacobianOperators.jl

SciMLJacobianOperators provides a convenient way to compute Jacobian-Vector Product (JVP)
and Vector-Jacobian Product (VJP) using
[SciMLOperators.jl](https://github.com/SciML/SciMLOperators.jl) and
[DifferentiationInterface.jl](https://github.com/gdalle/DifferentiationInterface.jl).

Currently we have interfaces for:

  - `NonlinearProblem`
  - `NonlinearLeastSquaresProblem`

and all autodiff backends supported by DifferentiationInterface.jl are supported.

## Example

```julia
using SciMLJacobianOperators, NonlinearSolve, Enzyme, ForwardDiff

# Define the problem
f(u, p) = u .* u .- p
u0 = ones(4)
p = 2.0
prob = NonlinearProblem(f, u0, p)
fu0 = f(u0, p)
v = ones(4) .* 2

# Construct the operator
jac_op = JacobianOperator(
    prob, fu0, u0;
    jvp_autodiff = AutoForwardDiff(),
    vjp_autodiff = AutoEnzyme(; mode = Enzyme.Reverse)
)
sjac_op = StatefulJacobianOperator(jac_op, u0, p)

sjac_op * v  # Computes the JVP
# 4-element Vector{Float64}:
#  4.0
#  4.0
#  4.0
#  4.0

sjac_op' * v  # Computes the VJP
# 4-element Vector{Float64}:
#  4.0
#  4.0
#  4.0
#  4.0

# What if we multiply the VJP and JVP?
snormal_form = sjac_op' * sjac_op

snormal_form * v  # Computes JᵀJ * v
# 4-element Vector{Float64}:
#  8.0
#  8.0
#  8.0
#  8.0
```
