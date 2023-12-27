# We want a general form of this in SciMLOperators. However, we use this extensively and we
# can have a custom implementation here till
# https://github.com/SciML/SciMLOperators.jl/issues/223 is resolved.
abstract type AbstractFunctionOperator end
