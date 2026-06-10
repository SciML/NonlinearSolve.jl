using NonlinearSolveFirstOrder
include("setup_corerootfindtesting.jl")

p = range(0.01, 2, length = 200)
@test nlprob_iterator_interface(quadratic_f, p, false, LevenbergMarquardt()) ≈ sqrt.(p)
@test nlprob_iterator_interface(quadratic_f!, p, true, LevenbergMarquardt()) ≈ sqrt.(p)
