using NonlinearSolveFirstOrder
include("setup_corerootfindtesting.jl")

p = range(0.01, 2, length = 200)
@test nlprob_iterator_interface(
    quadratic_f, p, false, PseudoTransient(; alpha_initial = 10.0)
) ≈ sqrt.(p)
@test nlprob_iterator_interface(
    quadratic_f!, p, true, PseudoTransient(; alpha_initial = 10.0)
) ≈ sqrt.(p)
