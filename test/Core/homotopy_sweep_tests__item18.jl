using NonlinearSolve

alg = HomotopySweep()
@test alg.max_step_factor ≈ 1.0
@test alg.expand_factor ≈ 2.0
@test alg.expand_threshold == 2
@test alg.expand_quality ≈ 0.25
@test alg.predictor === :secant

alg2 = HomotopySweep(;
    max_step_factor = 0.25, expand_factor = 1.5,
    expand_threshold = 3, predictor = :constant
)
@test alg2.max_step_factor ≈ 0.25
@test alg2.expand_factor ≈ 1.5
@test alg2.expand_threshold == 3
@test alg2.predictor === :constant

# expand_factor = 1 disables expansion and must be accepted
@test HomotopySweep(; expand_factor = 1).expand_factor == 1
# expand_quality = Inf disables the quality gate and must be accepted
@test HomotopySweep(; expand_quality = Inf).expand_quality == Inf

@test_throws ArgumentError HomotopySweep(; max_step_factor = 0.0)
@test_throws ArgumentError HomotopySweep(; max_step_factor = 1.5)
@test_throws ArgumentError HomotopySweep(; expand_factor = 0.5)
@test_throws ArgumentError HomotopySweep(; expand_threshold = 0)
@test_throws ArgumentError HomotopySweep(; expand_quality = 0.0)
@test_throws ArgumentError HomotopySweep(; expand_quality = -1.0)
@test_throws ArgumentError HomotopySweep(; predictor = :tangent)
