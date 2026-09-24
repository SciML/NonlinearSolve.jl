using NonlinearSolve

alg = ArcLengthContinuation()
@test alg isa ArcLengthContinuation
@test alg isa NonlinearSolve.AbstractNonlinearSolveAlgorithm
@test alg.inner === nothing
@test alg.initial_step_factor ≈ 0.1
@test alg.adaptive == true
@test alg.min_ds === nothing
@test alg.max_step_factor ≈ 1.0
@test alg.expand_factor ≈ 2.0
@test alg.expand_threshold == 2
@test alg.max_angle ≈ π / 6
@test alg.maxsteps == 10000

alg2 = ArcLengthContinuation(;
    inner = SimpleNewtonRaphson(), initial_step_factor = 0.05,
    max_angle = π / 4, maxsteps = 500
)
@test alg2.inner isa SimpleNewtonRaphson
@test alg2.initial_step_factor ≈ 0.05
@test alg2.max_angle ≈ π / 4
@test alg2.maxsteps == 500

@test_throws ArgumentError ArcLengthContinuation(; initial_step_factor = 0.0)
@test_throws ArgumentError ArcLengthContinuation(; initial_step_factor = 1.5)
@test_throws ArgumentError ArcLengthContinuation(; min_ds = 0.0)
@test_throws ArgumentError ArcLengthContinuation(; max_step_factor = 0.0)
@test_throws ArgumentError ArcLengthContinuation(; max_step_factor = 2.0)
@test_throws ArgumentError ArcLengthContinuation(; expand_factor = 0.5)
@test_throws ArgumentError ArcLengthContinuation(; expand_threshold = 0)
@test_throws ArgumentError ArcLengthContinuation(; max_angle = 0.0)
@test_throws ArgumentError ArcLengthContinuation(; max_angle = 4.0)   # > π
@test_throws ArgumentError ArcLengthContinuation(; maxsteps = 0)
