# Helpers shared between CUDA `@testitem`s in `cuda_tests.jl`. Each `@testitem`
# runs in a fresh module on its own worker, so this file is `include`d from
# inside each one.

# Variant of `@test_nowarn` that ignores benign GPUCompiler deprecation warnings
# emitted by CUDA.jl's libdevice linking on released CUDA.jl versions (<= v6.1.0).
# The upstream fix is on CUDA.jl master (commit dbddf1f46, "Adapt to
# GPUCompiler.jl changes, using lazy-linking") but is not in any tagged release
# yet, so we filter only those exact upstream warnings and continue to fail on
# anything else (including any warning that NonlinearSolve itself emits).
const _GPUCOMPILER_DEPRECATION_PATTERNS = (
    r"┌ Warning: 3-arg `link_libraries!\(job, mod, undefined_fns\)` is deprecated.*?\n(?:[│└].*?\n)+"s,
    r"┌ Warning: `GPUCompiler\.link_library!` is deprecated.*?\n(?:[│└].*?\n)+"s,
)

_strip_known_gpu_warnings(s::AbstractString) = foldl(
    (acc, pat) -> replace(acc, pat => ""),
    _GPUCOMPILER_DEPRECATION_PATTERNS;
    init = String(s),
)

macro test_nowarn_except_gpu_deprecations(ex)
    return quote
        Test.@test_warn (s -> begin
            filtered = $(_strip_known_gpu_warnings)(s)
            print(stderr, filtered)
            isempty(filtered)
        end) $(esc(ex))
    end
end
