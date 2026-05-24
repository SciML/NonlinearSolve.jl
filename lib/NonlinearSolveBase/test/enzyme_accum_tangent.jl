module EnzymeAccumTangentTests

using Test
using NonlinearSolveBase
import ChainRulesCore, Enzyme  # triggers NonlinearSolveBaseEnzymeExt
import SciMLStructures
import SciMLStructures: Tunable

mutable struct MockMTKParams
    tunable::Vector{Float64}
    caches::Tuple{Vector{Float64}}
end

SciMLStructures.isscimlstructure(::MockMTKParams) = true
SciMLStructures.ismutablescimlstructure(::MockMTKParams) = true
function SciMLStructures.canonicalize(::Tunable, p::MockMTKParams)
    return p.tunable, (val) -> MockMTKParams(collect(val), p.caches), true
end
function SciMLStructures.replace!(::Tunable, p::MockMTKParams, val)
    p.tunable .= val
    return nothing
end

const EXT = Base.get_extension(NonlinearSolveBase, :NonlinearSolveBaseEnzymeExt)

@testset "EnzymeExt._accum_tangent! walks non-Tunable fields (caches)" begin
    # Regression for SciML/NonlinearSolve.jl#935: when SciMLSensitivity's
    # `steadystatebackpass` returns a structured cotangent under
    # `diff_tunables = Val(false)` (e.g. SCC explicitfuns! coupling),
    # the gradient contribution lives in `caches`, not only `tunable`.
    # The accumulator must walk those non-Tunable fields too — otherwise
    # the meaningful cotangent is silently dropped and the user observes
    # a zero gradient.
    dval = MockMTKParams([0.0, 0.0], (zeros(3),))
    darg = MockMTKParams([1.0, 2.0], ([10.0, 20.0, 30.0],))

    EXT._accum_tangent!(dval, darg)

    @test dval.tunable == [1.0, 2.0]
    @test dval.caches[1] == [10.0, 20.0, 30.0]

    # Accumulate again — verify it adds, doesn't overwrite.
    darg2 = MockMTKParams([0.5, 0.5], ([1.0, 2.0, 3.0],))
    EXT._accum_tangent!(dval, darg2)
    @test dval.tunable == [1.5, 2.5]
    @test dval.caches[1] == [11.0, 22.0, 33.0]
end

end
