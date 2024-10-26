module NonlinearSolveBaseSparseMatrixColoringsExt

using ADTypes: ADTypes, AbstractADType
using NonlinearSolveBase: NonlinearSolveBase, Utils
using SciMLBase: SciMLBase, NonlinearFunction
using SparseMatrixColorings: ConstantColoringAlgorithm, GreedyColoringAlgorithm,
                             LargestFirst

Utils.is_extension_loaded(::Val{:SparseMatrixColorings}) = true

function NonlinearSolveBase.select_fastest_coloring_algorithm(::Val{:SparseMatrixColorings},
        prototype, f::NonlinearFunction, ad::AbstractADType)
    prototype === nothing && return GreedyColoringAlgorithm(LargestFirst())
    if SciMLBase.has_colorvec(f)
        return ConstantColoringAlgorithm{ifelse(
            ADTypes.mode(ad) isa ADTypes.ReverseMode, :row, :column)}(
            prototype, f.colorvec)
    end
    return GreedyColoringAlgorithm(LargestFirst())
end

end
