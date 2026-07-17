using SciMLTesting, SciMLJacobianOperators, Test

const NONLINEARSOLVE_DOCS_SRC = joinpath(@__DIR__, "..", "..", "..", "..", "docs", "src")

run_qa(
    SciMLJacobianOperators;
    explicit_imports = true,
    api_docs_kwargs = (; rendered = true, docs_src = NONLINEARSOLVE_DOCS_SRC),
    ei_kwargs = (;
        # Still non-public in its owning package after the make-public round:
        #   SciMLBase: AbstractNonlinearFunction
        all_explicit_imports_are_public = (;
            ignore = (:AbstractNonlinearFunction,),
        ),
    ),
)
