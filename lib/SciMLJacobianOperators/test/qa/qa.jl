using SciMLTesting, SciMLJacobianOperators, Test

run_qa(
    SciMLJacobianOperators;
    explicit_imports = true,
    ei_kwargs = (;
        # Still non-public in their owning packages (not covered by the public-API round):
        #   SciMLBase: AbstractNonlinearFunction
        #   SciMLOperators: AbstractSciMLOperator
        all_explicit_imports_are_public = (;
            ignore = (:AbstractNonlinearFunction, :AbstractSciMLOperator),
        ),
    ),
)
