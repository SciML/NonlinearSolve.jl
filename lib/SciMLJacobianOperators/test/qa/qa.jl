using SciMLTesting, SciMLJacobianOperators, Test

run_qa(
    SciMLJacobianOperators;
    explicit_imports = true,
    ei_kwargs = (;
        # Still non-public in their owning packages after the make-public round:
        #   SciMLBase: AbstractNonlinearFunction;  SciMLOperators: AbstractSciMLOperator
        all_explicit_imports_are_public = (;
            ignore = (:AbstractNonlinearFunction, :AbstractSciMLOperator),
        ),
    ),
)
