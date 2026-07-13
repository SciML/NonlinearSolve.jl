using SciMLTesting, SciMLJacobianOperators, Test

run_qa(
    SciMLJacobianOperators;
    explicit_imports = true,
    ei_kwargs = (;
        # Still non-public in its owning package after the make-public round:
        #   SciMLBase: AbstractNonlinearFunction
        all_explicit_imports_are_public = (;
            ignore = (:AbstractNonlinearFunction,),
        ),
    ),
)
