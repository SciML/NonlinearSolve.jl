using SciMLTesting, SciMLJacobianOperators, Test

run_qa(
    SciMLJacobianOperators;
    explicit_imports = true,
    ei_kwargs = (;
        # Non-public names qualified-accessed from their owning packages; they go
        # public as those packages adopt the `public` keyword:
        #   ArrayInterface: can_setindex, restructure
        #   SciMLBase: has_jac, has_jvp, has_vjp
        all_qualified_accesses_are_public = (;
            ignore = (
                :can_setindex, :restructure, :has_jac, :has_jvp, :has_vjp,
            ),
        ),
        # Non-public names explicitly imported from their owning packages:
        #   SciMLBase: AbstractNonlinearFunction, AbstractNonlinearProblem
        #   SciMLOperators: AbstractSciMLOperator
        all_explicit_imports_are_public = (;
            ignore = (
                :AbstractNonlinearFunction, :AbstractNonlinearProblem,
                :AbstractSciMLOperator,
            ),
        ),
    ),
)
