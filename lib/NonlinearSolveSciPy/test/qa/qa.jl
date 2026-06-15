using NonlinearSolveSciPy, Aqua, JET

Aqua.test_all(NonlinearSolveSciPy)
JET.test_package(NonlinearSolveSciPy; target_defined_modules = true)
