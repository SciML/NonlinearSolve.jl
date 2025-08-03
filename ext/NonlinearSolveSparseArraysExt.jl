module NonlinearSolveSparseArraysExt

using SparseArrays: SparseArrays

using NonlinearSolve: NonlinearSolve

# Re-export SparseArrays functionality that NonlinearSolve needs
# This extension is loaded when SparseArrays is explicitly imported by the user

# The main purpose of this extension is to ensure that SparseArrays 
# functionality is available when needed, but doesn't force loading
# SparseArrays for users who don't need sparse matrix support

end