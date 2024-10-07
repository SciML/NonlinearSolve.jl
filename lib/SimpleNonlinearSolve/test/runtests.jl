using TestItemRunner, InteractiveUtils, Pkg

@info sprint(InteractiveUtils.versioninfo)

const GROUP = lowercase(get(ENV, "GROUP", "All"))

(GROUP == "all" || GROUP == "cuda") && Pkg.add(["CUDA"])
(GROUP == "all" || GROUP == "adjoint") && Pkg.add(["SciMLSensitivity"])

if GROUP == "all"
    @run_package_tests
else
    @run_package_tests filter = ti -> (Symbol(GROUP) in ti.tags)
end
