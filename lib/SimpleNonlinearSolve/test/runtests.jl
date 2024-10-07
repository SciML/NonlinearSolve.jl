using TestItemRunner, InteractiveUtils, Pkg

@info sprint(InteractiveUtils.versioninfo)

const GROUP = lowercase(get(ENV, "GROUP", "All"))

if GROUP == "all"
    @run_package_tests
else
    @run_package_tests filter = ti -> (Symbol(GROUP) in ti.tags)
end
