using SafeTestsets

@safetestset "Clean implementation (non-trimmable)" begin
    using JET
    using SciMLBase: successful_retcode
    include("optimization_clean.jl")
    @test successful_retcode(TestModuleClean.minimize(1.0).retcode)
    # can't use `@test_opt` macro here because it would try to eval before
    # `using JET` is processed
    test_opt(TestModuleClean.minimize, (typeof(1.0),))
end

@safetestset "Trimmable implementation" begin
    using JET
    using SciMLBase: successful_retcode
    include("optimization_trimmable.jl")
    @test successful_retcode(TestModuleTrimmable.minimize(1.0).retcode)
    # can't use `@test_opt` macro here because it would try to eval before
    # `using JET` is processed
    test_opt(TestModuleTrimmable.minimize, (typeof(1.0),))
end

@safetestset "Run trim" begin
    # https://discourse.julialang.org/t/capture-stdout-and-stderr-in-case-a-command-fails/101772/3?u=romeov
    """
    Run a Cmd object, returning the stdout & stderr contents plus the exit code
    """
    function _execute(cmd::Cmd)
        out = Pipe()
        err = Pipe()
        # The pipes must be drained while the process runs: with `--trim=unsafe-warn`
        # juliac can emit far more than the 64KiB pipe buffer, and a full pipe blocks
        # the child while `run(...; wait = true)` blocks the parent — a deadlock that
        # hung CI until the 2h job timeout.
        process = run(pipeline(ignorestatus(cmd); stdout = out, stderr = err); wait = false)
        close(out.in)
        close(err.in)
        stdout_task = @async read(out, String)
        stderr_task = @async read(err, String)
        wait(process)
        return (;
            stdout = fetch(stdout_task), stderr = fetch(stderr_task),
            process.exitcode,
        )
    end

    JULIAC = normpath(
        joinpath(
            Sys.BINDIR, Base.DATAROOTDIR, "julia", "juliac",
            "juliac.jl"
        )
    )
    # Julia 1.13 removed `juliac.jl` from the distribution; juliac now lives
    # in the JuliaC package (a test dep of this project).
    JULIAC_CMD = isfile(JULIAC) ? `$(JULIAC)` :
        Cmd(["-e", "using JuliaC; JuliaC.main(ARGS)", "--"])
    @test isfile(JULIAC) || VERSION ≥ v"1.13-"

    for (mainfile, expectedtopass) in [
            ("main_trimmable.jl", true),
            #= The test below should verify that we indeed can't get a trimmed binary
    # for the "clean" implementation, but will trigger in the future if
    # it does start working. Unfortunately, right now it hangs indefinitely
    # so we are commenting it out. =#
            # ("main_clean.jl", false),
            ("main_segfault.jl", false),
        ]
        binpath = tempname()
        # JuliaC requires `--output-exe` to be a bare name, so run from the
        # output directory and pass absolute paths for project and entry file.
        cmd = `$(Base.julia_cmd()) --project=$(@__DIR__) --depwarn=error $(JULIAC_CMD) --experimental --trim=unsafe-warn --output-exe $(basename(binpath)) $(joinpath(@__DIR__, mainfile))`

        # since we are calling Julia from Julia, we first need to clean some
        # environment variables
        clean_env = copy(ENV)
        delete!(clean_env, "JULIA_PROJECT")
        delete!(clean_env, "JULIA_LOAD_PATH")
        # We could just check for success, but then failures are hard to debug.
        # Instead we use `_execute` to also capture `stdout` and `stderr`.
        # @test success(setenv(cmd, clean_env))
        trimcall = _execute(setenv(cmd, clean_env; dir = dirname(binpath)))
        if trimcall.exitcode != 0 && expectedtopass
            @show trimcall.stdout
            @show trimcall.stderr
        end
        @test trimcall.exitcode == 0 broken = !expectedtopass
        @test isfile(binpath) broken = !expectedtopass
        @test success(`$(binpath) 1.0`) broken = !expectedtopass
    end
end
