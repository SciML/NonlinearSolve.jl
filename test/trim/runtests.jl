using SafeTestsets

@safetestset "Clean implementation (non-trimmable)" begin
    using JET
    using SciMLBase: successful_retcode
    include("clean_optimization.jl")
    @test successful_retcode(minimize(1.0).retcode)
    # can't use `@test_opt` macro here because it would eval before `using JET`
    # is processed
    test_opt(minimize, (typeof(1.0),))
end

@safetestset "Trimmable implementation" begin
    using JET
    using SciMLBase: successful_retcode
    include("trimmable_optimization.jl")
    @test successful_retcode(minimize(1.0).retcode)
    # can't use `@test_opt` macro here because it would eval before `using JET`
    # is processed
    test_opt(minimize, (typeof(1.0),))
end

@safetestset "Run trim" begin
    # https://discourse.julialang.org/t/capture-stdout-and-stderr-in-case-a-command-fails/101772/3?u=romeov
    "Run a Cmd object, returning the stdout & stderr contents plus the exit code"
    function _execute(cmd::Cmd)
        out = Pipe()
        err = Pipe()
        process = run(pipeline(ignorestatus(cmd); stdout = out, stderr = err))
        close(out.in)
        close(err.in)
        out = (
            stdout = String(read(out)), stderr = String(read(err)),
            exitcode = process.exitcode,
        )
        return out
    end

    JULIAC = normpath(
        joinpath(
            Sys.BINDIR, Base.DATAROOTDIR, "julia", "juliac",
            "juliac.jl"
        )
    )
    @test isfile(JULIAC)
    binpath = tempname()
    cmd = `$(Base.julia_cmd()) --project=. --depwarn=error $(JULIAC) --experimental --trim=unsafe-warn --output-exe $(binpath) main.jl`

    # since we are calling Julia from Julia, we first need to clean some
    # environment variables
    clean_env = copy(ENV)
    delete!(clean_env, "JULIA_PROJECT")
    delete!(clean_env, "JULIA_LOAD_PATH")
    # We could just check for success, but then failures are hard to debug.
    # Instead we use `_execute` to also capture `stdout` and `stderr`.
    # @test success(setenv(cmd, clean_env))
    trimcall = _execute(setenv(cmd, clean_env; dir = @__DIR__))
    if trimcall.exitcode != 0
        @show trimcall.stdout
        @show trimcall.stderr
    end
    @test trimcall.exitcode == 0
    @test isfile(binpath)
    @test success(`$(binpath) 1.0`)
end
