using TestItemRunner

@testitem "Code quality (Aqua.jl)" tags = [:quality] begin
    using Aqua, RDE_Env
    Pkg = Aqua.Pkg
    PackageSpec = Aqua.PackageSpec
    # Aqua 0.8.16 swallows the wrapper's stderr (`Pkg.precompile(; io = devnull)`),
    # so a child crash is reported as a persistent-task failure with no cause.
    Aqua.test_all(RDE_Env; persistent_tasks = false)

    @testset "Persistent tasks" begin
        package_dir = Base.pkgdir(RDE_Env)
        wrapperdir = mktempdir()
        wrappername, _ = only(Pkg.generate(wrapperdir; io = devnull))
        prev = Base.active_project()
        statusfile = joinpath(wrapperdir, "done.log")
        errlog = joinpath(wrapperdir, "precompile-stderr.log")
        proc = try
            isdefined(Pkg, :respect_sysimage_versions) && Pkg.respect_sysimage_versions(false)
            Pkg.activate(wrapperdir; io = devnull)
            Pkg.develop(PackageSpec(; path = package_dir); io = devnull)
            write(
                joinpath(wrapperdir, "src", wrappername * ".jl"),
                """
                module $wrappername
                using RDE_Env
                open("$(escape_string(statusfile))", "w") do io
                    println(io, "done")
                    flush(io)
                end
                end
                """,
            )
            cmd = pipeline(
                `$(Base.julia_cmd()) --project=$wrapperdir -e 'push!(LOAD_PATH, "@stdlib"); using Pkg; Pkg.precompile()'`;
                stdout = errlog,
                stderr = errlog,
            )
            run(cmd; wait = false)
        finally
            isdefined(Pkg, :respect_sysimage_versions) && Pkg.respect_sysimage_versions(true)
            Pkg.activate(prev; io = devnull)
        end
        while !isfile(statusfile) && process_running(proc)
            sleep(0.5)
        end
        if !isfile(statusfile)
            wait(proc)
            log = isfile(errlog) ? read(errlog, String) : "<no stderr captured>"
            error(
                "Aqua wrapper exited before loading RDE_Env " *
                    "(exitcode=$(proc.exitcode), signal=$(proc.termsignal)). " *
                    "This is a precompile/load failure, not a persistent task.\n" *
                    log,
            )
        end
        t0 = time()
        while process_running(proc) && (time() - t0) < 30
            sleep(0.1)
        end
        hung = process_running(proc)
        hung && kill(proc, Base.SIGKILL)
        @test !hung
    end
end

@testitem "Code linting (JET.jl)" tags = [:quality] begin
    using JET, RDE_Env
    report = JET.report_package(RDE_Env; target_modules = (RDE_Env,), toplevel_logger = nothing)
    @test isempty(JET.get_reports(report))
end

@run_package_tests
