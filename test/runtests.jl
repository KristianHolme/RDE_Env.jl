using RDE_Env
using Test
using TestItems
using TestItemRunner

@testitem "Code quality (Aqua.jl)" tags = [:quality] begin
    using Aqua, RDE_Env
    Aqua.test_all(RDE_Env)
end

@testitem "Code linting (JET.jl)" tags = [:quality] begin
    using JET, RDE_Env
    report = JET.report_package(RDE_Env; target_modules = (RDE_Env,), toplevel_logger = nothing)
    @test isempty(JET.get_reports(report))
end

@run_package_tests
