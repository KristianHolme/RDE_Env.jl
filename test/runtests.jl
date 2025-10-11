# RDE_Env.jl/test/runtests.jl
using Test
using RDE_Env

@testset "RDE_Env.jl" begin
    include("actions_tests.jl")
    # include("vec_env_tests.jl")
    include("reward_tests.jl")
    include("RLenv_tests.jl")
    include("policy_tests.jl")
    include("utils_tests.jl")
    include("DRiLExt_tests.jl")
end
