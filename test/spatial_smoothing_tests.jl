@testitem "RDEEnv applies spatial kernel width" begin
    using RDE
    params = RDEParam(N = 32, tmax = 0.1f0)
    env = RDEEnv(;
        params,
        spatial_kernel_width = 8,
        action_strat = DirectVectorPressureAction(; n_sections = 4),
    )
    profile = RDE.inner_profile(env.prob.injection)
    @test profile isa SpatialMultiStepPressureProfile
    @test length(profile.kernel) == 9
end

@testitem "RDEEnv commit-time spatial smoothing reduces jump" begin
    using RDE
    params = RDEParam{Float32}(N = 64, tmax = 1.0f0)
    env = RDEEnv(;
        params,
        spatial_kernel_width = 9,
        τ_smooth = 0.1f0,
        action_strat = DirectVectorPressureAction(; n_sections = 4),
    )
    target = vcat(fill(0.0f0, params.N ÷ 2), fill(1.0f0, params.N ÷ 2))
    commit_schedule!(env.prob.injection, 0.0f0, 1.0f0, target)
    committed = current_u_p(env.prob.injection)
    boundary_index = params.N ÷ 2 + 1
    jump_raw = abs(target[boundary_index] - target[boundary_index - 1])
    jump_smoothed = abs(committed[boundary_index] - committed[boundary_index - 1])
    @test jump_smoothed < jump_raw
end
