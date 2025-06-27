using DRiL
function get_env()
    env_config = RDEEnvConfig(
        observation_strategy=SectionedStateObservation(),
        action_type=ScalarPressureAction(),
    )
    env = make_env(env_config)
    DRiLExt = Base.get_extension(RDE_Env, :RDE_EnvDRiLExt)
    env = DRiLExt.DRiLRDEEnv(env)
    return env
end

function get_multi_agent_env()
    env_config = RDEEnvConfig(
        observation_strategy=MultiCenteredObservation(n_sections=4),
        action_type=VectorPressureAction(n_sections=4),
        reward_type=MultiSectionPeriodMinimumReward(n_sections=4, target_shock_count=3, lowest_action_magnitude_reward=0.0f0, weights=[1f0, 1f0, 5f0, 1f0]),
    )
    env = make_env(env_config)
    DRiLExt = Base.get_extension(RDE_Env, :RDE_EnvDRiLExt)
    env = DRiLExt.DRiLMultiAgentRDEEnv(env)
    return env
end
function get_SAVA_env()
    env_config = RDEEnvConfig(
        observation_strategy=SectionedStateObservation(),
        action_type=VectorPressureAction(n_sections=4),
        reward_type=PeriodMinimumReward()
    )
    env = make_env(env_config)
    DRiLExt = Base.get_extension(RDE_Env, :RDE_EnvDRiLExt)
    env = DRiLExt.DRiLRDEEnv(env)
    return env
end
function test_single_env_usage(env)
    rand_obs = rand(observation_space(env))
    obs = observe(env)
    action = rand(action_space(env))
    reward = act!(env, action)
    random_obs = rand(observation_space(env))
    reset!(env)
    terminated(env)
    truncated(env)
    true
end
function test_parallel_env_usage(env)
    rand_obs = rand(observation_space(env))
    obs = observe(env)
    action = rand(action_space(env), number_of_envs(env))
    reward = act!(env, action)
    random_obs = rand(observation_space(env))
    reset!(env)
    terminated(env)
    truncated(env)
    true
end

@testset "DRiLExt Single env methods" begin
    env = get_env()
    @test hasmethod(DRiL.observation_space, (typeof(env),))
    @test hasmethod(DRiL.act!, (typeof(env), Any))
    @test hasmethod(DRiL.observe, (typeof(env),))
    @test hasmethod(DRiL.terminated, (typeof(env),))
    @test hasmethod(DRiL.truncated, (typeof(env),))
    @test hasmethod(DRiL.reset!, (typeof(env),))
    @test hasmethod(DRiL.action_space, (typeof(env),))

    @test test_single_env_usage(env)
end

@testset "DRiLExt single env usage" begin
    envs = [get_env() for _ in 1:4]
    env = BroadcastedParallelEnv(envs)
    @test hasmethod(DRiL.observation_space, (typeof(env),))
    @test hasmethod(DRiL.action_space, (typeof(env),))
    @test hasmethod(DRiL.number_of_envs, (typeof(env),))
    @test hasmethod(DRiL.reset!, (typeof(env),))
    @test hasmethod(DRiL.act!, (typeof(env), Any))
    @test hasmethod(DRiL.observe, (typeof(env),))
    @test hasmethod(DRiL.terminated, (typeof(env),))
    @test hasmethod(DRiL.truncated, (typeof(env),))

    @test test_parallel_env_usage(env)
    monitored_env = MonitorWrapperEnv(env)
    @test test_parallel_env_usage(monitored_env)
    norm_env = NormalizeWrapperEnv(monitored_env)
    @test test_parallel_env_usage(norm_env)
end

@testset "DRiLExt multi-agent env" begin
    env = get_multi_agent_env()
    @test hasmethod(DRiL.observation_space, (typeof(env),))
    @test hasmethod(DRiL.action_space, (typeof(env),))
    @test hasmethod(DRiL.number_of_envs, (typeof(env),))
    @test hasmethod(DRiL.reset!, (typeof(env),))
    @test hasmethod(DRiL.act!, (typeof(env), Any))
    @test hasmethod(DRiL.observe, (typeof(env),))
    @test hasmethod(DRiL.terminated, (typeof(env),))
    @test hasmethod(DRiL.truncated, (typeof(env),))

    @test test_parallel_env_usage(env)
end


@testset "DRiLExt multi-agent env usage" begin
    envs = [get_multi_agent_env() for _ in 1:4]
    env = MultiAgentParallelEnv(envs)
    @test test_parallel_env_usage(env)
    monitored_env = MonitorWrapperEnv(env)
    @test test_parallel_env_usage(monitored_env)
    norm_env = NormalizeWrapperEnv(monitored_env)
    @test test_parallel_env_usage(norm_env)
end

@testset "DRiLExt SAVA env" begin
    env = get_SAVA_env()
    @test hasmethod(DRiL.observation_space, (typeof(env),))
    @test hasmethod(DRiL.action_space, (typeof(env),))
    @test hasmethod(DRiL.number_of_envs, (typeof(env),))
    @test hasmethod(DRiL.reset!, (typeof(env),))
    @test hasmethod(DRiL.act!, (typeof(env), Any))
    @test hasmethod(DRiL.observe, (typeof(env),))
end

@testset "DRiLExt SAVA env usage" begin
    env = get_SAVA_env()
    @test test_single_env_usage(env)
    envs = [get_SAVA_env() for _ in 1:4]
    env = BroadcastedParallelEnv(envs)
    @test test_parallel_env_usage(env)
    monitored_env = MonitorWrapperEnv(env)
    @test test_parallel_env_usage(monitored_env)
    norm_env = NormalizeWrapperEnv(monitored_env)
    @test test_parallel_env_usage(norm_env)
end