@testitem "DRiLExt Single env methods" begin
    using DRiL
    using RDE

    # Helper functions
    function get_dril_ext()
        return Base.get_extension(RDE_Env, :RDE_EnvDRiLExt)
    end

    function get_env()
        env = RDEEnv(
            observation_strategy = SectionedStateObservation(),
            action_strat = ScalarPressureAction(),
            params = RDEParam(N = 512, tmax = 100.0f0)
        )
        DRiLExt = get_dril_ext()
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
        return true
    end

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

@testitem "DRiLExt single env usage" begin
    using DRiL
    using RDE

    # Helper functions
    function get_dril_ext()
        return Base.get_extension(RDE_Env, :RDE_EnvDRiLExt)
    end

    function get_env()
        env = RDEEnv(
            observation_strategy = SectionedStateObservation(),
            action_strat = ScalarPressureAction(),
            params = RDEParam(N = 512, tmax = 100.0f0)
        )
        DRiLExt = get_dril_ext()
        env = DRiLExt.DRiLRDEEnv(env)
        return env
    end

    function test_parallel_env_usage(env)
        rand_obs = rand(observation_space(env))
        obs = observe(env)
        @assert length(obs) == number_of_envs(env)
        actions = rand(action_space(env), number_of_envs(env))
        rewards, _, _, _ = act!(env, actions)
        @assert length(rewards) == number_of_envs(env) "length of rewards: $(length(rewards)) != number of envs: $(number_of_envs(env))"
        random_obs = rand(observation_space(env))
        reset!(env)
        terminated(env)
        truncated(env)
        return true
    end

    envs = [get_env() for _ in 1:4]
    env = BroadcastedParallelEnv(envs)
    @test hasmethod(DRiL.observation_space, (typeof(env),))
    @test hasmethod(DRiL.action_space, (typeof(env),))
    @test hasmethod(DRiL.number_of_envs, (typeof(env),))
    @test hasmethod(DRiL.reset!, (typeof(env),))
    # For parallel envs, act! takes a vector of actions
    @test hasmethod(DRiL.observe, (typeof(env),))
    @test hasmethod(DRiL.terminated, (typeof(env),))
    @test hasmethod(DRiL.truncated, (typeof(env),))

    @test test_parallel_env_usage(env)
    monitored_env = MonitorWrapperEnv(env)
    @test test_parallel_env_usage(monitored_env)
    norm_env = NormalizeWrapperEnv(monitored_env)
    @test test_parallel_env_usage(norm_env)
end

@testitem "DRiLExt multi-agent env" begin
    using DRiL
    using RDE

    # Helper functions
    function get_dril_ext()
        return Base.get_extension(RDE_Env, :RDE_EnvDRiLExt)
    end

    function get_multi_agent_env()
        env = RDEEnv(
            observation_strategy = MultiCenteredObservation(n_sections = 4),
            action_strat = VectorPressureAction(n_sections = 4),
            reward_type = MultiSectionPeriodMinimumReward(n_sections = 4, lowest_action_magnitude_reward = 0.0f0, weights = [1.0f0, 1.0f0, 5.0f0, 1.0f0]),
            params = RDEParam(N = 512, tmax = 100.0f0)
        )
        DRiLExt = get_dril_ext()
        env = DRiLExt.DRiLMultiAgentRDEEnv(env)
        return env
    end

    function test_parallel_env_usage(env)
        rand_obs = rand(observation_space(env))
        obs = observe(env)
        @assert length(obs) == number_of_envs(env)
        actions = rand(action_space(env), number_of_envs(env))
        rewards, _, _, _ = act!(env, actions)
        @assert length(rewards) == number_of_envs(env) "length of rewards: $(length(rewards)) != number of envs: $(number_of_envs(env))"
        random_obs = rand(observation_space(env))
        reset!(env)
        terminated(env)
        truncated(env)
        return true
    end

    env = get_multi_agent_env()
    @test hasmethod(DRiL.observation_space, (typeof(env),))
    @test hasmethod(DRiL.action_space, (typeof(env),))
    @test hasmethod(DRiL.number_of_envs, (typeof(env),))
    @test hasmethod(DRiL.reset!, (typeof(env),))
    # Multi-agent envs act like parallel envs
    @test hasmethod(DRiL.observe, (typeof(env),))
    @test hasmethod(DRiL.terminated, (typeof(env),))
    @test hasmethod(DRiL.truncated, (typeof(env),))

    @test test_parallel_env_usage(env)
end

@testitem "DRiLExt multi-agent env usage" begin
    using DRiL
    using RDE

    # Helper functions
    function get_dril_ext()
        return Base.get_extension(RDE_Env, :RDE_EnvDRiLExt)
    end

    function get_multi_agent_env()
        env = RDEEnv(
            observation_strategy = MultiCenteredObservation(n_sections = 4),
            action_strat = VectorPressureAction(n_sections = 4),
            reward_type = MultiSectionPeriodMinimumReward(n_sections = 4, lowest_action_magnitude_reward = 0.0f0, weights = [1.0f0, 1.0f0, 5.0f0, 1.0f0]),
            params = RDEParam(N = 512, tmax = 100.0f0)
        )
        DRiLExt = get_dril_ext()
        env = DRiLExt.DRiLMultiAgentRDEEnv(env)
        return env
    end

    function test_parallel_env_usage(env)
        rand_obs = rand(observation_space(env))
        obs = observe(env)
        @assert length(obs) == number_of_envs(env)
        actions = rand(action_space(env), number_of_envs(env))
        rewards, _, _, _ = act!(env, actions)
        @assert length(rewards) == number_of_envs(env) "length of rewards: $(length(rewards)) != number of envs: $(number_of_envs(env))"
        random_obs = rand(observation_space(env))
        reset!(env)
        terminated(env)
        truncated(env)
        return true
    end

    envs = [get_multi_agent_env() for _ in 1:4]
    env = MultiAgentParallelEnv(envs)
    @test test_parallel_env_usage(env)
    monitored_env = MonitorWrapperEnv(env)
    @test test_parallel_env_usage(monitored_env)
    norm_env = NormalizeWrapperEnv(monitored_env)
    @test test_parallel_env_usage(norm_env)
end

@testitem "DRiLExt SAVA env" begin
    using DRiL
    using RDE

    # Helper functions
    function get_dril_ext()
        return Base.get_extension(RDE_Env, :RDE_EnvDRiLExt)
    end

    function get_SAVA_env()
        env = RDEEnv(
            observation_strategy = SectionedStateObservation(),
            action_strat = VectorPressureAction(n_sections = 4),
            reward_type = PeriodMinimumReward(),
            params = RDEParam(N = 512, tmax = 100.0f0)
        )
        DRiLExt = get_dril_ext()
        env = DRiLExt.DRiLRDEEnv(env)
        return env
    end

    env = get_SAVA_env()
    @test hasmethod(DRiL.observation_space, (typeof(env),))
    @test hasmethod(DRiL.action_space, (typeof(env),))
    # Single envs don't have number_of_envs method
    @test hasmethod(DRiL.reset!, (typeof(env),))
    @test hasmethod(DRiL.act!, (typeof(env), typeof(rand(action_space(env)))))
    @test hasmethod(DRiL.observe, (typeof(env),))
    @test hasmethod(DRiL.terminated, (typeof(env),))
    @test hasmethod(DRiL.truncated, (typeof(env),))
end

@testitem "DRiLExt SAVA env usage" begin
    using DRiL
    using RDE

    # Helper functions
    function get_dril_ext()
        return Base.get_extension(RDE_Env, :RDE_EnvDRiLExt)
    end

    function get_SAVA_env()
        env = RDEEnv(
            observation_strategy = SectionedStateObservation(),
            action_strat = VectorPressureAction(n_sections = 4),
            reward_type = PeriodMinimumReward(),
            params = RDEParam(N = 512, tmax = 100.0f0)
        )
        DRiLExt = get_dril_ext()
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
        return true
    end

    function test_parallel_env_usage(env)
        rand_obs = rand(observation_space(env))
        obs = observe(env)
        @assert length(obs) == number_of_envs(env)
        actions = rand(action_space(env), number_of_envs(env))
        rewards, _, _, _ = act!(env, actions)
        @assert length(rewards) == number_of_envs(env) "length of rewards: $(length(rewards)) != number of envs: $(number_of_envs(env))"
        random_obs = rand(observation_space(env))
        reset!(env)
        terminated(env)
        truncated(env)
        return true
    end

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
