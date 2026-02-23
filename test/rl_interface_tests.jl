using RDE_Env

@testitem "Drill Single env methods" begin
    using Drill
    using RDE

    function get_env()
        env = RDEEnv(
            observation_strat = FullStateObservation(),
            action_strat = DirectScalarPressureAction(),
            params = RDEParam(N = 512, tmax = 100.0f0)
        )
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
    @test hasmethod(Drill.observation_space, (typeof(env),))
    @test hasmethod(Drill.act!, (typeof(env), Any))
    @test hasmethod(Drill.observe, (typeof(env),))
    @test hasmethod(Drill.terminated, (typeof(env),))
    @test hasmethod(Drill.truncated, (typeof(env),))
    @test hasmethod(Drill.reset!, (typeof(env),))
    @test hasmethod(Drill.action_space, (typeof(env),))

    @test test_single_env_usage(env)
end

@testitem "Drill single env usage" begin
    using Drill
    using RDE

    function get_env()
        env = RDEEnv(
            observation_strat = FullStateObservation(),
            action_strat = DirectScalarPressureAction(),
            params = RDEParam(N = 512, tmax = 100.0f0)
        )
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
    @test hasmethod(Drill.observation_space, (typeof(env),))
    @test hasmethod(Drill.action_space, (typeof(env),))
    @test hasmethod(Drill.number_of_envs, (typeof(env),))
    @test hasmethod(Drill.reset!, (typeof(env),))
    # For parallel envs, act! takes a vector of actions
    @test hasmethod(Drill.observe, (typeof(env),))
    @test hasmethod(Drill.terminated, (typeof(env),))
    @test hasmethod(Drill.truncated, (typeof(env),))

    @test test_parallel_env_usage(env)
    monitored_env = MonitorWrapperEnv(env)
    @test test_parallel_env_usage(monitored_env)
    norm_env = NormalizeWrapperEnv(monitored_env)
    @test test_parallel_env_usage(norm_env)
end

@testitem "Drill multi-agent env" begin
    using Drill
    using RDE

    function get_multi_agent_env()
        n_sections = 4
        env = RDEEnv(
            observation_strat = FullStateCenteredObservation(n_sections = n_sections),
            action_strat = DirectVectorPressureAction(n_sections = n_sections),
            reward_strat = ScalarToVectorReward(USpanReward(), n_sections),
            params = RDEParam(N = 512, tmax = 100.0f0)
        )
        return MultiAgentRDEEnv(env)
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
    @test hasmethod(Drill.observation_space, (typeof(env),))
    @test hasmethod(Drill.action_space, (typeof(env),))
    @test hasmethod(Drill.number_of_envs, (typeof(env),))
    @test hasmethod(Drill.reset!, (typeof(env),))
    # Multi-agent envs act like parallel envs
    @test hasmethod(Drill.observe, (typeof(env),))
    @test hasmethod(Drill.terminated, (typeof(env),))
    @test hasmethod(Drill.truncated, (typeof(env),))

    @test test_parallel_env_usage(env)
end

@testitem "Drill multi-agent env usage" begin
    using Drill
    using RDE

    function get_multi_agent_env()
        n_sections = 4
        env = RDEEnv(
            observation_strat = FullStateCenteredObservation(n_sections = n_sections),
            action_strat = DirectVectorPressureAction(n_sections = n_sections),
            reward_strat = ScalarToVectorReward(USpanReward(), n_sections),
            params = RDEParam(N = 512, tmax = 100.0f0)
        )
        return MultiAgentRDEEnv(env)
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

@testitem "Drill SAVA env" begin
    using Drill
    using RDE

    function get_SAVA_env()
        n_sections = 4
        env = RDEEnv(
            observation_strat = FullStateObservation(),
            action_strat = DirectVectorPressureAction(n_sections = n_sections),
            reward_strat = USpanReward(),
            params = RDEParam(N = 512, tmax = 100.0f0)
        )
        return env
    end

    env = get_SAVA_env()
    @test hasmethod(Drill.observation_space, (typeof(env),))
    @test hasmethod(Drill.action_space, (typeof(env),))
    # Single envs don't have number_of_envs method
    @test hasmethod(Drill.reset!, (typeof(env),))
    @test hasmethod(Drill.act!, (typeof(env), typeof(rand(action_space(env)))))
    @test hasmethod(Drill.observe, (typeof(env),))
    @test hasmethod(Drill.terminated, (typeof(env),))
    @test hasmethod(Drill.truncated, (typeof(env),))
end

@testitem "Drill SAVA env usage" begin
    using Drill
    using RDE

    function get_SAVA_env()
        n_sections = 4
        env = RDEEnv(
            observation_strat = FullStateObservation(),
            action_strat = DirectVectorPressureAction(n_sections = n_sections),
            reward_strat = USpanReward(),
            params = RDEParam(N = 512, tmax = 100.0f0)
        )
        return env
    end

    function test_single_env_usage(env)
        rand_obs = rand(observation_space(env))
        obs = observe(env)
        action = rand(action_space(env))
        reward = act!(env, action)
        @test reward isa Float32
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
        @test rewards isa Vector{Float32}
        @test length(rewards) == number_of_envs(env)
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
