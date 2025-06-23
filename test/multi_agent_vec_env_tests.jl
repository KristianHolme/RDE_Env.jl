using Test
using RDE
using RDE_Env
using Random
using CommonRLInterface

@testset "MultiAgentVecEnv Tests" begin
    @testset "Construction" begin
        # Create environments with 2 agents each
        envs = [RDEEnv(
            dt=0.1,
            τ_smooth=0.01,
            observation_strategy=MultiSectionObservation(2)  # 2 agents
        ) for _ in 1:3]
        vec_env = MultiAgentRDEVecEnv(envs)
        
        # Test basic properties
        @test vec_env.n_envs == 3
        @test vec_env.n_agents_per_env == 2
        @test length(vec_env.envs) == 3
        @test size(vec_env.observations, 2) == 6  # 3 envs * 2 agents
    end
    
    @testset "Reset and Seed" begin
        envs = [RDEEnv(
            dt=0.1,
            τ_smooth=0.01,
            observation_strategy=MultiSectionObservation(2),
            reset_strategy=RandomCombination()
        ) for _ in 1:2]
        vec_env = MultiAgentRDEVecEnv(envs)
        
        # Test initial reset
        _reset!(vec_env)
        obs1 = observe(vec_env)
        @test size(obs1, 2) == 4  # 2 envs * 2 agents
        @test !any(isnan, obs1)
        
        # Test seeding produces deterministic results
        seed!(vec_env, 42)
        _reset!(vec_env)
        obs2 = observe(vec_env)
        
        seed!(vec_env, 42)
        _reset!(vec_env)
        obs3 = observe(vec_env)
        
        @test obs2 == obs3  # Same seed should give same results
        
        seed!(vec_env, 43)
        _reset!(vec_env)
        obs4 = observe(vec_env)
        
        @test obs2 != obs4  # Different seeds should give different results
    end
    
    @testset "Step" begin
        envs = [RDEEnv(
            dt=0.1,
            τ_smooth=0.01,
            observation_strategy=MultiSectionObservation(2)
        ) for _ in 1:2]
        vec_env = MultiAgentRDEVecEnv(envs)
        _reset!(vec_env)
        
        # Create test actions (2 agents per env, 2 envs)
        action_length = action_dim(envs[1].action_type)
        actions = zeros(Float32, action_length, 4)  # 4 total agents
        actions[:, 1] .= 0.5  # Different action for first agent
        
        # Test act!
        rewards = act!(vec_env, actions)
        obs = observe(vec_env)
        dones = terminated(vec_env)
        
        # Check dimensions
        @test size(obs, 2) == 4  # 2 envs * 2 agents
        @test length(rewards) == 4
        @test length(dones) == 4
        
        # Check values
        @test !any(isnan, obs)
        @test !any(isnan, rewards)
        
        # Test different actions give different results
        @test obs[:, 1] != obs[:, 2]  # First agent should differ due to different action
    end
    
    @testset "Termination" begin
        # Create envs with short time limit
        envs = [RDEEnv(
            dt=0.1,
            τ_smooth=0.01,
            params=RDEParam(tmax=0.2),
            observation_strategy=MultiSectionObservation(2)
        ) for _ in 1:2]
        vec_env = MultiAgentRDEVecEnv(envs)
        _reset!(vec_env)
        
        # Run until termination
        action_length = action_dim(envs[1].action_type)
        actions = zeros(Float32, action_length, 4)  # 4 total agents
        is_terminated = false
        n_steps = 0
        
        while !is_terminated && n_steps < 100
            act!(vec_env, actions)
            dones = terminated(vec_env)
            is_terminated = any(dones)
            n_steps += 1
        end
        
        @test is_terminated
        @test n_steps < 100  # Should terminate due to time limit
        
        # Check auto-reset
        @test !any(isnan, vec_env.observations)  # Environments should be reset
    end
    
    @testset "Thread Safety" begin
        envs = [RDEEnv(
            dt=0.1,
            τ_smooth=0.01,
            observation_strategy=MultiSectionObservation(2)
        ) for _ in 1:4]
        vec_env = MultiAgentRDEVecEnv(envs)
        _reset!(vec_env)
        
        action_length = action_dim(envs[1].action_type)
        actions = zeros(Float32, action_length, 8)  # 4 envs * 2 agents
        
        # Run multiple steps in parallel
        _act!(vec_env, actions)
        
        # If we got here without errors, threading worked
        @test true
    end
end 