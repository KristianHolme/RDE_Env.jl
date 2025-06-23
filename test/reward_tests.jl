using Test
using RDE
using CommonRLInterface

@testset "Reward Interface Tests" begin
    @testset "ShockSpanReward" begin
        # Create environment with ShockSpanReward
        env = RDEEnv(;
            dt=1.0,
            reward_type=ShockSpanReward(
                target_shock_count=3,
                span_scale=4.0f0,
                shock_weight=5.0f0
            )
        )

        # Test initial reward
        _reset!(env)
        @test env.reward isa Float32
        @test !isnan(env.reward)
        @test !isinf(env.reward)

        # Test reward with no shocks
        env.state .= 1.0  # Constant state = no shocks
        set_reward!(env, env.reward_type)
        @test env.reward < 0  # Should be negative with no shocks

        # Test reward with target number of shocks
        # Create a state with 3 shocks
        N = env.prob.params.N
        env.state[1:N] .= 1.0
        env.state[N÷4] = 2.0  # First shock
        env.state[N÷2] = 2.0  # Second shock
        env.state[3N÷4] = 2.0  # Third shock
        set_reward!(env, env.reward_type)
        @test env.reward > 0  # Should be positive with target shocks
    end

    @testset "ShockPreservingReward" begin
        # Create environment with ShockPreservingReward
        params = RDEParam(tmax=1.0)
        env = RDEEnv(params;
            dt=0.1,
            reward_type=ShockPreservingReward(
                target_shock_count=3,
                abscence_limit=0.01f0
            )
        )

        # Test initial reward
        _reset!(env)
        @test env.reward isa Float32
        @test !isnan(env.reward)
        @test !isinf(env.reward)

        # Test truncation with wrong number of shocks
        env.state .= 1.0  # Constant state = no shocks
        set_reward!(env, env.reward_type)
        _act!(env, 0.0f0)
        _act!(env, 0.0f0) #act twice to outrun abscence limit

        @test env.terminated  # Should be terminated with wrong shock count
        @test env.reward ≈ -2.0  # Should get penalty reward

        # Test reward with correct number of shocks
        env.terminated = false  # Reset termination flag
        N = env.prob.params.N
        env.state[1:N] .= 1.0
        # Create 3 evenly spaced shocks
        env.state[N÷3] = 2.0
        env.state[2N÷3] = 2.0
        env.state[N] = 2.0
        set_reward!(env, env.reward_type)
        @test !env.terminated  # Should not be terminated
        @test env.reward > 0  # Should get positive reward for evenly spaced shocks
        @test env.reward ≤ 1.0  # Reward should be normalized
    end

    @testset "Policy Reward Behavior" begin
        # Test parameters
        tmax = 1.0
        dt = 0.1
        n_steps = Int(tmax / dt)

        @testset "Constant Policy with ShockSpanReward" begin
            env = RDEEnv(;
                dt=dt,
                params=RDEParam(tmax=tmax),
                reward_type=ShockSpanReward(
                    target_shock_count=3,
                    span_scale=4.0f0,
                    shock_weight=5.0f0
                )
            )
            policy = ConstantRDEPolicy(env)

            # Run policy and collect rewards
            _reset!(env)
            rewards = Float32[]
            for _ in 1:n_steps
                action = _predict_action(policy, _observe(env))
                push!(rewards, _act!(env, action))
                @test !isnan(env.reward)
                @test !isinf(env.reward)
            end
            @test length(rewards) == n_steps
            @test !any(isnan.(rewards))
            @test !any(isinf.(rewards))
        end

        @testset "Random Policy with ShockPreservingReward" begin
            env = RDEEnv(;
                dt=dt,
                params=RDEParam(tmax=tmax),
                reward_type=ShockPreservingReward(
                    target_shock_count=2
                )
            )
            policy = RandomRDEPolicy(env)

            # Run policy and collect rewards
            _reset!(env)
            rewards = Float32[]
            terminations = Int[]
            for step in 1:n_steps
                if env.done
                    break
                end
                action = _predict_action(policy, _observe(env))
                push!(rewards, _act!(env, action))
                if env.terminated
                    push!(terminations, step)
                end
                @test !isnan(env.reward)
                @test !isinf(env.reward)
            end

            # Test reward properties
            @test !isempty(rewards)
            @test !any(isnan.(rewards))
            @test !any(isinf.(rewards))

            # Test truncation behavior
            if !isempty(terminations)
                @test all(rewards[terminations] .≈ -2.0)  # All truncated steps should have penalty reward
            end
        end

        @testset "Policy Run Data Collection" begin
            env = RDEEnv(;
                dt=dt,
                params=RDEParam(tmax=tmax),
                reward_type=ShockSpanReward(
                    target_shock_count=3,
                    span_scale=4.0f0,
                    shock_weight=5.0f0
                )
            )
            policy = ConstantRDEPolicy(env)

            # Run policy and collect data
            data = run_policy(policy, env)

            # Test data properties
            @test !isempty(data.rewards)
            @test !any(isnan.(data.rewards))
            @test !any(isinf.(data.rewards))
            @test length(data.action_ts) == length(data.rewards)
            @test all(diff(data.action_ts) .≈ dt)  # Time steps should be consistent
        end
    end

    @testset "TimeAggCompositeReward" begin
        # Create environment with TimeAggCompositeReward
        env = RDEEnv(;
            dt=1.0,
            reward_type=TimeAggCompositeReward(
                aggregation=TimeMin(),
                target_shock_count=3,
                lowest_action_magnitude_reward=0.5f0,
                weights=[0.25f0, 0.25f0, 0.25f0, 0.25f0]
            )
        )

        # Test initial reward
        _reset!(env)
        @test env.reward isa Float32
        @test !isnan(env.reward)
        @test !isinf(env.reward)

        # Test reward with no shocks
        env.state .= 1.0  # Constant state = no shocks
        set_reward!(env, env.reward_type)
        @test env.reward ≈ 0  # Should be zero with no shocks

        # Test reward with target number of shocks
        N = env.prob.params.N
        env.state[1:N] .= 1.0
        env.state[N÷4] = 2.0  # First shock
        env.state[N÷2] = 2.0  # Second shock
        env.state[3N÷4] = 2.0  # Third shock
        set_reward!(env, env.reward_type)
        @test env.reward > 0  # Should be positive with target shocks

        # Test different aggregation methods
        for agg in [TimeMin(), TimeMax(), TimeAvg(), TimeSum(), TimeProd()]
            env.reward_type = TimeAggCompositeReward(
                aggregation=agg,
                target_shock_count=3,
                lowest_action_magnitude_reward=0.5f0,
                weights=[0.25f0, 0.25f0, 0.25f0, 0.25f0]
            )
            set_reward!(env, env.reward_type)
            @test !isnan(env.reward)
            @test !isinf(env.reward)
        end
    end

    @testset "TimeAggMultiSectionReward" begin
        # Create environment with TimeAggMultiSectionReward
        env = RDEEnv(;
            dt=1.0,
            reward_type=TimeAggMultiSectionReward(
                aggregation=TimeMin(),
                n_sections=4,
                target_shock_count=3,
                lowest_action_magnitude_reward=0.5f0,
                weights=[1f0, 1f0, 5f0, 1f0]
            )
        )

        # Test initial reward
        _reset!(env)
        @test env.reward isa Vector{Float32}
        @test length(env.reward) == 4  # One reward per section
        @test !any(isnan.(env.reward))
        @test !any(isinf.(env.reward))

        # Test reward with no shocks
        env.state .= 1.0  # Constant state = no shocks
        set_reward!(env, env.reward_type)
        @test all(env.reward .< 0)  # All sections should have negative reward with no shocks

        # Test reward with target number of shocks
        N = env.prob.params.N
        env.state[1:N] .= 1.0
        env.state[N÷4] = 2.0  # First shock
        env.state[N÷2] = 2.0  # Second shock
        env.state[3N÷4] = 2.0  # Third shock
        set_reward!(env, env.reward_type)
        @test all(env.reward .> 0)  # All sections should have positive reward with target shocks

        # Test different aggregation methods
        for agg in [TimeMin(), TimeMax(), TimeAvg(), TimeSum(), TimeProd()]
            env.reward_type.aggregation = agg
            set_reward!(env, env.reward_type)
            @test !any(isnan.(env.reward))
            @test !any(isinf.(env.reward))
            @test length(env.reward) == 4  # Should maintain section count
        end
    end

    @testset "MultiplicativeReward" begin
        @testset "Scalar rewards multiplication" begin
            # Create two scalar rewards
            reward1 = ShockSpanReward(
                target_shock_count=3,
                span_scale=4.0f0,
                shock_weight=0.5f0
            )

            reward2 = TimeDiffNormReward(
                threshold=10.0f0,
                threshold_reward=0.1f0
            )

            # Create environment with MultiplicativeReward
            env = RDEEnv(;
                dt=1.0,
                reward_type=MultiplicativeReward(reward1, reward2)
            )

            # Test initial reward
            _reset!(env)
            @test env.reward isa Float32
            @test !isnan(env.reward)
            @test !isinf(env.reward)

            # Compare with individual rewards
            r1 = compute_reward(env, reward1)
            r2 = compute_reward(env, reward2)
            @test env.reward ≈ r1 * r2

            # Test with different state
            N = env.prob.params.N
            env.state[1:N] .= 1.0
            env.state[N÷4] = 2.0  # First shock
            env.state[N÷2] = 2.0  # Second shock
            env.state[3N÷4] = 2.0  # Third shock

            set_reward!(env, env.reward_type)
            r1 = compute_reward(env, reward1)
            r2 = compute_reward(env, reward2)
            @test env.reward ≈ r1 * r2
        end

        @testset "Vector rewards multiplication" begin
            # Create two vector rewards
            reward1 = TimeAggMultiSectionReward(
                aggregation=TimeMin(),
                n_sections=4,
                target_shock_count=3,
                lowest_action_magnitude_reward=0.5f0,
                weights=[1f0, 1f0, 5f0, 1f0]
            )

            reward2 = MultiSectionReward(
                n_sections=4,
                target_shock_count=3,
                lowest_action_magnitude_reward=0.8f0
            )

            # Create environment with MultiplicativeReward
            env = RDEEnv(;
                dt=1.0,
                reward_type=MultiplicativeReward(reward1, reward2)
            )

            # Test initial reward
            _reset!(env)
            @test env.reward isa Vector{Float32}
            @test length(env.reward) == 4  # Should maintain section count
            @test !any(isnan.(env.reward))
            @test !any(isinf.(env.reward))

            # Compare with individual rewards
            r1 = compute_reward(env, reward1)
            r2 = compute_reward(env, reward2)
            @test all(env.reward .≈ r1 .* r2)

            # Test with different state
            N = env.prob.params.N
            env.state[1:N] .= 1.0
            env.state[N÷4] = 2.0  # First shock
            env.state[N÷2] = 2.0  # Second shock
            env.state[3N÷4] = 2.0  # Third shock

            set_reward!(env, env.reward_type)
            r1 = compute_reward(env, reward1)
            r2 = compute_reward(env, reward2)
            @test all(env.reward .≈ r1 .* r2)
        end

        @testset "Mixed scalar and vector rewards" begin
            # Create a scalar reward
            scalar_reward = ShockSpanReward(
                target_shock_count=3,
                span_scale=4.0f0,
                shock_weight=0.5f0
            )

            # Create a vector reward
            vector_reward = MultiSectionReward(
                n_sections=4,
                target_shock_count=3,
                lowest_action_magnitude_reward=0.8f0
            )

            # Create environment with MultiplicativeReward
            env = RDEEnv(;
                dt=1.0,
                reward_type=MultiplicativeReward(scalar_reward, vector_reward)
            )

            # Test initial reward
            _reset!(env)
            @test env.reward isa Vector{Float32}
            @test length(env.reward) == 4  # Should maintain section count from vector reward
            @test !any(isnan.(env.reward))
            @test !any(isinf.(env.reward))

            # Compare with individual rewards
            r1 = compute_reward(env, scalar_reward)
            r2 = compute_reward(env, vector_reward)
            @test all(env.reward .≈ r1 .* r2)

            # Test scalar reward first vs. vector reward first
            env2 = RDEEnv(;
                dt=1.0,
                reward_type=MultiplicativeReward(vector_reward, scalar_reward)
            )
            _reset!(env2)
            set_reward!(env2, env2.reward_type)
            @test all(env2.reward .≈ env.reward)  # Order shouldn't matter

            # Test with multiple scalar rewards
            scalar_reward2 = TimeDiffNormReward(threshold=10.0f0)
            env3 = RDEEnv(;
                dt=1.0,
                reward_type=MultiplicativeReward(scalar_reward, vector_reward, scalar_reward2)
            )
            _reset!(env3)
            set_reward!(env3, env3.reward_type)
            r3 = compute_reward(env, scalar_reward2)
            @test all(env3.reward .≈ r1 .* r2 .* r3)
        end

        @testset "Policy with MultiplicativeReward" begin
            # Test parameters
            tmax = 1.0
            dt = 0.1
            n_steps = Int(tmax / dt)

            # Create a multiplicative reward with mixed types
            scalar_reward = ShockSpanReward(target_shock_count=3)
            vector_reward = MultiSectionReward(n_sections=4)

            env = RDEEnv(;
                dt=dt,
                params=RDEParam(tmax=tmax),
                reward_type=MultiplicativeReward(scalar_reward, vector_reward)
            )
            policy = ConstantRDEPolicy(env)

            # Run policy and collect rewards
            _reset!(env)
            for _ in 1:n_steps
                action = _predict_action(policy, _observe(env))
                _act!(env, action)
                @test !any(isnan.(env.reward))
                @test !any(isinf.(env.reward))
            end

            # Run policy and collect data
            data = run_policy(policy, env)

            # Test data properties
            @test !isempty(data.rewards)
            @test all(x -> x isa Vector{Float32}, data.rewards)
            @test all(x -> !any(isnan.(x)), data.rewards)
            @test all(x -> !any(isinf.(x)), data.rewards)
            @test length(data.action_ts) == length(data.rewards)
        end
    end
end