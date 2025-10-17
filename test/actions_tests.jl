@testitem "linear_action_to_control composite" begin
    # Test with no momentum (α=0): should equal linear_control_target
    action = 1.0f0
    c_prev = 0.6f0
    c_max = 1.2f0
    α = 0.0f0
    @test RDE_Env.linear_action_to_control(action, c_prev, c_max, α) ≈ 1.2f0

    action = -1.0f0
    @test RDE_Env.linear_action_to_control(action, c_prev, c_max, α) ≈ 0.0f0

    # Test with full momentum (α=1): should equal c_prev
    α = 1.0f0
    action = 0.5f0
    @test RDE_Env.linear_action_to_control(action, c_prev, c_max, α) ≈ c_prev

    # Test with intermediate momentum
    α = 0.5f0
    action = 0.0f0
    c_prev = 0.3f0
    c_max = 1.2f0
    target = RDE_Env.linear_control_target(action, c_max)  # 0.6
    expected = 0.5f0 * c_prev + 0.5f0 * target  # 0.5*0.3 + 0.5*0.6 = 0.45
    @test RDE_Env.linear_action_to_control(action, c_prev, c_max, α) ≈ expected

    # Test another case
    α = 0.9f0
    action = 0.5f0
    c_prev = 0.2f0
    c_max = 1.0f0
    target = RDE_Env.linear_control_target(action, c_max)  # 0.75
    expected = 0.9f0 * c_prev + 0.1f0 * target
    @test RDE_Env.linear_action_to_control(action, c_prev, c_max, α) ≈ expected
end

@testitem "LinearScalarPressureAction comprehensive" begin
    using RDE
    using Accessors

    # Create minimal environment
    params = RDEParam(N = 512, tmax = 0.01)
    action_type = LinearScalarPressureAction(α = 0.5f0)
    env = RDEEnv(params, action_type = action_type)
    _reset!(env)

    # Store initial values
    u_p_initial = copy(env.prob.method.cache.u_p_current)
    s_initial = copy(env.prob.method.cache.s_current)

    # Test basic action application
    action = 0.5f0
    RDE_Env.apply_action!(env, action)

    # Verify control is updated correctly
    expected = RDE_Env.linear_action_to_control(action, u_p_initial[1], env.u_pmax, RDE_Env.momentum(env.action_type))
    @test all(env.prob.method.cache.u_p_current .≈ expected)

    # Verify state preservation: u_p_previous stores old state
    @test all(env.prob.method.cache.u_p_previous .≈ u_p_initial)

    # Verify s_current unchanged
    @test all(env.prob.method.cache.s_current .≈ s_initial)

    # Verify s_previous was updated
    @test all(env.prob.method.cache.s_previous .≈ s_initial)

    # Verify cache.action is updated correctly
    @test all(env.cache.action[:, 1] .≈ 0.0f0)  # s channel zero
    @test all(env.cache.action[:, 2] .≈ action)  # u_p channel stores action

    # Test with different action
    u_p_prev = env.prob.method.cache.u_p_current[1]
    action2 = -0.8f0
    RDE_Env.apply_action!(env, action2)
    expected2 = RDE_Env.linear_action_to_control(action2, u_p_prev, env.u_pmax, RDE_Env.momentum(env.action_type))
    @test all(env.prob.method.cache.u_p_current .≈ expected2)

    # Test array input (convenience interface)
    _reset!(env)
    u_p_reset = env.prob.method.cache.u_p_current[1]
    action3 = 0.4f0
    RDE_Env.apply_action!(env, [action3])
    expected3 = RDE_Env.linear_action_to_control(action3, u_p_reset, env.u_pmax, RDE_Env.momentum(env.action_type))
    @test all(env.prob.method.cache.u_p_current .≈ expected3)

    # Test momentum extremes: α=0.0 (no momentum - immediate)
    env_immediate = RDEEnv(params, action_type = LinearScalarPressureAction())
    _reset!(env_immediate)
    @reset env_immediate.action_type.α = 0.0f0
    env_immediate.prob.method.cache.u_p_current .= 0.2f0
    action_m = 0.5f0
    RDE_Env.apply_action!(env_immediate, action_m)
    expected_immediate = RDE_Env.linear_control_target(action_m, env_immediate.u_pmax)
    @test env_immediate.prob.method.cache.u_p_current[1] ≈ expected_immediate

    # Test momentum extremes: α=1.0 (full momentum - frozen)
    env_frozen = RDEEnv(params, action_type = LinearScalarPressureAction())
    _reset!(env_frozen)
    @reset env_frozen.action_type.α = 1.0f0
    initial_value = 0.3f0
    env_frozen.prob.method.cache.u_p_current .= initial_value
    RDE_Env.apply_action!(env_frozen, 0.5f0)
    @test env_frozen.prob.method.cache.u_p_current[1] ≈ initial_value

    # Test varying u_pmax
    for u_pmax in [1.0f0, 2.0f0, 0.5f0]
        env_umax = RDEEnv(params, action_type = LinearScalarPressureAction(), u_pmax = u_pmax)
        _reset!(env_umax)
        @reset env_umax.action_type.α = 0.0f0
        RDE_Env.apply_action!(env_umax, 0.0f0)
        @test env_umax.prob.method.cache.u_p_current[1] ≈ 0.5f0 * u_pmax
    end

    # Test at boundaries (should not throw)
    RDE_Env.apply_action!(env, 1.0f0)
    RDE_Env.apply_action!(env, -1.0f0)
    @test true
end

@testitem "LinearScalarPressureAction bounds" begin
    using RDE

    params = RDEParam(N = 512, tmax = 0.01)
    env = RDEEnv(params, action_type = LinearScalarPressureAction())
    _reset!(env)

    # Test out-of-bounds warning
    @test_logs (:warn,) RDE_Env.apply_action!(env, 1.5f0)
    @test_logs (:warn,) RDE_Env.apply_action!(env, -1.1f0)

    # Test boundary values don't warn
    @test_logs RDE_Env.apply_action!(env, 1.0f0)
    @test_logs RDE_Env.apply_action!(env, -1.0f0)
    @test_logs RDE_Env.apply_action!(env, 0.0f0)
end

@testitem "LinearVectorPressureAction comprehensive" begin
    using RDE

    # Test single section
    params = RDEParam(N = 512, tmax = 0.01)
    env1 = RDEEnv(params, action_type = LinearVectorPressureAction(n_sections = 1))
    _reset!(env1)
    action1 = [0.5f0]
    u_p_initial1 = env1.prob.method.cache.u_p_current[1]
    RDE_Env.apply_action!(env1, action1)
    expected1 = RDE_Env.linear_action_to_control(0.5f0, u_p_initial1, env1.u_pmax, RDE_Env.momentum(env1.action_type))
    @test all(env1.prob.method.cache.u_p_current .≈ expected1)

    # Test uniform actions across multiple sections
    env_uniform = RDEEnv(params, action_type = LinearVectorPressureAction(n_sections = 4))
    _reset!(env_uniform)
    action_uniform = [0.5f0, 0.5f0, 0.5f0, 0.5f0]
    points_per_section = 512 ÷ 4
    u_p_initial_uniform = [env_uniform.prob.method.cache.u_p_current[1 + (i - 1) * points_per_section] for i in 1:4]
    RDE_Env.apply_action!(env_uniform, action_uniform)
    for i in 1:4
        start_idx = (i - 1) * points_per_section + 1
        end_idx = i * points_per_section
        expected = RDE_Env.linear_action_to_control(0.5f0, u_p_initial_uniform[i], env_uniform.u_pmax, RDE_Env.momentum(env_uniform.action_type))
        @test all(env_uniform.prob.method.cache.u_p_current[start_idx:end_idx] .≈ expected)
    end

    # Test independent sections with different actions
    env_independent = RDEEnv(params, action_type = LinearVectorPressureAction(n_sections = 4))
    _reset!(env_independent)
    action_independent = [-1.0f0, -0.3f0, 0.3f0, 1.0f0]
    u_p_initial_independent = [env_independent.prob.method.cache.u_p_current[1 + (i - 1) * points_per_section] for i in 1:4]
    RDE_Env.apply_action!(env_independent, action_independent)
    for i in 1:4
        start_idx = (i - 1) * points_per_section + 1
        end_idx = i * points_per_section
        expected = RDE_Env.linear_action_to_control(action_independent[i], u_p_initial_independent[i], env_independent.u_pmax, RDE_Env.momentum(env_independent.action_type))
        @test all(env_independent.prob.method.cache.u_p_current[start_idx:end_idx] .≈ expected)
    end

    # Test cache per section
    env_cache = RDEEnv(params, action_type = LinearVectorPressureAction(n_sections = 2))
    _reset!(env_cache)
    action_cache = [-0.5f0, 0.8f0]
    RDE_Env.apply_action!(env_cache, action_cache)
    @test all(env_cache.cache.action[1:256, 2] .≈ -0.5f0)
    @test all(env_cache.cache.action[257:512, 2] .≈ 0.8f0)
end

@testitem "LinearVectorPressureAction section boundaries" begin
    using RDE

    # Test various N and n_sections combinations
    test_cases = [
        (N = 512, n_sections = 2),
        (N = 512, n_sections = 8),
        (N = 1024, n_sections = 4),
    ]

    for (N, n_sections) in test_cases
        params = RDEParam(N = N, tmax = 0.01)
        env = RDEEnv(params, action_type = LinearVectorPressureAction(n_sections = n_sections))
        _reset!(env)

        # Create distinct action for each section
        action = Float32[i / n_sections for i in 1:n_sections]
        points_per_section = N ÷ n_sections

        # Store initial values
        u_p_initial = [env.prob.method.cache.u_p_current[1 + (i - 1) * points_per_section] for i in 1:n_sections]

        RDE_Env.apply_action!(env, action)

        # Sample one point from each section and verify
        for i in 1:n_sections
            sample_idx = (i - 1) * points_per_section + 1
            expected = RDE_Env.linear_action_to_control(action[i], u_p_initial[i], env.u_pmax, RDE_Env.momentum(env.action_type))
            @test env.prob.method.cache.u_p_current[sample_idx] ≈ expected
        end
    end
end

@testitem "LinearVectorPressureAction validation" begin
    using RDE

    params = RDEParam(N = 512, tmax = 0.01)
    env = RDEEnv(params, action_type = LinearVectorPressureAction(n_sections = 4))
    _reset!(env)

    # Test assertion for wrong action length
    @test_throws AssertionError RDE_Env.apply_action!(env, [0.5f0, 0.5f0])
    @test_throws AssertionError RDE_Env.apply_action!(env, [0.1f0, 0.2f0, 0.3f0, 0.4f0, 0.5f0])
end

@testitem "Linear actions type stability" begin
    using RDE
    using Accessors
    # Test with Float32
    params32 = RDEParam{Float32}(tmax = 0.01f0)
    env32 = RDEEnv(params32, action_type = LinearScalarPressureAction())
    _reset!(env32)

    action32 = 0.5f0
    RDE_Env.apply_action!(env32, action32)
    @test eltype(env32.prob.method.cache.u_p_current) == Float32
    @test env32.prob.method.cache.u_p_current[1] isa Float32

    # Test with Float64
    params64 = RDEParam{Float64}(tmax = 0.01)
    env64 = RDEEnv(params64, action_type = LinearScalarPressureAction(α = 0.0))
    _reset!(env64)

    action64 = 0.5
    RDE_Env.apply_action!(env64, action64)
    @test eltype(env64.prob.method.cache.u_p_current) == Float64
    @test env64.prob.method.cache.u_p_current[1] isa Float64

    # Test vector action with Float32
    env32_vec = RDEEnv(params32, action_type = LinearVectorPressureAction(n_sections = 2))
    _reset!(env32_vec)
    RDE_Env.apply_action!(env32_vec, [0.3f0, 0.7f0])
    @test eltype(env32_vec.prob.method.cache.u_p_current) == Float32
end

@testitem "Linear actions convergence" begin
    using RDE
    using Accessors

    params = RDEParam(N = 512, tmax = 0.01)
    env = RDEEnv(params, action_type = LinearScalarPressureAction())
    _reset!(env)

    # Set high momentum
    @reset env.action_type.α = 0.9f0
    env.prob.method.cache.u_p_current .= 0.0f0

    # Apply action=1.0 repeatedly
    action = 1.0f0
    target = RDE_Env.linear_control_target(action, env.u_pmax)

    # After many iterations, should converge toward target
    for i in 1:50
        RDE_Env.apply_action!(env, action)
    end

    # Should be close to target (with high momentum, not exact)
    @test env.prob.method.cache.u_p_current[1] > 0.99f0 * target

    # Test convergence from opposite direction
    env2 = RDEEnv(params, action_type = LinearScalarPressureAction())
    _reset!(env2)
    @reset env2.action_type.α = 0.9f0
    env2.prob.method.cache.u_p_current .= env2.u_pmax  # Start at max

    action2 = -1.0f0  # Target is 0
    for i in 1:50
        RDE_Env.apply_action!(env2, action2)
    end

    @test env2.prob.method.cache.u_p_current[1] < 0.01f0 * env2.u_pmax
end
