@testitem "direct action" begin
    actions = [0.0f0, 1.0f0, -1.0f0]
    known_targets = [0.6f0, 1.2f0, 0.0f0]
    u_p_max = 1.2f0
    c_prev = rand(Float32)
    for i in eachindex(actions)
        action = actions[i]
        known_target = known_targets[i]
        target = RDE_Env.direct_action_to_control(action, c_prev, u_p_max, 0.0f0)
        @test target ≈ known_target
    end
end

@testitem "direct_control_target mapping" begin
    # Test boundary values
    @test RDE_Env.direct_control_target(-1.0f0, 1.2f0) ≈ 0.0f0
    @test RDE_Env.direct_control_target(0.0f0, 1.2f0) ≈ 0.6f0
    @test RDE_Env.direct_control_target(1.0f0, 1.2f0) ≈ 1.2f0

    # Test intermediate values
    @test RDE_Env.direct_control_target(0.5f0, 1.2f0) ≈ 0.9f0
    @test RDE_Env.direct_control_target(-0.5f0, 1.2f0) ≈ 0.3f0

    # Test with different c_max
    @test RDE_Env.direct_control_target(0.0f0, 2.0f0) ≈ 1.0f0
    @test RDE_Env.direct_control_target(1.0f0, 0.5f0) ≈ 0.5f0

    # Test linearity: f(0.5*(a+b)) ≈ 0.5*(f(a) + f(b))
    a, b = -0.3f0, 0.7f0
    c_max = 1.0f0
    mid = 0.5f0 * (a + b)
    @test RDE_Env.direct_control_target(mid, c_max) ≈
        0.5f0 * (RDE_Env.direct_control_target(a, c_max) + RDE_Env.direct_control_target(b, c_max))
end

@testitem "momentum_target blending" begin
    # Test momentum extremes
    @test RDE_Env.momentum_target(0.8f0, 0.4f0, 0.0f0) ≈ 0.8f0  # No momentum
    @test RDE_Env.momentum_target(0.8f0, 0.4f0, 1.0f0) ≈ 0.4f0  # Full momentum

    # Test intermediate momentum
    @test RDE_Env.momentum_target(0.8f0, 0.4f0, 0.5f0) ≈ 0.6f0  # Half momentum
    @test RDE_Env.momentum_target(0.8f0, 0.4f0, 0.9f0) ≈ 0.44f0  # High momentum

    # Test another case
    @test RDE_Env.momentum_target(1.0f0, 0.0f0, 0.9f0) ≈ 0.1f0

    # Verify weighted average formula: α * prev + (1-α) * target
    control_target = 0.6f0
    previous = 0.2f0
    α = 0.7f0
    expected = α * previous + (1.0f0 - α) * control_target
    @test RDE_Env.momentum_target(control_target, previous, α) ≈ expected
end

@testitem "direct_action_to_control composite" begin
    # Test with no momentum (α=0): should equal direct_control_target
    action = 1.0f0
    c_prev = 0.6f0
    c_max = 1.2f0
    α = 0.0f0
    @test RDE_Env.direct_action_to_control(action, c_prev, c_max, α) ≈ 1.2f0

    action = -1.0f0
    @test RDE_Env.direct_action_to_control(action, c_prev, c_max, α) ≈ 0.0f0

    # Test with full momentum (α=1): should equal c_prev
    α = 1.0f0
    action = 0.5f0
    @test RDE_Env.direct_action_to_control(action, c_prev, c_max, α) ≈ c_prev

    # Test with intermediate momentum
    α = 0.5f0
    action = 0.0f0
    c_prev = 0.3f0
    c_max = 1.2f0
    target = RDE_Env.direct_control_target(action, c_max)  # 0.6
    expected = 0.5f0 * c_prev + 0.5f0 * target  # 0.5*0.3 + 0.5*0.6 = 0.45
    @test RDE_Env.direct_action_to_control(action, c_prev, c_max, α) ≈ expected

    # Test another case
    α = 0.9f0
    action = 0.5f0
    c_prev = 0.2f0
    c_max = 1.0f0
    target = RDE_Env.direct_control_target(action, c_max)  # 0.75
    expected = 0.9f0 * c_prev + 0.1f0 * target
    @test RDE_Env.direct_action_to_control(action, c_prev, c_max, α) ≈ expected
end

@testitem "DirectScalarPressureAction basic" begin
    using RDE

    # Create minimal environment
    params = RDEParam(N = 512, tmax = 0.01)
    env = RDEEnv(params, action_type = DirectScalarPressureAction())
    _reset!(env)

    # Store initial value
    u_p_initial = env.prob.method.cache.u_p_current[1]

    # Apply action
    action = 0.5f0
    RDE_Env.apply_action!(env, action)

    # Verify control is updated correctly
    expected = RDE_Env.direct_action_to_control(action, u_p_initial, env.u_pmax, env.α)
    @test all(env.prob.method.cache.u_p_current .≈ expected)

    # Test with different action
    u_p_prev = env.prob.method.cache.u_p_current[1]
    action2 = -0.8f0
    RDE_Env.apply_action!(env, action2)
    expected2 = RDE_Env.direct_action_to_control(action2, u_p_prev, env.u_pmax, env.α)
    @test all(env.prob.method.cache.u_p_current .≈ expected2)

    # Test at boundaries
    RDE_Env.apply_action!(env, 1.0f0)
    RDE_Env.apply_action!(env, -1.0f0)
    @test true  # Should not throw
end

@testitem "DirectScalarPressureAction state preservation" begin
    using RDE

    params = RDEParam(N = 512, tmax = 0.01)
    env = RDEEnv(params, action_type = DirectScalarPressureAction())
    _reset!(env)

    # Store references
    u_p_initial = copy(env.prob.method.cache.u_p_current)
    s_initial = copy(env.prob.method.cache.s_current)

    # Apply action
    action = 0.3f0
    RDE_Env.apply_action!(env, action)

    # Verify u_p_previous stores old state
    @test all(env.prob.method.cache.u_p_previous .≈ u_p_initial)

    # Verify s_current unchanged
    @test all(env.prob.method.cache.s_current .≈ s_initial)

    # Verify s_previous was updated (copyto! called even though values same)
    @test all(env.prob.method.cache.s_previous .≈ s_initial)
end

@testitem "DirectScalarPressureAction cache" begin
    using RDE

    params = RDEParam(N = 512, tmax = 0.01)
    env = RDEEnv(params, action_type = DirectScalarPressureAction())
    _reset!(env)

    # Apply action
    action = 0.3f0
    RDE_Env.apply_action!(env, action)

    # Verify cache.action[:, 1] is zero (s channel)
    @test all(env.cache.action[:, 1] .≈ 0.0f0)

    # Verify cache.action[:, 2] stores raw action (u_p channel)
    @test all(env.cache.action[:, 2] .≈ action)
end

@testitem "DirectScalarPressureAction bounds" begin
    using RDE

    params = RDEParam(N = 512, tmax = 0.01)
    env = RDEEnv(params, action_type = DirectScalarPressureAction())
    _reset!(env)

    # Test out-of-bounds warning
    @test_logs (:warn,) RDE_Env.apply_action!(env, 1.5f0)
    @test_logs (:warn,) RDE_Env.apply_action!(env, -1.1f0)

    # Test boundary values don't warn
    @test_logs RDE_Env.apply_action!(env, 1.0f0)
    @test_logs RDE_Env.apply_action!(env, -1.0f0)
    @test_logs RDE_Env.apply_action!(env, 0.0f0)
end

@testitem "DirectScalarPressureAction array input" begin
    using RDE

    params = RDEParam(N = 512, tmax = 0.01)
    env = RDEEnv(params, action_type = DirectScalarPressureAction())
    _reset!(env)

    # Test with scalar
    action = 0.4f0
    RDE_Env.apply_action!(env, action)
    result_scalar = copy(env.prob.method.cache.u_p_current)

    # Reset and test with array
    _reset!(env)
    RDE_Env.apply_action!(env, [action])
    result_array = copy(env.prob.method.cache.u_p_current)

    # Should produce same result
    @test all(result_scalar .≈ result_array)
end

@testitem "DirectVectorPressureAction single section" begin
    using RDE

    params = RDEParam(N = 512, tmax = 0.01)
    env = RDEEnv(params, action_type = DirectVectorPressureAction(n_sections = 1))
    _reset!(env)

    # Apply single-element action
    action = [0.5f0]
    u_p_initial = env.prob.method.cache.u_p_current[1]
    RDE_Env.apply_action!(env, action)

    # All points should have same value
    expected = RDE_Env.direct_action_to_control(0.5f0, u_p_initial, env.u_pmax, env.α)
    @test all(env.prob.method.cache.u_p_current .≈ expected)
end

@testitem "DirectVectorPressureAction uniform actions" begin
    using RDE

    params = RDEParam(N = 512, tmax = 0.01)
    env = RDEEnv(params, action_type = DirectVectorPressureAction(n_sections = 4))
    _reset!(env)

    # Apply uniform action across all sections
    action = [0.5f0, 0.5f0, 0.5f0, 0.5f0]
    points_per_section = 512 ÷ 4

    # Store initial values for each section
    u_p_initial = [env.prob.method.cache.u_p_current[1 + (i - 1) * points_per_section] for i in 1:4]

    RDE_Env.apply_action!(env, action)

    # All sections should have same value (since action is uniform and initial might be too)
    # But each section calculated independently
    for i in 1:4
        start_idx = (i - 1) * points_per_section + 1
        end_idx = i * points_per_section
        expected = RDE_Env.direct_action_to_control(0.5f0, u_p_initial[i], env.u_pmax, env.α)
        @test all(env.prob.method.cache.u_p_current[start_idx:end_idx] .≈ expected)
    end
end

@testitem "DirectVectorPressureAction independent sections" begin
    using RDE

    params = RDEParam(N = 512, tmax = 0.01)
    env = RDEEnv(params, action_type = DirectVectorPressureAction(n_sections = 4))
    _reset!(env)

    # Apply different actions to each section
    action = [-1.0f0, -0.3f0, 0.3f0, 1.0f0]
    points_per_section = 512 ÷ 4

    # Store initial values for each section
    u_p_initial = [env.prob.method.cache.u_p_current[1 + (i - 1) * points_per_section] for i in 1:4]

    RDE_Env.apply_action!(env, action)

    # Verify each section has correct independent value
    for i in 1:4
        start_idx = (i - 1) * points_per_section + 1
        end_idx = i * points_per_section
        expected = RDE_Env.direct_action_to_control(action[i], u_p_initial[i], env.u_pmax, env.α)
        @test all(env.prob.method.cache.u_p_current[start_idx:end_idx] .≈ expected)
    end
end

@testitem "DirectVectorPressureAction section boundaries" begin
    using RDE

    # Test various N and n_sections combinations
    test_cases = [
        (N = 512, n_sections = 2),
        (N = 512, n_sections = 8),
        (N = 1024, n_sections = 4),
    ]

    for (N, n_sections) in test_cases
        params = RDEParam(N = N, tmax = 0.01)
        env = RDEEnv(params, action_type = DirectVectorPressureAction(n_sections = n_sections))
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
            expected = RDE_Env.direct_action_to_control(action[i], u_p_initial[i], env.u_pmax, env.α)
            @test env.prob.method.cache.u_p_current[sample_idx] ≈ expected
        end
    end
end

@testitem "DirectVectorPressureAction cache per section" begin
    using RDE

    params = RDEParam(N = 512, tmax = 0.01)
    env = RDEEnv(params, action_type = DirectVectorPressureAction(n_sections = 2))
    _reset!(env)

    # Apply different actions to each section
    action = [-0.5f0, 0.8f0]
    RDE_Env.apply_action!(env, action)

    # Verify cache.action filled correctly per section
    @test all(env.cache.action[1:256, 2] .≈ -0.5f0)
    @test all(env.cache.action[257:512, 2] .≈ 0.8f0)
end

@testitem "DirectVectorPressureAction validation" begin
    using RDE

    params = RDEParam(N = 512, tmax = 0.01)
    env = RDEEnv(params, action_type = DirectVectorPressureAction(n_sections = 4))
    _reset!(env)

    # Test assertion for wrong action length
    @test_throws AssertionError RDE_Env.apply_action!(env, [0.5f0, 0.5f0])
    @test_throws AssertionError RDE_Env.apply_action!(env, [0.1f0, 0.2f0, 0.3f0, 0.4f0, 0.5f0])
end

@testitem "Direct action independence" begin
    using RDE

    # Create two environments with different initial controls
    params = RDEParam(N = 512, tmax = 0.01)
    env1 = RDEEnv(params, action_type = DirectScalarPressureAction())
    env2 = RDEEnv(params, action_type = DirectScalarPressureAction())

    _reset!(env1)
    _reset!(env2)

    # Set different initial u_p_current values
    env1.prob.method.cache.u_p_current .= 0.2f0
    env2.prob.method.cache.u_p_current .= 0.8f0

    # Apply same action with no momentum
    action = 0.5f0
    old_α1 = env1.α
    old_α2 = env2.α
    env1.α = 0.0f0
    env2.α = 0.0f0

    RDE_Env.apply_action!(env1, action)
    RDE_Env.apply_action!(env2, action)

    # Both should produce same result (independence from c_prev)
    @test all(env1.prob.method.cache.u_p_current .≈ env2.prob.method.cache.u_p_current)

    # Restore for another test with momentum
    _reset!(env1)
    _reset!(env2)
    env1.prob.method.cache.u_p_current .= 0.2f0
    env2.prob.method.cache.u_p_current .= 0.8f0
    env1.α = 0.5f0
    env2.α = 0.5f0

    RDE_Env.apply_action!(env1, action)
    RDE_Env.apply_action!(env2, action)

    # With momentum they differ (because they're moving toward same target from different starts)
    @test !all(env1.prob.method.cache.u_p_current .≈ env2.prob.method.cache.u_p_current)

    # But the target they're moving toward is the same
    target = RDE_Env.direct_control_target(action, env1.u_pmax)
    # env1: 0.5 * 0.2 + 0.5 * target
    # env2: 0.5 * 0.8 + 0.5 * target
    # Both should be equidistant from target
    @test abs(env1.prob.method.cache.u_p_current[1] - target) ≈
        abs(env2.prob.method.cache.u_p_current[1] - target) - abs(0.2f0 - 0.8f0) * 0.5f0

    # Restore α values
    env1.α = old_α1
    env2.α = old_α2
end

@testitem "Direct vs non-Direct contrast" begin
    using RDE

    params = RDEParam(N = 512, tmax = 0.01)
    env_direct = RDEEnv(params, action_type = DirectScalarPressureAction())
    env_standard = RDEEnv(params, action_type = ScalarPressureAction())

    _reset!(env_direct)
    _reset!(env_standard)

    # Set different initial u_p_current values
    env_direct.prob.method.cache.u_p_current .= 0.2f0
    env_standard.prob.method.cache.u_p_current .= 0.2f0

    # Store second set with different values
    env_direct2 = RDEEnv(params, action_type = DirectScalarPressureAction())
    env_standard2 = RDEEnv(params, action_type = ScalarPressureAction())
    _reset!(env_direct2)
    _reset!(env_standard2)
    env_direct2.prob.method.cache.u_p_current .= 0.8f0
    env_standard2.prob.method.cache.u_p_current .= 0.8f0

    # Apply same action with no momentum
    action = 0.0f0  # Maps to midpoint
    env_direct.α = 0.0f0
    env_standard.α = 0.0f0
    env_direct2.α = 0.0f0
    env_standard2.α = 0.0f0

    RDE_Env.apply_action!(env_direct, action)
    RDE_Env.apply_action!(env_standard, action)
    RDE_Env.apply_action!(env_direct2, action)
    RDE_Env.apply_action!(env_standard2, action)

    # For Direct: both should give same result
    @test all(
        env_direct.prob.method.cache.u_p_current .≈
            env_direct2.prob.method.cache.u_p_current
    )

    # For non-Direct: should give different results
    # action=0 maps to current control, so both should stay at their values
    @test env_standard.prob.method.cache.u_p_current[1] ≈ 0.2f0
    @test env_standard2.prob.method.cache.u_p_current[1] ≈ 0.8f0

    # Direct action=0 should map to 0.5*u_pmax regardless of previous
    expected_direct = 0.5f0 * env_direct.u_pmax
    @test env_direct.prob.method.cache.u_p_current[1] ≈ expected_direct
    @test env_direct2.prob.method.cache.u_p_current[1] ≈ expected_direct
end

@testitem "Direct actions type stability" begin
    using RDE

    # Test with Float32
    params32 = RDEParam{Float32}(N = 16, tmax = 0.01f0)
    env32 = RDEEnv(params32, action_type = DirectScalarPressureAction())
    _reset!(env32)

    action32 = 0.5f0
    RDE_Env.apply_action!(env32, action32)
    @test eltype(env32.prob.method.cache.u_p_current) == Float32
    @test env32.prob.method.cache.u_p_current[1] isa Float32

    # Test with Float64
    params64 = RDEParam{Float64}(N = 16, tmax = 0.01)
    env64 = RDEEnv(params64, action_type = DirectScalarPressureAction())
    _reset!(env64)

    action64 = 0.5
    RDE_Env.apply_action!(env64, action64)
    @test eltype(env64.prob.method.cache.u_p_current) == Float64
    @test env64.prob.method.cache.u_p_current[1] isa Float64

    # Test vector action with Float32
    env32_vec = RDEEnv(params32, action_type = DirectVectorPressureAction(n_sections = 2))
    _reset!(env32_vec)
    RDE_Env.apply_action!(env32_vec, [0.3f0, 0.7f0])
    @test eltype(env32_vec.prob.method.cache.u_p_current) == Float32
end

@testitem "Direct actions momentum extremes" begin
    using RDE

    params = RDEParam(N = 512, tmax = 0.01)

    # Test α=0.0 (no momentum - immediate application)
    env_immediate = RDEEnv(params, action_type = DirectScalarPressureAction())
    _reset!(env_immediate)
    env_immediate.α = 0.0f0
    env_immediate.prob.method.cache.u_p_current .= 0.2f0

    action = 0.5f0
    RDE_Env.apply_action!(env_immediate, action)
    expected = RDE_Env.direct_control_target(action, env_immediate.u_pmax)
    @test env_immediate.prob.method.cache.u_p_current[1] ≈ expected

    # Test α=1.0 (full momentum - frozen, no change)
    env_frozen = RDEEnv(params, action_type = DirectScalarPressureAction())
    _reset!(env_frozen)
    env_frozen.α = 1.0f0
    initial_value = 0.3f0
    env_frozen.prob.method.cache.u_p_current .= initial_value

    # Apply multiple actions - should never change
    for action in [0.5f0, -0.8f0, 1.0f0, -1.0f0]
        RDE_Env.apply_action!(env_frozen, action)
        @test env_frozen.prob.method.cache.u_p_current[1] ≈ initial_value
    end
end

@testitem "Direct actions varying u_pmax" begin
    using RDE

    params = RDEParam(N = 512, tmax = 0.01)

    # Test with u_pmax = 1.0
    env1 = RDEEnv(params, action_type = DirectScalarPressureAction(), u_pmax = 1.0f0)
    _reset!(env1)
    env1.α = 0.0f0
    RDE_Env.apply_action!(env1, 0.0f0)
    @test env1.prob.method.cache.u_p_current[1] ≈ 0.5f0

    # Test with u_pmax = 2.0
    env2 = RDEEnv(params, action_type = DirectScalarPressureAction(), u_pmax = 2.0f0)
    _reset!(env2)
    env2.α = 0.0f0
    RDE_Env.apply_action!(env2, 0.0f0)
    @test env2.prob.method.cache.u_p_current[1] ≈ 1.0f0

    # Test with u_pmax = 0.5
    env3 = RDEEnv(params, action_type = DirectScalarPressureAction(), u_pmax = 0.5f0)
    _reset!(env3)
    env3.α = 0.0f0
    RDE_Env.apply_action!(env3, 1.0f0)
    @test env3.prob.method.cache.u_p_current[1] ≈ 0.5f0
end

@testitem "Direct actions convergence" begin
    using RDE

    params = RDEParam(N = 512, tmax = 0.01)
    env = RDEEnv(params, action_type = DirectScalarPressureAction())
    _reset!(env)

    # Set high momentum
    env.α = 0.9f0
    env.prob.method.cache.u_p_current .= 0.0f0

    # Apply action=1.0 repeatedly
    action = 1.0f0
    target = RDE_Env.direct_control_target(action, env.u_pmax)

    # After many iterations, should converge toward target
    for i in 1:50
        RDE_Env.apply_action!(env, action)
    end

    # Should be close to target (with high momentum, not exact)
    @test env.prob.method.cache.u_p_current[1] > 0.99f0 * target

    # Test convergence from opposite direction
    env2 = RDEEnv(params, action_type = DirectScalarPressureAction())
    _reset!(env2)
    env2.α = 0.9f0
    env2.prob.method.cache.u_p_current .= env2.u_pmax  # Start at max

    action2 = -1.0f0  # Target is 0
    for i in 1:50
        RDE_Env.apply_action!(env2, action2)
    end

    @test env2.prob.method.cache.u_p_current[1] < 0.01f0 * env2.u_pmax
end
