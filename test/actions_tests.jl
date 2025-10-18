@testitem "LinearScalarPressureAction" begin
    using RDE
    using Accessors

    params = RDEParam(N = 512, tmax = 0.01)
    env = RDEEnv(params, action_type = LinearScalarPressureAction())
    _reset!(env)

    # Test basic action application with scalar input
    u_p_initial = env.prob.method.cache.u_p_current[1]
    action = 0.5f0
    RDE_Env.apply_action!(env, action)
    expected = RDE_Env.linear_action_to_control(action, u_p_initial, env.u_pmax, RDE_Env.momentum(env.action_type))
    @test all(env.prob.method.cache.u_p_current .≈ expected)

    # Test with array input
    _reset!(env)
    RDE_Env.apply_action!(env, [0.3f0])
    @test true  # Should not throw

    # Test momentum: α=0 gives direct mapping
    env_no_momentum = RDEEnv(params, action_type = LinearScalarPressureAction())
    _reset!(env_no_momentum)
    @reset env_no_momentum.action_type.momentum = 0.0f0
    RDE_Env.apply_action!(env_no_momentum, 0.0f0)
    @test env_no_momentum.prob.method.cache.u_p_current[1] ≈ 0.5f0 * env_no_momentum.u_pmax

    # Test bounds checking
    @test_logs (:warn,) RDE_Env.apply_action!(env, 1.5f0)
    @test_logs (:warn,) RDE_Env.apply_action!(env, -1.1f0)
    @test_logs RDE_Env.apply_action!(env, 1.0f0)
    @test_logs RDE_Env.apply_action!(env, -1.0f0)
end

@testitem "LinearVectorPressureAction" begin
    using RDE

    params = RDEParam(N = 512, tmax = 0.01)

    # Test single section behaves like scalar
    env1 = RDEEnv(params, action_type = LinearVectorPressureAction(n_sections = 1))
    _reset!(env1)
    RDE_Env.apply_action!(env1, [0.5f0])
    @test true  # Should not throw

    # Test multiple independent sections
    env4 = RDEEnv(params, action_type = LinearVectorPressureAction(n_sections = 4))
    _reset!(env4)
    action = [-1.0f0, -0.3f0, 0.3f0, 1.0f0]
    points_per_section = 512 ÷ 4
    u_p_initial = [env4.prob.method.cache.u_p_current[1 + (i - 1) * points_per_section] for i in 1:4]

    RDE_Env.apply_action!(env4, action)

    # Verify each section has correct independent value
    for i in 1:4
        sample_idx = (i - 1) * points_per_section + 1
        expected = RDE_Env.linear_action_to_control(action[i], u_p_initial[i], env4.u_pmax, RDE_Env.momentum(env4.action_type))
        @test env4.prob.method.cache.u_p_current[sample_idx] ≈ expected
    end

    # Test validation for wrong action length
    @test_throws AssertionError RDE_Env.apply_action!(env4, [0.5f0, 0.5f0])
    @test_throws AssertionError RDE_Env.apply_action!(env4, [0.1f0, 0.2f0, 0.3f0, 0.4f0, 0.5f0])
end

@testitem "Linear actions type stability" begin
    using RDE

    # Test Float32
    params32 = RDEParam{Float32}(tmax = 0.01f0)
    env32 = RDEEnv(params32, action_type = LinearScalarPressureAction())
    _reset!(env32)
    RDE_Env.apply_action!(env32, 0.5f0)
    @test eltype(env32.prob.method.cache.u_p_current) == Float32

    # Test Float64
    params64 = RDEParam{Float64}(tmax = 0.01)
    env64 = RDEEnv(params64, action_type = LinearScalarPressureAction(momentum = 0.0))
    _reset!(env64)
    RDE_Env.apply_action!(env64, 0.5)
    @test eltype(env64.prob.method.cache.u_p_current) == Float64

    # Test vector action type stability
    env32_vec = RDEEnv(params32, action_type = LinearVectorPressureAction(n_sections = 2))
    _reset!(env32_vec)
    RDE_Env.apply_action!(env32_vec, [0.3f0, 0.7f0])
    @test eltype(env32_vec.prob.method.cache.u_p_current) == Float32
end
