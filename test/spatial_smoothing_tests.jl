@testitem "RDEEnv applies spatial kernel width" begin
    using RDE
    params = RDEParam(N = 32, tmax = 0.1f0)
    env = RDEEnv(; params, spatial_kernel_width = 8)
    cache = env.prob.method.cache
    @test cache.spatial_kernel_width == 9
    @test length(cache.spatial_kernel) == 9
end

@testitem "RDEEnv RDE_RHS! uses smoothed controls" begin
    using RDE
    params = RDEParam{Float32}(N = 64, tmax = 1.0f0)
    env = RDEEnv(; params, spatial_kernel_width = 9, τ_smooth = 1.0f0)
    cache = env.prob.method.cache

    cache.control_time = 0.0f0
    cache.u_p_current .= vcat(fill(0.0f0, params.N ÷ 2), fill(1.0f0, params.N ÷ 2))
    cache.u_p_previous .= cache.u_p_current
    cache.s_current .= 1.0f0
    cache.s_previous .= cache.s_current

    uλ = vcat(env.prob.u0, env.prob.λ0)
    duλ = zeros(Float32, length(uλ))
    RDE.RDE_RHS!(duλ, uλ, env.prob, 1.0f0)

    boundary_index = params.N ÷ 2 + 1
    jump_raw = abs(cache.u_p_current[boundary_index] - cache.u_p_current[boundary_index - 1])
    jump_smoothed = abs(cache.u_p_t_shifted[boundary_index] - cache.u_p_t_shifted[boundary_index - 1])
    @test jump_smoothed < jump_raw
end
