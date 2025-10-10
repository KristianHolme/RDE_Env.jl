function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::ShockSpanReward) where {T, A, O, R, V, OBS}
    target_shock_count = rt.target_shock_count
    max_span = rt.span_scale
    λ = rt.shock_weight

    u, = RDE.split_sol_view(env.state)
    dx = env.prob.params.L / env.prob.params.N
    shocks = T(RDE.count_shocks(u, dx))
    u_min, u_max = RDE.turbo_extrema(u)
    span = u_max - u_min
    span_reward = span / max_span
    if shocks >= target_shock_count
        shock_reward = one(T)
    elseif shocks > 0
        shock_reward = shocks / (2 * target_shock_count)
    else
        shock_reward = T(-1)
    end

    return λ * shock_reward + (1 - λ) * span_reward
end

reward_value_type(::Type{T}, ::ShockSpanReward) where {T} = T
