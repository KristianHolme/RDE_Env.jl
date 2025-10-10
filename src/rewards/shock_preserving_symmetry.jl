function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::ShockPreservingSymmetryReward) where {T, A, O, R, V, OBS}
    target_shock_count = rt.target_shock_count
    N = env.prob.params.N
    u = env.state[1:N]

    errs = zeros(T, target_shock_count - 1)
    cache = rt.cache
    shift_steps = N ÷ target_shock_count
    for i in 1:(target_shock_count - 1)
        cache .= u
        circshift!(cache, u, -shift_steps * i)
        errs[i] = RDE.turbo_diff_norm(u, cache) / sqrt(T(N))
    end
    maxerr = RDE.turbo_maximum(errs)
    return T(1) - (maxerr - T(0.1)) / T(0.5)
end

reward_value_type(::Type{T}, ::ShockPreservingSymmetryReward) where {T} = T
