# Shared reward helpers and utilities

function action_magnitude_factor(lowest_action_magnitude_reward::AbstractFloat, action_magnitudes)
    α = lowest_action_magnitude_reward
    action_magnitude_inv = 1 .- action_magnitudes
    return α .+ (1 - α) .* action_magnitude_inv
end

function calculate_periodicity_reward(u::AbstractVector{T}, N::Int, target_shock_count::Int, cache::AbstractVector{T}) where {T}
    inv_sqrt_N = 1 / sqrt(N)
    if target_shock_count > 1
        errs = zeros(T, target_shock_count - 1)
        shift_steps = N ÷ target_shock_count
        for i in 1:(target_shock_count - 1)
            circshift!(cache, u, -shift_steps * i)
            errs[i] = RDE.turbo_diff_norm(u, cache) * inv_sqrt_N
        end
        maxerr = RDE.turbo_maximum(errs)
        periodicity_reward = one(T) - (max(maxerr - T(0.08), zero(T)) / sqrt(T(3)))
        periodicity_reward = sigmoid_to_linear(periodicity_reward)
        return periodicity_reward
    end
    return one(T)
end

function calculate_shock_reward(shocks::T, target_shock_count::Int, max_shocks::Int) where {T <: AbstractFloat}
    safe_max_shocks = T(max_shocks) + T(1.0e-6)
    shock_reward = max(min(shocks / target_shock_count, (shocks - safe_max_shocks) / (target_shock_count - safe_max_shocks)), zero(T))
    return sigmoid_to_linear(shock_reward)
end

function calculate_shock_rewards(u::AbstractVector{T}, dx::T, L::T, N::Int, target_shock_count::Int) where {T <: AbstractFloat}
    shock_inds = RDE.shock_indices(u, dx)
    shocks = T(length(shock_inds))
    max_shocks = 4
    shock_reward = calculate_shock_reward(shocks, target_shock_count, max_shocks)

    if shocks > 1
        optimal_spacing = L / target_shock_count
        shock_spacing = mod.(RDE.periodic_diff(shock_inds), N) .* dx
        shock_spacing_reward = one(T) - RDE.turbo_maximum_abs((shock_spacing .- optimal_spacing) ./ optimal_spacing)
    elseif shocks == 1 && target_shock_count == 1
        shock_spacing_reward = one(T)
    else
        shock_spacing_reward = zero(T)
    end
    shock_spacing_reward = sigmoid_to_linear(shock_spacing_reward)
    return shock_reward, shock_spacing_reward, shocks
end

function calculate_span_rewards(u::AbstractVector{T}, shocks::T) where {T <: AbstractFloat}
    mn, mx = RDE.turbo_extrema(u)
    span = mx - mn
    abs_span_punishment_threshold = T(0.08)
    target_span = T(2) - T(0.3) * shocks
    span_reward = span / target_span
    low_span_punishment = RDE.smooth_g(span / abs_span_punishment_threshold)
    return span_reward, low_span_punishment
end

function global_rewards(u::AbstractVector{T}, L::T, dx::T, rt::CachedCompositeReward) where {T <: AbstractFloat}
    N = length(u)
    @assert length(rt.cache) == N "cache length must match state length, cache length: $(length(rt.cache)), state length: $N"

    periodicity_reward = calculate_periodicity_reward(u, N, rt.target_shock_count, rt.cache)
    shock_reward, shock_spacing_reward, shocks = calculate_shock_rewards(u, dx, L, N, rt.target_shock_count)
    span_reward, low_span_punishment = calculate_span_rewards(u, shocks)

    return span_reward, periodicity_reward, shock_reward, shock_spacing_reward, low_span_punishment
end

function global_reward(u::AbstractVector{T}, L::T, dx::T, rt::CachedCompositeReward) where {T <: AbstractFloat}
    span_reward, periodicity_reward, shock_reward, shock_spacing_reward, low_span_punishment = global_rewards(u, L, dx, rt)
    weighted_rewards = [span_reward, periodicity_reward, shock_reward, shock_spacing_reward]' * rt.weights / sum(rt.weights)
    global_reward = low_span_punishment * sum(weighted_rewards)
    return global_reward
end

function global_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::CachedCompositeReward) where {T, A, O, R, V, OBS}
    N = env.prob.params.N
    dx = env.prob.x[2] - env.prob.x[1]
    L = env.prob.params.L
    u = env.state[1:N]
    return global_reward(u, L, dx, rt)
end
