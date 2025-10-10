struct PeriodMinimumVariationReward{T <: AbstractFloat} <: CachedCompositeReward
    target_shock_count::Int
    cache::Vector{T}
    lowest_action_magnitude_reward::T
    weights::Vector{T}
    variation_penalties::Vector{T}
end

function PeriodMinimumVariationReward{T}(; target_shock_count::Int = 4,
        lowest_action_magnitude_reward::T = one(T),
        weights::Vector{T} = [one(T), one(T), T(5), one(T)],
        variation_penalties::Vector{T} = [T(2), T(2), T(2), T(2)],
        N::Int = 512) where {T <: AbstractFloat}
    return PeriodMinimumVariationReward{T}(
        target_shock_count,
        zeros(T, N),
        lowest_action_magnitude_reward,
        weights,
        variation_penalties
    )
end

function PeriodMinimumVariationReward(; target_shock_count::Int = 4,
        lowest_action_magnitude_reward::Float32 = 1.0f0,
        weights::Vector{Float32} = Float32[1, 1, 5, 1],
        variation_penalties::Vector{Float32} = Float32[2.0, 2.0, 2.0, 2.0],
        N::Int = 512)
    return PeriodMinimumVariationReward{Float32}(
        target_shock_count = target_shock_count,
        lowest_action_magnitude_reward = lowest_action_magnitude_reward,
        weights = weights,
        variation_penalties = variation_penalties,
        N = N
    )
end

function Base.show(io::IO, rt::PeriodMinimumVariationReward)
    return print(io, "PeriodMinimumVariationReward(target_shock_count=$(rt.target_shock_count))")
end

function Base.show(io::IO, ::MIME"text/plain", rt::PeriodMinimumVariationReward)
    println(io, "PeriodMinimumVariationReward:")
    println(io, "  target_shock_count: $(rt.target_shock_count)")
    println(io, "  lowest_action_magnitude_reward: $(rt.lowest_action_magnitude_reward)")
    println(io, "  weights: $(rt.weights)")
    println(io, "  variation_penalties: $(rt.variation_penalties)")
    return println(io, "  cache size: $(length(rt.cache))")
end

reward_value_type(::Type{T}, ::PeriodMinimumVariationReward) where {T} = T

function calculate_variation_punishment(values::Vector{T}, α::T) where {T <: AbstractFloat}
    if length(values) <= 1
        return one(T)
    end

    mean_abs_value = mean(abs.(values))
    if mean_abs_value < T(1.0e-6)
        return one(T)
    end

    coefficient_of_variation = std(values) / mean_abs_value
    return exp(-α * coefficient_of_variation)
end

function time_minimum_global_reward_with_variation(env::RDEEnv{T, A, O, R, V, OBS}, rt::PeriodMinimumVariationReward) where {T <: AbstractFloat, A, O, R, V, OBS}
    N = env.prob.params.N
    if isnothing(env.prob.sol)
        return zero(T)
    end

    L = env.prob.params.L::T
    dx = (env.prob.x[2] - env.prob.x[1])::T
    weight_sum = sum(rt.weights)::T

    sol_states = env.prob.sol.u
    n_states = length(sol_states)

    all_span_rewards = Vector{T}(undef, n_states)
    all_periodicity_rewards = Vector{T}(undef, n_states)
    all_shock_rewards = Vector{T}(undef, n_states)
    all_shock_spacing_rewards = Vector{T}(undef, n_states)
    all_low_span_punishments = Vector{T}(undef, n_states)

    @inbounds for (i, state) in enumerate(sol_states)
        u = @view state[1:N]

        span_rew, periodicity_rew, shock_rew, shock_spacing_rew, low_span_punishment =
            global_rewards(u, L, dx, rt)

        all_span_rewards[i] = span_rew
        all_periodicity_rewards[i] = periodicity_rew
        all_shock_rewards[i] = shock_rew
        all_shock_spacing_rewards[i] = shock_spacing_rew
        all_low_span_punishments[i] = low_span_punishment
    end

    span_var_punishment = calculate_variation_punishment(all_span_rewards, T(rt.variation_penalties[1]))
    periodicity_var_punishment = calculate_variation_punishment(all_periodicity_rewards, T(rt.variation_penalties[2]))
    shock_var_punishment = calculate_variation_punishment(all_shock_rewards, T(rt.variation_penalties[3]))
    shock_spacing_var_punishment = calculate_variation_punishment(all_shock_spacing_rewards, T(rt.variation_penalties[4]))

    punished_span = minimum(all_span_rewards) * span_var_punishment
    punished_periodicity = minimum(all_periodicity_rewards) * periodicity_var_punishment
    punished_shock = minimum(all_shock_rewards) * shock_var_punishment
    punished_shock_spacing = minimum(all_shock_spacing_rewards) * shock_spacing_var_punishment

    min_low_span_punishment = minimum(all_low_span_punishments)

    weighted_rewards = (
        punished_span * rt.weights[1] +
        punished_periodicity * rt.weights[2] +
        punished_shock * rt.weights[3] +
        punished_shock_spacing * rt.weights[4]
    ) / weight_sum

    return min_low_span_punishment * weighted_rewards
end

function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::PeriodMinimumVariationReward) where {T, A, O, R, V, OBS}
    global_reward = time_minimum_global_reward_with_variation(env, rt)
    action_magnitude = RDE.turbo_maximum_abs(env.cache.action)
    action_magnitude_modifier = action_magnitude_factor(rt.lowest_action_magnitude_reward, action_magnitude)
    return global_reward * action_magnitude_modifier
end
