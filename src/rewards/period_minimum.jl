struct PeriodMinimumReward{T <: AbstractFloat} <: CachedCompositeReward
    target_shock_count::Int
    cache::Vector{T}
    lowest_action_magnitude_reward::T
    weights::Vector{T}
    function PeriodMinimumReward{T}(; target_shock_count::Int = 4,
            lowest_action_magnitude_reward::T = zero(T),
            weights::Vector{T} = [one(T), one(T), T(5), one(T)],
            N::Int = 512) where {T <: AbstractFloat}
        return new{T}(target_shock_count, zeros(T, N), lowest_action_magnitude_reward, weights)
    end
    function PeriodMinimumReward(; target_shock_count::Int = 4,
            lowest_action_magnitude_reward::Float32 = 0.0f0,
            weights::Vector{Float32} = Float32[1, 1, 5, 1],
            N::Int = 512)
        return PeriodMinimumReward{Float32}(
            target_shock_count = target_shock_count,
            lowest_action_magnitude_reward = lowest_action_magnitude_reward,
            weights = weights,
            N = N)
    end
end

function Base.show(io::IO, rt::PeriodMinimumReward)
    return print(io, "PeriodMinimumReward(target_shock_count=$(rt.target_shock_count))")
end

function Base.show(io::IO, ::MIME"text/plain", rt::PeriodMinimumReward)
    println(io, "PeriodMinimumReward:")
    println(io, "  target_shock_count: $(rt.target_shock_count)")
    println(io, "  lowest_action_magnitude_reward: $(rt.lowest_action_magnitude_reward)")
    println(io, "  weights: $(rt.weights)")
    return println(io, "  cache size: $(length(rt.cache))")
end

function time_minimum_global_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::CachedCompositeReward)::T where {T <: AbstractFloat, A, O, R, V, OBS}
    N::Int = env.prob.params.N
    if isnothing(env.prob.sol)
        return zero(T)
    end

    L::T = env.prob.params.L
    dx::T = RDE.get_dx(env.prob)
    weight_sum::T = sum(rt.weights)

    min_rewards::Vector{T} = fill(T(10000), 5)

    sol_states = env.prob.sol.u::Vector{Vector{T}}

    @inbounds for state::Vector{T} in sol_states
        u = @view state[1:N]

        span_rew, periodicity_rew, shock_rew, shock_spacing_rew, low_span_punishment =
            global_rewards(u, L, dx, rt)

        min_rewards[1] = min(min_rewards[1], span_rew)
        min_rewards[2] = min(min_rewards[2], periodicity_rew)
        min_rewards[3] = min(min_rewards[3], shock_rew)
        min_rewards[4] = min(min_rewards[4], shock_spacing_rew)
        min_rewards[5] = min(min_rewards[5], low_span_punishment)
    end

    weighted_rewards = (
        min_rewards[1] * rt.weights[1] +
        min_rewards[2] * rt.weights[2] +
        min_rewards[3] * rt.weights[3] +
        min_rewards[4] * rt.weights[4]
    ) / weight_sum

    return min_rewards[5] * weighted_rewards
end

function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::PeriodMinimumReward) where {T, A, O, R, V, OBS}
    global_reward = time_minimum_global_reward(env, rt)
    action_magnitude = RDE.turbo_maximum_abs(env.cache.action)
    action_magnitude_modifier = action_magnitude_factor(rt.lowest_action_magnitude_reward, action_magnitude)
    return global_reward * action_magnitude_modifier
end

reward_value_type(::Type{T}, ::PeriodMinimumReward) where {T} = T
