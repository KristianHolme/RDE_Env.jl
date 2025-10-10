mutable struct CompositeReward{T <: AbstractFloat} <: CachedCompositeReward
    target_shock_count::Int
    cache::Vector{T}
    lowest_action_magnitude_reward::T
    span_reward::Bool
    weights::Vector{T}
    function CompositeReward{T}(; target_shock_count::Int = 4,
            lowest_action_magnitude_reward::T = one(T),
            span_reward::Bool = true,
            weights::Vector{T} = [T(0.25), T(0.25), T(0.25), T(0.25)],
            N::Int = 512) where {T <: AbstractFloat}
        return new{T}(target_shock_count, zeros(T, N), lowest_action_magnitude_reward, span_reward, weights)
    end
    function CompositeReward(; target_shock_count::Int = 4,
            lowest_action_magnitude_reward::Float32 = 1.0f0,
            span_reward::Bool = true,
            weights::Vector{Float32} = [0.25f0, 0.25f0, 0.25f0, 0.25f0],
            N::Int = 512)
        return CompositeReward{Float32}(
            target_shock_count = target_shock_count,
            lowest_action_magnitude_reward = lowest_action_magnitude_reward,
            span_reward = span_reward,
            weights = weights,
            N = N)
    end
end

function Base.show(io::IO, rt::CompositeReward)
    return print(io, "CompositeReward(target_shock_count=$(rt.target_shock_count))")
end

function Base.show(io::IO, ::MIME"text/plain", rt::CompositeReward)
    println(io, "CompositeReward:")
    println(io, "  target_shock_count: $(rt.target_shock_count)")
    println(io, "  lowest_action_magnitude_reward: $(rt.lowest_action_magnitude_reward)")
    println(io, "  span_reward: $(rt.span_reward)")
    println(io, "  weights: $(rt.weights)")
    return println(io, "  cache size: $(length(rt.cache))")
end

function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::CompositeReward) where {T, A, O, R, V, OBS}
    reward = global_reward(env, rt)
    action_magnitude = RDE.turbo_maximum_abs(env.cache.action)
    action_magnitude_modifier = action_magnitude_factor(rt.lowest_action_magnitude_reward, action_magnitude)
    reward *= action_magnitude_modifier
    return reward
end

reward_value_type(::Type{T}, ::CachedCompositeReward) where {T} = T

struct TimeAvg <: TimeAggregation end
struct TimeMax <: TimeAggregation end
struct TimeMin <: TimeAggregation end
struct TimeSum <: TimeAggregation end
struct TimeProd <: TimeAggregation end

struct TimeAggCompositeReward{T <: AbstractFloat} <: CachedCompositeReward
    aggregation::TimeAggregation
    target_shock_count::Int
    cache::Vector{T}
    lowest_action_magnitude_reward::T
    span_reward::Bool
    weights::Vector{T}
    function TimeAggCompositeReward{T}(; aggregation::TimeAggregation = TimeMin(), target_shock_count::Int = 4,
            lowest_action_magnitude_reward::T = one(T), span_reward::Bool = true,
            weights::Vector{T} = [T(0.25), T(0.25), T(0.25), T(0.25)], N::Int = 512) where {T <: AbstractFloat}
        return new{T}(aggregation, target_shock_count, zeros(T, N), lowest_action_magnitude_reward, span_reward, weights)
    end
    function TimeAggCompositeReward(; aggregation::TimeAggregation = TimeMin(), target_shock_count::Int = 4,
            lowest_action_magnitude_reward::T = T(1.0), span_reward::Bool = true,
            weights::Vector{T} = [T(0.25), T(0.25), T(0.25), T(0.25)], N::Int = 512) where {T <: AbstractFloat}
        return TimeAggCompositeReward{T}(aggregation = aggregation, target_shock_count = target_shock_count,
            lowest_action_magnitude_reward = lowest_action_magnitude_reward, span_reward = span_reward, weights = weights, N = N)
    end
end

function Base.show(io::IO, rt::TimeAggCompositeReward)
    return print(io, "TimeAggCompositeReward(aggregation=$(rt.aggregation), target_shock_count=$(rt.target_shock_count))")
end

function Base.show(io::IO, ::MIME"text/plain", rt::TimeAggCompositeReward)
    println(io, "TimeAggCompositeReward:")
    println(io, "  aggregation: $(rt.aggregation)")
    println(io, "  target_shock_count: $(rt.target_shock_count)")
    println(io, "  lowest_action_magnitude_reward: $(rt.lowest_action_magnitude_reward)")
    println(io, "  span_reward: $(rt.span_reward)")
    println(io, "  weights: $(rt.weights)")
    return println(io, "  cache size: $(length(rt.cache))")
end

reward_value_type(::Type{T}, ::TimeAggCompositeReward) where {T} = T

function aggregate(rewards::AbstractVector, ::TimeAvg)
    return mean(rewards)
end
function aggregate(rewards::AbstractVector, ::TimeMax)
    return maximum(rewards)
end
function aggregate(rewards::AbstractVector, ::TimeMin)
    return minimum(rewards)
end
function aggregate(rewards::AbstractVector, ::TimeSum)
    return sum(rewards)
end
function aggregate(rewards::AbstractVector, ::TimeProd)
    return prod(rewards)
end

function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::TimeAggCompositeReward) where {T, A, O, R, V, OBS}
    N = env.prob.params.N
    dx = env.prob.x[2] - env.prob.x[1]
    L = env.prob.params.L
    sol = env.prob.sol
    if !isnothing(sol)
        us = [uλ[1:N] for uλ in sol.u[2:end]]
        rewards = zeros(T, length(us))
    else
        @debug "No solution found for env at time $(env.t), step $(env.steps_taken)"
        us = [env.state[1:N]]
        rewards = zeros(T, 1)
    end
    for (i, u) in enumerate(us)
        rewards[i] = global_reward(u, L, dx, rt)
    end
    agg_reward = aggregate(rewards, rt.aggregation)

    action_magnitude = RDE.turbo_maximum_abs(env.cache.action)
    action_magnitude_modifier = action_magnitude_factor(rt.lowest_action_magnitude_reward, action_magnitude)
    return agg_reward * action_magnitude_modifier
end
