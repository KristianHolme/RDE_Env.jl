mutable struct MultiSectionReward{T <: AbstractFloat} <: MultiAgentCachedCompositeReward
    n_sections::Int
    target_shock_count::Int
    cache::Vector{T}
    lowest_action_magnitude_reward::T
    weights::Vector{T}
end

function MultiSectionReward{T}(; n_sections::Int = 4, target_shock_count::Int = 3, N::Int = 512,
        lowest_action_magnitude_reward::T = zero(T), weights::Vector{T} = [one(T), one(T), T(5), one(T)]) where {T <: AbstractFloat}
    return MultiSectionReward{T}(n_sections, target_shock_count, zeros(T, N), lowest_action_magnitude_reward, weights)
end

function MultiSectionReward(; n_sections::Int = 4, target_shock_count::Int = 3, N::Int = 512,
        lowest_action_magnitude_reward::Float32 = 0.0f0, weights::Vector{Float32} = [1.0f0, 1.0f0, 5.0f0, 1.0f0])
    return MultiSectionReward{Float32}(n_sections, target_shock_count, zeros(Float32, N), lowest_action_magnitude_reward, weights)
end

reward_value_type(::Type{T}, ::MultiSectionReward{T}) where {T} = Vector{T}

function Base.show(io::IO, rt::MultiSectionReward)
    return print(io, "MultiSectionReward(n_sections=$(rt.n_sections))")
end

function Base.show(io::IO, ::MIME"text/plain", rt::MultiSectionReward)
    println(io, "MultiSectionReward:")
    println(io, "  n_sections: $(rt.n_sections)")
    println(io, "  target_shock_count: $(rt.target_shock_count)")
    println(io, "  lowest_action_magnitude_reward: $(rt.lowest_action_magnitude_reward)")
    println(io, "  weights: $(rt.weights)")
    return println(io, "  cache size: $(length(rt.cache))")
end

function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::MultiSectionReward) where {T, A, O, R, V, OBS}
    common_reward = global_reward(env, rt)
    N = env.prob.params.N
    n_sections = rt.n_sections
    points_per_section = N ÷ n_sections
    first_element_in_sections = collect(1:points_per_section:N)

    α = rt.lowest_action_magnitude_reward
    full_engine_pressure_action = env.cache.action[:, 2]
    section_actions = full_engine_pressure_action[first_element_in_sections]
    action_magnitudes = abs.(section_actions)
    individual_modifiers = action_magnitude_factor(α, action_magnitudes)
    return common_reward .* individual_modifiers
end

mutable struct TimeAggMultiSectionReward{T <: AbstractFloat} <: MultiAgentCachedCompositeReward
    aggregation::TimeAggregation
    n_sections::Int
    target_shock_count::Int
    cache::Vector{T}
    lowest_action_magnitude_reward::T
    weights::Vector{T}
end

reward_value_type(::Type{T}, ::TimeAggMultiSectionReward{T}) where {T} = Vector{T}

function Base.show(io::IO, rt::TimeAggMultiSectionReward)
    return print(io, "TimeAggMultiSectionReward(n_sections=$(rt.n_sections), aggregation=$(typeof(rt.aggregation)))")
end

function Base.show(io::IO, ::MIME"text/plain", rt::TimeAggMultiSectionReward)
    println(io, "TimeAggMultiSectionReward:")
    println(io, "  aggregation: $(typeof(rt.aggregation))")
    println(io, "  n_sections: $(rt.n_sections)")
    println(io, "  target_shock_count: $(rt.target_shock_count)")
    println(io, "  lowest_action_magnitude_reward: $(rt.lowest_action_magnitude_reward)")
    println(io, "  weights: $(rt.weights)")
    return println(io, "  cache size: $(length(rt.cache))")
end

function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::TimeAggMultiSectionReward) where {T, A, O, R, V, OBS}
    N = env.prob.params.N
    n_sections = rt.n_sections
    points_per_section = N ÷ n_sections
    first_element_in_sections = collect(1:points_per_section:N)

    if !isnothing(env.prob.sol)
        us = [uλ[1:N] for uλ in env.prob.sol.u[2:end]]
    else
        us = [env.state[1:N]]
    end
    section_rewards = zeros(T, n_sections, length(us))

    for (i, u) in enumerate(us)
        common_reward = global_reward(u, env.prob.params.L, env.prob.x[2] - env.prob.x[1], rt)
        full_engine_pressure_action = env.cache.action[:, 2]
        section_actions = full_engine_pressure_action[first_element_in_sections]
        action_magnitudes = abs.(section_actions)
        individual_modifiers = action_magnitude_factor(rt.lowest_action_magnitude_reward, action_magnitudes)
        section_rewards[:, i] = common_reward .* individual_modifiers
    end

    agg_section_rewards = zeros(T, n_sections)
    for i in 1:n_sections
        agg_section_rewards[i] = aggregate(section_rewards[i, :], rt.aggregation)
    end

    return agg_section_rewards
end

function TimeAggMultiSectionReward{T}(; aggregation::TimeAggregation = TimeMin(), n_sections::Int = 4,
        target_shock_count::Int = 3, N::Int = 512, lowest_action_magnitude_reward::T = zero(T),
        weights::Vector{T} = [one(T), one(T), T(5), one(T)]) where {T <: AbstractFloat}
    return TimeAggMultiSectionReward{T}(aggregation, n_sections, target_shock_count, zeros(T, N), lowest_action_magnitude_reward, weights)
end

function TimeAggMultiSectionReward(; aggregation::TimeAggregation = TimeMin(), n_sections::Int = 4,
        target_shock_count::Int = 3, N::Int = 512, lowest_action_magnitude_reward::Float32 = 0.0f0,
        weights::Vector{Float32} = [1.0f0, 1.0f0, 5.0f0, 1.0f0])
    return TimeAggMultiSectionReward{Float32}(aggregation, n_sections, target_shock_count, zeros(Float32, N), lowest_action_magnitude_reward, weights)
end
