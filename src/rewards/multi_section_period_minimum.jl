mutable struct MultiSectionPeriodMinimumReward{T <: AbstractFloat} <: MultiAgentCachedCompositeReward
    n_sections::Int
    target_shock_count::Int
    cache::Vector{T}
    lowest_action_magnitude_reward::T
    weights::Vector{T}
end

function MultiSectionPeriodMinimumReward(; weights::Vector{T} = [1.0f0, 1.0f0, 5.0f0, 1.0f0],
        n_sections::Int = 4, target_shock_count::Int = 3, lowest_action_magnitude_reward::T = 0.0f0) where {T <: AbstractFloat}
    cache = zeros(T, 512)
    return MultiSectionPeriodMinimumReward{T}(n_sections, target_shock_count, cache, lowest_action_magnitude_reward, weights)
end

function MultiSectionPeriodMinimumReward{T}(n_sections::Int = 4, target_shock_count::Int = 3,
        lowest_action_magnitude_reward::T = zero(T), weights::Vector{T} = [one(T), one(T), T(5), one(T)]) where {T <: AbstractFloat}
    cache = zeros(T, 512)
    return MultiSectionPeriodMinimumReward{T}(n_sections, target_shock_count, cache, lowest_action_magnitude_reward, weights)
end

function Base.show(io::IO, rt::MultiSectionPeriodMinimumReward)
    return print(io, "MultiSectionPeriodMinimumReward(n_sections=$(rt.n_sections))")
end

function Base.show(io::IO, ::MIME"text/plain", rt::MultiSectionPeriodMinimumReward)
    println(io, "MultiSectionPeriodMinimumReward:")
    println(io, "  n_sections: $(rt.n_sections)")
    println(io, "  target_shock_count: $(rt.target_shock_count)")
    println(io, "  lowest_action_magnitude_reward: $(rt.lowest_action_magnitude_reward)")
    println(io, "  weights: $(rt.weights)")
    return println(io, "  cache size: $(length(rt.cache))")
end

reward_value_type(::Type{T}, ::MultiSectionPeriodMinimumReward{T}) where {T} = Vector{T}

function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::MultiSectionPeriodMinimumReward) where {T, A, O, R, V, OBS}
    common_reward = time_minimum_global_reward(env, rt)

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
