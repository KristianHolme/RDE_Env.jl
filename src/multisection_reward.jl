@kwdef mutable struct MultiSectionReward <: AbstractRDEReward
    n_sections::Int = 4
    target_shock_count::Int = 3
    cache::Vector{Float32} = zeros(Float32, 512)
    lowest_action_magnitude_reward::Float32 = 1f0 #reward will be \in [lowest_action_magnitude_reward, 1]
end

function Base.show(io::IO, rt::MultiSectionReward)
    print(io, "MultiSectionReward(n_sections=$(rt.n_sections), target_shock_count=$(rt.target_shock_count), lowest_action_magnitude_reward=$(rt.lowest_action_magnitude_reward))")
end

function set_reward!(env::AbstractRDEEnv, rt::MultiSectionReward)
    common_reward = global_reward(env, rt)
    N = env.prob.params.N
    n_sections = rt.n_sections
    points_per_section = N ÷ n_sections
    first_element_in_sections = collect(1:points_per_section:N)
    
    α = rt.lowest_action_magnitude_reward
    full_engine_pressure_action = env.cache.action[:, 2]
    section_actions = full_engine_pressure_action[first_element_in_sections]
    action_magnitudes = abs.(section_actions)
    individual_modifiers = 1f0 .- action_magnitudes .* (1f0-α)
    individual_rewards = common_reward .* individual_modifiers
    env.reward = individual_rewards
    nothing
end

function global_reward(env::AbstractRDEEnv{T}, rt::MultiSectionReward) where T
    N = env.prob.params.N
    dx = env.prob.x[2] - env.prob.x[1]
    L = env.prob.params.L
    u = env.state[1:N]
    target_shock_count = rt.target_shock_count  
    
    target_span = 2f0 .- 0.3f0.*target_shock_count
    span = maximum(u) - minimum(u)
    span_reward = 1f0 - (max(target_span - span, 0f0)/target_span)

    abs_span_punishment_threshold = 0.08f0
    abs_span_reward = RDE.smooth_g(span/abs_span_punishment_threshold)

    if target_shock_count > 1
        errs = zeros(T, target_shock_count-1)
        cache = rt.cache
        shift_steps = N ÷ target_shock_count
        for i in 1:(target_shock_count-1)
            cache .= u
            RDE.apply_periodic_shift!(cache, u, shift_steps * i)
            errs[i] = norm(u - cache)/sqrt(N)
        end
        maxerr = maximum(errs)
        periodicity_reward = 1f0 - (max(maxerr-0.08f0, 0f0)/sqrt(3f0))
    else
        periodicity_reward = 1f0
    end

    shock_inds = RDE.shock_indices(u, dx)
    shocks = T(length(shock_inds))
    shock_reward = 1f0 - (abs(shocks - target_shock_count)/target_shock_count)
    if shocks > 0
        # optimal_spacing = L/shocks
        optimal_spacing = L/target_shock_count
        shock_spacing = mod.(RDE.periodic_diff(shock_inds), N) .* dx
        shock_spacing_reward = 1f0 - maximum(abs.((shock_spacing .- optimal_spacing)./optimal_spacing))
    else
        shock_spacing_reward = 1f0
    end

    @debug "span_reward: $span_reward" 
    @debug "abs_span_reward: $abs_span_reward"
    @debug "periodicity_reward: $periodicity_reward"
    @debug "shock_reward: $shock_reward"
    @debug "shock_spacing_reward: $shock_spacing_reward"

    reward = abs_span_reward*exp(span_reward + periodicity_reward + shock_reward + shock_spacing_reward - 4f0)
    return reward
end