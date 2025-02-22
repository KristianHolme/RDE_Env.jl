function set_reward!(env::AbstractRDEEnv, rt::AbstractRDEReward)
    @error "No reward set for type $(typeof(rt))"
end
function set_reward!(env::AbstractRDEEnv{T}, rt::ShockSpanReward) where T
    target_shock_count = rt.target_shock_count
    max_span = rt.span_scale
    λ = rt.shock_weight

    u, = RDE.split_sol_view(env.state)
    dx = env.prob.params.L/env.prob.params.N
    shocks = Float32(RDE.count_shocks(u, dx))
    span = maximum(u) - minimum(u)
    span_reward = span/max_span
    if shocks >= target_shock_count
        shock_reward = one(T)
    elseif shocks > 0
        shock_reward = shocks/(2*target_shock_count)
    else
        shock_reward = T(-1.0)
    end
    
    env.reward = λ*shock_reward + (1-λ)*span_reward
    nothing
end

"""
    set_reward!(env::AbstractRDEEnv, rt::ShockPreservingReward)

Reward for preserving a given number of shocks. terminated/truncated if the number of shocks is not preserved.
    penalize reward if shocks are not evenly spaced, reward for large span.
"""
function set_reward!(env::AbstractRDEEnv{T}, rt::ShockPreservingReward) where T
    target_shock_count = rt.target_shock_count
    max_span = rt.span_scale
    λ = rt.shock_weight

    u, = RDE.split_sol_view(env.state)
    dx = env.prob.x[2] - env.prob.x[1]
    N = env.prob.params.N
    L = env.prob.params.L
    shock_inds = RDE.shock_indices(u, dx)

    span = maximum(u) - minimum(u)
    span_reward = span/max_span

    if length(shock_inds) != target_shock_count
        if isnothing(rt.abscence_start)
            rt.abscence_start = env.t
        elseif env.t - rt.abscence_start > rt.abscence_limit
            env.terminated = true
            env.reward = T(-2.0)
            return nothing
        end
        shock_reward = T(-1.0)
    else
        optimal_spacing = L/target_shock_count
        shock_spacing = mod.(RDE.periodic_diff(shock_inds), N) .* dx
        shock_reward = T(1.0) - mean(abs.((shock_spacing .- optimal_spacing)./optimal_spacing))
    end
    env.reward = λ*shock_reward + (1-λ)*span_reward
    nothing
end

function set_reward!(env::AbstractRDEEnv, rt::ShockPreservingSymmetryReward)
    target_shock_count = rt.target_shock_count
    N = env.prob.params.N
    u = env.state[1:N]
    
    errs = zeros(target_shock_count-1)
    cache = rt.cache
    shift_steps = N ÷ target_shock_count
    for i in 1:(target_shock_count-1)
        cache .= u
        RDE.apply_periodic_shift!(cache, u, shift_steps * i)
        errs[i] = norm(u - cache)/sqrt(N)
    end
    maxerr = maximum(errs)
    env.reward = 1f0 - (maxerr-0.1f0)/0.5f0
    nothing
end

function set_reward!(env::AbstractRDEEnv, rt::PeriodicityReward)
    u, = RDE.split_sol_view(env.state)
    N = env.prob.params.N
    dx = env.prob.x[2] - env.prob.x[1]
    L = env.prob.params.L
    shock_inds = RDE.shock_indices(u, dx)
    shocks = length(shock_inds)

    cache = rt.cache
    if shocks > 1
        shift_steps = N ÷ shocks
        errs = zeros(shocks-1)
        for i in 1:(shocks-1)
            cache .= u
            RDE.apply_periodic_shift!(cache, u, shift_steps * i)
            errs[i] = norm(u - cache)/sqrt(N)
        end
        maxerr = maximum(errs)
        periodicity_reward = 1f0 - (max(maxerr-0.08f0, 0f0)/sqrt(3f0))
    else
        periodicity_reward = 1f0
    end


    if shocks > 1
        optimal_spacing = L/shocks
        shock_spacing = mod.(RDE.periodic_diff(shock_inds), N) .* dx
        shock_spacing_reward = 1f0 - maximum(abs.((shock_spacing .- optimal_spacing)./optimal_spacing))
    else
        shock_spacing_reward = 1f0
    end


    env.reward = (Float32(periodicity_reward + shock_spacing_reward))/2f0
    nothing
end

function Base.show(io::IO, rt::PeriodicityReward)
    print(io, "PeriodicityReward(N=$(length(rt.cache)))")
end


@kwdef mutable struct MultiSectionReward <: CachedCompositeReward
    n_sections::Int = 4
    target_shock_count::Int = 3
    cache::Vector{Float32} = zeros(Float32, 512)
    lowest_action_magnitude_reward::Float32 = 0.3f0 #reward will be \in [lowest_action_magnitude_reward, 1]
    weights::Vector{Float32} = [1f0,1f0,5f0,1f0]
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

function calculate_periodicity_reward(u::AbstractVector{T}, N::Int, target_shock_count::Int, cache::AbstractVector{T}) where T
    if target_shock_count > 1
        errs = zeros(T, target_shock_count-1)
        shift_steps = N ÷ target_shock_count
        for i in 1:(target_shock_count-1)
            cache .= u
            RDE.apply_periodic_shift!(cache, u, shift_steps * i)
            errs[i] = norm(u - cache)/sqrt(N)
        end
        maxerr = maximum(errs)
        periodicity_reward = 1f0 - (max(maxerr-0.08f0, 0f0)/sqrt(3f0))
        periodicity_reward = sigmoid_to_linear(periodicity_reward)
        return periodicity_reward
    end
    return one(T)
end

function calculate_shock_rewards(u::AbstractVector, dx::T, L::T, N::Int, target_shock_count::Int) where T
    shock_inds = RDE.shock_indices(u, dx)
    shocks = T(length(shock_inds))
    @logmsg LogLevel(-10000) "shocks: $shocks"
    ϵ = T(1e-6)
    max_shocks = T(4 + ϵ)
    shock_reward = max(min(shocks/target_shock_count, (shocks-max_shocks)/(target_shock_count-max_shocks)), zero(T))
    # shock_reward = shocks == target_shock_count ? one(T) : zero(T)
    shock_reward = sigmoid_to_linear(shock_reward)

    if shocks > 1
        optimal_spacing = L/target_shock_count
        shock_spacing = mod.(RDE.periodic_diff(shock_inds), N) .* dx
        shock_spacing_reward = one(T) - maximum(abs.((shock_spacing .- optimal_spacing)./optimal_spacing))
    elseif shocks == 1 && target_shock_count == 1
        shock_spacing_reward = one(T)
    else
        shock_spacing_reward = zero(T)
    end
    shock_spacing_reward = sigmoid_to_linear(shock_spacing_reward)
    return shock_reward, shock_spacing_reward, shocks
end

function calculate_span_rewards(u::AbstractVector, shocks::T) where T
    span = maximum(u) - minimum(u)
    abs_span_punishment_threshold = 0.08f0
    target_span = 2.0f0 - 0.3f0*shocks
    span_reward = span/target_span
    # span_reward = linear_to_sigmoid(span_reward)
    low_span_punishment = RDE.smooth_g(span/abs_span_punishment_threshold)
    return span_reward, low_span_punishment
end

function global_reward(env::AbstractRDEEnv{T}, rt::CachedCompositeReward) where T
    N = env.prob.params.N
    dx = env.prob.x[2] - env.prob.x[1]
    L = env.prob.params.L
    u = env.state[1:N]
    
    periodicity_reward = calculate_periodicity_reward(u, N, rt.target_shock_count, rt.cache)
    shock_reward, shock_spacing_reward, shocks = calculate_shock_rewards(u, dx, L, N, rt.target_shock_count)
    span_reward, low_span_punishment = calculate_span_rewards(u, shocks)

    @logmsg LogLevel(-10000) "low_span_punishment: $low_span_punishment"
    @logmsg LogLevel(-10000) "span_reward: $span_reward" 
    @logmsg LogLevel(-10000) "periodicity_reward: $periodicity_reward"
    @logmsg LogLevel(-10000) "shock_reward: $shock_reward"
    @logmsg LogLevel(-10000) "shock_spacing_reward: $shock_spacing_reward"

    weighted_rewards = [span_reward, periodicity_reward, shock_reward, shock_spacing_reward]' * rt.weights / sum(rt.weights)
    global_reward = low_span_punishment * sum(weighted_rewards)
    @logmsg LogLevel(-10000) "global_reward: $global_reward"
    return global_reward
end


mutable struct CompositeReward <: CachedCompositeReward
    target_shock_count::Int
    cache::Vector{Float32}
    lowest_action_magnitude_reward::Float32 #reward will be \in [lowest_action_magnitude_reward, 1]
    span_reward::Bool
    weights::Vector{Float32}
    function CompositeReward(;target_shock_count::Int=4,
                              lowest_action_magnitude_reward::Float32=1f0,
                              span_reward::Bool=true,
                              weights::Vector{Float32}=[0.25f0, 0.25f0, 0.25f0, 0.25f0])
        return new(target_shock_count,
                   zeros(Float32, 512),
                   lowest_action_magnitude_reward,
                   span_reward, weights)
    end
end

function Base.show(io::IO, rt::CompositeReward)
    print(io, "CompositeReward(target_shock_count=$(rt.target_shock_count), lowest_action_magnitude_reward=$(rt.lowest_action_magnitude_reward))")
end

function set_reward!(env::AbstractRDEEnv{T}, rt::CompositeReward) where T
    N = env.prob.params.N

    reward = global_reward(env, rt)
    if rt.lowest_action_magnitude_reward < 1f0
        action_magnitude_inv = 1f0 - maximum(abs.(env.cache.action))
        α = rt.lowest_action_magnitude_reward
        @logmsg LogLevel(-10000) "action_magnitude factor: $(α + (1f0 - α)*action_magnitude_inv)" maximum(abs.(env.cache.action))
        reward *= α + (1f0 - α)*action_magnitude_inv
    end
    @logmsg LogLevel(-10000) "set_reward!: $reward"
    env.reward = reward
    nothing
end


struct ConstantTargetReward <: AbstractRDEReward
    target::Float32
    function ConstantTargetReward(;target::Float32=0.64f0)
        return new(target)
    end
end

function set_reward!(env::AbstractRDEEnv{T}, rt::ConstantTargetReward) where T
    env.reward = -abs(rt.target - mean(env.prob.method.cache.u_p_current)) + T(1.0)
    nothing
end