function set_reward!(env::AbstractRDEEnv, rew_strat::AbstractRewardStrategy)
    #not doing in place because reward can change size
    #TODO: initialize env.reward to correct size so we can do .= here
    env.reward = compute_reward(env, rew_strat)
    return nothing
end

# External API - delegates to internal implementation with cache
compute_reward(env::AbstractRDEEnv, rew_strat::AbstractRewardStrategy) = _compute_reward(env, rew_strat, env.cache.reward_cache)

# Internal implementation - takes cache as explicit parameter
# Each reward type implements _compute_reward instead of compute_reward

# ============================================================================
# Cache Types
# ============================================================================

struct RewardShiftBufferCache{T <: AbstractFloat} <: AbstractCache
    shift_buffer::Vector{T}
end

struct CompositeRewardCache{T <: Tuple} <: AbstractCache
    caches::T
end

function reset_cache!(cache::CompositeRewardCache)
    for c in cache.caches
        reset_cache!(c)
    end
    return nothing
end

struct WrappedRewardCache{IC <: AbstractCache, OC <: AbstractCache} <: AbstractCache
    inner_cache::IC  # wrapped reward's cache
    outer_cache::OC  # wrapper's own state
end

function reset_cache!(cache::WrappedRewardCache)
    reset_cache!(cache.inner_cache)
    reset_cache!(cache.outer_cache)
    return nothing
end

mutable struct ExponentialAverageCache{T <: AbstractFloat} <: AbstractCache
    average::Union{Nothing, T, Vector{T}}
end

function reset_cache!(cache::ExponentialAverageCache)
    cache.average = nothing
    return nothing
end

mutable struct TransitionBasedCache{T <: AbstractFloat} <: AbstractCache
    past_rewards::Vector{T}
    past_shock_counts::Vector{Int}
    transition_found::Bool
end

function reset_cache!(cache::TransitionBasedCache)
    empty!(cache.past_rewards)
    empty!(cache.past_shock_counts)
    cache.transition_found = false
    return nothing
end

# ============================================================================
# Helper Functions
# ============================================================================

function action_magnitude_factor(lowest_action_magnitude_reward::AbstractFloat, action_magnitudes)
    α = lowest_action_magnitude_reward
    action_magnitude_inv = 1 .- action_magnitudes
    return α .+ (1 - α) .* action_magnitude_inv
end

function span_variation_reward(values::Vector{T}, α::T; threshold::T = T(4.0f-3)) where {T <: AbstractFloat}
    v_min, v_max = RDE.turbo_extrema(values)
    span = v_max - v_min
    span = max(span - threshold, zero(T))
    variation = span / mean(abs.(values))
    return exp.(-α * variation)
end

"""
    compute_section_control_efforts(env::RDEEnv{T}, n_sections::Int) where {T}

Compute per-section control efforts based on Δu_p normalized by u_pmax.
Returns a vector of length `n_sections` where each element is the maximum absolute
difference in control pressure within that section.
"""
function compute_section_control_efforts(env::RDEEnv{T}, n_sections::Int) where {T}
    N = env.prob.params.N
    method_cache = env.prob.method.cache
    points_per_section = N ÷ n_sections
    efforts = zeros(T, n_sections)
    for i in 1:n_sections
        s = (i - 1) * points_per_section + 1
        e = i * points_per_section
        efforts[i] = RDE.turbo_maximum_abs_diff(@view(method_cache.u_p_current[s:e]), @view(method_cache.u_p_previous[s:e])) / env.u_pmax
    end
    return efforts
end

function calculate_periodicity_reward(u::AbstractVector{T}, N::Int, target_shock_count::Int, cache::AbstractVector{T}) where {T}
    inv_sqrt_N = 1 / sqrt(N)
    if target_shock_count > 1
        errs = zeros(T, target_shock_count - 1)
        shift_steps = N ÷ target_shock_count
        for i in 1:(target_shock_count - 1)
            # cache .= u #maybe not necessary?
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

#TODO: dont use shock_indices(performance), use shock_locations instead
function calculate_shock_rewards(u::AbstractVector{T}, dx::T, L::T, N::Int, target_shock_count::Int) where {T <: AbstractFloat}

    shock_inds = RDE.shock_indices(u, dx)
    shocks = T(length(shock_inds))
    @logmsg LogLevel(-10000) "shocks: $shocks"
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
    # span_reward = linear_to_sigmoid(span_reward)
    low_span_punishment = RDE.smooth_g(span / abs_span_punishment_threshold)
    return span_reward, low_span_punishment
end

function global_rewards(u::AbstractVector{T}, L::T, dx::T, rew_strat::CachedCompositeReward, cache::AbstractVector{T}, target_shock_count::Int) where {T <: AbstractFloat}
    N = length(u)

    periodicity_reward = calculate_periodicity_reward(u, N, target_shock_count, cache)
    shock_reward, shock_spacing_reward, shocks = calculate_shock_rewards(u, dx, L, N, target_shock_count)
    span_reward, low_span_punishment = calculate_span_rewards(u, shocks)

    @logmsg LogLevel(-10000) "low_span_punishment: $low_span_punishment"
    @logmsg LogLevel(-10000) "span_reward: $span_reward"
    @logmsg LogLevel(-10000) "periodicity_reward: $periodicity_reward"
    @logmsg LogLevel(-10000) "shock_reward: $shock_reward"
    @logmsg LogLevel(-10000) "shock_spacing_reward: $shock_spacing_reward"
    return span_reward, periodicity_reward, shock_reward, shock_spacing_reward, low_span_punishment
end

function global_reward(u::AbstractVector{T}, L::T, dx::T, rew_strat::CachedCompositeReward, cache::AbstractVector{T}, target_shock_count::Int) where {T <: AbstractFloat}
    span_reward, periodicity_reward, shock_reward, shock_spacing_reward, low_span_punishment = global_rewards(u, L, dx, rew_strat, cache, target_shock_count)
    weighted_rewards = [span_reward, periodicity_reward, shock_reward, shock_spacing_reward]' * rew_strat.weights / sum(rew_strat.weights)
    global_reward = low_span_punishment * sum(weighted_rewards)
    @logmsg LogLevel(-10000) "global_reward: $global_reward"
    return global_reward
end

function global_reward(env::RDEEnv{T, A, O, R, G, V, OBS}, rew_strat::CachedCompositeReward, cache::RewardShiftBufferCache{T}) where {T, A, O, R, G, V, OBS}
    N = env.prob.params.N
    dx = env.prob.x[2] - env.prob.x[1]
    L = env.prob.params.L
    u = env.state[1:N]
    shift_buffer = cache.shift_buffer
    target_shock_count = get_target_shock_count(env)

    return global_reward(u, L, dx, rew_strat, shift_buffer, target_shock_count)
end

# ============================================================================
# Reward Types and Implementations
# ============================================================================

# ----------------------------------------------------------------------------
# ShockSpanReward
# ----------------------------------------------------------------------------

@kwdef struct ShockSpanReward <: AbstractScalarRewardStrategy
    span_scale::Float32 = 4.0f0
    shock_weight::Float32 = 0.8f0
end

function _compute_reward(env::RDEEnv{T, A, O, R, G, V, OBS}, rew_strat::ShockSpanReward, ::NoCache) where {T, A, O, R, G, V, OBS}
    target_shock_count = get_target_shock_count(env)
    max_span = rew_strat.span_scale
    λ = rew_strat.shock_weight

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
# ----------------------------------------------------------------------------
# ShockPreservingReward
# ----------------------------------------------------------------------------

@kwdef mutable struct ShockPreservingReward <: AbstractScalarRewardStrategy
    span_scale::Float32 = 4.0f0
    shock_weight::Float32 = 0.8f0
    abscence_limit::Float32 = 5.0f0
    abscence_start::Union{Float32, Nothing} = nothing
end

function _compute_reward(env::RDEEnv{T, A, O, R, G, V, OBS}, rew_strat::ShockPreservingReward, ::NoCache) where {T, A, O, R, G, V, OBS}
    target_shock_count = get_target_shock_count(env)
    max_span = rew_strat.span_scale
    λ = rew_strat.shock_weight

    u, = RDE.split_sol_view(env.state)
    dx = env.prob.x[2] - env.prob.x[1]
    N = env.prob.params.N
    L = env.prob.params.L
    shock_inds = RDE.shock_indices(u, dx)

    u_min, u_max = RDE.turbo_extrema(u)
    span = u_max - u_min
    span_reward = span / max_span

    if length(shock_inds) != target_shock_count
        if isnothing(rew_strat.abscence_start)
            rew_strat.abscence_start = env.t
        elseif env.t - rew_strat.abscence_start > rew_strat.abscence_limit
            env.terminated = true
            return T(-2.0)
        end
        shock_reward = T(-1.0)
    else
        optimal_spacing = L / target_shock_count
        shock_spacing = mod.(RDE.periodic_diff(shock_inds), N) .* dx
        shock_reward = T(1.0) - mean(abs.((shock_spacing .- optimal_spacing) ./ optimal_spacing))
    end
    return λ * shock_reward + (1 - λ) * span_reward
end

# ----------------------------------------------------------------------------
# ShockPreservingSymmetryReward
# ----------------------------------------------------------------------------

mutable struct ShockPreservingSymmetryReward <: AbstractScalarRewardStrategy end

initialize_cache(::ShockPreservingSymmetryReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

function _compute_reward(env::RDEEnv{T, A, O, R, G, V, OBS}, rew_strat::ShockPreservingSymmetryReward, cache::RewardShiftBufferCache{T}) where {T, A, O, R, G, V, OBS}
    target_shock_count = get_target_shock_count(env)
    N = env.prob.params.N
    u = env.state[1:N]

    errs = zeros(T, target_shock_count - 1)
    shift_buffer = cache.shift_buffer
    shift_steps = N ÷ target_shock_count
    for i in 1:(target_shock_count - 1)
        shift_buffer .= u
        circshift!(shift_buffer, u, -shift_steps * i)
        errs[i] = RDE.turbo_diff_norm(u, shift_buffer) / sqrt(T(N))
    end
    maxerr = RDE.turbo_maximum(errs)
    return T(1) - (maxerr - T(0.1)) / T(0.5)
end

# ----------------------------------------------------------------------------
# PeriodicityReward
# ----------------------------------------------------------------------------

struct PeriodicityReward <: AbstractScalarRewardStrategy end

initialize_cache(::PeriodicityReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

function _compute_reward(env::RDEEnv{T, A, O, R, G, V, OBS}, ::PeriodicityReward, cache::RewardShiftBufferCache{T}) where {T <: AbstractFloat, A, O, R, G, V, OBS}
    u, = RDE.split_sol_view(env.state)
    N::Int = env.prob.params.N
    dx::T = env.prob.x[2] - env.prob.x[1]
    L::T = env.prob.params.L
    shock_inds = RDE.shock_indices(u, dx)
    shocks::Int = length(shock_inds)

    shift_buffer = cache.shift_buffer
    if shocks > 1
        shift_steps::Int = N ÷ shocks
        errs::Vector{T} = zeros(T, shocks - 1)
        for i in 1:(shocks - 1)
            shift_buffer .= u
            circshift!(shift_buffer, u, -shift_steps * i)
            errs[i]::T = RDE.turbo_diff_norm(u, shift_buffer) / sqrt(N)
        end
        maxerr::T = RDE.turbo_maximum(errs)
        periodicity_reward = T(1) - (max(maxerr - T(0.08), zero(T)) / sqrt(T(3)))
    else
        periodicity_reward = T(1)
    end

    if shocks > 1
        optimal_spacing::T = L / shocks
        shock_spacing::Vector{T} = mod.(RDE.periodic_diff(shock_inds), N) .* dx
        shock_spacing_reward = T(1) - RDE.turbo_maximum_abs((shock_spacing .- optimal_spacing) ./ optimal_spacing)
    else
        shock_spacing_reward = T(1)
    end

    return (periodicity_reward::T + shock_spacing_reward::T) / T(2)
end


function Base.show(io::IO, ::PeriodicityReward)
    return print(io, "PeriodicityReward()")
end

function Base.show(io::IO, ::MIME"text/plain", ::PeriodicityReward)
    println(io, "PeriodicityReward")
    return nothing
end

# ----------------------------------------------------------------------------
# StabilityReward
# ----------------------------------------------------------------------------
"""
A reward that measures the stability of the solution over time. 
It consists of three parts:
An amplitude stability reward, a shock profile similarity and a shock spacing reward. 
The latter two are combined into a spatial stability reward by multiplication. 
The final reward is the average of the amplitude stability and spatial stability.

When the target shock count is not achieved, the reward could be as high as ~0.5, 
because of the amplitude stability reward. With the correct number of shocks,
the reward can approach 1.
"""
@kwdef struct StabilityReward{T <: AbstractFloat} <: AbstractScalarRewardStrategy
    variation_scaling::T = 4.0f0
end

initialize_cache(::StabilityReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

function _compute_reward(env::RDEEnv{T, A, O, R, G, V, OBS}, rew_strat::StabilityReward, cache::RewardShiftBufferCache{T}) where {T <: AbstractFloat, A, O, R, G, V, OBS}
    if isnothing(env.prob.sol)
        return zero(T)
    end

    # Extract all states from solution
    t_current = env.t
    sol_times = env.prob.sol.t::Vector{T}
    #TODO: maybe move this to reward_strategy struct?
    stability_window_duration = T(5) #only view states within the last stability_window_duration seconds (for long dt's to not punish if we start bad and end good)
    lookback_index::Int = findfirst(sol_times .>= (t_current - stability_window_duration))
    # only use states after
    sol_states = env.prob.sol.u[lookback_index:end]::Vector{Vector{T}}
    n_states::Int = length(sol_states)

    # Get constants
    N = env.prob.params.N
    dx = RDE.get_dx(env.prob)
    L = env.prob.params.L
    target_shock_count = get_target_shock_count(env)
    shift_buffer = cache.shift_buffer

    # Collect data across time
    min_values = Vector{T}(undef, n_states)
    max_values = Vector{T}(undef, n_states)
    periodicity_rewards = Vector{T}(undef, n_states)
    shock_spacing_rewards = Vector{T}(undef, n_states)
    for (i, state) in enumerate(sol_states)
        u = @view state[1:N]

        # Collect span data
        u_min, u_max = RDE.turbo_extrema(u)
        min_values[i] = u_min
        max_values[i] = u_max


        # Calculate periodicity using target shock count
        if target_shock_count > 1
            shift_steps = N ÷ target_shock_count
            errs = zeros(T, target_shock_count - 1)
            for j in 1:(target_shock_count - 1)
                shift_buffer .= u
                circshift!(shift_buffer, u, -shift_steps * j)
                err = RDE.turbo_diff_norm(u, shift_buffer) / sqrt(N)
                err = max(err - T(0.037), zero(T)) #since N ÷ 3 is not a divisor of N, we set the baseline a bit lower
                errs[j] = err
            end
            maxerr = RDE.turbo_maximum(errs)
            periodicity_rewards[i] = exp(-rew_strat.variation_scaling * maxerr)
        else
            periodicity_rewards[i] = T(1)
        end

        # Calculate shock spacing using target shock count
        u, _ = RDE.split_sol_view(state)
        shock_inds = RDE.shock_indices(u, dx)
        shocks = length(shock_inds)
        if shocks > 0
            optimal_spacing = L / target_shock_count
            if shocks == 1
                shock_spacing = [L]
            else
                shock_spacing = mod.(RDE.periodic_diff(shock_inds), N) .* dx
            end
            # Use exponential reward function for spacing
            spacing_diffs = abs.(shock_spacing .- optimal_spacing)
            spacing_diffs = max.(spacing_diffs .- T(7.8f-3), zero(T)) #since N ÷ 3 is not a divisor of N, we set the baseline a bit lower
            spacing_errors = spacing_diffs ./ optimal_spacing
            spacing_rewards = exp.(-rew_strat.variation_scaling .* spacing_errors)
            shock_spacing_rewards[i] = RDE.turbo_minimum(spacing_rewards)
        else
            shock_spacing_rewards[i] = T(0)
        end
    end

    min_variation_rew = span_variation_reward(min_values, rew_strat.variation_scaling * 2)
    max_variation_rew = span_variation_reward(max_values, rew_strat.variation_scaling * 2)
    span_variation_rew = min(min_variation_rew, max_variation_rew)

    # Punish no span, to discourage getting constant solutions and 0.25 total reward.
    spans = max_values - min_values
    max_span = maximum(spans)
    small_span_threshold = T(0.1)
    small_span_punishment = min(T(1), inv(small_span_threshold) * max_span)

    # Calculate worst periodicity and shock spacing over time
    worst_periodicity = minimum(periodicity_rewards)
    worst_shock_spacing = minimum(shock_spacing_rewards)

    @debug "span_variation_rew: $span_variation_rew"
    @debug "small_span_punishment: $small_span_punishment"

    amplitude_stability = span_variation_rew * small_span_punishment
    spatial_stability = worst_periodicity * worst_shock_spacing
    stability = (amplitude_stability + spatial_stability) / T(2)
    @debug "amplitude_stability: $amplitude_stability"
    @debug "spatial_stability: $spatial_stability"
    @debug "stability: $stability"
    return stability
end

# ----------------------------------------------------------------------------
# StabilityTargetReward
# ----------------------------------------------------------------------------
#TODO: constructor and types etc...
struct StabilityTargetReward{T <: AbstractFloat} <: AbstractScalarRewardStrategy
    stability_weight::T
    target_weight::T
    stability_reward::StabilityReward{T}
end

function StabilityTargetReward(;
        stability_weight::T = 0.5f0,
        target_weight::T = 0.5f0,
        kwargs...
    ) where {T <: AbstractFloat}
    return StabilityTargetReward{T}(
        stability_weight,
        target_weight,
        StabilityReward{T}(; kwargs...)
    )
end

initialize_cache(rew_strat::StabilityTargetReward, N::Int, ::Type{T}) where {T} =
    WrappedRewardCache(
    initialize_cache(rew_strat.stability_reward, N, T),
    NoCache()
)

function _compute_reward(env::RDEEnv{T, A, O, R, G, V, OBS}, rew_strat::StabilityTargetReward, cache::WrappedRewardCache) where {T <: AbstractFloat, A, O, R, G, V, OBS}
    # Compute stability reward using inner cache
    stability_value = _compute_reward(env, rew_strat.stability_reward, cache.inner_cache)

    # Count current shocks
    N = env.prob.params.N
    current_u = @view env.state[1:N]
    dx = RDE.get_dx(env.prob)
    current_shock_count = RDE.count_shocks(current_u, dx)

    # Get target from environment's goal cache
    target_shock_count = get_target_shock_count(env)

    # Compute target reward (1.0 if match, 0.0 otherwise)
    target_value = (current_shock_count == target_shock_count) ? one(T) : zero(T)

    # Weighted combination
    return rew_strat.stability_weight * stability_value + rew_strat.target_weight * target_value
end

function Base.show(io::IO, rew_strat::StabilityTargetReward)
    return print(io, "StabilityTargetReward(stability_weight=$(rew_strat.stability_weight), target_weight=$(rew_strat.target_weight))")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::StabilityTargetReward)
    println(io, "StabilityTargetReward:")
    println(io, "  stability_weight: $(rew_strat.stability_weight)")
    println(io, "  target_weight: $(rew_strat.target_weight)")
    return println(io, "  stability_reward: $(rew_strat.stability_reward)")
end

# ----------------------------------------------------------------------------
# MultiSectionReward
# ----------------------------------------------------------------------------

@kwdef mutable struct MultiSectionReward{T <: AbstractFloat} <: MultiAgentCachedCompositeReward
    n_sections::Int = 8
    lowest_action_magnitude_reward::T = 0.0f0 #reward will be \in [lowest_action_magnitude_reward, 1]
    weights::Vector{T} = Float32[1, 1, 5, 1]
end

initialize_cache(::MultiSectionReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

reward_value_type(::Type{T}, ::MultiSectionReward{T}) where {T} = Vector{T}

function Base.show(io::IO, rew_strat::MultiSectionReward)
    return print(io, "MultiSectionReward(n_sections=$(rew_strat.n_sections))")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::MultiSectionReward)
    println(io, "MultiSectionReward:")
    println(io, "  n_sections: $(rew_strat.n_sections)")
    println(io, "  lowest_action_magnitude_reward: $(rew_strat.lowest_action_magnitude_reward)")
    return println(io, "  weights: $(rew_strat.weights)")
end

function _compute_reward(env::RDEEnv{T, A, O, R, G, V, OBS}, rew_strat::MultiSectionReward, cache::RewardShiftBufferCache{T}) where {T, A, O, R, G, V, OBS}
    common_reward = global_reward(env, rew_strat, cache)
    efforts = compute_section_control_efforts(env, rew_strat.n_sections)
    individual_modifiers = action_magnitude_factor(rew_strat.lowest_action_magnitude_reward, efforts)
    return common_reward .* individual_modifiers
end


# ----------------------------------------------------------------------------
# CompositeReward
# ----------------------------------------------------------------------------

reward_value_type(::Type{T}, ::CachedCompositeReward) where {T} = T

mutable struct CompositeReward{T <: AbstractFloat} <: CachedCompositeReward
    lowest_action_magnitude_reward::T #reward will be \in [lowest_action_magnitude_reward, 1]
    span_reward::Bool
    weights::Vector{T}
    function CompositeReward{T}(;
            lowest_action_magnitude_reward::T = one(T),
            span_reward::Bool = true,
            weights::Vector{T} = [T(0.25), T(0.25), T(0.25), T(0.25)],
        ) where {T <: AbstractFloat}
        return new{T}(
            lowest_action_magnitude_reward,
            span_reward, weights
        )
    end
    function CompositeReward(;
            lowest_action_magnitude_reward::Float32 = 1.0f0,
            span_reward::Bool = true,
            weights::Vector{Float32} = [0.25f0, 0.25f0, 0.25f0, 0.25f0],
        )
        return CompositeReward{Float32}(
            lowest_action_magnitude_reward = lowest_action_magnitude_reward,
            span_reward = span_reward,
            weights = weights
        )
    end
end

initialize_cache(::CompositeReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

function Base.show(io::IO, ::CompositeReward)
    return print(io, "CompositeReward()")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::CompositeReward)
    println(io, "CompositeReward:")
    println(io, "  lowest_action_magnitude_reward: $(rew_strat.lowest_action_magnitude_reward)")
    println(io, "  span_reward: $(rew_strat.span_reward)")
    return println(io, "  weights: $(rew_strat.weights)")
end

function _compute_reward(env::RDEEnv{T, A, O, R, G, V, OBS}, rew_strat::CompositeReward, cache::RewardShiftBufferCache{T}) where {T, A, O, R, G, V, OBS}
    reward = global_reward(env, rew_strat, cache)
    # Control-effort scalar modifier based on Δu_p
    Δmax = RDE.turbo_maximum_abs_diff(env.prob.method.cache.u_p_current, env.prob.method.cache.u_p_previous) / env.u_pmax
    action_effort_modifier = action_magnitude_factor(rew_strat.lowest_action_magnitude_reward, Δmax)
    reward *= action_effort_modifier
    @logmsg LogLevel(-10000) "set_reward!: $reward"
    return reward
end

# ----------------------------------------------------------------------------
# TimeAggCompositeReward
# ----------------------------------------------------------------------------

abstract type TimeAggregation end
struct TimeAvg <: TimeAggregation end
struct TimeMax <: TimeAggregation end
struct TimeMin <: TimeAggregation end
struct TimeSum <: TimeAggregation end
struct TimeProd <: TimeAggregation end

struct TimeAggCompositeReward{T <: AbstractFloat} <: CachedCompositeReward
    aggregation::TimeAggregation
    lowest_action_magnitude_reward::T #reward will be \in [lowest_action_magnitude_reward, 1]
    span_reward::Bool
    weights::Vector{T}
    function TimeAggCompositeReward{T}(;
            aggregation::TimeAggregation = TimeMin(),
            lowest_action_magnitude_reward::T = one(T),
            span_reward::Bool = true,
            weights::Vector{T} = [T(0.25), T(0.25), T(0.25), T(0.25)],
        ) where {T <: AbstractFloat}
        return new{T}(
            aggregation,
            lowest_action_magnitude_reward,
            span_reward, weights
        )
    end
    function TimeAggCompositeReward(;
            aggregation::TimeAggregation = TimeMin(),
            lowest_action_magnitude_reward::T = T(1.0),
            span_reward::Bool = true,
            weights::Vector{T} = [T(0.25), T(0.25), T(0.25), T(0.25)],
        ) where {T <: AbstractFloat}
        return TimeAggCompositeReward{T}(
            aggregation = aggregation,
            lowest_action_magnitude_reward = lowest_action_magnitude_reward,
            span_reward = span_reward,
            weights = weights
        )
    end
end

initialize_cache(::TimeAggCompositeReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

function Base.show(io::IO, rew_strat::TimeAggCompositeReward)
    return print(io, "TimeAggCompositeReward(aggregation=$(rew_strat.aggregation))")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::TimeAggCompositeReward)
    println(io, "TimeAggCompositeReward:")
    println(io, "  aggregation: $(rew_strat.aggregation)")
    println(io, "  lowest_action_magnitude_reward: $(rew_strat.lowest_action_magnitude_reward)")
    println(io, "  span_reward: $(rew_strat.span_reward)")
    return println(io, "  weights: $(rew_strat.weights)")
end

reward_value_type(::Type{T}, ::TimeAggCompositeReward) where {T} = T

function aggregate(rewards::AbstractVector, aggregation::TimeAvg)
    return mean(rewards)
end
function aggregate(rewards::AbstractVector, aggregation::TimeMax)
    return maximum(rewards)
end
function aggregate(rewards::AbstractVector, aggregation::TimeMin)
    return minimum(rewards)
end
function aggregate(rewards::AbstractVector, aggregation::TimeSum)
    return sum(rewards)
end
function aggregate(rewards::AbstractVector, aggregation::TimeProd)
    return prod(rewards)
end

function _compute_reward(env::RDEEnv{T, A, O, R, G, V, OBS}, rew_strat::TimeAggCompositeReward, cache::RewardShiftBufferCache{T}) where {T, A, O, R, G, V, OBS}
    N = env.prob.params.N
    dx = env.prob.x[2] - env.prob.x[1]
    L = env.prob.params.L
    shift_buffer = cache.shift_buffer
    target_shock_count = get_target_shock_count(env)
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
        rewards[i] = global_reward(u, L, dx, rew_strat, shift_buffer, target_shock_count)
    end
    agg_reward = aggregate(rewards, rew_strat.aggregation)

    Δmax = RDE.turbo_maximum_abs_diff(env.prob.method.cache.u_p_current, env.prob.method.cache.u_p_previous) / env.u_pmax
    action_effort_modifier = action_magnitude_factor(rew_strat.lowest_action_magnitude_reward, Δmax)
    return agg_reward * action_effort_modifier
end

# ----------------------------------------------------------------------------
# ConstantTargetReward
# ----------------------------------------------------------------------------

@kwdef struct ConstantTargetReward{T <: AbstractFloat} <: AbstractScalarRewardStrategy
    target::T = 0.64f0
end

function Base.show(io::IO, rew_strat::ConstantTargetReward)
    return print(io, "ConstantTargetReward(target=$(rew_strat.target))")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::ConstantTargetReward)
    println(io, "ConstantTargetReward:")
    return println(io, "  target: $(rew_strat.target)")
end

_compute_reward(env::RDEEnv{T, A, O, R, G, V, OBS}, rew_strat::ConstantTargetReward{T}, cache) where {T, A, O, R, G, V, OBS} =
    -abs(rew_strat.target - mean(env.prob.method.cache.u_p_current)) + one(T)

# ----------------------------------------------------------------------------
# TimeAggMultiSectionReward
# ----------------------------------------------------------------------------

mutable struct TimeAggMultiSectionReward{T <: AbstractFloat} <: MultiAgentCachedCompositeReward
    aggregation::TimeAggregation
    n_sections::Int
    lowest_action_magnitude_reward::T #reward will be \in [lowest_action_magnitude_reward, 1]
    weights::Vector{T}
end
reward_value_type(::Type{T}, ::TimeAggMultiSectionReward{T}) where {T} = Vector{T}

function Base.show(io::IO, rew_strat::TimeAggMultiSectionReward)
    return print(io, "TimeAggMultiSectionReward(n_sections=$(rew_strat.n_sections), aggregation=$(typeof(rew_strat.aggregation)))")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::TimeAggMultiSectionReward)
    println(io, "TimeAggMultiSectionReward:")
    println(io, "  aggregation: $(typeof(rew_strat.aggregation))")
    println(io, "  n_sections: $(rew_strat.n_sections)")
    println(io, "  lowest_action_magnitude_reward: $(rew_strat.lowest_action_magnitude_reward)")
    return println(io, "  weights: $(rew_strat.weights)")
end

function _compute_reward(env::RDEEnv{T, A, O, R, G, V, OBS}, rew_strat::TimeAggMultiSectionReward, cache::RewardShiftBufferCache{T}) where {T, A, O, R, G, V, OBS}
    N = env.prob.params.N
    n_sections = rew_strat.n_sections
    shift_buffer = cache.shift_buffer
    target_shock_count = get_target_shock_count(env)

    # Get all states from the solution
    if !isnothing(env.prob.sol)
        us = [uλ[1:N] for uλ in env.prob.sol.u[2:end]]
    else
        us = [env.state[1:N]]
    end
    section_rewards = zeros(T, n_sections, length(us))

    # Calculate rewards for each section and time step
    for (i, u) in enumerate(us)
        common_reward = global_reward(u, env.prob.params.L, env.prob.x[2] - env.prob.x[1], rew_strat, shift_buffer, target_shock_count)
        efforts = compute_section_control_efforts(env, n_sections)
        individual_modifiers = action_magnitude_factor(rew_strat.lowest_action_magnitude_reward, efforts)
        section_rewards[:, i] = common_reward .* individual_modifiers
    end

    # Aggregate rewards over time for each section
    agg_section_rewards = zeros(T, n_sections)
    for i in 1:n_sections
        agg_section_rewards[i] = aggregate(section_rewards[i, :], rew_strat.aggregation)
    end

    return agg_section_rewards
end

# Ensure T is known when building defaults
function TimeAggMultiSectionReward{T}(;
        aggregation::TimeAggregation = TimeMin(),
        n_sections::Int = 4,
        lowest_action_magnitude_reward::T = zero(T),
        weights::Vector{T} = [one(T), one(T), T(5), one(T)]
    ) where {T <: AbstractFloat}
    return TimeAggMultiSectionReward{T}(
        aggregation,
        n_sections,
        lowest_action_magnitude_reward,
        weights
    )
end

function TimeAggMultiSectionReward(;
        aggregation::TimeAggregation = TimeMin(),
        n_sections::Int = 4,
        lowest_action_magnitude_reward::Float32 = 0.0f0,
        weights::Vector{Float32} = [1.0f0, 1.0f0, 5.0f0, 1.0f0]
    )
    return TimeAggMultiSectionReward{Float32}(
        aggregation,
        n_sections,
        lowest_action_magnitude_reward,
        weights
    )
end

initialize_cache(::TimeAggMultiSectionReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

# ----------------------------------------------------------------------------
# TimeDiffNormReward
# ----------------------------------------------------------------------------

@kwdef struct TimeDiffNormReward{T <: AbstractFloat} <: AbstractScalarRewardStrategy
    threshold::T = 1.1f0
    threshold_reward::T = 0.3f0
end

function _compute_reward(env::RDEEnv{T, A, O, R, G, V, OBS}, rew_strat::TimeDiffNormReward{T}, cache) where {T, A, O, R, G, V, OBS}
    if isnothing(env.prob.sol)
        return zero(T)
    end
    # Spatial size for each snapshot
    N = env.prob.params.N
    # Extract only the velocity field u from each saved state [u; λ]
    us = map(v -> v[1:N], env.prob.sol.u)
    n = length(us)
    # We will compute pairwise spectral-distance between all time snapshots.
    # There are n*(n-1)/2 unique pairs, store their norms in a flat vector.
    diff_norms = zeros(T, Int((n^2 - n) // 2))
    ind = 1
    # FFT magnitude spectra for each snapshot; use real FFT for efficiency.
    ft_us = rfft.(us)
    abs_ft_us = map(v -> abs.(v), ft_us)
    for i in 1:n, j in (i + 1):n
        # L2 distance between magnitude spectra, normalized by sqrt(N)
        diff_norms[ind] = RDE.turbo_diff_norm(abs_ft_us[i], abs_ft_us[j]) / sqrt(N)
        ind += 1
    end
    # Use fast turbo maximum for the worst-case spectral change across time
    max_norm = RDE.turbo_maximum(diff_norms)
    # Map max_norm to (0,1] via exponential shaping so that
    #   reward = threshold_reward when max_norm == threshold
    a = log(rew_strat.threshold_reward) / rew_strat.threshold
    return exp(a * max_norm)
end

# ----------------------------------------------------------------------------
# PeriodMinimumReward
# ----------------------------------------------------------------------------

struct PeriodMinimumReward{T <: AbstractFloat} <: CachedCompositeReward
    lowest_action_magnitude_reward::T #reward will be \in [lowest_action_magnitude_reward, 1]
    weights::Vector{T}
    function PeriodMinimumReward{T}(;
            lowest_action_magnitude_reward::T = zero(T),
            weights::Vector{T} = [one(T), one(T), T(5), one(T)],
        ) where {T <: AbstractFloat}
        return new{T}(
            lowest_action_magnitude_reward,
            weights
        )
    end
    function PeriodMinimumReward(;
            lowest_action_magnitude_reward::Float32 = 0.0f0,
            weights::Vector{Float32} = Float32[1, 1, 5, 1],
        )
        return PeriodMinimumReward{Float32}(
            lowest_action_magnitude_reward = lowest_action_magnitude_reward,
            weights = weights
        )
    end
end

initialize_cache(::PeriodMinimumReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

function Base.show(io::IO, rew_strat::PeriodMinimumReward)
    return print(io, "PeriodMinimumReward()")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::PeriodMinimumReward)
    println(io, "PeriodMinimumReward:")
    println(io, "  lowest_action_magnitude_reward: $(rew_strat.lowest_action_magnitude_reward)")
    return println(io, "  weights: $(rew_strat.weights)")
end

function time_minimum_global_reward(env::RDEEnv{T, A, O, R, G, V, OBS}, rew_strat::CachedCompositeReward, cache::RewardShiftBufferCache{T})::T where {T <: AbstractFloat, A, O, R, G, V, OBS}
    N::Int = env.prob.params.N
    if isnothing(env.prob.sol)
        return zero(T)
    end

    # Pre-compute constants to avoid type instability in loop
    L::T = env.prob.params.L
    dx::T = RDE.get_dx(env.prob)
    weight_sum::T = sum(rew_strat.weights)
    shift_buffer = cache.shift_buffer
    target_shock_count = get_target_shock_count(env)

    # Initialize with large values to find minimum
    min_rewards::Vector{T} = fill(T(10000), 5)

    # Type-stable iteration - convert to Vector{Vector{T}} if needed
    sol_states = env.prob.sol.u::Vector{Vector{T}}

    for state::Vector{T} in sol_states
        # Use view to avoid allocation, with fallback type assertion
        u = @view state[1:N]

        span_rew, periodicity_rew, shock_rew, shock_spacing_rew, low_span_punishment =
            global_rewards(u, L, dx, rew_strat, shift_buffer, target_shock_count)

        # Update minimums efficiently
        min_rewards[1] = min(min_rewards[1], span_rew)
        min_rewards[2] = min(min_rewards[2], periodicity_rew)
        min_rewards[3] = min(min_rewards[3], shock_rew)
        min_rewards[4] = min(min_rewards[4], shock_spacing_rew)
        min_rewards[5] = min(min_rewards[5], low_span_punishment)
    end

    # Efficient weighted sum without temporary arrays - manually unrolled
    weighted_rewards = (
        min_rewards[1] * rew_strat.weights[1] +
            min_rewards[2] * rew_strat.weights[2] +
            min_rewards[3] * rew_strat.weights[3] +
            min_rewards[4] * rew_strat.weights[4]
    ) / weight_sum
    return min_rewards[5] * weighted_rewards  # low_span_punishment * weighted_sum
end

function _compute_reward(env::RDEEnv{T, A, O, R, G, V, OBS}, rew_strat::PeriodMinimumReward, cache::RewardShiftBufferCache{T}) where {T, A, O, R, G, V, OBS}
    global_reward = time_minimum_global_reward(env, rew_strat, cache)
    Δmax = RDE.turbo_maximum_abs_diff(env.prob.method.cache.u_p_current, env.prob.method.cache.u_p_previous) / env.u_pmax
    action_effort_modifier = action_magnitude_factor(rew_strat.lowest_action_magnitude_reward, Δmax)
    return global_reward * action_effort_modifier
end

reward_value_type(::Type{T}, ::PeriodMinimumReward) where {T} = T

struct PeriodMinimumVariationReward{T <: AbstractFloat} <: CachedCompositeReward
    lowest_action_magnitude_reward::T #reward will be \in [lowest_action_magnitude_reward, 1]
    weights::Vector{T}
    variation_penalties::Vector{T}  # α values for each component [span, periodicity, shock, shock_spacing]
end
function PeriodMinimumVariationReward{T}(;
        lowest_action_magnitude_reward::T = one(T),
        weights::Vector{T} = [one(T), one(T), T(5), one(T)],
        variation_penalties::Vector{T} = [T(2), T(2), T(2), T(2)],
    ) where {T <: AbstractFloat}
    return PeriodMinimumVariationReward{T}(
        lowest_action_magnitude_reward,
        weights,
        variation_penalties
    )
end
function PeriodMinimumVariationReward(;
        lowest_action_magnitude_reward::Float32 = 1.0f0,
        weights::Vector{Float32} = Float32[1, 1, 5, 1],
        variation_penalties::Vector{Float32} = Float32[2.0, 2.0, 2.0, 2.0],
    )
    return PeriodMinimumVariationReward{Float32}(
        lowest_action_magnitude_reward = lowest_action_magnitude_reward,
        weights = weights,
        variation_penalties = variation_penalties
    )
end

initialize_cache(::PeriodMinimumVariationReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

function Base.show(io::IO, ::PeriodMinimumVariationReward)
    return print(io, "PeriodMinimumVariationReward()")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::PeriodMinimumVariationReward)
    println(io, "PeriodMinimumVariationReward:")
    println(io, "  lowest_action_magnitude_reward: $(rew_strat.lowest_action_magnitude_reward)")
    println(io, "  weights: $(rew_strat.weights)")
    return println(io, "  variation_penalties: $(rew_strat.variation_penalties)")
end

reward_value_type(::Type{T}, ::PeriodMinimumVariationReward) where {T} = T

function calculate_variation_punishment(values::Vector{T}, α::T) where {T <: AbstractFloat}
    if length(values) <= 1
        return one(T)  # No variation with single value
    end

    mean_abs_value = mean(abs.(values))
    if mean_abs_value < T(1.0e-6)
        return one(T)  # Avoid division by zero
    end

    coefficient_of_variation = std(values) / mean_abs_value
    return exp(-α * coefficient_of_variation)
end

function time_minimum_global_reward_with_variation(env::RDEEnv{T, A, O, R, G, V, OBS}, rew_strat::PeriodMinimumVariationReward, cache::RewardShiftBufferCache{T}) where {T <: AbstractFloat, A, O, R, G, V, OBS}
    N = env.prob.params.N
    if isnothing(env.prob.sol)
        return zero(T)
    end

    # Pre-compute constants to avoid type instability in loop
    L = env.prob.params.L::T
    dx = (env.prob.x[2] - env.prob.x[1])::T
    weight_sum = sum(rew_strat.weights)::T
    shift_buffer = cache.shift_buffer
    target_shock_count = get_target_shock_count(env)

    # Type-stable iteration
    sol_states = env.prob.sol.u
    n_states = length(sol_states)

    # Pre-allocate reward vectors with correct size
    all_span_rewards = Vector{T}(undef, n_states)
    all_periodicity_rewards = Vector{T}(undef, n_states)
    all_shock_rewards = Vector{T}(undef, n_states)
    all_shock_spacing_rewards = Vector{T}(undef, n_states)
    all_low_span_punishments = Vector{T}(undef, n_states)

    for (i, state) in enumerate(sol_states)
        # Use view to avoid allocation
        u = @view state[1:N]

        span_rew, periodicity_rew, shock_rew, shock_spacing_rew, low_span_punishment =
            global_rewards(u, L, dx, rew_strat, shift_buffer, target_shock_count)

        all_span_rewards[i] = span_rew
        all_periodicity_rewards[i] = periodicity_rew
        all_shock_rewards[i] = shock_rew
        all_shock_spacing_rewards[i] = shock_spacing_rew
        all_low_span_punishments[i] = low_span_punishment
    end

    # Calculate variation punishment for each component
    span_var_punishment = calculate_variation_punishment(all_span_rewards, T(rew_strat.variation_penalties[1]))
    periodicity_var_punishment = calculate_variation_punishment(all_periodicity_rewards, T(rew_strat.variation_penalties[2]))
    shock_var_punishment = calculate_variation_punishment(all_shock_rewards, T(rew_strat.variation_penalties[3]))
    shock_spacing_var_punishment = calculate_variation_punishment(all_shock_spacing_rewards, T(rew_strat.variation_penalties[4]))

    variation_punishment = minimum([span_var_punishment, periodicity_var_punishment, shock_var_punishment, shock_spacing_var_punishment])
    # Apply variation punishment to each component before taking minimum
    min_span_rew = minimum(all_span_rewards)
    min_periodicity_rew = minimum(all_periodicity_rewards)
    min_shock_rew = minimum(all_shock_rewards)
    min_shock_spacing_rew = minimum(all_shock_spacing_rewards)

    # Low span punishment: only minimum, no variation punishment
    min_low_span_punishment = minimum(all_low_span_punishments)

    # Efficient weighted sum
    weighted_rewards = (
        min_span_rew * rew_strat.weights[1] +
            min_periodicity_rew * rew_strat.weights[2] +
            min_shock_rew * rew_strat.weights[3] +
            min_shock_spacing_rew * rew_strat.weights[4]
    ) / weight_sum

    return min_low_span_punishment * variation_punishment * weighted_rewards
end

function _compute_reward(env::RDEEnv{T, A, O, R, G, V, OBS}, rew_strat::PeriodMinimumVariationReward, cache::RewardShiftBufferCache{T}) where {T, A, O, R, G, V, OBS}
    global_reward = time_minimum_global_reward_with_variation(env, rew_strat, cache)
    Δmax = RDE.turbo_maximum_abs_diff(env.prob.method.cache.u_p_current, env.prob.method.cache.u_p_previous) / env.u_pmax
    action_effort_modifier = action_magnitude_factor(rew_strat.lowest_action_magnitude_reward, Δmax)
    return global_reward * action_effort_modifier
end

@kwdef mutable struct MultiSectionPeriodMinimumReward{T <: AbstractFloat} <: MultiAgentCachedCompositeReward
    n_sections::Int = 8
    lowest_action_magnitude_reward::T = 0.0f0
    weights::Vector{T} = Float32[1, 1, 5, 1]
end

initialize_cache(::MultiSectionPeriodMinimumReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

function Base.show(io::IO, rew_strat::MultiSectionPeriodMinimumReward)
    return print(io, "MultiSectionPeriodMinimumReward(n_sections=$(rew_strat.n_sections))")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::MultiSectionPeriodMinimumReward)
    println(io, "MultiSectionPeriodMinimumReward:")
    println(io, "  n_sections: $(rew_strat.n_sections)")
    println(io, "  lowest_action_magnitude_reward: $(rew_strat.lowest_action_magnitude_reward)")
    return println(io, "  weights: $(rew_strat.weights)")
end

reward_value_type(::Type{T}, ::MultiSectionPeriodMinimumReward{T}) where {T} = Vector{T}

function _compute_reward(env::RDEEnv{T, A, O, R, G, V, OBS}, rew_strat::MultiSectionPeriodMinimumReward, cache::RewardShiftBufferCache{T}) where {T, A, O, R, G, V, OBS}
    # Compute the time minimum global reward (same as PeriodMinimumReward)
    common_reward = time_minimum_global_reward(env, rew_strat, cache)

    # Multi-section control-effort punishment
    efforts = compute_section_control_efforts(env, rew_strat.n_sections)
    individual_modifiers = action_magnitude_factor(rew_strat.lowest_action_magnitude_reward, efforts)

    return common_reward .* individual_modifiers
end

"""
    MultiplicativeReward <: AbstractRewardStrategy

A reward that multiplies the rewards from multiple reward types.

# Fields
- `rewards::Vector{<:AbstractRewardStrategy}`: Vector of reward components to multiply

# Notes
- If all wrapped rewards return scalar values, this returns a scalar
- If any wrapped reward returns a vector, this returns a vector (element-wise multiplication)
"""
struct MultiplicativeReward <: AbstractRewardStrategy
    rewards::Vector{<:AbstractRewardStrategy}

    function MultiplicativeReward(rewards::Vector{<:AbstractRewardStrategy})
        return new(rewards)
    end

    function MultiplicativeReward(rewards::AbstractRewardStrategy...)
        return new([rewards...])
    end
end

function Base.show(io::IO, rew_strat::MultiplicativeReward)
    return print(io, "MultiplicativeReward($(length(rew_strat.rewards)) components)")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::MultiplicativeReward)
    println(io, "MultiplicativeReward:")
    println(io, "  components: $(length(rew_strat.rewards))")
    for (i, reward) in enumerate(rew_strat.rewards)
        println(io, "  reward $i: $(typeof(reward))")
    end
    return
end

initialize_cache(rew_strat::MultiplicativeReward, N::Int, ::Type{T}) where {T} =
    CompositeRewardCache(Tuple(initialize_cache(r, N, T) for r in rew_strat.rewards))

function _compute_reward(env::RDEEnv{T, A, O, R, G, V, OBS}, rew_strat::MultiplicativeReward, cache::CompositeRewardCache) where {T, A, O, R, G, V, OBS}
    # Compute rewards with their respective caches
    reward = _compute_reward(env, rew_strat.rewards[1], cache.caches[1])

    for (i, r) in enumerate(rew_strat.rewards[2:end])
        next_reward = _compute_reward(env, r, cache.caches[i + 1])
        reward = next_reward .* reward
    end

    return reward
end

function reward_value_from_wrapper(::Type{T}, rts::Vector{<:AbstractRewardStrategy}) where {T}
    value_types = reward_value_type.(T, rts)
    if any(value_types .== Vector{T})
        return Vector{T}
    else
        return T
    end
end

function reward_value_type(::Type{T}, rew_strat::MultiplicativeReward) where {T}
    return reward_value_from_wrapper(T, rew_strat.rewards)
end

"""
ExponentialAverageReward{T<:AbstractRewardStrategy} <: AbstractRewardStrategy

A reward wrapper that applies exponential averaging to smooth reward signals.

The exponential average is computed as:
`new_avg = α * current_reward + (1-α) * old_avg`

# Fields
- `wrapped_reward::T`: The underlying reward to wrap
- `α::Float32`: Averaging parameter (0 < α ≤ 1). Higher values give more weight to recent rewards

# Notes
- If wrapped reward returns scalar, this returns scalar exponential average
- If wrapped reward returns vector, this returns element-wise exponential averages
- The average state is stored in the cache and reset on `reset_cache!()` calls
"""
struct ExponentialAverageReward{R <: AbstractRewardStrategy, T <: AbstractFloat} <: AbstractRewardStrategy
    wrapped_reward::R
    α::T  # averaging parameter

    function ExponentialAverageReward{R, T}(wrapped_reward::R; α::T = T(0.2)) where {R <: AbstractRewardStrategy, T <: AbstractFloat}
        return new{R, T}(wrapped_reward, α)
    end
    function ExponentialAverageReward(wrapped_reward::R; α::Float32 = 0.2f0) where {R <: AbstractRewardStrategy}
        return ExponentialAverageReward{R, Float32}(wrapped_reward, α = α)
    end
end

function Base.show(io::IO, rew_strat::ExponentialAverageReward)
    return print(io, "ExponentialAverageReward(α=$(rew_strat.α), wrapped=$(typeof(rew_strat.wrapped_reward)))")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::ExponentialAverageReward)
    println(io, "ExponentialAverageReward:")
    println(io, "  α: $(rew_strat.α)")
    return println(io, "  wrapped_reward: $(typeof(rew_strat.wrapped_reward))")
end

function reward_value_type(::Type{T}, rew_strat::ExponentialAverageReward) where {T}
    return reward_value_type(T, rew_strat.wrapped_reward)
end

initialize_cache(rew_strat::ExponentialAverageReward, N::Int, ::Type{T}) where {T} =
    WrappedRewardCache(
    initialize_cache(rew_strat.wrapped_reward, N, T),
    ExponentialAverageCache{T}(nothing)
)

function _compute_reward(env::RDEEnv{T, A, O, R, G, V, OBS}, rew_strat::ExponentialAverageReward, cache::WrappedRewardCache) where {T, A, O, R, G, V, OBS}
    # Use the inner cache for the wrapped reward
    current_reward = _compute_reward(env, rew_strat.wrapped_reward, cache.inner_cache)

    if isnothing(cache.outer_cache.average)
        # Initialize average with first reward
        cache.outer_cache.average = current_reward
        return current_reward
    else
        # Apply exponential averaging: new_avg = α * current + (1-α) * old_avg
        cache.outer_cache.average = rew_strat.α .* current_reward .+ (1 - rew_strat.α) .* cache.outer_cache.average
        return cache.outer_cache.average
    end
end

"""
    TransitionBasedReward{T<:AbstractRewardStrategy} <: AbstractRewardStrategy

A reward wrapper that gives -1*dt reward until a successful transition is detected,
then terminates the environment. Uses transition detection logic similar to detect_transition.

# Fields
- `wrapped_reward::T`: The underlying reward to wrap and monitor
- `reward_stability_length::Float32`: Time duration (in env time units) to maintain stable rewards
- `reward_threshold::Float32`: Minimum reward threshold for stability check

# Notes
- Returns -1.0*dt until transition is found
- When transition is detected, sets env.terminated = true
- Transition occurs when: shock count reaches target AND rewards stay above threshold for stability_length time
- Internal state (past_rewards, past_shock_counts, transition_found) is stored in the cache
"""
@kwdef struct TransitionBasedReward{R <: AbstractScalarRewardStrategy, T <: AbstractFloat} <: AbstractScalarRewardStrategy
    wrapped_reward::R = StabilityReward(1.0f0)
    reward_stability_length::T = 20.0f0 # in time units
    reward_threshold::T = 0.9f0
end

function TransitionBasedReward(
        wrapped_reward::AbstractScalarRewardStrategy;
        kwargs...
    )
    return TransitionBasedReward(; wrapped_reward, kwargs...)
end

function Base.show(io::IO, rew_strat::TransitionBasedReward)
    return print(io, "TransitionBasedReward(wrapped=$(typeof(rew_strat.wrapped_reward)))")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::TransitionBasedReward)
    println(io, "TransitionBasedReward:")
    println(io, "  reward_stability_length: $(rew_strat.reward_stability_length)")
    println(io, "  reward_threshold: $(rew_strat.reward_threshold)")
    return println(io, "  wrapped_reward: $(typeof(rew_strat.wrapped_reward))")
end

initialize_cache(rew_strat::TransitionBasedReward, N::Int, ::Type{T}) where {T} =
    WrappedRewardCache(
    initialize_cache(rew_strat.wrapped_reward, N, T),
    TransitionBasedCache{T}(T[], Int[], false)
)

function _compute_reward(env::RDEEnv{T, A, O, R, G, V, OBS}, rew_strat::TransitionBasedReward{S, T}, cache::WrappedRewardCache) where {T, A, O, R, G, V, OBS, S}
    # Compute the wrapped reward using inner cache
    wrapped_reward_value = _compute_reward(env, rew_strat.wrapped_reward, cache.inner_cache)

    # Handle vector rewards by taking minimum (similar to detect_transition)
    reward_scalar = if wrapped_reward_value isa AbstractVector
        minimum(wrapped_reward_value)
    else
        wrapped_reward_value
    end

    # Count current shocks
    N = env.prob.params.N
    u = @view env.state[1:N]
    dx = env.prob.x[2] - env.prob.x[1]
    current_shock_count = RDE.count_shocks(u, dx)
    target_shock_count = get_target_shock_count(env)

    # Update history in outer cache
    push!(cache.outer_cache.past_rewards, T(reward_scalar))
    push!(cache.outer_cache.past_shock_counts, current_shock_count)

    # Check for transition (only if we have enough history)
    if length(cache.outer_cache.past_shock_counts) >= 2
        cache.outer_cache.transition_found = detect_transition_realtime(rew_strat, cache.outer_cache, env.dt, target_shock_count)

        if cache.outer_cache.transition_found
            env.terminated = true
            @logmsg LogLevel(-500) "Transition detected! Terminating environment at t=$(env.t)"
            env.info["Termination.Reason"] = "Transition detected"
            env.info["Termination.env_t"] = env.t
            return T(100.0)  # Positive reward when transition found
        end
    end

    # Return -1*dt until transition
    return T(-env.dt)
end

function detect_transition_realtime(rew_strat::TransitionBasedReward{T, U}, cache::TransitionBasedCache{U}, dt::U, target_shock_count::Int) where {T, U <: AbstractFloat}
    stability_steps = round(Int, rew_strat.reward_stability_length / dt)
    if length(cache.past_shock_counts) < stability_steps
        @debug "Not enough shock counts to detect transition"
        return false
    end


    # Check if we've had stable rewards for long enough since then
    stability_rewards = cache.past_rewards[(end - stability_steps + 1):end]
    stability_shock_counts = cache.past_shock_counts[(end - stability_steps + 1):end]
    if all(stability_shock_counts .== target_shock_count) && minimum(stability_rewards) > rew_strat.reward_threshold
        return true
    end
    return false
end

struct ScalarToVectorReward{T <: AbstractRewardStrategy} <: AbstractVectorRewardStrategy
    wrapped_reward::T
    n::Int
end

initialize_cache(rew_strat::ScalarToVectorReward, N::Int, ::Type{T}) where {T} =
    WrappedRewardCache(
    initialize_cache(rew_strat.wrapped_reward, N, T),
    NoCache()
)

function _compute_reward(env::RDEEnv{T, A, O, R, G, V, OBS}, rew_strat::ScalarToVectorReward, cache::WrappedRewardCache) where {T, A, O, R, G, V, OBS}
    reward = _compute_reward(env, rew_strat.wrapped_reward, cache.inner_cache)
    return fill(reward, rew_strat.n)
end
