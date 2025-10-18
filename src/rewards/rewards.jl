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

# ============================================================================
# Reward Types and Implementations
# ============================================================================

# ----------------------------------------------------------------------------
# ShockSpanReward
# ----------------------------------------------------------------------------

@kwdef struct ShockSpanReward <: AbstractRDEReward
    span_scale::Float32 = 4.0f0
    shock_weight::Float32 = 0.8f0
end

reward_value_type(::Type{T}, ::ShockSpanReward) where {T} = T
initialize_cache(::ShockSpanReward, N::Int, ::Type{T}) where {T} = NoCache()
function _compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::ShockSpanReward, cache) where {T, A, O, R, V, OBS}
    target_shock_count = get_target_shock_count(env)
    max_span = rt.span_scale
    λ = rt.shock_weight

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

@kwdef mutable struct ShockPreservingReward <: AbstractRDEReward
    target_shock_count::Int = 3
    span_scale::Float32 = 4.0f0
    shock_weight::Float32 = 0.8f0
    abscence_limit::Float32 = 5.0f0
    abscence_start::Union{Float32, Nothing} = nothing
end

reward_value_type(::Type{T}, ::ShockPreservingReward) where {T} = T
initialize_cache(::ShockPreservingReward, N::Int, ::Type{T}) where {T} = NoCache()

# ----------------------------------------------------------------------------
# ShockPreservingSymmetryReward
# ----------------------------------------------------------------------------

mutable struct ShockPreservingSymmetryReward <: AbstractRDEReward
    target_shock_count::Int
    function ShockPreservingSymmetryReward(;
            target_shock_count::Int = 4
        )
        return new(target_shock_count)
    end
end

reward_value_type(::Type{T}, ::ShockPreservingSymmetryReward) where {T} = T
initialize_cache(::ShockPreservingSymmetryReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

# ----------------------------------------------------------------------------
# PeriodicityReward
# ----------------------------------------------------------------------------

struct PeriodicityReward <: AbstractRDEReward end

reward_value_type(::Type{T}, ::PeriodicityReward) where {T} = T
initialize_cache(::PeriodicityReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))
# Default reset interface for rewards - does nothing by default
function reset_reward!(rt::AbstractRDEReward)
    return nothing
end

function set_reward!(env::AbstractRDEEnv, rt::AbstractRDEReward)
    #not doing in place because reward can change size
    #TODO: initialize env.reward to correct size so we can do .= here
    env.reward = compute_reward(env, rt)
    return nothing
end

# External API - delegates to internal implementation with cache
compute_reward(env::AbstractRDEEnv, rt::AbstractRDEReward) = _compute_reward(env, rt, env.cache.reward_cache)

# Internal implementation - takes cache as explicit parameter
# Each reward type implements _compute_reward instead of compute_reward


function _compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::ShockPreservingReward, cache) where {T, A, O, R, V, OBS}
    target_shock_count = rt.target_shock_count  # Note: This reward uses its own target, not env.cache.goal
    max_span = rt.span_scale
    λ = rt.shock_weight

    u, = RDE.split_sol_view(env.state)
    dx = env.prob.x[2] - env.prob.x[1]
    N = env.prob.params.N
    L = env.prob.params.L
    shock_inds = RDE.shock_indices(u, dx)

    u_min, u_max = RDE.turbo_extrema(u)
    span = u_max - u_min
    span_reward = span / max_span

    if length(shock_inds) != target_shock_count
        if isnothing(rt.abscence_start)
            rt.abscence_start = env.t
        elseif env.t - rt.abscence_start > rt.abscence_limit
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

reward_value_type(::Type{T}, ::ShockPreservingReward) where {T} = T

function _compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::ShockPreservingSymmetryReward, cache::RewardShiftBufferCache{T}) where {T, A, O, R, V, OBS}
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

reward_value_type(::Type{T}, ::ShockPreservingSymmetryReward) where {T} = T

function _compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::PeriodicityReward, cache::RewardShiftBufferCache{T}) where {T <: AbstractFloat, A, O, R, V, OBS}
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

reward_value_type(::Type{T}, ::PeriodicityReward) where {T} = T

function Base.show(io::IO, rt::PeriodicityReward)
    return print(io, "PeriodicityReward()")
end

function Base.show(io::IO, ::MIME"text/plain", rt::PeriodicityReward)
    println(io, "PeriodicityReward")
    return nothing
end

function action_magnitude_factor(lowest_action_magnitude_reward::AbstractFloat, action_magnitudes)
    α = lowest_action_magnitude_reward
    action_magnitude_inv = 1 .- action_magnitudes
    return α .+ (1 - α) .* action_magnitude_inv
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

mutable struct MultiSectionReward{T <: AbstractFloat} <: MultiAgentCachedCompositeReward
    n_sections::Int
    lowest_action_magnitude_reward::T #reward will be \in [lowest_action_magnitude_reward, 1]
    weights::Vector{T}
end

# Ensure T is known when building defaults
function MultiSectionReward{T}(;
        n_sections::Int = 4,
        lowest_action_magnitude_reward::T = zero(T),
        weights::Vector{T} = [one(T), one(T), T(5), one(T)]
    ) where {T <: AbstractFloat}
    return MultiSectionReward{T}(
        n_sections,
        lowest_action_magnitude_reward,
        weights
    )
end

function MultiSectionReward(;
        n_sections::Int = 4,
        target_shock_count::Int = 3,  # Ignored, kept for backward compat
        N::Int = 512,  # Ignored, kept for backward compat
        lowest_action_magnitude_reward::Float32 = 0.0f0,
        weights::Vector{Float32} = [1.0f0, 1.0f0, 5.0f0, 1.0f0]
    )
    return MultiSectionReward{Float32}(
        n_sections,
        lowest_action_magnitude_reward,
        weights
    )
end

initialize_cache(::MultiSectionReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

reward_value_type(::Type{T}, ::MultiSectionReward{T}) where {T} = Vector{T}

function Base.show(io::IO, rt::MultiSectionReward)
    return print(io, "MultiSectionReward(n_sections=$(rt.n_sections))")
end

function Base.show(io::IO, ::MIME"text/plain", rt::MultiSectionReward)
    println(io, "MultiSectionReward:")
    println(io, "  n_sections: $(rt.n_sections)")
    println(io, "  lowest_action_magnitude_reward: $(rt.lowest_action_magnitude_reward)")
    return println(io, "  weights: $(rt.weights)")
end

function _compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::MultiSectionReward, cache::RewardShiftBufferCache{T}) where {T, A, O, R, V, OBS}
    common_reward = global_reward(env, rt, cache)
    efforts = compute_section_control_efforts(env, rt.n_sections)
    individual_modifiers = action_magnitude_factor(rt.lowest_action_magnitude_reward, efforts)
    return common_reward .* individual_modifiers
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

function global_rewards(u::AbstractVector{T}, L::T, dx::T, rt::CachedCompositeReward, cache::AbstractVector{T}, target_shock_count::Int) where {T <: AbstractFloat}
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

function global_reward(u::AbstractVector{T}, L::T, dx::T, rt::CachedCompositeReward, cache::AbstractVector{T}, target_shock_count::Int) where {T <: AbstractFloat}
    span_reward, periodicity_reward, shock_reward, shock_spacing_reward, low_span_punishment = global_rewards(u, L, dx, rt, cache, target_shock_count)
    weighted_rewards = [span_reward, periodicity_reward, shock_reward, shock_spacing_reward]' * rt.weights / sum(rt.weights)
    global_reward = low_span_punishment * sum(weighted_rewards)
    @logmsg LogLevel(-10000) "global_reward: $global_reward"
    return global_reward
end

function global_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::CachedCompositeReward, cache::RewardShiftBufferCache{T}) where {T, A, O, R, V, OBS}
    N = env.prob.params.N
    dx = env.prob.x[2] - env.prob.x[1]
    L = env.prob.params.L
    u = env.state[1:N]
    shift_buffer = cache.shift_buffer
    target_shock_count = get_target_shock_count(env)

    return global_reward(u, L, dx, rt, shift_buffer, target_shock_count)
end

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
            target_shock_count::Int = 4,  # Ignored, kept for backward compat
            lowest_action_magnitude_reward::Float32 = 1.0f0,
            span_reward::Bool = true,
            weights::Vector{Float32} = [0.25f0, 0.25f0, 0.25f0, 0.25f0],
            N::Int = 512  # Ignored, kept for backward compat
        )
        return CompositeReward{Float32}(
            lowest_action_magnitude_reward = lowest_action_magnitude_reward,
            span_reward = span_reward,
            weights = weights
        )
    end
end

initialize_cache(::CompositeReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

function Base.show(io::IO, rt::CompositeReward)
    return print(io, "CompositeReward()")
end

function Base.show(io::IO, ::MIME"text/plain", rt::CompositeReward)
    println(io, "CompositeReward:")
    println(io, "  lowest_action_magnitude_reward: $(rt.lowest_action_magnitude_reward)")
    println(io, "  span_reward: $(rt.span_reward)")
    return println(io, "  weights: $(rt.weights)")
end

function _compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::CompositeReward, cache::RewardShiftBufferCache{T}) where {T, A, O, R, V, OBS}
    reward = global_reward(env, rt, cache)
    # Control-effort scalar modifier based on Δu_p
    Δmax = RDE.turbo_maximum_abs_diff(env.prob.method.cache.u_p_current, env.prob.method.cache.u_p_previous) / env.u_pmax
    action_effort_modifier = action_magnitude_factor(rt.lowest_action_magnitude_reward, Δmax)
    reward *= action_effort_modifier
    @logmsg LogLevel(-10000) "set_reward!: $reward"
    return reward
end

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
            N::Int = 512  # Ignored, kept for backward compat
        ) where {T <: AbstractFloat}
        return new{T}(
            aggregation,
            lowest_action_magnitude_reward,
            span_reward, weights
        )
    end
    function TimeAggCompositeReward(;
            aggregation::TimeAggregation = TimeMin(),
            target_shock_count::Int = 4,  # Ignored, kept for backward compat
            lowest_action_magnitude_reward::T = T(1.0),
            span_reward::Bool = true,
            weights::Vector{T} = [T(0.25), T(0.25), T(0.25), T(0.25)],
            N::Int = 512  # Ignored, kept for backward compat
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

function Base.show(io::IO, rt::TimeAggCompositeReward)
    return print(io, "TimeAggCompositeReward(aggregation=$(rt.aggregation))")
end

function Base.show(io::IO, ::MIME"text/plain", rt::TimeAggCompositeReward)
    println(io, "TimeAggCompositeReward:")
    println(io, "  aggregation: $(rt.aggregation)")
    println(io, "  lowest_action_magnitude_reward: $(rt.lowest_action_magnitude_reward)")
    println(io, "  span_reward: $(rt.span_reward)")
    return println(io, "  weights: $(rt.weights)")
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

function _compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::TimeAggCompositeReward, cache::RewardShiftBufferCache{T}) where {T, A, O, R, V, OBS}
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
        rewards[i] = global_reward(u, L, dx, rt, shift_buffer, target_shock_count)
    end
    agg_reward = aggregate(rewards, rt.aggregation)

    Δmax = RDE.turbo_maximum_abs_diff(env.prob.method.cache.u_p_current, env.prob.method.cache.u_p_previous) / env.u_pmax
    action_effort_modifier = action_magnitude_factor(rt.lowest_action_magnitude_reward, Δmax)
    return agg_reward * action_effort_modifier
end


struct ConstantTargetReward{T <: AbstractFloat} <: AbstractRDEReward
    target::T
    function ConstantTargetReward{T}(; target::T = T(0.64)) where {T <: AbstractFloat}
        return new{T}(target)
    end
    function ConstantTargetReward(; target::T = 0.64f0) where {T <: AbstractFloat}
        return ConstantTargetReward{T}(target = target)
    end
end

function Base.show(io::IO, rt::ConstantTargetReward)
    return print(io, "ConstantTargetReward(target=$(rt.target))")
end

function Base.show(io::IO, ::MIME"text/plain", rt::ConstantTargetReward)
    println(io, "ConstantTargetReward:")
    return println(io, "  target: $(rt.target)")
end

_compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::ConstantTargetReward{T}, cache) where {T, A, O, R, V, OBS} =
    -abs(rt.target - mean(env.prob.method.cache.u_p_current)) + one(T)

mutable struct TimeAggMultiSectionReward{T <: AbstractFloat} <: MultiAgentCachedCompositeReward
    aggregation::TimeAggregation
    n_sections::Int
    lowest_action_magnitude_reward::T #reward will be \in [lowest_action_magnitude_reward, 1]
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
    println(io, "  lowest_action_magnitude_reward: $(rt.lowest_action_magnitude_reward)")
    return println(io, "  weights: $(rt.weights)")
end

function _compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::TimeAggMultiSectionReward, cache::RewardShiftBufferCache{T}) where {T, A, O, R, V, OBS}
    N = env.prob.params.N
    n_sections = rt.n_sections
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
        common_reward = global_reward(u, env.prob.params.L, env.prob.x[2] - env.prob.x[1], rt, shift_buffer, target_shock_count)
        efforts = compute_section_control_efforts(env, n_sections)
        individual_modifiers = action_magnitude_factor(rt.lowest_action_magnitude_reward, efforts)
        section_rewards[:, i] = common_reward .* individual_modifiers
    end

    # Aggregate rewards over time for each section
    agg_section_rewards = zeros(T, n_sections)
    for i in 1:n_sections
        agg_section_rewards[i] = aggregate(section_rewards[i, :], rt.aggregation)
    end

    return agg_section_rewards
end

# Ensure T is known when building defaults
function TimeAggMultiSectionReward{T}(;
        aggregation::TimeAggregation = TimeMin(),
        n_sections::Int = 4,
        target_shock_count::Int = 3,  # Ignored, kept for backward compat
        N::Int = 512,  # Ignored, kept for backward compat
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
        target_shock_count::Int = 3,  # Ignored, kept for backward compat
        N::Int = 512,  # Ignored, kept for backward compat
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

struct TimeDiffNormReward{T <: AbstractFloat} <: AbstractRDEReward
    threshold::T
    threshold_reward::T
end

function TimeDiffNormReward{T}(; threshold::T = 1.1f0, threshold_reward::T = 0.3f0) where {T <: AbstractFloat}
    return TimeDiffNormReward{T}(threshold, threshold_reward)
end
function TimeDiffNormReward(; threshold::T = 1.1f0, threshold_reward::T = 0.3f0) where {T <: AbstractFloat}
    return TimeDiffNormReward{T}(threshold, threshold_reward)
end


function _compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::TimeDiffNormReward{T}, cache) where {T, A, O, R, V, OBS}
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
    a = log(rt.threshold_reward) / rt.threshold
    return exp(a * max_norm)
end

reward_value_type(::Type{T}, ::TimeDiffNormReward) where {T} = T

struct PeriodMinimumReward{T <: AbstractFloat} <: CachedCompositeReward
    lowest_action_magnitude_reward::T #reward will be \in [lowest_action_magnitude_reward, 1]
    weights::Vector{T}
    function PeriodMinimumReward{T}(;
            target_shock_count::Int = 4,  # Ignored, kept for backward compat
            lowest_action_magnitude_reward::T = zero(T),
            weights::Vector{T} = [one(T), one(T), T(5), one(T)],
            N::Int = 512  # Ignored, kept for backward compat
        ) where {T <: AbstractFloat}
        return new{T}(
            lowest_action_magnitude_reward,
            weights
        )
    end
    function PeriodMinimumReward(;
            target_shock_count::Int = 4,  # Ignored, kept for backward compat
            lowest_action_magnitude_reward::Float32 = 0.0f0,
            weights::Vector{Float32} = Float32[1, 1, 5, 1],
            N::Int = 512  # Ignored, kept for backward compat
        )
        return PeriodMinimumReward{Float32}(
            lowest_action_magnitude_reward = lowest_action_magnitude_reward,
            weights = weights
        )
    end
end

initialize_cache(::PeriodMinimumReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

function Base.show(io::IO, rt::PeriodMinimumReward)
    return print(io, "PeriodMinimumReward()")
end

function Base.show(io::IO, ::MIME"text/plain", rt::PeriodMinimumReward)
    println(io, "PeriodMinimumReward:")
    println(io, "  lowest_action_magnitude_reward: $(rt.lowest_action_magnitude_reward)")
    return println(io, "  weights: $(rt.weights)")
end

function time_minimum_global_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::CachedCompositeReward, cache::RewardShiftBufferCache{T})::T where {T <: AbstractFloat, A, O, R, V, OBS}
    N::Int = env.prob.params.N
    if isnothing(env.prob.sol)
        return zero(T)
    end

    # Pre-compute constants to avoid type instability in loop
    L::T = env.prob.params.L
    dx::T = RDE.get_dx(env.prob)
    weight_sum::T = sum(rt.weights)
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
            global_rewards(u, L, dx, rt, shift_buffer, target_shock_count)

        # Update minimums efficiently
        min_rewards[1] = min(min_rewards[1], span_rew)
        min_rewards[2] = min(min_rewards[2], periodicity_rew)
        min_rewards[3] = min(min_rewards[3], shock_rew)
        min_rewards[4] = min(min_rewards[4], shock_spacing_rew)
        min_rewards[5] = min(min_rewards[5], low_span_punishment)
    end

    # Efficient weighted sum without temporary arrays - manually unrolled
    weighted_rewards = (
        min_rewards[1] * rt.weights[1] +
            min_rewards[2] * rt.weights[2] +
            min_rewards[3] * rt.weights[3] +
            min_rewards[4] * rt.weights[4]
    ) / weight_sum

    return min_rewards[5] * weighted_rewards  # low_span_punishment * weighted_sum
end

function _compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::PeriodMinimumReward, cache::RewardShiftBufferCache{T}) where {T, A, O, R, V, OBS}
    global_reward = time_minimum_global_reward(env, rt, cache)
    Δmax = RDE.turbo_maximum_abs_diff(env.prob.method.cache.u_p_current, env.prob.method.cache.u_p_previous) / env.u_pmax
    action_effort_modifier = action_magnitude_factor(rt.lowest_action_magnitude_reward, Δmax)
    return global_reward * action_effort_modifier
end

reward_value_type(::Type{T}, ::PeriodMinimumReward) where {T} = T

struct PeriodMinimumVariationReward{T <: AbstractFloat} <: CachedCompositeReward
    lowest_action_magnitude_reward::T #reward will be \in [lowest_action_magnitude_reward, 1]
    weights::Vector{T}
    variation_penalties::Vector{T}  # α values for each component [span, periodicity, shock, shock_spacing]
end
function PeriodMinimumVariationReward{T}(;
        target_shock_count::Int = 4,  # Ignored, kept for backward compat
        lowest_action_magnitude_reward::T = one(T),
        weights::Vector{T} = [one(T), one(T), T(5), one(T)],
        variation_penalties::Vector{T} = [T(2), T(2), T(2), T(2)],
        N::Int = 512  # Ignored, kept for backward compat
    ) where {T <: AbstractFloat}
    return PeriodMinimumVariationReward{T}(
        lowest_action_magnitude_reward,
        weights,
        variation_penalties
    )
end
function PeriodMinimumVariationReward(;
        target_shock_count::Int = 4,  # Ignored, kept for backward compat
        lowest_action_magnitude_reward::Float32 = 1.0f0,
        weights::Vector{Float32} = Float32[1, 1, 5, 1],
        variation_penalties::Vector{Float32} = Float32[2.0, 2.0, 2.0, 2.0],
        N::Int = 512  # Ignored, kept for backward compat
    )
    return PeriodMinimumVariationReward{Float32}(
        lowest_action_magnitude_reward = lowest_action_magnitude_reward,
        weights = weights,
        variation_penalties = variation_penalties
    )
end

initialize_cache(::PeriodMinimumVariationReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

function Base.show(io::IO, rt::PeriodMinimumVariationReward)
    return print(io, "PeriodMinimumVariationReward()")
end

function Base.show(io::IO, ::MIME"text/plain", rt::PeriodMinimumVariationReward)
    println(io, "PeriodMinimumVariationReward:")
    println(io, "  lowest_action_magnitude_reward: $(rt.lowest_action_magnitude_reward)")
    println(io, "  weights: $(rt.weights)")
    return println(io, "  variation_penalties: $(rt.variation_penalties)")
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

function time_minimum_global_reward_with_variation(env::RDEEnv{T, A, O, R, V, OBS}, rt::PeriodMinimumVariationReward, cache::RewardShiftBufferCache{T}) where {T <: AbstractFloat, A, O, R, V, OBS}
    N = env.prob.params.N
    if isnothing(env.prob.sol)
        return zero(T)
    end

    # Pre-compute constants to avoid type instability in loop
    L = env.prob.params.L::T
    dx = (env.prob.x[2] - env.prob.x[1])::T
    weight_sum = sum(rt.weights)::T
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
            global_rewards(u, L, dx, rt, shift_buffer, target_shock_count)

        all_span_rewards[i] = span_rew
        all_periodicity_rewards[i] = periodicity_rew
        all_shock_rewards[i] = shock_rew
        all_shock_spacing_rewards[i] = shock_spacing_rew
        all_low_span_punishments[i] = low_span_punishment
    end

    # Calculate variation punishment for each component
    span_var_punishment = calculate_variation_punishment(all_span_rewards, T(rt.variation_penalties[1]))
    periodicity_var_punishment = calculate_variation_punishment(all_periodicity_rewards, T(rt.variation_penalties[2]))
    shock_var_punishment = calculate_variation_punishment(all_shock_rewards, T(rt.variation_penalties[3]))
    shock_spacing_var_punishment = calculate_variation_punishment(all_shock_spacing_rewards, T(rt.variation_penalties[4]))

    # Apply variation punishment to each component before taking minimum
    punished_span = minimum(all_span_rewards) * span_var_punishment
    punished_periodicity = minimum(all_periodicity_rewards) * periodicity_var_punishment
    punished_shock = minimum(all_shock_rewards) * shock_var_punishment
    punished_shock_spacing = minimum(all_shock_spacing_rewards) * shock_spacing_var_punishment

    # Low span punishment: only minimum, no variation punishment
    min_low_span_punishment = minimum(all_low_span_punishments)

    # Efficient weighted sum
    weighted_rewards = (
        punished_span * rt.weights[1] +
            punished_periodicity * rt.weights[2] +
            punished_shock * rt.weights[3] +
            punished_shock_spacing * rt.weights[4]
    ) / weight_sum

    return min_low_span_punishment * weighted_rewards
end

function _compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::PeriodMinimumVariationReward, cache::RewardShiftBufferCache{T}) where {T, A, O, R, V, OBS}
    global_reward = time_minimum_global_reward_with_variation(env, rt, cache)
    Δmax = RDE.turbo_maximum_abs_diff(env.prob.method.cache.u_p_current, env.prob.method.cache.u_p_previous) / env.u_pmax
    action_effort_modifier = action_magnitude_factor(rt.lowest_action_magnitude_reward, Δmax)
    return global_reward * action_effort_modifier
end

mutable struct MultiSectionPeriodMinimumReward{T <: AbstractFloat} <: MultiAgentCachedCompositeReward
    n_sections::Int
    lowest_action_magnitude_reward::T
    weights::Vector{T}
end

# Default constructor with Float32 type
function MultiSectionPeriodMinimumReward(;
        weights::Vector{T} = [1.0f0, 1.0f0, 5.0f0, 1.0f0],
        n_sections::Int = 4,
        target_shock_count::Int = 3,  # Ignored, kept for backward compat
        lowest_action_magnitude_reward::T = 0.0f0
    ) where {T <: AbstractFloat}
    return MultiSectionPeriodMinimumReward{T}(
        n_sections,
        lowest_action_magnitude_reward,
        weights
    )
end

# Constructor with explicit type parameter
function MultiSectionPeriodMinimumReward{T}(
        n_sections::Int = 4,
        target_shock_count::Int = 3,  # Ignored, kept for backward compat
        lowest_action_magnitude_reward::T = zero(T),
        weights::Vector{T} = [one(T), one(T), T(5), one(T)]
    ) where {T <: AbstractFloat}
    return MultiSectionPeriodMinimumReward{T}(
        n_sections,
        lowest_action_magnitude_reward,
        weights
    )
end

initialize_cache(::MultiSectionPeriodMinimumReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

function Base.show(io::IO, rt::MultiSectionPeriodMinimumReward)
    return print(io, "MultiSectionPeriodMinimumReward(n_sections=$(rt.n_sections))")
end

function Base.show(io::IO, ::MIME"text/plain", rt::MultiSectionPeriodMinimumReward)
    println(io, "MultiSectionPeriodMinimumReward:")
    println(io, "  n_sections: $(rt.n_sections)")
    println(io, "  lowest_action_magnitude_reward: $(rt.lowest_action_magnitude_reward)")
    return println(io, "  weights: $(rt.weights)")
end

reward_value_type(::Type{T}, ::MultiSectionPeriodMinimumReward{T}) where {T} = Vector{T}

function _compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::MultiSectionPeriodMinimumReward, cache::RewardShiftBufferCache{T}) where {T, A, O, R, V, OBS}
    # Compute the time minimum global reward (same as PeriodMinimumReward)
    common_reward = time_minimum_global_reward(env, rt, cache)

    # Multi-section control-effort punishment
    efforts = compute_section_control_efforts(env, rt.n_sections)
    individual_modifiers = action_magnitude_factor(rt.lowest_action_magnitude_reward, efforts)

    return common_reward .* individual_modifiers
end

"""
    MultiplicativeReward <: AbstractRDEReward

A reward that multiplies the rewards from multiple reward types.

# Fields
- `rewards::Vector{AbstractRDEReward}`: Vector of reward components to multiply

# Notes
- If all wrapped rewards return scalar values, this returns a scalar
- If any wrapped reward returns a vector, this returns a vector (element-wise multiplication)
"""
struct MultiplicativeReward <: AbstractRDEReward
    rewards::Vector{AbstractRDEReward}

    function MultiplicativeReward(rewards::Vector{AbstractRDEReward})
        return new(rewards)
    end

    function MultiplicativeReward(rewards::AbstractRDEReward...)
        return new([rewards...])
    end
end

function Base.show(io::IO, rt::MultiplicativeReward)
    return print(io, "MultiplicativeReward($(length(rt.rewards)) components)")
end

function Base.show(io::IO, ::MIME"text/plain", rt::MultiplicativeReward)
    println(io, "MultiplicativeReward:")
    println(io, "  components: $(length(rt.rewards))")
    for (i, reward) in enumerate(rt.rewards)
        println(io, "  reward $i: $(typeof(reward))")
    end
    return
end

initialize_cache(rt::MultiplicativeReward, N::Int, ::Type{T}) where {T} =
    CompositeRewardCache(Tuple(initialize_cache(r, N, T) for r in rt.rewards))

function _compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::MultiplicativeReward, cache::CompositeRewardCache) where {T, A, O, R, V, OBS}
    # Compute rewards with their respective caches
    reward = _compute_reward(env, rt.rewards[1], cache.caches[1])

    for (i, r) in enumerate(rt.rewards[2:end])
        next_reward = _compute_reward(env, r, cache.caches[i + 1])
        reward = next_reward .* reward
    end

    return reward
end

function reward_value_from_wrapper(::Type{T}, rts::Vector{<:AbstractRDEReward}) where {T}
    value_types = reward_value_type.(T, rts)
    if any(value_types .== Vector{T})
        return Vector{T}
    else
        return T
    end
end

function reward_value_type(::Type{T}, rt::MultiplicativeReward) where {T}
    return reward_value_from_wrapper(T, rt.rewards)
end

# Reset method for MultiplicativeReward - pass through to sub-rewards
function reset_reward!(rt::MultiplicativeReward)
    for r in rt.rewards
        reset_reward!(r)
    end
    return nothing
end


"""
ExponentialAverageReward{T<:AbstractRDEReward} <: AbstractRDEReward

A reward wrapper that applies exponential averaging to smooth reward signals.

The exponential average is computed as:
`new_avg = α * current_reward + (1-α) * old_avg`

# Fields
- `wrapped_reward::T`: The underlying reward to wrap
- `α::Float32`: Averaging parameter (0 < α ≤ 1). Higher values give more weight to recent rewards
- `average::Union{Nothing, Float32, Vector{Float32}}`: Current exponential average (initialized on first use)

# Notes
- If wrapped reward returns scalar, this returns scalar exponential average
- If wrapped reward returns vector, this returns element-wise exponential averages
- The average is reset to `nothing` on `reset_reward!()` calls
"""
mutable struct ExponentialAverageReward{T <: AbstractRDEReward, U <: AbstractFloat} <: AbstractRDEReward
    wrapped_reward::T
    α::U  # averaging parameter
    average::Union{Nothing, U, Vector{U}}  # current exponential average

    function ExponentialAverageReward{T, U}(wrapped_reward::T; α::U = U(0.2)) where {T <: AbstractRDEReward, U <: AbstractFloat}
        return new{T, U}(wrapped_reward, α, nothing)
    end
    function ExponentialAverageReward(wrapped_reward::T; α::Float32 = 0.2f0) where {T <: AbstractRDEReward}
        return ExponentialAverageReward{T, Float32}(wrapped_reward, α = α)
    end
end

function Base.show(io::IO, rt::ExponentialAverageReward)
    return print(io, "ExponentialAverageReward(α=$(rt.α), wrapped=$(typeof(rt.wrapped_reward)))")
end

function Base.show(io::IO, ::MIME"text/plain", rt::ExponentialAverageReward)
    println(io, "ExponentialAverageReward:")
    println(io, "  α: $(rt.α)")
    println(io, "  wrapped_reward: $(typeof(rt.wrapped_reward))")
    return println(io, "  current_average: $(rt.average)")
end

function reward_value_type(::Type{T}, rt::ExponentialAverageReward) where {T}
    return reward_value_type(T, rt.wrapped_reward)
end

initialize_cache(rt::ExponentialAverageReward, N::Int, ::Type{T}) where {T} =
    initialize_cache(rt.wrapped_reward, N, T)

function _compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::ExponentialAverageReward, cache) where {T, A, O, R, V, OBS}
    # Use the cache for the wrapped reward (no composite needed since only one sub-reward)
    current_reward = _compute_reward(env, rt.wrapped_reward, cache)

    if isnothing(rt.average)
        # Initialize average with first reward
        rt.average = current_reward
        return current_reward
    else
        # Apply exponential averaging: new_avg = α * current + (1-α) * old_avg
        rt.average = rt.α .* current_reward .+ (1 - rt.α) .* rt.average
        return rt.average
    end
end

function reset_reward!(rt::ExponentialAverageReward)
    rt.average = nothing
    reset_reward!(rt.wrapped_reward)  # Reset the wrapped reward too
    return nothing
end

"""
    TransitionBasedReward{T<:AbstractRDEReward} <: AbstractRDEReward

A reward wrapper that gives -1*dt reward until a successful transition is detected,
then terminates the environment. Uses transition detection logic similar to detect_transition.

# Fields
- `wrapped_reward::T`: The underlying reward to wrap and monitor
- `target_shocks::Int`: Target number of shocks for transition
- `reward_stability_length::Float32`: Time duration (in env time units) to maintain stable rewards
- `reward_threshold::Float32`: Minimum reward threshold for stability check
- `past_rewards::Vector{Float32}`: History of wrapped reward values
- `past_shock_counts::Vector{Int}`: History of shock counts
- `past_times::Vector{Float32}`: History of time stamps
- `transition_found::Bool`: Whether transition has been detected

# Notes
- Returns -1.0*dt until transition is found
- When transition is detected, sets env.terminated = true
- Transition occurs when: shock count reaches target AND rewards stay above threshold for stability_length time
"""
mutable struct TransitionBasedReward{T <: AbstractRDEReward, U <: AbstractFloat} <: AbstractRDEReward
    wrapped_reward::T
    target_shocks::Int
    reward_stability_length::U  # in time units
    reward_threshold::U

    # Internal state tracking
    past_rewards::Vector{U}
    past_shock_counts::Vector{Int}
    transition_found::Bool

    function TransitionBasedReward{T, U}(
            wrapped_reward::T;
            target_shocks::Int = 3,
            reward_stability_length::U = U(20),
            reward_threshold::U = U(0.99)
        ) where {T <: AbstractRDEReward, U <: AbstractFloat}
        return new{T, U}(
            wrapped_reward, target_shocks, reward_stability_length, reward_threshold,
            U[], Int[], false
        )
    end
    function TransitionBasedReward(
            wrapped_reward::T;
            target_shocks::Int = 3,
            reward_stability_length::Float32 = 20.0f0,
            reward_threshold::Float32 = 0.99f0
        ) where {T <: AbstractRDEReward}
        return TransitionBasedReward{T, Float32}(
            wrapped_reward, target_shocks = target_shocks, reward_stability_length = reward_stability_length, reward_threshold = reward_threshold
        )
    end
end

function Base.show(io::IO, rt::TransitionBasedReward)
    return print(io, "TransitionBasedReward(target_shocks=$(rt.target_shocks), wrapped=$(typeof(rt.wrapped_reward)))")
end

function Base.show(io::IO, ::MIME"text/plain", rt::TransitionBasedReward)
    println(io, "TransitionBasedReward:")
    println(io, "  target_shocks: $(rt.target_shocks)")
    println(io, "  reward_stability_length: $(rt.reward_stability_length)")
    println(io, "  reward_threshold: $(rt.reward_threshold)")
    println(io, "  wrapped_reward: $(typeof(rt.wrapped_reward))")
    println(io, "  transition_found: $(rt.transition_found)")
    return println(io, "  history_length: $(length(rt.past_rewards))")
end
reward_value_type(::Type{T}, ::TransitionBasedReward) where {T} = T

initialize_cache(rt::TransitionBasedReward, N::Int, ::Type{T}) where {T} =
    initialize_cache(rt.wrapped_reward, N, T)

function _compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::TransitionBasedReward{S, T}, cache) where {T, A, O, R, V, OBS, S}
    # Compute the wrapped reward
    wrapped_reward_value = _compute_reward(env, rt.wrapped_reward, cache)

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

    # Update history
    push!(rt.past_rewards, T(reward_scalar))
    push!(rt.past_shock_counts, current_shock_count)

    # Check for transition (only if we have enough history)
    if length(rt.past_shock_counts) >= 2
        rt.transition_found = detect_transition_realtime(rt, env.dt)

        if rt.transition_found
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

function detect_transition_realtime(rt::TransitionBasedReward{T, U}, dt::U) where {T, U <: AbstractFloat}
    stability_steps = round(Int, rt.reward_stability_length / dt)
    if length(rt.past_shock_counts) < stability_steps
        @debug "Not enough shock counts to detect transition"
        return false
    end


    # Check if we've had stable rewards for long enough since then
    stability_rewards = rt.past_rewards[(end - stability_steps + 1):end]
    stability_shock_counts = rt.past_shock_counts[(end - stability_steps + 1):end]
    if all(stability_shock_counts .== rt.target_shocks) && minimum(stability_rewards) > rt.reward_threshold
        return true
    end
    return false
end

function reset_reward!(rt::TransitionBasedReward)
    empty!(rt.past_rewards)
    empty!(rt.past_shock_counts)
    rt.transition_found = false
    reset_reward!(rt.wrapped_reward)  # Reset the wrapped reward too
    return nothing
end

struct ScalarToVectorReward{T <: AbstractRDEReward} <: AbstractRDEReward
    wrapped_reward::T
    n::Int
end

reward_value_type(::Type{T}, ::ScalarToVectorReward) where {T} = Vector{T}

initialize_cache(rt::ScalarToVectorReward, N::Int, ::Type{T}) where {T} =
    initialize_cache(rt.wrapped_reward, N, T)

function _compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::ScalarToVectorReward, cache) where {T, A, O, R, V, OBS}
    reward = _compute_reward(env, rt.wrapped_reward, cache)
    return fill(reward, rt.n)
end

function reset_reward!(rt::ScalarToVectorReward)
    return reset_reward!(rt.wrapped_reward)
end
