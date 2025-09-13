# Default reset interface for rewards - does nothing by default
function reset_reward!(rt::AbstractRDEReward)
    return nothing
end

function compute_reward(env::AbstractRDEEnv, rt::AbstractRDEReward)
    @error "No reward computation implemented for type $(typeof(rt))"
    return zero(T)
end

function set_reward!(env::AbstractRDEEnv, rt::AbstractRDEReward)
    env.reward = compute_reward(env, rt)
    return nothing
end

function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::ShockSpanReward) where {T, A, O, R, V, OBS}
    target_shock_count = rt.target_shock_count
    max_span = rt.span_scale
    λ = rt.shock_weight

    u, = RDE.split_sol_view(env.state)
    dx = env.prob.params.L / env.prob.params.N
    shocks = T(RDE.count_shocks(u, dx))
    span = RDE.turbo_maximum(u) - RDE.turbo_minimum(u)
    span_reward = span / max_span
    if shocks >= target_shock_count
        shock_reward = one(T)
    elseif shocks > 0
        shock_reward = shocks / (2 * target_shock_count)
    else
        shock_reward = T(-1.0)
    end

    return λ * shock_reward + (1 - λ) * span_reward
end

reward_value_type(::Type{T}, ::ShockSpanReward) where {T} = T

"""
    compute_reward(env::AbstractRDEEnv, rt::ShockPreservingReward)

Reward for preserving a given number of shocks. terminated/truncated if the number of shocks is not preserved.
    penalize reward if shocks are not evenly spaced, reward for large span.
"""
function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::ShockPreservingReward) where {T, A, O, R, V, OBS}
    target_shock_count = rt.target_shock_count
    max_span = rt.span_scale
    λ = rt.shock_weight

    u, = RDE.split_sol_view(env.state)
    dx = env.prob.x[2] - env.prob.x[1]
    N = env.prob.params.N
    L = env.prob.params.L
    shock_inds = RDE.shock_indices(u, dx)

    span = RDE.turbo_maximum(u) - RDE.turbo_minimum(u)
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

function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::ShockPreservingSymmetryReward) where {T, A, O, R, V, OBS}
    target_shock_count = rt.target_shock_count
    N = env.prob.params.N
    u = env.state[1:N]

    errs = zeros(T, target_shock_count - 1)
    cache = rt.cache
    shift_steps = N ÷ target_shock_count
    for i in 1:(target_shock_count - 1)
        cache .= u
        circshift!(cache, u, -shift_steps * i)
        errs[i] = RDE.turbo_diff_norm(u, cache) / sqrt(T(N))
    end
    maxerr = RDE.turbo_maximum(errs)
    return T(1) - (maxerr - T(0.1)) / T(0.5)
end

reward_value_type(::Type{T}, ::ShockPreservingSymmetryReward) where {T} = T

function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::PeriodicityReward) where {T <: AbstractFloat, A, O, R, V, OBS}
    u, = RDE.split_sol_view(env.state)
    N::Int = env.prob.params.N
    dx::T = env.prob.x[2] - env.prob.x[1]
    L::T = env.prob.params.L
    shock_inds = RDE.shock_indices(u, dx)
    shocks::Int = length(shock_inds)

    cache = rt.cache
    if shocks > 1
        shift_steps::Int = N ÷ shocks
        errs::Vector{T} = zeros(T, shocks - 1)
        for i in 1:(shocks - 1)
            cache .= u
            circshift!(cache, u, -shift_steps * i)
            errs[i]::T = RDE.turbo_diff_norm(u, cache) / sqrt(N)
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
    return print(io, "PeriodicityReward(N=$(length(rt.cache)))")
end

function Base.show(io::IO, ::MIME"text/plain", rt::PeriodicityReward)
    println(io, "PeriodicityReward:")
    return println(io, "  cache size: $(length(rt.cache))")
end

function action_magnitude_factor(lowest_action_magnitude_reward::AbstractFloat, action_magnitudes)
    α = lowest_action_magnitude_reward
    action_magnitude_inv = 1 .- action_magnitudes
    return α .+ (1 - α) .* action_magnitude_inv
end

@kwdef mutable struct MultiSectionReward{T <: AbstractFloat} <: MultiAgentCachedCompositeReward
    n_sections::Int = 4
    target_shock_count::Int = 3
    cache::Vector{T} = zeros(T, 512)
    lowest_action_magnitude_reward::T = zero(T) #reward will be \in [lowest_action_magnitude_reward, 1]
    weights::Vector{T} = [one(T), one(T), T(5), one(T)]
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

function global_rewards(u::AbstractVector{T}, L::T, dx::T, rt::CachedCompositeReward) where {T <: AbstractFloat}
    N = length(u)
    @assert length(rt.cache) == N "cache length must match state length, cache length: $(length(rt.cache)), state length: $N"

    periodicity_reward = calculate_periodicity_reward(u, N, rt.target_shock_count, rt.cache)
    shock_reward, shock_spacing_reward, shocks = calculate_shock_rewards(u, dx, L, N, rt.target_shock_count)
    span_reward, low_span_punishment = calculate_span_rewards(u, shocks)

    @logmsg LogLevel(-10000) "low_span_punishment: $low_span_punishment"
    @logmsg LogLevel(-10000) "span_reward: $span_reward"
    @logmsg LogLevel(-10000) "periodicity_reward: $periodicity_reward"
    @logmsg LogLevel(-10000) "shock_reward: $shock_reward"
    @logmsg LogLevel(-10000) "shock_spacing_reward: $shock_spacing_reward"
    return span_reward, periodicity_reward, shock_reward, shock_spacing_reward, low_span_punishment
end

function global_reward(u::AbstractVector{T}, L::T, dx::T, rt::CachedCompositeReward) where {T <: AbstractFloat}
    span_reward, periodicity_reward, shock_reward, shock_spacing_reward, low_span_punishment = global_rewards(u, L, dx, rt)
    weighted_rewards = [span_reward, periodicity_reward, shock_reward, shock_spacing_reward]' * rt.weights / sum(rt.weights)
    global_reward = low_span_punishment * sum(weighted_rewards)
    @logmsg LogLevel(-10000) "global_reward: $global_reward"
    return global_reward
end

function global_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::CachedCompositeReward) where {T, A, O, R, V, OBS}
    N = env.prob.params.N
    dx = env.prob.x[2] - env.prob.x[1]
    L = env.prob.params.L
    u = env.state[1:N]

    return global_reward(u, L, dx, rt)
end

reward_value_type(::Type{T}, ::CachedCompositeReward) where {T} = T

mutable struct CompositeReward{T <: AbstractFloat} <: CachedCompositeReward
    target_shock_count::Int
    cache::Vector{T}
    lowest_action_magnitude_reward::T #reward will be \in [lowest_action_magnitude_reward, 1]
    span_reward::Bool
    weights::Vector{T}
    function CompositeReward{T}(;
            target_shock_count::Int = 4,
            lowest_action_magnitude_reward::T = one(T),
            span_reward::Bool = true,
            weights::Vector{T} = [T(0.25), T(0.25), T(0.25), T(0.25)],
            N::Int = 512
        ) where {T <: AbstractFloat}
        return new{T}(
            target_shock_count,
            zeros(T, N),
            lowest_action_magnitude_reward,
            span_reward, weights
        )
    end
    function CompositeReward(;
            target_shock_count::Int = 4,
            lowest_action_magnitude_reward::Float32 = 1.0f0,
            span_reward::Bool = true,
            weights::Vector{Float32} = [0.25f0, 0.25f0, 0.25f0, 0.25f0],
            N::Int = 512
        )
        return CompositeReward{Float32}(
            target_shock_count = target_shock_count,
            lowest_action_magnitude_reward = lowest_action_magnitude_reward,
            span_reward = span_reward,
            weights = weights,
            N = N
        )
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
    # if rt.lowest_action_magnitude_reward < 1f0
    #     action_magnitude_inv = 1f0 - RDE.turbo_maximum_abs(env.cache.action)
    #     α = rt.lowest_action_magnitude_reward
    #     @logmsg LogLevel(-10000) "action_magnitude factor: $(α + (1f0 - α)*action_magnitude_inv)" RDE.turbo_maximum_abs(env.cache.action)
    #     reward *= α + (1f0 - α)*action_magnitude_inv
    # end
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
    target_shock_count::Int
    cache::Vector{T}
    lowest_action_magnitude_reward::T #reward will be \in [lowest_action_magnitude_reward, 1]
    span_reward::Bool
    weights::Vector{T}
    function TimeAggCompositeReward{T}(;
            aggregation::TimeAggregation = TimeMin(),
            target_shock_count::Int = 4,
            lowest_action_magnitude_reward::T = one(T),
            span_reward::Bool = true,
            weights::Vector{T} = [T(0.25), T(0.25), T(0.25), T(0.25)],
            N::Int = 512
        ) where {T <: AbstractFloat}
        return new{T}(
            aggregation,
            target_shock_count,
            zeros(T, N),
            lowest_action_magnitude_reward,
            span_reward, weights
        )
    end
    function TimeAggCompositeReward(;
            aggregation::TimeAggregation = TimeMin(),
            target_shock_count::Int = 4,
            lowest_action_magnitude_reward::Float32 = 1.0f0,
            span_reward::Bool = true,
            weights::Vector{Float32} = [0.25f0, 0.25f0, 0.25f0, 0.25f0],
            N::Int = 512
        )
        return TimeAggCompositeReward{Float32}(
            aggregation = aggregation,
            target_shock_count = target_shock_count,
            lowest_action_magnitude_reward = lowest_action_magnitude_reward,
            span_reward = span_reward,
            weights = weights,
            N = N
        )
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


struct ConstantTargetReward{T <: AbstractFloat} <: AbstractRDEReward
    target::T
    function ConstantTargetReward{T}(; target::T = T(0.64)) where {T <: AbstractFloat}
        return new{T}(target)
    end
    function ConstantTargetReward(; target::Float32 = 0.64f0)
        return ConstantTargetReward{Float32}(target = target)
    end
end

function Base.show(io::IO, rt::ConstantTargetReward)
    return print(io, "ConstantTargetReward(target=$(rt.target))")
end

function Base.show(io::IO, ::MIME"text/plain", rt::ConstantTargetReward)
    println(io, "ConstantTargetReward:")
    return println(io, "  target: $(rt.target)")
end

function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::ConstantTargetReward{T}) where {T, A, O, R, V, OBS}
    return -abs(rt.target - mean(env.prob.method.cache.u_p_current)) + one(T)
end

@kwdef mutable struct TimeAggMultiSectionReward{T <: AbstractFloat} <: MultiAgentCachedCompositeReward
    aggregation::TimeAggregation = TimeMin()
    n_sections::Int = 4
    target_shock_count::Int = 3
    cache::Vector{T} = zeros(T, 512)
    lowest_action_magnitude_reward::T = zero(T) #reward will be \in [lowest_action_magnitude_reward, 1]
    weights::Vector{T} = [one(T), one(T), T(5), one(T)]
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

    # Get all states from the solution
    if !isnothing(env.prob.sol)
        us = [uλ[1:N] for uλ in env.prob.sol.u[2:end]]
    else
        us = [env.state[1:N]]
    end
    section_rewards = zeros(T, n_sections, length(us))

    # Calculate rewards for each section and time step
    for (i, u) in enumerate(us)
        common_reward = global_reward(u, env.prob.params.L, env.prob.x[2] - env.prob.x[1], rt)
        full_engine_pressure_action = env.cache.action[:, 2]
        section_actions = full_engine_pressure_action[first_element_in_sections]
        action_magnitudes = abs.(section_actions)
        individual_modifiers = action_magnitude_factor(rt.lowest_action_magnitude_reward, action_magnitudes)
        section_rewards[:, i] = common_reward .* individual_modifiers
    end

    # Aggregate rewards over time for each section
    agg_section_rewards = zeros(T, n_sections)
    for i in 1:n_sections
        agg_section_rewards[i] = aggregate(section_rewards[i, :], rt.aggregation)
    end

    return agg_section_rewards
end

@kwdef struct TimeDiffNormReward{T <: AbstractFloat} <: AbstractRDEReward
    threshold::T = T(1.1)
    threshold_reward::T = T(0.3)
end

function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::TimeDiffNormReward{T}) where {T, A, O, R, V, OBS}
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
    target_shock_count::Int
    cache::Vector{T}
    lowest_action_magnitude_reward::T #reward will be \in [lowest_action_magnitude_reward, 1]
    weights::Vector{T}
    function PeriodMinimumReward{T}(;
            target_shock_count::Int = 4,
            lowest_action_magnitude_reward::T = one(T),
            weights::Vector{T} = [one(T), one(T), T(5), one(T)],
            N::Int = 512
        ) where {T <: AbstractFloat}
        return new{T}(
            target_shock_count,
            zeros(T, N),
            lowest_action_magnitude_reward,
            weights
        )
    end
    function PeriodMinimumReward(;
            target_shock_count::Int = 4,
            lowest_action_magnitude_reward::Float32 = 1.0f0,
            weights::Vector{Float32} = Float32[1, 1, 5, 1],
            N::Int = 512
        )
        return PeriodMinimumReward{Float32}(
            target_shock_count = target_shock_count,
            lowest_action_magnitude_reward = lowest_action_magnitude_reward,
            weights = weights,
            N = N
        )
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

    # Pre-compute constants to avoid type instability in loop
    L::T = env.prob.params.L
    dx::T = RDE.get_dx(env.prob)
    weight_sum::T = sum(rt.weights)

    # Initialize with large values to find minimum
    min_rewards::Vector{T} = fill(T(10000), 5)

    # Type-stable iteration - convert to Vector{Vector{T}} if needed
    sol_states = env.prob.sol.u::Vector{Vector{T}}

    @inbounds for state::Vector{T} in sol_states
        # Use view to avoid allocation, with fallback type assertion
        u = @view state[1:N]

        span_rew, periodicity_rew, shock_rew, shock_spacing_rew, low_span_punishment =
            global_rewards(u, L, dx, rt)

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

function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::PeriodMinimumReward) where {T, A, O, R, V, OBS}
    global_reward = time_minimum_global_reward(env, rt)
    action_magnitude = RDE.turbo_maximum_abs(env.cache.action)
    action_magnitude_modifier = action_magnitude_factor(rt.lowest_action_magnitude_reward, action_magnitude)
    return global_reward * action_magnitude_modifier
end

reward_value_type(::Type{T}, ::PeriodMinimumReward) where {T} = T

struct PeriodMinimumVariationReward{T <: AbstractFloat} <: CachedCompositeReward
    target_shock_count::Int
    cache::Vector{T}
    lowest_action_magnitude_reward::T #reward will be \in [lowest_action_magnitude_reward, 1]
    weights::Vector{T}
    variation_penalties::Vector{T}  # α values for each component [span, periodicity, shock, shock_spacing]
end
function PeriodMinimumVariationReward{T}(;
        target_shock_count::Int = 4,
        lowest_action_magnitude_reward::T = one(T),
        weights::Vector{T} = [one(T), one(T), T(5), one(T)],
        variation_penalties::Vector{T} = [T(2), T(2), T(2), T(2)],
        N::Int = 512
    ) where {T <: AbstractFloat}
    return PeriodMinimumVariationReward{T}(
        target_shock_count,
        zeros(T, N),
        lowest_action_magnitude_reward,
        weights,
        variation_penalties
    )
end
function PeriodMinimumVariationReward(;
        target_shock_count::Int = 4,
        lowest_action_magnitude_reward::Float32 = 1.0f0,
        weights::Vector{Float32} = Float32[1, 1, 5, 1],
        variation_penalties::Vector{Float32} = Float32[2.0, 2.0, 2.0, 2.0],
        N::Int = 512
    )
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
        return one(T)  # No variation with single value
    end

    mean_abs_value = mean(abs.(values))
    if mean_abs_value < T(1.0e-6)
        return one(T)  # Avoid division by zero
    end

    coefficient_of_variation = std(values) / mean_abs_value
    return exp(-α * coefficient_of_variation)
end

function time_minimum_global_reward_with_variation(env::RDEEnv{T, A, O, R, V, OBS}, rt::PeriodMinimumVariationReward) where {T <: AbstractFloat, A, O, R, V, OBS}
    N = env.prob.params.N
    if isnothing(env.prob.sol)
        return zero(T)
    end

    # Pre-compute constants to avoid type instability in loop
    L = env.prob.params.L::T
    dx = (env.prob.x[2] - env.prob.x[1])::T
    weight_sum = sum(rt.weights)::T

    # Type-stable iteration
    sol_states = env.prob.sol.u
    n_states = length(sol_states)

    # Pre-allocate reward vectors with correct size
    all_span_rewards = Vector{T}(undef, n_states)
    all_periodicity_rewards = Vector{T}(undef, n_states)
    all_shock_rewards = Vector{T}(undef, n_states)
    all_shock_spacing_rewards = Vector{T}(undef, n_states)
    all_low_span_punishments = Vector{T}(undef, n_states)

    @inbounds for (i, state) in enumerate(sol_states)
        # Use view to avoid allocation
        u = @view state[1:N]

        span_rew, periodicity_rew, shock_rew, shock_spacing_rew, low_span_punishment =
            global_rewards(u, L, dx, rt)

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

function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::PeriodMinimumVariationReward) where {T, A, O, R, V, OBS}
    global_reward = time_minimum_global_reward_with_variation(env, rt)
    action_magnitude = RDE.turbo_maximum_abs(env.cache.action)
    action_magnitude_modifier = action_magnitude_factor(rt.lowest_action_magnitude_reward, action_magnitude)
    return global_reward * action_magnitude_modifier
end

@kwdef mutable struct MultiSectionPeriodMinimumReward{T <: AbstractFloat} <: MultiAgentCachedCompositeReward
    n_sections::Int = 4
    target_shock_count::Int = 3
    cache::Vector{T} = zeros(T, 512)
    lowest_action_magnitude_reward::T = zero(T) #reward will be \in [lowest_action_magnitude_reward, 1]
    weights::Vector{T} = [one(T), one(T), T(5), one(T)]
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
    # Compute the time minimum global reward (same as PeriodMinimumReward)
    common_reward = time_minimum_global_reward(env, rt)

    # Multi-section action magnitude punishment (same as MultiSectionReward)
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

function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::MultiplicativeReward) where {T, A, O, R, V, OBS}
    # Simply compute all rewards and use prod to multiply them
    # Julia's prod will handle broadcasting automatically when mixing scalar and vector rewards
    reward = compute_reward(env, rt.rewards[1])
    # @show reward
    for r in rt.rewards[2:end]
        next_reward = compute_reward(env, r)
        # @show next_reward
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
function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::ExponentialAverageReward) where {T, A, O, R, V, OBS}
    current_reward = compute_reward(env, rt.wrapped_reward)

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

function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::TransitionBasedReward{S, T}) where {T, A, O, R, V, OBS, S}
    # Compute the wrapped reward
    wrapped_reward_value = compute_reward(env, rt.wrapped_reward)

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
    stability_steps = rt.reward_stability_length / dt |> Int
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

function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::ScalarToVectorReward) where {T, A, O, R, V, OBS}
    reward = compute_reward(env, rt.wrapped_reward)
    return fill(reward, rt.n)
end

function reset_reward!(rt::ScalarToVectorReward)
    return reset_reward!(rt.wrapped_reward)
end
