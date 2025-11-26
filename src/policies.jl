abstract type AbstractRDEPolicy end

"""
    _predict_action(policy::AbstractRDEPolicy, obs)
"""
function _predict_action end

"""
    PolicyRunData{T<:AbstractFloat}

Container for data collected during policy execution.

# Fields
- `action_ts::Vector{T}`: Time points for actions
- `ss::Union{Vector{T}, Vector{Vector{T}}}`: Control parameter s at each action
- `u_ps::Union{Vector{T}, Vector{Vector{T}}}`: Control parameter u_p at each action
- `rewards::Union{Vector{T}, Vector{Vector{T}}}`: Rewards at each action
- `actions::Union{Vector{T}, Vector{Vector{T}}}`: Raw actions at each step
- `state_ts::Vector{T}`: Time points for states
- `states::Vector{Vector{T}}`: States at each time point
- `observations::Union{Vector{Vector{T}}, Vector{Matrix{T}}}`: Observations at each action step
"""
struct PolicyRunData{T <: AbstractFloat}
    action_ts::Vector{T}
    ss::Union{Vector{T}, Vector{Vector{T}}}
    u_ps::Union{Vector{T}, Vector{Vector{T}}}
    rewards::Union{Vector{T}, Vector{Vector{T}}}
    actions::Union{Vector{T}, Vector{Vector{T}}}
    state_ts::Vector{T}
    states::Vector{Vector{T}}
    observations::Union{Vector{Vector{T}}, Vector{Matrix{T}}}
end

function Base.show(io::IO, data::PolicyRunData)
    # Compact display
    return if length(data.action_ts) == length(data.state_ts)
        print(io, "PolicyRunData(steps: $(length(data.action_ts)))")
    else
        print(io, "PolicyRunData(steps: $(length(data.action_ts)), states: $(length(data.state_ts)))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", data::PolicyRunData)
    # Detailed display (used in REPL and notebooks)
    println(io, "PolicyRunData:")
    println(io, "  action_ts: $(typeof(data.action_ts))($(length(data.action_ts)))")
    println(io, "  ss: $(typeof(data.ss))($(length(data.ss)))")
    println(io, "  u_ps: $(typeof(data.u_ps))($(length(data.u_ps)))")
    println(io, "  rewards: $(typeof(data.rewards))($(length(data.rewards)))")
    println(io, "  actions: $(typeof(data.actions))($(length(data.actions)))")
    println(io, "  state_ts: $(typeof(data.state_ts))($(length(data.state_ts)))")
    println(io, "  states: $(typeof(data.states))($(length(data.states)))")
    return println(io, "  observations: $(typeof(data.observations))($(length(data.observations)))")
end

"""
    run_policy(π::AbstractRDEPolicy, env::RDEEnv{T}; saves_per_action=10) where {T}

Run a policy `π` on the RDE environment and collect trajectory data.

# Arguments
- `π::AbstractRDEPolicy`: Policy to execute
- `env::RDEEnv{T}`: RDE environment to run the policy in
- `saves_per_action=10`: Number of intermediate states to save per action

# Returns
`PolicyRunData` containing:
- `action_ts`: Time points for each action (length N)
- `ss`: Control parameter s values (length N)
- `u_ps`: Control parameter u_p values (length N)
- `rewards`: Rewards received (length N)
- `actions`: Raw actions taken (length N)
- `state_ts`: Time points for each state (length 1 + N*saves_per_action)
- `states`: Full system state at each time point (length 1 + N*saves_per_action)
- `observations`: Observations at each action step (length N)

# Example
```julia
env = RDEEnv()
policy = ConstantRDEPolicy(env)
data = run_policy(policy, env, saves_per_action=10)
```
"""
function run_policy(policy::AbstractRDEPolicy, env::RDEEnv{T}; saves_per_action = 10) where {T}
    _reset!(env)
    @assert saves_per_action ≥ 1 "saves_per_action must be at least 1"


    dt = env.dt
    max_actions = ceil(env.prob.params.tmax / dt) + 1 |> Int
    N = env.prob.params.N

    # Preallocate action-aligned arrays
    action_ts = Vector{T}(undef, max_actions)
    ss, u_ps = get_init_control_data(env, env.action_strat, max_actions)
    rewards = get_init_rewards(env, env.reward_strat, max_actions)

    # Preallocate actions using action_dim
    actions = if action_dim(env.action_strat) == 1
        Vector{T}(undef, max_actions)
    else
        Vector{Vector{T}}(undef, max_actions)
    end

    # Preallocate observations
    if env.observation_strat isa AbstractMultiAgentObservationStrategy
        observations = Vector{Matrix{T}}(undef, max_actions)
    else
        observations = Vector{Vector{T}}(undef, max_actions)
    end

    # Preallocate state arrays (initial + saves_per_action per action)
    max_state_points = 1 + max_actions * saves_per_action
    states = Vector{Vector{T}}(undef, max_state_points)
    state_ts = Vector{T}(undef, max_state_points)

    # Save initial state
    states[1] = copy(env.state)
    state_ts[1] = env.t
    total_state_steps = 1

    step = 0
    while !env.done && step < max_actions
        step += 1

        # Pre-action logging
        action_ts[step] = env.t
        observations[step] = _observe(env)
        # Record control summaries (pre-action values)
        if eltype(ss) <: AbstractVector
            sections = env.action_strat.n_sections
            ss[step] = section_reduction(env.prob.method.cache.s_current, sections)
        elseif eltype(ss) <: Number
            ss[step] = mean(env.prob.method.cache.s_current)
        end
        if eltype(u_ps) <: AbstractVector
            sections = env.action_strat.n_sections
            u_ps[step] = section_reduction(env.prob.method.cache.u_p_current, sections)
        elseif eltype(u_ps) <: Number
            u_ps[step] = mean(env.prob.method.cache.u_p_current)
        end

        # Compute action
        action = _predict_action(policy, copy(observations[step]))
        if env.observation_strat isa AbstractMultiAgentObservationStrategy && action isa Vector{Vector{T}}
            action = vcat(action...)
            @assert action isa Vector{T}
        end

        # Save raw action
        if action_dim(env.action_strat) == 1 && action isa Vector{T}
            actions[step] = action[1]
        else
            actions[step] = action
        end

        @debug "action: $action"

        # Step environment
        _act!(env, action; saves_per_action)

        # Collect solver states (drop first which is pre-action state)
        if typeof(env.prob.sol) == Nothing
            @info "sol is nothing at step $step"
            @info env.info
            step_states = Vector{Vector{T}}()
            step_ts = Vector{T}()
        else
            if length(env.prob.sol.t) != saves_per_action + 1
                @debug "length(env.prob.sol.t) ($(length(env.prob.sol.t))) != saves_per_action + 1 ($(saves_per_action + 1))"
                if env.prob.sol.t[end] - env.prob.sol.t[end - 1] < env.dt / 10
                    step_states = env.prob.sol.u[2:(end - 1)]
                    step_ts = env.prob.sol.t[2:(end - 1)]
                else
                    @warn "Too many states, but last two indices are not similar, using all states"
                    step_states = env.prob.sol.u[2:end]
                    step_ts = env.prob.sol.t[2:end]
                end
            else
                step_states = env.prob.sol.u[2:end]
                step_ts = env.prob.sol.t[2:end]
            end
            if length(step_states) != saves_per_action
                @warn "length(step_states) ($(length(step_states))) != saves_per_action ($(saves_per_action))"
            end
        end

        n_states = length(step_states)
        start_idx = total_state_steps + 1
        end_idx = total_state_steps + n_states

        # Ensure we have enough space
        if end_idx > max_state_points
            new_size = end_idx + max_actions * saves_per_action
            resize!(states, new_size)
            resize!(state_ts, new_size)
            max_state_points = new_size
        end

        # Append copies of states
        for i in 1:n_states
            states[start_idx + i - 1] = copy(step_states[i])
        end
        state_ts[start_idx:end_idx] = step_ts
        total_state_steps += n_states

        # Post-action reward
        if eltype(rewards) <: AbstractVector
            rewards[step] = env.reward
        else
            rewards[step] = env.reward
        end

        if env.terminated && env.verbose > 0
            @info "Env terminated at step $step"
            @info env.info
            if env.done
                @info "Env done at step $step"
            end
        end
        if env.truncated && env.verbose > 0
            @info "Env truncated at step $step"
            @info env.info
        end
    end

    n_actions = step

    # Trim arrays to actual size
    action_ts = action_ts[1:n_actions]
    ss = ss[1:n_actions]
    u_ps = u_ps[1:n_actions]
    rewards = rewards[1:n_actions]
    actions = actions[1:n_actions]
    observations = observations[1:n_actions]
    state_ts = state_ts[1:total_state_steps]
    states = states[1:total_state_steps]

    return PolicyRunData{T}(action_ts, ss, u_ps, rewards, actions, state_ts, states, observations)
end

function get_init_rewards(env::RDEEnv{T}, reward_strat::AbstractRewardStrategy, max_steps::Int) where {T}
    reward_strat = typeof(env.reward)
    return Vector{reward_strat}(undef, max_steps)
end

# function get_init_rewards(env::RDEEnv{T}, reward_strat::AbstractRewardStrategy, max_steps::Int) where {T}
#     return Vector{T}(undef, max_steps)
# end

# function get_init_rewards(env::RDEEnv{T}, reward_strat::MultiAgentCachedCompositeReward, max_steps::Int) where {T}
#     # n_section = reward_strat.n_sections
#     return Vector{Vector{T}}(undef, max_steps)
# end

# function get_init_rewards(env::RDEEnv{T}, reward_strat::ScalarToVectorReward, max_steps::Int) where {T}
#     return Vector{Vector{T}}(undef, max_steps)
# end

function get_init_rewards(env::RDEEnv{T}, reward_strat::MultiplicativeReward, max_steps::Int) where {T}
    # Check if any of the component rewards is a multi-agent reward
    if any(r isa MultiAgentCachedCompositeReward for r in reward_strat.rewards)
        return Vector{Vector{T}}(undef, max_steps)
    else
        return Vector{T}(undef, max_steps)
    end
end

#TODO: remove need for these
function get_init_control_data(env::RDEEnv{T}, action_strat::AbstractActionStrategy, max_steps::Int) where {T}
    return Vector{T}(undef, max_steps), Vector{T}(undef, max_steps)
end

function get_init_control_data(env::RDEEnv{T}, action_strat::VectorPressureAction, max_steps::Int) where {T}
    return Vector{T}(undef, max_steps), Vector{Vector{T}}(undef, max_steps)
end

function get_init_control_data(env::RDEEnv{T}, action_strat::DirectVectorPressureAction, max_steps::Int) where {T}
    return Vector{T}(undef, max_steps), Vector{Vector{T}}(undef, max_steps)
end

function get_init_control_data(env::RDEEnv{T}, action_strat::LinearVectorPressureAction, max_steps::Int) where {T}
    return Vector{T}(undef, max_steps), Vector{Vector{T}}(undef, max_steps)
end


function section_reduction(v::Vector{T}, sections::Int) where {T}
    N = length(v)
    if N % sections != 0
        @warn "Vector length $N is not divisible by section length $sections"
    end
    section_length = Int(round(N // sections))
    m = reshape(v, section_length, :)
    return vec(mean(m, dims = 1))
end

function get_env(π::AbstractRDEPolicy)
    if hasfield(typeof(π), :env)
        return getfield(π, :env)
    else
        return nothing
    end
end

"""
    ConstantRDEPolicy <: AbstractRDEPolicy

Policy that maintains constant control values.

# Fields
- `env::RDEEnv`: RDE environment

# Notes
Returns [0.0, 0.0] for ScalarAreaScalarPressureAction
Returns 0.0 for ScalarPressureAction
"""
struct ConstantRDEPolicy <: AbstractRDEPolicy
    env::RDEEnv
end

function _predict_action(π::ConstantRDEPolicy, s::AbstractVector{T}) where {T <: AbstractFloat}
    @assert !(π.env.action_strat isa DirectScalarPressureAction) && !(π.env.action_strat isa DirectVectorPressureAction) "ConstantRDEPolicy not supported with direct actions"
    if π.env.action_strat isa ScalarAreaScalarPressureAction
        return zeros(T, 2)
    elseif π.env.action_strat isa ScalarPressureAction
        return zero(T)
    elseif π.env.action_strat isa VectorPressureAction
        return zeros(T, π.env.action_strat.n_sections)
    else
        @error "Unknown action type $(typeof(π.env.action_strat)) for ConstantRDEPolicy"
    end
end
Base.show(io::IO, π::ConstantRDEPolicy) = print(io, "ConstantRDEPolicy()")
Base.show(io::IO, ::MIME"text/plain", π::ConstantRDEPolicy) = println(io, "ConstantRDEPolicy()")
"""
    SinusoidalRDEPolicy{T<:AbstractFloat} <: AbstractRDEPolicy

Policy that applies sinusoidal control signals.

# Fields
- `env::RDEEnv{T}`: RDE environment
- `w_1::T`: Phase speed parameter for first action
- `w_2::T`: Phase speed parameter for second action

# Constructor
```julia
SinusoidalRDEPolicy(env::RDEEnv{T}; w_1::T=1.0, w_2::T=2.0) where {T<:AbstractFloat}
```
"""
struct SinusoidalRDEPolicy{T <: AbstractFloat} <: AbstractRDEPolicy
    env::RDEEnv{T}
    w_1::T
    w_2::T
    scale::T
    function SinusoidalRDEPolicy(env::RDEEnv{T}; w_1::T = 1.0f0, w_2::T = 2.0f, scale = 0.2f0) where {T <: AbstractFloat}
        return new{T}(env, w_1, w_2, scale)
    end
end

function _predict_action(π::SinusoidalRDEPolicy, s::AbstractVector{T}) where {T <: AbstractFloat}
    @assert !(π.env.action_strat isa DirectScalarPressureAction) && !(π.env.action_strat isa DirectVectorPressureAction) "SinusoidalRDEPolicy not supported with direct actions"
    t = π.env.t
    action1 = T(sin(π.w_1 * t)) * π.scale
    action2 = T(sin(π.w_2 * t)) * π.scale
    if π.env.action_strat isa ScalarAreaScalarPressureAction
        return [action1, action2]
    elseif π.env.action_strat isa ScalarPressureAction
        return action2
    else
        @error "Unknown action type $(typeof(π.env.action_strat)) for SinusoidalRDEPolicy"
    end
end

function Base.show(io::IO, π::SinusoidalRDEPolicy)
    return print(io, "SinusoidalRDEPolicy()")
end

function Base.show(io::IO, ::MIME"text/plain", π::SinusoidalRDEPolicy)
    PeriodMinimumVariationReward(
            weights = [1.0f0, 1.0f0, 5.0f0, 1.0f0],
            lowest_action_magnitude_reward = 0.0f0,
            variation_penalties = Float32[1, 1, 1, 1]
        ),
        println(io, "SinusoidalRDEPolicy:")
    println(io, "  w₁: $(π.w_1)")
    println(io, "  w₂: $(π.w_2)")
    println(io, "  scale: $(π.scale)")
    return println(io, "  env: $(typeof(π.env))")
end

"""
    StepwiseRDEPolicy{T<:AbstractFloat} <: AbstractRDEPolicy

Policy that applies predefined control values at specified times.

# Fields
- `env::RDEEnv{T}`: RDE environment
- `ts::Vector{T}`: Vector of time steps
- `c::Vector{Vector{T}}`: Vector of control actions

# Notes
- Only supports ScalarAreaScalarPressureAction
- Requires sorted time steps
- Each control action must have 2 elements
"""
struct StepwiseRDEPolicy{T <: AbstractFloat} <: AbstractRDEPolicy
    env::RDEEnv{T}
    ts::Vector{T}  # Vector of time steps
    c::Union{Vector{Vector{T}}, Vector{T}}  # Vector of control actions

    function StepwiseRDEPolicy(env::RDEEnv{T}, ts::Vector{T}, c::Union{Vector{Vector{T}}, Vector{T}}) where {T <: AbstractFloat}
        @assert env.action_strat isa DirectScalarPressureAction || env.action_strat isa DirectVectorPressureAction "StepwiseRDEPolicy requires direct action types"
        @assert length(ts) == length(c) "Length of time steps and control actions must be equal"
        @assert issorted(ts) "Time steps must be in ascending order"
        if env.action_strat isa DirectVectorPressureAction
            @assert eltype(c) <: Vector{T} "Control actions must be a vector of vectors for DirectVectorPressureAction"
            @assert all(length(action) == env.action_strat.n_sections for action in c) "Each control action must match n_sections"
        else
            @assert eltype(c) <: T "Control actions must be a vector of scalars for DirectScalarPressureAction"
        end
        return new{T}(env, ts, c)
    end
end

function _predict_action(π::StepwiseRDEPolicy, s::AbstractVector{T}) where {T <: AbstractFloat}
    t = π.env.t
    cache = π.env.prob.method.cache
    past = π.ts .≤ t
    idx = findlast(past)

    # If before first time step, return current pressure (maintain status quo)
    if isnothing(idx)
        if π.env.action_strat isa DirectVectorPressureAction
            # Sample current pressure at section control points
            N = π.env.prob.params.N
            points_per_section = N ÷ π.env.action_strat.n_sections
            return cache.u_p_current[1:points_per_section:end]
        elseif π.env.action_strat isa DirectScalarPressureAction
            # Return first value of current pressure
            return cache.u_p_current[1]
        end
    end

    if π.env.action_strat isa DirectVectorPressureAction
        u_pmax = π.env.u_pmax
        return clamp.(π.c[idx], zero(T), u_pmax)
    elseif π.env.action_strat isa DirectScalarPressureAction
        u_pmax = π.env.u_pmax
        return clamp(π.c[idx], zero(T), u_pmax)
    else
        @error "Unknown action type $(typeof(π.env.action_strat)) for StepwiseRDEPolicy"
    end
end

function Base.show(io::IO, π::StepwiseRDEPolicy)
    return print(io, "StepwiseRDEPolicy($(length(π.ts)) steps)")
end

function Base.show(io::IO, ::MIME"text/plain", π::StepwiseRDEPolicy)
    println(io, "StepwiseRDEPolicy:")
    println(io, "  steps: $(length(π.ts))")
    println(io, "  env: $(typeof(π.env))")
    return println(io, "  time range: [$(minimum(π.ts)), $(maximum(π.ts))]")
end

"""
    RandomRDEPolicy{T<:AbstractFloat} <: AbstractRDEPolicy

Policy that applies random control values.

# Fields
- `env::RDEEnv{T}`: RDE environment

# Notes
Generates random values in [-1, 1] for each control dimension
"""
struct RandomRDEPolicy{T <: AbstractFloat} <: AbstractRDEPolicy
    env::RDEEnv{T}
    scale::T
    function RandomRDEPolicy(env::RDEEnv{T}, scale::T = 1.0f0) where {T <: AbstractFloat}
        return new{T}(env, scale)
    end
end

function _predict_action(π::RandomRDEPolicy, state::AbstractVector{T}) where {T <: AbstractFloat}
    action1 = π.scale * (2 * rand(T) - 1)
    action2 = π.scale * (2 * rand(T) - 1)
    if π.env.action_strat isa ScalarAreaScalarPressureAction
        return [action1, action2]
    elseif π.env.action_strat isa ScalarPressureAction
        return action2
    else
        @error "Unknown action type $(typeof(π.env.action_strat)) for RandomRDEPolicy"
    end
end

function Base.show(io::IO, π::RandomRDEPolicy)
    return print(io, "RandomRDEPolicy(scale=$(π.scale))")
end

function Base.show(io::IO, ::MIME"text/plain", π::RandomRDEPolicy)
    println(io, "RandomRDEPolicy:")
    return println(io, "  env: $(typeof(π.env))")
    return println(io, "  scale: $(π.scale)")
end

"""
    DelayedPolicy{T<:AbstractFloat} <: AbstractRDEPolicy

A policy wrapper that delays the start of another policy until a specified time.
Before the start time, returns zero actions.

# Fields
- `policy::Policy`: The policy to be delayed
- `start_time::T`: Time at which to start using the wrapped policy
- `env::RDEEnv{T}`: RDE environment (needed to access current time)

# Example
```julia
base_policy = RandomRDEPolicy(env)
delayed_policy = DelayedPolicy(base_policy, 100.0f0, env)
```
"""
struct DelayedPolicy{T <: AbstractFloat, P <: AbstractRDEPolicy} <: AbstractRDEPolicy
    policy::P
    start_time::T
    env::RDEEnv{T}
end

function _predict_action(π::DelayedPolicy, s::Union{AbstractVector{T}, Matrix{T}}) where {T <: AbstractFloat}
    t = π.env.t
    if t < π.start_time
        if π.env.action_strat isa ScalarAreaScalarPressureAction
            return [zeros(T, 2)]
        elseif π.env.action_strat isa ScalarPressureAction
            return zero(T)
        elseif π.env.action_strat isa VectorPressureAction
            return zeros(T, π.env.action_strat.n_sections)
        elseif π.env.action_strat isa LinearScalarPressureAction || π.env.action_strat isa LinearVectorPressureAction
            u_p = mean(π.env.prob.method.cache.u_p_current)
            u_pmax = π.env.u_pmax
            action = 2 * u_p / u_pmax - 1
            control_from_action = linear_control_target(action, u_pmax)
            @assert abs(u_p - control_from_action) < 1.0e-6 "action $(action) and control_from_action $(control_from_action) are not close"
            if π.env.action_strat isa LinearScalarPressureAction
                return [action]
            elseif π.env.action_strat isa LinearVectorPressureAction
                return fill(action, π.env.action_strat.n_sections)
            end
        else
            @error "Unknown action type $(typeof(π.env.action_strat)) for DelayedPolicy"
        end
    else
        return _predict_action(π.policy, s)
    end
end

function Base.show(io::IO, π::DelayedPolicy)
    return print(io, "DelayedPolicy(t>$(π.start_time))")
end

function Base.show(io::IO, ::MIME"text/plain", π::DelayedPolicy)
    println(io, "DelayedPolicy:")
    println(io, "  start_time: $(π.start_time)")
    println(io, "  policy: $(π.policy)")
    return println(io, "  env: $(typeof(π.env))")
end

struct ScaledPolicy{T <: AbstractFloat, P <: AbstractRDEPolicy} <: AbstractRDEPolicy
    policy::P
    scale::T
end

function get_env(π::ScaledPolicy)
    return get_env(π.policy)
end

function _predict_action(π::ScaledPolicy, s::AbstractVector{T}) where {T <: AbstractFloat}
    return π.scale::T .* _predict_action(π.policy, s)
end

function Base.show(io::IO, π::ScaledPolicy)
    return print(io, "ScaledPolicy(scale=$(π.scale))")
end

function Base.show(io::IO, ::MIME"text/plain", π::ScaledPolicy)
    println(io, "ScaledPolicy:")
    println(io, "  scale: $(π.scale)")
    return println(io, "  policy: $(π.policy)")
end

struct LinearPolicy{T <: AbstractFloat} <: AbstractRDEPolicy
    env::RDEEnv{T}
    ts::Vector{T}  # Vector of time points
    c::Union{Vector{Vector{T}}, Vector{T}}  # Vector of control values at each time point

    function LinearPolicy(env::RDEEnv{T}, ts::Vector{T}, c::Union{Vector{Vector{T}}, Vector{T}}) where {T <: AbstractFloat}
        @assert env.action_strat isa DirectScalarPressureAction || env.action_strat isa DirectVectorPressureAction "LinearPolicy requires direct action types"
        @assert length(ts) == length(c) "Length of time points and control values must be equal"
        @assert issorted(ts) "Time points must be in ascending order"
        @assert length(ts) >= 2 "At least two time points are required for linear interpolation"
        if env.action_strat isa DirectVectorPressureAction
            @assert eltype(c) <: Vector{T} "Control values must be a vector of vectors for DirectVectorPressureAction"
            @assert all(length(action) == env.action_strat.n_sections for action in c) "Each control value must match n_sections"
        else
            @assert eltype(c) <: T "Control values must be a vector of scalars for DirectScalarPressureAction"
        end
        return new{T}(env, ts, c)
    end
end

function _predict_action(π::LinearPolicy, s)
    t = π.env.t
    cache = π.env.prob.method.cache

    # Find the time points to interpolate between
    past = π.ts .≤ t
    idx = findlast(past)

    if isnothing(idx)
        target_value = π.c[1]
    elseif idx == length(π.ts)
        target_value = π.c[end]
    else
        t1, t2 = π.ts[idx], π.ts[idx + 1]
        c1, c2 = π.c[idx], π.c[idx + 1]
        if π.env.action_strat isa DirectVectorPressureAction
            target_value = c1 .+ (c2 .- c1) .* (t - t1) / (t2 - t1)
        elseif π.env.action_strat isa DirectScalarPressureAction
            target_value = c1 + (c2 - c1) * (t - t1) / (t2 - t1)
        else
            @error "Unknown action type $(typeof(π.env.action_strat)) for LinearPolicy"
        end
    end

    if π.env.action_strat isa DirectVectorPressureAction
        u_pmax = π.env.u_pmax
        return clamp.(target_value, zero(eltype(target_value)), u_pmax)
    elseif π.env.action_strat isa DirectScalarPressureAction
        u_pmax = π.env.u_pmax
        return clamp(target_value, zero(typeof(target_value)), u_pmax)
    else
        @error "Unknown action type $(typeof(π.env.action_strat)) for LinearPolicy"
    end
end

function Base.show(io::IO, π::LinearPolicy)
    return print(io, "LinearPolicy($(length(π.ts)) points)")
end

function Base.show(io::IO, ::MIME"text/plain", π::LinearPolicy)
    println(io, "LinearPolicy:")
    println(io, "  points: $(length(π.ts))")
    println(io, "  env: $(typeof(π.env))")
    return println(io, "  time range: [$(minimum(π.ts)), $(maximum(π.ts))]")
end

"""
    SawtoothPolicy{T<:AbstractFloat} <: AbstractRDEPolicy

Policy that implements a sawtooth control pattern for pressure control, periodically increasing from min to max value.

# Fields
- `env::RDEEnv{T}`: RDE environment
- `timescale::T`: Time period for one complete sawtooth cycle
- `max_value::T`: Maximum pressure value
- `min_value::T`: Minimum pressure value (value to drop to)

# Notes
- Only compatible with ScalarPressureAction
- Starts from the current injection pressure and gradually increases
"""
struct SawtoothPolicy{T <: AbstractFloat} <: AbstractRDEPolicy
    env::RDEEnv{T}
    timescale::T
    max_value::T
    min_value::T

    function SawtoothPolicy(env::RDEEnv{T}, timescale::T, max_value::T, min_value::T) where {T <: AbstractFloat}
        @assert env.action_strat isa DirectScalarPressureAction "SawtoothPolicy requires direct scalar action"
        @assert timescale > 0 "Timescale must be positive"
        @assert max_value > min_value "Max value must be greater than min value"
        return new{T}(env, timescale, max_value, min_value)
    end
end

function _predict_action(π::SawtoothPolicy, s)
    t = π.env.t
    cache = π.env.prob.method.cache

    # Calculate the current phase in the sawtooth cycle
    phase = mod(t, π.timescale)
    period_number = t ÷ π.timescale + 1
    # Calculate target value based on phase
    # Linear increase from min to max
    max_value = π.max_value
    min_value = period_number == 1 ? π.env.prob.params.u_p : π.min_value
    target_value = min_value + (max_value - min_value) * phase / π.timescale

    # Direct control target (clamped)
    return clamp(target_value, zero(typeof(target_value)), π.env.u_pmax)
end

function Base.show(io::IO, π::SawtoothPolicy)
    return print(io, "SawtoothPolicy(timescale=$(π.timescale))")
end

function Base.show(io::IO, ::MIME"text/plain", π::SawtoothPolicy)
    println(io, "SawtoothPolicy:")
    println(io, "  timescale: $(π.timescale)")
    println(io, "  max_value: $(π.max_value)")
    println(io, "  min_value: $(π.min_value)")
    return println(io, "  env: $(typeof(π.env))")
end

"""
    PIDControllerPolicy{T<:AbstractFloat} <: AbstractRDEPolicy

A policy that outputs constant PID gains for use with PIDAction environments.
The environment handles all PID computation (error, integral, derivative).

# Requirements
- Must be used with an environment that has PIDAction action type
- Environment derives target from RDE.SHOCK_PRESSURES[target_shock_count]

# Fields
- `Kp::T`: Proportional gain
- `Ki::T`: Integral gain
- `Kd::T`: Derivative gain

# Notes
- Policy is stateless - all PID state managed by environment's PIDActionCache
- To change target, use set_target_shock_count!(env, n)
"""
struct PIDControllerPolicy{T <: AbstractFloat} <: AbstractRDEPolicy
    Kp::T
    Ki::T
    Kd::T
end

# Convenience constructor with default Float32
PIDControllerPolicy(; Kp::Real, Ki::Real, Kd::Real) =
    PIDControllerPolicy{Float32}(Float32(Kp), Float32(Ki), Float32(Kd))

function _predict_action(π::PIDControllerPolicy, o)
    # Simply return the PID gains - environment does the computation
    return [π.Kp, π.Ki, π.Kd]
end

function Base.show(io::IO, π::PIDControllerPolicy)
    return print(io, "PIDControllerPolicy(Kp=$(π.Kp), Ki=$(π.Ki), Kd=$(π.Kd))")
end

function Base.show(io::IO, ::MIME"text/plain", π::PIDControllerPolicy)
    println(io, "PIDControllerPolicy:")
    println(io, "  Kp: $(π.Kp)")
    println(io, "  Ki: $(π.Ki)")
    return println(io, "  Kd: $(π.Kd)")
end
