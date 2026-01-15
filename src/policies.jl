abstract type AbstractRDEPolicy <: DRiL.AbstractPolicy end

"""
    _predict_action(policy::AbstractRDEPolicy, obs)
"""
function _predict_action end

function _predict_action(policy::DRiL.AbstractPolicy, observation::Vector)
    return policy(observation; deterministic = true)
end

function _predict_action(policy::DRiL.AbstractPolicy, observation::Matrix)
    obs_batch = collect(eachcol(observation))
    return policy(obs_batch; deterministic = true)
end

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
    control_shifts::Vector{<:AbstractControlShift} # moving reference velocity.
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
    println(io, "  observations: $(typeof(data.observations))($(length(data.observations)))")
    println(io, "  control_shifts: $(typeof(data.control_shifts))($(length(data.control_shifts)))")
    return nothing
end

"""
    run_policy(π::AbstractPolicy, env::RDEEnv{T}; saves_per_action=10) where {T}

Run a policy `π` on the RDE environment and collect trajectory data.

# Arguments
- `π::AbstractPolicy`: Policy to execute
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
function run_policy(policy::AbstractPolicy, env::RDEEnv{T}; saves_per_action = 10) where {T}
    _reset!(env)
    @assert saves_per_action ≥ 1 "saves_per_action must be at least 1"


    dt = env.dt
    max_actions = ceil(env.prob.params.tmax / dt) + 1 |> Int
    N = env.prob.params.N

    # Preallocate action-aligned arrays
    action_ts = Vector{T}(undef, max_actions)
    ss, u_ps = get_init_control_data(env, env.action_strat, max_actions)
    rewards = get_init_rewards(env, env.reward_strat, max_actions)
    control_shifts = Vector{typeof(env.prob.control_shift_strategy)}(undef, max_actions)
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
        control_shifts[step] = deepcopy(env.prob.control_shift_strategy)

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

        # Record control summaries. This is the controls during the action step.
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

        # Collect solver states (drop first which is pre-action state)
        if typeof(env.prob.sol) == Nothing
            @info "sol is nothing at step $step"
            @info env.info
            step_states = Vector{Vector{T}}()
            step_ts = Vector{T}()
        else
            @debug "length(env.prob.sol.t) ($(length(env.prob.sol.t))) != saves_per_action + 1 ($(saves_per_action + 1))"
            last_ministep = env.prob.sol.t[end] - env.prob.sol.t[end - 1]
            step_states = env.prob.sol.u[2:end]
            step_ts = env.prob.sol.t[2:end]
            if length(env.prob.sol.t) > 2
                second_last_ministep = env.prob.sol.t[end - 1] - env.prob.sol.t[end - 2]
                if last_ministep < second_last_ministep / 10 #sometimes last step is very small, so we disregard the secon_to_last step
                    steps = length(env.prob.sol.t)
                    inds = [collect(1:(steps - 2)); steps]
                    step_states = env.prob.sol.u[inds]
                    step_ts = env.prob.sol.t[inds]
                end
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
        rewards[step] = env.reward

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
    control_shifts = control_shifts[1:n_actions]

    return PolicyRunData{T}(action_ts, ss, u_ps, rewards, actions, state_ts, states, observations, control_shifts)
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

# function get_init_rewards(env::RDEEnv{T}, reward_strat::MultiplicativeReward, max_steps::Int) where {T}
#     # Check if any of the component rewards is a multi-agent reward
#     if any(r isa MultiAgentCachedCompositeReward for r in reward_strat.rewards)
#         return Vector{Vector{T}}(undef, max_steps)
#     else
#         return Vector{T}(undef, max_steps)
#     end
# end

#TODO: remove need for these
# function get_init_control_data(env::RDEEnv{T}, action_strat::AbstractActionStrategy, max_steps::Int) where {T}
#     return Vector{T}(undef, max_steps), Vector{T}(undef, max_steps)
# end

# function get_init_control_data(env::RDEEnv{T}, action_strat::VectorPressureAction, max_steps::Int) where {T}
#     return Vector{T}(undef, max_steps), Vector{Vector{T}}(undef, max_steps)
# end

# function get_init_control_data(env::RDEEnv{T}, action_strat::DirectVectorPressureAction, max_steps::Int) where {T}
#     return Vector{T}(undef, max_steps), Vector{Vector{T}}(undef, max_steps)
# end

# function get_init_control_data(env::RDEEnv{T}, action_strat::LinearVectorPressureAction, max_steps::Int) where {T}
#     return Vector{T}(undef, max_steps), Vector{Vector{T}}(undef, max_steps)
# end
function get_init_control_data(env::RDEEnv{T}, action_strat::AbstractActionStrategy, max_steps::Int) where {T}
    a_space = DRiL.action_space(env)
    rand_action = rand(a_space)
    if rand_action isa AbstractVector && length(rand_action) == 1
        A = T
    else
        A = typeof(rand_action)
    end
    return Vector{T}(undef, max_steps), Vector{A}(undef, max_steps)
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
