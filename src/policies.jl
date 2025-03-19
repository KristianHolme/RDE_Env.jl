"""
    PolicyRunData{T<:AbstractFloat}

Container for data collected during policy execution.

# Fields
- `action_ts::Vector{T}`: Time points for actions
- `ss::Vector{T}`: Control parameter s at each action
- `u_ps::Vector{T}`: Control parameter u_p at each action
- `rewards::Vector{T}`: Rewards at each action
- `energy_bal::Vector{T}`: Energy balance at each state
- `chamber_p::Vector{T}`: Chamber pressure at each state
- `state_ts::Vector{T}`: Time points for states
- `states::Vector{Vector{T}}`: States at each time point
"""
struct PolicyRunData{T<:AbstractFloat}
    action_ts::Vector{T} #time points for actions
    ss::Union{Vector{T}, Vector{Vector{T}}} #control parameter s at each action
    u_ps::Union{Vector{T}, Vector{Vector{T}}} #control parameter u_p at each action
    rewards::Union{Vector{T}, Vector{Vector{T}}} #rewards at each action
    energy_bal::Vector{T} #energy balance at each state
    chamber_p::Vector{T} #chamber pressure at each state
    state_ts::Vector{T} #time points for states
    states::Vector{Vector{T}} #states at each time point
    observations::Vector{Union{Vector{T}, Matrix{T}}} #observations at each time point
end

function Base.show(io::IO, data::PolicyRunData)
    println(io, "PolicyRunData:")
    println(io, "  action_ts: $(typeof(data.action_ts))($(length(data.action_ts)))")
    println(io, "  ss: $(typeof(data.ss))($(length(data.ss)))")
    println(io, "  u_ps: $(typeof(data.u_ps))($(length(data.u_ps)))")
    println(io, "  rewards: $(typeof(data.rewards))($(length(data.rewards)))")
    println(io, "  energy_bal: $(typeof(data.energy_bal))($(length(data.energy_bal)))")
    println(io, "  chamber_p: $(typeof(data.chamber_p))($(length(data.chamber_p)))")
    println(io, "  state_ts: $(typeof(data.state_ts))($(length(data.state_ts)))")
    println(io, "  states: $(typeof(data.states))($(length(data.states)))")
    println(io, "  observations: $(typeof(data.observations))($(length(data.observations)))")
end

"""
    run_policy(π::Policy, env::RDEEnv{T}; saves_per_action=1) where {T}

Run a policy `π` on the RDE environment and collect trajectory data.

# Arguments
- `π::Policy`: Policy to execute
- `env::RDEEnv{T}`: RDE environment to run the policy in
- `saves_per_action=1`: Save full state every `saves_per_action` steps

# Returns
`PolicyRunData` containing:
- `ts`: Time points for each action
- `ss`: Control parameter s values
- `u_ps`: Control parameter u_p values  
- `rewards`: Rewards received
- `energy_bal`: Energy balance at each step
- `chamber_p`: Chamber pressure at each step
- `state_ts`: Time points for each state
- `states`: Full system state at each time point

# Example
```julia
env = RDEEnv()
policy = ConstantRDEPolicy(env)
data = run_policy(policy, env, saves_per_action=10)
```
"""
function run_policy(π::Policy, env::AbstractRDEEnv{T}; saves_per_action=1) where {T}
    reset!(env)
    dt = env.dt
    max_steps = ceil(env.prob.params.tmax / dt) + 2 |> Int # +1 for initial state, +1 for overshoot
    N = env.prob.params.N
    
    # Initialize vectors for action data
    ts = Vector{T}(undef, max_steps)
    ss, u_ps = get_init_control_data(env, env.action_type, max_steps)
    # ss = Vector{T}(undef, max_steps)
    # u_ps = Vector{T}(undef, max_steps)
    rewards = get_init_rewards(env, env.reward_type, max_steps)
    # rewards = Vector{T}(undef, max_steps)
    
    # For saves_per_action > 0, we need more space for state data
    max_state_points = if saves_per_action == 0
        max_steps  # Only save at action points
    else
        max_steps * (saves_per_action + 1)  # +1 to account for potential extra points
    end
    
    energy_bal = Vector{T}(undef, max_state_points)
    chamber_p = Vector{T}(undef, max_state_points)
    states = Vector{Vector{T}}(undef, max_state_points)
    state_ts = Vector{T}(undef, max_state_points)
    if env.observation_strategy isa AbstractMultiAgentObservationStrategy
        observations = Vector{Matrix{T}}(undef, max_state_points)
    else
        observations = Vector{Vector{T}}(undef, max_state_points)
    end
    
    step = 0
    total_state_steps = 0
    
    function log!(step)
        ind = step + 1
        ts[ind] = env.t
        if eltype(ss) <: AbstractVector
            sections = env.action_type.n_sections
            ss[ind] = section_reduction(env.prob.method.cache.s_current, sections)
        elseif eltype(ss) <: Number
            ss[ind] = mean(env.prob.method.cache.s_current)
        end
        if eltype(u_ps) <: AbstractVector
            sections = env.action_type.n_sections
            u_ps[ind] = section_reduction(env.prob.method.cache.u_p_current, sections)
        elseif eltype(u_ps) <: Number
            u_ps[ind] = mean(env.prob.method.cache.u_p_current)
        end
        if eltype(rewards) <: AbstractVector
            rewards[ind] = env.reward
        else
            rewards[ind] = env.reward
        end
        observations[ind] = observe(env)

        if step > 0
            step_states = env.prob.sol.u[2:end]
            step_ts = env.prob.sol.t[2:end]
            n_states = length(step_states)
        else
            step_states = [env.state]
            step_ts = [env.t]
            n_states = 1
        end

        # Calculate indices for this step's data
        start_idx = total_state_steps + 1
        end_idx = total_state_steps + n_states
        
        # Ensure we have enough space
        if end_idx > max_state_points
            # Extend arrays if needed
            new_size = end_idx + max_steps * (saves_per_action + 1)
            resize!(energy_bal, new_size)
            resize!(chamber_p, new_size)
            resize!(states, new_size)
            resize!(state_ts, new_size)
            max_state_points = new_size
        end
        
        # Save states and timestamps
        state_ts[start_idx:end_idx] = step_ts
        states[start_idx:end_idx] = step_states
        
        # Save energy balance and chamber pressure
        energy_bal[start_idx:end_idx] = energy_balance.(step_states, Ref(env.prob.params))
        chamber_p[start_idx:end_idx] = chamber_pressure.(step_states, Ref(env.prob.params))
        
        total_state_steps += n_states
    end

    log!(step)
    while !env.done && step < max_steps
        action = POMDPs.action(π, observe(env))
        act!(env, action, saves_per_action=saves_per_action)
        step += 1
        log!(step)
    end
    
    # Trim arrays to actual size
    ts = ts[1:step+1]
    ss = ss[1:step+1]
    u_ps = u_ps[1:step+1]
    rewards = rewards[1:step+1]
    energy_bal = energy_bal[1:total_state_steps]
    chamber_p = chamber_p[1:total_state_steps]
    state_ts = state_ts[1:total_state_steps]
    states = states[1:total_state_steps]
    observations = observations[1:step+1]

    return PolicyRunData{T}(ts, ss, u_ps, rewards, energy_bal, chamber_p, state_ts, states, observations)
end


function get_init_rewards(env::RDEEnv{T}, reward_type::AbstractRDEReward, max_steps::Int) where {T}
    return Vector{T}(undef, max_steps)
end

function get_init_rewards(env::RDEEnv{T}, reward_type::MultiSectionReward, max_steps::Int) where {T}
    # n_section = reward_type.n_sections
    return Vector{Vector{T}}(undef, max_steps)
end

function get_init_control_data(env::RDEEnv{T}, action_type::AbstractActionType, max_steps::Int) where {T}
    return Vector{T}(undef, max_steps), Vector{T}(undef, max_steps)
end

function get_init_control_data(env::RDEEnv{T}, action_type::VectorPressureAction, max_steps::Int) where {T}
    return Vector{T}(undef, max_steps), Vector{Vector{T}}(undef, max_steps)
end

function section_reduction(v::Vector{T}, sections::Int) where {T}
    N = length(v)
    if N % sections != 0
        @warn "Vector length $N is not divisible by section length $sections"
    end
    section_length = Int(round(N // sections))
    m = reshape(v, section_length, :)
    return vec(mean(m, dims=1))
end


"""
    ConstantRDEPolicy <: Policy

Policy that maintains constant control values.

# Fields
- `env::RDEEnv`: RDE environment

# Notes
Returns [0.0, 0.0] for ScalarAreaScalarPressureAction
Returns 0.0 for ScalarPressureAction
"""
struct ConstantRDEPolicy <: Policy
    env::RDEEnv
    ConstantRDEPolicy(env::RDEEnv=RDEEnv()) = new(env)
end

function POMDPs.action(π::ConstantRDEPolicy, s)
    if π.env.action_type isa ScalarAreaScalarPressureAction
        return [0.0, 0.0]
    elseif π.env.action_type isa ScalarPressureAction
        return 0.0
    else
        @error "Unknown action type $(typeof(π.env.action_type)) for ConstantRDEPolicy"
    end
end

Base.show(io::IO, π::ConstantRDEPolicy) = print(io, "ConstantRDEPolicy($(typeof(π.env)))")

"""
    SinusoidalRDEPolicy{T<:AbstractFloat} <: Policy

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
struct SinusoidalRDEPolicy{T<:AbstractFloat} <: Policy
    env::RDEEnv{T}
    w_1::T  
    w_2::T  
    scale::T
    function SinusoidalRDEPolicy(env::RDEEnv{T}; w_1::T=1.0, w_2::T=2.0) where {T<:AbstractFloat}
        new{T}(env, w_1, w_2)
    end
end

function POMDPs.action(π::SinusoidalRDEPolicy, s)
    t = π.env.t
    action1 = sin(π.w_1 * t)
    action2 = sin(π.w_2 * t)
    if π.env.action_type isa ScalarAreaScalarPressureAction 
        return [action1, action2]
    elseif π.env.action_type isa ScalarPressureAction
        return action2
    else
        @error "Unknown action type $(typeof(π.env.action_type)) for SinusoidalRDEPolicy"
    end
end

Base.show(io::IO, π::SinusoidalRDEPolicy) = print(io, "SinusoidalRDEPolicy($(typeof(π.env)))")

"""
    StepwiseRDEPolicy{T<:AbstractFloat} <: Policy

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
struct StepwiseRDEPolicy{T<:AbstractFloat} <: Policy
    env::RDEEnv{T}
    ts::Vector{T}  # Vector of time steps
    c::Union{Vector{Vector{T}}, Vector{T}}  # Vector of control actions

    function StepwiseRDEPolicy(env::RDEEnv{T}, ts::Vector{T}, c::Union{Vector{Vector{T}}, Vector{T}}) where {T<:AbstractFloat}
        @assert length(ts) == length(c) "Length of time steps and control actions must be equal"
        @assert issorted(ts) "Time steps must be in ascending order"
        if env.action_type isa ScalarAreaScalarPressureAction
            @assert all(length(action) == 2 for action in c) "Each control action must have 2 elements"
            @assert eltype(c) <: Vector{T} "Control actions must be a vector of vectors"
        else
            @assert eltype(c) <: T "Control actions must be a vector of scalars"
        end
        env.α = 0.0 #to assure that get_scaled_control works
        new{T}(env, ts, c)
    end
end

function POMDPs.action(π::StepwiseRDEPolicy, s)
    t = π.env.t
    cache = π.env.prob.method.cache
    past = π.ts .≤ t    
    idx = findlast(past)
    if isnothing(idx)
        return zero(π.c[1])
    end
    if π.env.action_type isa ScalarAreaScalarPressureAction
        return get_scaled_control.([cache.s_current[1], cache.u_p_current[1]], [π.env.smax, π.env.u_pmax], π.c[idx])
    elseif π.env.action_type isa ScalarPressureAction
        return get_scaled_control(cache.u_p_current[1], π.env.u_pmax, π.c[idx])
    else
        @error "Unknown action type $(typeof(π.env.action_type)) for StepwiseRDEPolicy"
    end
end

Base.show(io::IO, π::StepwiseRDEPolicy) = print(io, "StepwiseRDEPolicy($(typeof(π.env)))")

"""
    get_scaled_control(current, max_val, target)

Scale control value to [-1, 1] range based on current value and target.

# Arguments
- `current`: Current control value
- `max_val`: Maximum allowed value
- `target`: Target control value

# Returns
Scaled control value in [-1, 1]

# Notes
Assumes zero momentum (env.α = 0)
"""
function get_scaled_control(current, max_val, target)
    if target < current
        return target / current - 1.0
    else
        return (target - current) / (max_val - current)
    end
end

"""
    RandomRDEPolicy{T<:AbstractFloat} <: Policy

Policy that applies random control values.

# Fields
- `env::RDEEnv{T}`: RDE environment

# Notes
Generates random values in [-1, 1] for each control dimension
"""
struct RandomRDEPolicy{T<:AbstractFloat} <: Policy
    env::RDEEnv{T}
    function RandomRDEPolicy(env::RDEEnv{T}) where {T<:AbstractFloat}
        new{T}(env)
    end
end

function POMDPs.action(π::RandomRDEPolicy, state)
    action1 = 2 * rand() - 1
    action2 = 2 * rand() - 1
    if π.env.action_type isa ScalarAreaScalarPressureAction
        return [action1, action2]
    elseif π.env.action_type isa ScalarPressureAction
        return action2
    else
        @error "Unknown action type $(typeof(π.env.action_type)) for RandomRDEPolicy"
    end
end

Base.show(io::IO, π::RandomRDEPolicy) = print(io, "RandomRDEPolicy($(typeof(π.env)))")

"""
    DelayedPolicy{T<:AbstractFloat} <: Policy

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
struct DelayedPolicy{T<:AbstractFloat} <: Policy
    policy::Policy
    start_time::T
    env::AbstractRDEEnv{T}
end

function POMDPs.action(π::DelayedPolicy, s)
    t = π.env.t
    if t < π.start_time
        if π.env.action_type isa ScalarAreaScalarPressureAction
            return [0.0f0, 0.0f0]
        elseif π.env.action_type isa ScalarPressureAction
            return 0.0f0
        elseif π.env.action_type isa VectorPressureAction
            return zeros(Float32, π.env.action_type.n_sections)
        else
            @error "Unknown action type $(typeof(π.env.action_type)) for DelayedPolicy"
        end
    else
        return POMDPs.action(π.policy, s)
    end
end

Base.show(io::IO, π::DelayedPolicy) = print(io, "DelayedPolicy($(π.policy), $(π.start_time))")

struct ScaledPolicy{T<:AbstractFloat} <: Policy
    policy::Policy
    scale::T
end

function POMDPs.action(π::ScaledPolicy, s)
    return π.scale .* POMDPs.action(π.policy, s)
end

Base.show(io::IO, π::ScaledPolicy) = print(io, "ScaledPolicy($(π.policy), $(π.scale))")

struct LinearPolicy{T<:AbstractFloat} <: Policy
    env::RDEEnv{T}
    start_value::Union{Vector{T}, T}
    end_value::Union{Vector{T}, T}
    function LinearPolicy(env::RDEEnv{T}, start_value::Union{Vector{T}, T}, end_value::Union{Vector{T}, T}, start_time::T, end_time::T) where {T<:AbstractFloat}
        if env.action_type isa ScalarAreaScalarPressureAction
            @assert length(start_value) == 2 && length(end_value) == 2 "Start and end values must have 2 elements"
        elseif env.action_type isa ScalarPressureAction
            @assert length(start_value) == 1 && length(end_value) == 1 "Start and end values must have 1 element"
        else
            @error "Unknown action type $(typeof(env.action_type)) for LinearPolicy"
        end
        new{T}(env, start_value, end_value, start_time, end_time)
    end
end

function POMDPs.action(π::LinearPolicy, s)
    t = π.env.t
    tmax = π.env.prob.params.tmax
    if π.env.action_type isa ScalarAreaScalarPressureAction
        target_values = π.start_value .+ (π.end_value .- π.start_value) .* t / tmax
        return get_scaled_control.([π.env.prob.method.cache.s_current[1], π.env.prob.method.cache.u_p_current[1]], [π.env.smax, π.env.u_pmax], target_values)
    elseif π.env.action_type isa ScalarPressureAction
        target_value = π.start_value + (π.end_value - π.start_value) * t/tmax
        return get_scaled_control(π.env.prob.method.cache.u_p_current[1], π.env.u_pmax, target_value)
    else
        @error "Unknown action type $(typeof(π.env.action_type)) for LinearPolicy"
    end
end

Base.show(io::IO, π::LinearPolicy) = print(io, "LinearPolicy($(typeof(π.env)), $(π.start_value), $(π.end_value)")
