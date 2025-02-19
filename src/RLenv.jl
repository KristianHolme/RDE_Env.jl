"""
    RDEEnvCache{T<:AbstractFloat}

Cache for RDE environment computations and state tracking.

# Fields
- `circ_u::CircularVector{T}`: Circular buffer for velocity field
- `circ_λ::CircularVector{T}`: Circular buffer for reaction progress
- `prev_u::Vector{T}`: Previous velocity field
- `prev_λ::Vector{T}`: Previous reaction progress
"""
mutable struct RDEEnvCache{T<:AbstractFloat}
    circ_u::CircularVector{T, Vector{T}}
    circ_λ::CircularVector{T, Vector{T}}
    prev_u::Vector{T}  # Previous step's u values
    prev_λ::Vector{T}  # Previous step's λ values
    action::Matrix{T} # column 1 = s action, column 2 = u_p action
    function RDEEnvCache{T}(N::Int) where {T<:AbstractFloat}
        # Initialize all arrays with zeros instead of undefined values
        circ_u = CircularArray(zeros(T, N))
        circ_λ = CircularArray(zeros(T, N))
        prev_u = zeros(T, N)
        prev_λ = zeros(T, N)
        action = zeros(T, N, 2)    
        return new{T}(circ_u, circ_λ, prev_u, prev_λ, action)
    end
end

function Base.show(io::IO, cache::RDEEnvCache)
    println(io, "RDEEnvCache{$(eltype(cache.circ_u))} with:")
    println(io, "  circ_u: $(typeof(cache.circ_u)) of size $(length(cache.circ_u))")
    println(io, "  circ_λ: $(typeof(cache.circ_λ)) of size $(length(cache.circ_λ))")
    println(io, "  prev_u: $(typeof(cache.prev_u)) of size $(length(cache.prev_u))")
    println(io, "  prev_λ: $(typeof(cache.prev_λ)) of size $(length(cache.prev_λ))")
    println(io, "  action: $(typeof(cache.action)) of size $(size(cache.action))")
end

"""
    RDEEnv{T<:AbstractFloat} <: AbstractRDEEnv

Reinforcement learning environment for the RDE system.

# Fields
- `prob::RDEProblem{T}`: Underlying RDE problem
- `state::Vector{T}`: Current system state
- `observation::Vector{T}`: Current observation vector
- `dt::T`: Time step
- `t::T`: Current time
- `done::Bool`: Episode termination flag
- `truncated::Bool`: Episode truncation flag
- `terminated::Bool`: Episode termination flag
- `reward::T`: Current reward
- `smax::T`: Maximum value for s parameter
- `u_pmax::T`: Maximum value for u_p parameter
- `α::T`: Action momentum parameter
- `τ_smooth::T`: Control smoothing time constant
- `cache::RDEEnvCache{T}`: Environment cache
- `action_type::AbstractActionType`: Type of control actions
- `reward_type::AbstractRDEReward`: Type of reward function
- `verbose::Bool`: Control solver output
# Constructor
```julia
RDEEnv{T}(;
    dt=10.0,
    smax=4.0,
    u_pmax=1.2,
    params::RDEParam{T}=RDEParam{T}(),
    momentum=0.5,
    τ_smooth=1.25,
    fft_terms::Int=32,
    observation_strategy::AbstractObservationStrategy=FourierObservation(fft_terms),
    action_type::AbstractActionType=ScalarPressureAction(),
    reward_type::AbstractRDEReward=ShockSpanReward(target_shock_count=3),
    verbose::Bool=true,
    kwargs...
) where {T<:AbstractFloat}
```

# Example
```julia
env = RDEEnv(dt=5.0, smax=3.0)
```
"""
mutable struct RDEEnv{T<:AbstractFloat} <: AbstractRDEEnv{T}
    prob::RDEProblem{T}                  # RDE problem
    state::Vector{T}
    observation::AbstractArray{T}
    dt::T                       # time step
    t::T                        # Current time
    done::Bool                        # Termination flag
    truncated::Bool
    terminated::Bool
    reward::Union{T, Vector{T}}
    smax::T
    u_pmax::T
    α::T #action momentum
    τ_smooth::T #smoothing time constant
    cache::RDEEnvCache{T}
    action_type::AbstractActionType
    observation_strategy::AbstractObservationStrategy
    reward_type::AbstractRDEReward
    verbose::Bool               # Control solver output
    info::Dict{String, Any}
    steps_taken::Int
    function RDEEnv{T}(;
        dt=1.0,
        smax=4.0,
        u_pmax=1.2,
        params::RDEParam{T}=RDEParam{T}(),
        momentum=0.0,
        τ_smooth=0.01,
        observation_strategy::AbstractObservationStrategy=FourierObservation(16),
        action_type::AbstractActionType=ScalarPressureAction(),
        reward_type::AbstractRDEReward=ShockSpanReward(target_shock_count=3),
        verbose::Bool=true,
        kwargs...) where {T<:AbstractFloat}

        if τ_smooth > dt
            @warn "τ_smooth > dt, this will cause discontinuities in the control signal"
            @info "Setting τ_smooth = $(dt/8)"
            τ_smooth = dt/8
        end

        prob = RDEProblem(params; kwargs...)
        prob.method.cache.τ_smooth = τ_smooth

        # Set N in action_type
        set_N!(action_type, params.N)

        initial_state = vcat(prob.u0, prob.λ0)
        init_observation = get_init_observation(observation_strategy, params.N)
        cache = RDEEnvCache{T}(params.N)
        return new{T}(prob, initial_state, init_observation,
                      dt, T(0.0), false, false, false, T(0.0), smax, u_pmax,
                      momentum, τ_smooth, cache,
                      action_type, observation_strategy, 
                      reward_type, verbose, Dict{String, Any}(), 0)
    end
end

RDEEnv(; kwargs...) = RDEEnv{Float32}(; kwargs...)
RDEEnv(params::RDEParam{T}; kwargs...) where {T<:AbstractFloat} = RDEEnv{T}(; params=params, kwargs...)

function Base.show(io::IO, env::RDEEnv)
    println(io, "RDEEnv:")
    println(io, "  dt: $(env.dt)")
    println(io, "  t: $(env.t)")
    println(io, "  truncated: $(env.truncated)")
    println(io, "  terminated: $(env.terminated)")
    println(io, "  action type: $(env.action_type)")
    println(io, "  observation strategy: $(env.observation_strategy)")
    println(io, "  reward type: $(env.reward_type)")
    println(io, "  steps taken: $(env.steps_taken)")
end


"""
    CommonRLInterface.act!(env::RDEEnv{T}, action; saves_per_action::Int=0) where {T<:AbstractFloat}

Take an action in the environment.

# Arguments
- `env::RDEEnv{T}`: RDE environment
- `action`: Control action to take
- `saves_per_action::Int=0`: Number of intermediate saves per action

# Returns
- Current reward

# Notes
- Updates environment state and reward
- Handles smooth control transitions
- Supports multiple action types
"""
function CommonRLInterface.act!(env::RDEEnv{T}, action; saves_per_action::Int=0) where {T<:AbstractFloat}
    # Store current state before taking action
    @logmsg LogLevel(-10000) "Starting act! for environment on thread $(Threads.threadid())"
    N = env.prob.params.N
    env.cache.prev_u .= @view env.state[1:N]
    env.cache.prev_λ .= @view env.state[N+1:end]
    @logmsg LogLevel(-10000) "Stored previous state" prev_u=env.cache.prev_u prev_λ=env.cache.prev_λ

    t_span = (env.t, env.t + env.dt)
    @logmsg LogLevel(-10000) "Set time span" t_span=t_span
    env.prob.method.cache.control_time = env.t
    @logmsg LogLevel(-10000) "Set control time" control_time=env.prob.method.cache.control_time

    prev_controls = [env.prob.method.cache.s_current, env.prob.method.cache.u_p_current]
    c = [env.prob.method.cache.s_current, env.prob.method.cache.u_p_current]
    c_max = [env.smax, env.u_pmax]
    @logmsg LogLevel(-10000) "Initial controls" prev_controls=prev_controls c_max=c_max

    normalized_standard_actions = get_standard_normalized_actions(env.action_type, action)
    env.cache.action[:,1] = normalized_standard_actions[1]
    env.cache.action[:,2] = normalized_standard_actions[2]
    @logmsg LogLevel(-10000) "Normalized actions" actions=normalized_standard_actions
    
    for i in 1:2
        a = normalized_standard_actions[i]
        if any(abs.(a) .> 1)
            @warn "action $a out of bounds [-1,1]"
        end
        c_prev = c[i]
        c_hat = @. ifelse(a < 0, c_prev .* (a .+ 1), c_prev .+ (c_max[i] .- c_prev) .* a)
        c[i] = env.α .* c_prev .+ (1 - env.α) .* c_hat
    end

    env.prob.method.cache.s_previous = env.prob.method.cache.s_current
    env.prob.method.cache.u_p_previous = env.prob.method.cache.u_p_current
    env.prob.method.cache.s_current = c[1]
    env.prob.method.cache.u_p_current = c[2]

    @logmsg LogLevel(-10000) "taking action $action at time $(env.t), controls: $(mean.(prev_controls)) to $(mean.(c))"

    prob_ode = ODEProblem{true, SciMLBase.FullSpecialize}(RDE_RHS!, env.state, t_span, env.prob)
    
    if saves_per_action == 0
        sol = OrdinaryDiffEq.solve(prob_ode, Tsit5(), save_on=false, 
                                   isoutofdomain=RDE.outofdomain, verbose=env.verbose)
    else
        saveat = env.dt / saves_per_action
        sol = OrdinaryDiffEq.solve(prob_ode, Tsit5(), saveat=saveat, 
                                   isoutofdomain=RDE.outofdomain, verbose=env.verbose)
    end

    
    #Check termination caused by ODE solver
    if sol.retcode != :Success || any(isnan.(sol.u[end]))
        if any(isnan.(sol.u[end]))
            @warn "NaN state detected"
        end
        @logmsg LogLevel(-10000) "ODE solver failed, controls: $(mean(prev_controls)) to $(mean(c))"
        env.terminated = true
        env.done = true
        set_termination_reward!(env, -2.0)
        env.info["Termination.Reason"] = "ODE solver failed"
        env.info["Termination.ReturnCode"] = sol.retcode
        env.info["Termination.env_t"] = env.t
        @logmsg LogLevel(-500) "ODE solver failed, t=$(env.t), terminating"
    else #advance environment
        env.prob.sol = sol
        env.t = sol.t[end]
        env.state = sol.u[end]

        env.steps_taken += 1

        set_reward!(env, env.reward_type)
        if env.terminated #maybe reward caused termination
            set_termination_reward!(env, -2.0)
            env.done = true;
            @logmsg LogLevel(-10000) "termination caused by reward"
            @logmsg LogLevel(-500) "terminated, t=$(env.t), from reward?"
        end
        if env.t ≥ env.prob.params.tmax
            env.done = true
            env.truncated = true 
            @logmsg LogLevel(-10000) "tmax reached, t=$(env.t)"
            env.info["Truncation.Reason"] = "tmax reached"
        end
    end

    if env.done != xor(env.truncated, env.terminated)
        @warn "done is not xor(truncated, terminated), at t=$(env.t)" env.done, env.truncated, env.terminated
        @info "info: $(env.info)"
        @logmsg LogLevel(-500) sol.retcode
    end
    @logmsg LogLevel(-10000) "End of step reward: $(env.reward)"
    return env.reward
end



# CommonRLInterface implementations
CommonRLInterface.state(env::RDEEnv) = vcat(env.state, env.t)
CommonRLInterface.terminated(env::RDEEnv) = env.done
function CommonRLInterface.observe(env::RDEEnv)
    return compute_observation(env, env.observation_strategy)
end

function CommonRLInterface.actions(env::RDEEnv)
    n = action_dim(env.action_type)
    return [(-1 .. 1) for _ in 1:n]
end

function CommonRLInterface.clone(env::RDEEnv)
    env2 = deepcopy(env)
    @logmsg LogLevel(-10000) "env is copied!"
    return env2
end

function CommonRLInterface.setstate!(env::RDEEnv, s)
    env.state = s[1:end-1]
    env.t = s[end]
end

function POMDPs.initialobs(RLEnvPOMDP, s)
    return [CommonRLInterface.observe(RLEnvPOMDP.env)]
end

"""
    CommonRLInterface.reset!(env::RDEEnv)

Reset the environment to its initial state.

# Arguments
- `env::RDEEnv`: Environment to reset

# Effects
- Resets time to 0
- Resets state to initial conditions
- Resets reward to 0
- Resets control parameters to initial values
- Initializes previous state tracking
"""
function CommonRLInterface.reset!(env::RDEEnv)
    env.t = 0
    RDE.reset_state_and_pressure!(env.prob, env.prob.reset_strategy)
    env.state = vcat(env.prob.u0, env.prob.λ0)
    set_termination_reward!(env, 0.0)
    env.steps_taken = 0
    env.done = false
    env.truncated = false
    env.terminated = false
    set_reward!(env, env.reward_type)

    env.prob.method.cache.τ_smooth = env.τ_smooth
    env.prob.method.cache.u_p_previous = fill(env.prob.params.u_p, env.prob.params.N)
    env.prob.method.cache.u_p_current = fill(env.prob.params.u_p, env.prob.params.N)
    env.prob.method.cache.s_previous = fill(env.prob.params.s, env.prob.params.N)
    env.prob.method.cache.s_current = fill(env.prob.params.s, env.prob.params.N)

    # Initialize previous state
    N = env.prob.params.N
    env.cache.prev_u .= @view env.state[1:N]
    env.cache.prev_λ .= @view env.state[N+1:end]
    
    nothing
end

function set_termination_reward!(env::RDEEnv, value::Number)
    value = eltype(env.reward)(value)
    if env.reward isa Number
        env.reward = value
    else
        env.reward .= value
    end
end