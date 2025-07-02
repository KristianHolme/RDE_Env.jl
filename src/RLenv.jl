"""
    RDEEnv{T<:AbstractFloat} <: AbstractRDEEnv

Reinforcement learning environment for the RDE system.

# Fields
- `prob::RDEProblem{T, M, R, C}`: Underlying RDE problem
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

function RDEEnv(;
    dt=1.0f0,
    smax=4.0f0,
    u_pmax=1.2f0,
    params::RDEParam{T}=RDEParam(),
    momentum=0.0f0,
    τ_smooth=0.1f0,
    action_type::A=ScalarPressureAction(),
    observation_strategy::O=SectionedStateObservation(),
    reward_type::R=PeriodMinimumReward(),
    verbose::Bool=false,
    kwargs...) where {T<:AbstractFloat,A<:AbstractActionType,O<:AbstractObservationStrategy,R<:AbstractRDEReward}

    if τ_smooth > dt
        @warn "τ_smooth > dt, this will cause discontinuities in the control signal"
        @info "Setting τ_smooth = $(dt/8)"
        τ_smooth = dt / 8
    end

    prob = RDEProblem(params; kwargs...)
    RDE.reset_state_and_pressure!(prob, prob.reset_strategy)
    reset_cache!(prob.method.cache, τ_smooth=τ_smooth, params=params)

    # Set N in action_type
    set_N!(action_type, params.N)

    initial_state = vcat(prob.u0, prob.λ0)
    init_observation = get_init_observation(observation_strategy, params.N)
    cache = RDEEnvCache{T}(params.N)
    ode_problem = ODEProblem{true,SciMLBase.FullSpecialize}(RDE_RHS!, initial_state, (zero(T), dt), prob)

    # Use helper functions to determine type parameters
    V = reward_value_type(T, reward_type)
    OBS = observation_array_type(T, observation_strategy)

    # Initialize reward with correct type
    initial_reward = if V <: Vector
        # For multi-section rewards, determine the number of sections
        n_sections = if hasfield(typeof(reward_type), :n_sections)
            reward_type.n_sections
        else
            1  # fallback
        end
        zeros(T, n_sections)
    else
        zero(T)
    end

    return RDEEnv{T,A,O,R,V,OBS}(prob, initial_state, init_observation,
        dt, T(0.0), false, false, false, initial_reward, smax, u_pmax,
        momentum, τ_smooth, cache,
        action_type, observation_strategy,
        reward_type, verbose, Dict{String,Any}(), 0, ode_problem)
end

RDEEnv(params::RDEParam{T}; kwargs...) where {T<:AbstractFloat} = RDEEnv(; params=params, kwargs...)

_observe(env::RDEEnv) = copy(env.observation)

"""
    _act!(env::RDEEnv{T}, action; saves_per_action::Int=0) where {T<:AbstractFloat}

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
function _act!(env::RDEEnv{T,A,O,R,V,OBS}, action; saves_per_action::Int=10) where {T,A,O,R,V,OBS}
    t_start = time()

    if env.t > env.prob.params.tmax
        @warn "t > tmax! ($(env.t) > $(env.prob.params.tmax))"
    end
    # Store current state before taking action
    @logmsg LogLevel(-10000) "Starting act! for environment on thread $(Threads.threadid())"
    N = env.prob.params.N
    env.cache.prev_u .= @view env.state[1:N]
    env.cache.prev_λ .= @view env.state[N+1:end]
    @logmsg LogLevel(-10000) "Stored previous state" prev_u = env.cache.prev_u prev_λ = env.cache.prev_λ

    t_span = (env.t, env.t + env.dt)
    @logmsg LogLevel(-10000) "Set time span" t_span = t_span
    env.prob.method.cache.control_time = env.t
    @logmsg LogLevel(-10000) "Set control time" control_time = env.prob.method.cache.control_time

    prev_controls = [env.prob.method.cache.s_current, env.prob.method.cache.u_p_current]
    c = [env.prob.method.cache.s_current, env.prob.method.cache.u_p_current]
    c_max = [env.smax, env.u_pmax]
    @logmsg LogLevel(-10000) "Initial controls" prev_controls = prev_controls c_max = c_max

    normalized_standard_actions = get_standardized_actions(env.action_type, action)
    env.cache.action[:, 1] = normalized_standard_actions[1]
    env.cache.action[:, 2] = normalized_standard_actions[2]
    @logmsg LogLevel(-10000) "Normalized actions" actions = normalized_standard_actions

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

    #TODO use remake instead of recreating ??
    # prob_ode = ODEProblem{true, SciMLBase.FullSpecialize}(RDE_RHS!, env.state, t_span, env.prob)
    env.ode_problem = remake(env.ode_problem, u0=env.state, tspan=t_span)

    if saves_per_action == 0
        sol = OrdinaryDiffEq.solve(env.ode_problem, Tsit5(), save_on=false,
            isoutofdomain=RDE.outofdomain, verbose=env.verbose)
    else
        saveat = env.dt / saves_per_action
        sol = OrdinaryDiffEq.solve(env.ode_problem, Tsit5(), saveat=saveat,
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
        set_termination_reward!(env, -100.0)
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
        env.observation .= compute_observation(env, env.observation_strategy)::OBS
        if env.terminated #maybe reward caused termination
            # set_termination_reward!(env, -2.0)
            env.done = true
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
    t_end = time()
    t_elapsed = t_end - t_start
    @debug "act! took $(round(t_elapsed*1000, digits=3)) ms"
    return env.reward
end

"""
    _reset!(env::RDEEnv)

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
function _reset!(env::RDEEnv{T,A,O,R,V,OBS}) where {T,A,O,R,V,OBS}
    env.t = 0
    RDE.reset_state_and_pressure!(env.prob, env.prob.reset_strategy)
    env.state = vcat(env.prob.u0, env.prob.λ0)
    set_termination_reward!(env, 0.0)
    env.steps_taken = 0
    env.done = false
    env.truncated = false
    env.terminated = false
    reset_reward!(env.reward_type)  # Reset reward state (e.g., exponential averages)
    set_reward!(env, env.reward_type)
    env.observation .= compute_observation(env, env.observation_strategy)::OBS
    env.info = Dict{String,Any}()

    reset_cache!(env.prob.method.cache, τ_smooth=env.τ_smooth, params=env.prob.params)
    env.prob.sol = nothing
    # Initialize previous state
    N = env.prob.params.N
    env.cache.prev_u .= @view env.state[1:N]
    env.cache.prev_λ .= @view env.state[N+1:end]

    nothing
end

function set_termination_reward!(env::RDEEnv{T,A,O,R,V,OBS}, value::Number) where {T,A,O,R,V<:Vector,OBS}
    fill!(env.reward, T(value))
end
function set_termination_reward!(env::RDEEnv{T,A,O,R,V,OBS}, value::Number) where {T,A,O,R,V<:Number,OBS}
    env.reward = T(value)
end