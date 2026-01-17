import DiffEqCallbacks: StepsizeLimiter
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
- `τ_smooth::T`: Control smoothing time constant
- `cache::RDEEnvCache{T}`: Environment cache
- `action_strat::AbstractActionStrategy`: Type of control actions
- `reward_strat::AbstractRewardStrategy`: Type of reward function
- `verbose::Bool`: Control solver output
# Constructor
```julia
RDEEnv{T}(;
    dt=10.0,
    smax=4.0,
    u_pmax=1.2,
    params::RDEParam{T}=RDEParam{T}(),

    τ_smooth=1.25,
    fft_terms::Int=32,
    observation_strat::AbstractObservationStrategy=FullStateObservation(),
    action_strat::AbstractActionStrategy=DirectScalarPressureAction(),
    reward_strat::AbstractRewardStrategy=USpanReward(),
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
        dt = 1.0f0,
        smax = 4.0f0,
        u_pmax = 1.2f0,
        params::RDEParam{T} = RDEParam(),
        τ_smooth = 0.1f0,
        action_strat::A = DirectScalarPressureAction(),
        observation_strat::O = FullStateObservation(),
        reward_strat::RW = USpanReward(),
        context_strat::CS = NoContextStrategy(),
        verbose::Bool = false,
        kwargs...
    ) where {
        T <: AbstractFloat, A <: AbstractActionStrategy,
        O <: AbstractObservationStrategy, RW <: Union{AbstractScalarRewardStrategy, AbstractVectorRewardStrategy},
        CS <: AbstractContextStrategy,
    }

    if τ_smooth > dt
        @warn "τ_smooth > dt, this will cause discontinuities in the control signal"
        @info "Setting τ_smooth = $(dt / 8)"
        τ_smooth = dt / 8
    end

    # RDE.reset_state_and_pressure!(prob, prob.reset_strategy)
    # RDE.reset_cache!(prob.method.cache; τ_smooth, params)

    # Initialize typed subcaches (no env dependency)
    reward_cache = initialize_cache(reward_strat, params.N, T)
    action_cache = initialize_cache(action_strat, params.N, T)
    observation_cache = initialize_cache(observation_strat, params.N, T)
    context = initialize_cache(context_strat, params.N, T)
    cache = RDEEnvCache{T, typeof(reward_cache), typeof(action_cache), typeof(observation_cache), typeof(context)}(
        params.N; reward_cache, action_cache, observation_cache, context
    )

    #TODO should do this somewhere else
    # if observation_strat isa MultiCenteredMovingFrameObservation || observation_strat isa SectionedStateMovingFrameObservation
    #     control_shift_strategy = MovingFrameControlShift()
    # else
    #     control_shift_strategy = ZeroControlShift()
    # end
    #TODO: remember to supply control shift strategy in kwargs if applicable
    prob = RDEProblem(params; kwargs...)
    initial_state = vcat(prob.u0, prob.λ0)
    ode_problem = ODEProblem{true, SciMLBase.FullSpecialize}(RDE_RHS!, initial_state, (zero(T), dt), prob)

    is_vector_reward = reward_strat isa AbstractVectorRewardStrategy
    V = is_vector_reward ? Vector{T} : T
    obs_space = _observation_space(params, observation_strat)
    init_observation = rand(obs_space)
    OBS = typeof(init_observation)

    # Initialize reward with correct type
    initial_reward = if is_vector_reward
        # For multi-section rewards, determine the number of sections
        n_sections = if hasfield(typeof(reward_strat), :n_sections)
            reward_strat.n_sections
        else
            1  # fallback
        end
        zeros(T, n_sections)
    else
        zero(T)
    end


    M = typeof(prob.method)
    RS = typeof(prob.reset_strategy)
    C = typeof(prob.control_shift_strategy)

    env = RDEEnv{T, A, O, RW, CS, V, OBS, M, RS, C}(
        prob, initial_state, init_observation,
        dt, T(0.0), false, false, false, initial_reward, smax, u_pmax,
        τ_smooth, cache,
        action_strat, observation_strat,
        reward_strat, context_strat, verbose, Dict{String, Any}(), 0, ode_problem
    )
    compute_observation!(env.observation, env, observation_strat, env.cache.observation_cache, env.cache.context)
    RDE.RDE_RHS!(zeros(T, 2 * params.N), initial_state, prob, T(0.0)) #to update caches
    return env
end

RDEEnv(params::RDEParam{T}; kwargs...) where {T <: AbstractFloat} = RDEEnv(; params = params, kwargs...)

_observe(env::RDEEnv) = copy(env.observation)


@inline get_saveat(env::RDEEnv{T}, saves_per_action::Int) where {T <: AbstractFloat} = saves_per_action == 0 ? nothing : (env.dt / saves_per_action)

function solve_step(env::RDEEnv{T, A, O, R, CS, V, OBS}; saves_per_action::Int = 10) where {T <: AbstractFloat, A, O, R, CS, V, OBS}
    t0, t1 = env.ode_problem.tspan::Tuple{T, T}
    saveat = collect(range(t0, t1; length = saves_per_action + 1))
    sol = RDE.solve_pde_step(env.prob, env.ode_problem; saveat = saveat)
    if sol.retcode != :Success
        @warn "Failed to solve PDE step for $(typeof(env.prob.method))"
    end
    return sol
end

function _act_postprocessing!(env::RDEEnv{T}, sol::SciMLBase.ODESolution) where {T <: AbstractFloat}
    tvec = sol.t::Vector{T}
    sol_u = sol.u::Vector{Vector{T}}
    last_u = sol_u[end]

    if !SciMLBase.successful_retcode(sol) || any(isnan, last_u)
        if any(isnan, last_u)
            @warn "NaN state detected"
        end
        env.terminated = true
        env.done = true
        set_termination_reward!(env, -100.0)
        env.info["Termination.Reason"] = "ODE solver failed"
        env.info["Termination.ReturnCode"] = sol.retcode
        env.info["Termination.env_t"] = env.t
        @logmsg LogLevel(-500) "ODE solver failed, t=$(env.t), terminating"
    else #advance environment
        env.prob.sol = sol
        env.t = tvec[end]::T
        env.state .= last_u

        env.steps_taken += 1

        on_step!(env.cache.context, env.context_strat, env)
        set_reward!(env, env.reward_strat)
        compute_observation!(env.observation, env, env.observation_strat, env.cache.observation_cache, env.cache.context)
        if env.terminated #maybe reward caused termination
            # set_termination_reward!(env, -2.0)
            env.done = true
            @logmsg LogLevel(-10000) "termination caused by reward"
            @logmsg LogLevel(-500) "terminated, t=$(env.t), from reward?"
        elseif env.t ≥ env.prob.params.tmax #dont mark as truncated if terminated by reward or solver
            env.done = true
            env.truncated = true
            @logmsg LogLevel(-10000) "tmax reached, t=$(env.t)"
            env.info["Truncation.Reason"] = "tmax reached"
        end
    end

    if env.done != xor(env.truncated, env.terminated)
        @warn "done is not xor(truncated, terminated), at t=$(env.t)" env.done, env.truncated, env.terminated
        @info "info: $(env.info)"
    end
    return env
end

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
function _act!(env::RDEEnv{T, A, O, RW, CS, V, OBS, M, RS, C}, action; saves_per_action::Int = 10) where {T <: AbstractFloat, A, O, RW, CS, V, OBS, M, RS, C}
    params = env.prob.params::RDEParam{T}
    prob = env.prob::RDEProblem{T}
    tmax = params.tmax
    t = env.t
    method_cache = prob.method.cache
    env_cache = env.cache
    if t > tmax
        @warn "t > tmax! ($(t) > $(tmax))"
    end
    # Store current state before taking action
    N::Int = params.N
    env_cache.prev_u .= @view env.state[1:N]
    env_cache.prev_λ .= @view env.state[(N + 1):end]

    method_cache.control_time::T = t

    action_to_apply = action
    if env.action_strat isa AbstractScalarActionStrategy && action isa AbstractVector
        @assert length(action) == 1 "Expected scalar action as length-1 vector"
        action_to_apply = action[1]
    end
    apply_action!(env, action_to_apply, env.action_strat, env.cache.action_cache, env.cache.context)
    dt = env.dt
    t_span = (t, t + dt)::Tuple{T, T}

    # Construct a fully-specialized ODEProblem to keep types concrete
    env.ode_problem = ODEProblem{true, SciMLBase.FullSpecialize}(RDE_RHS!, env.state, t_span, env.prob)
    #env.ode_problem = remake(env.ode_problem; u0 = env.state, tspan = t_span)::SciMLBase.ODEProblem

    #adjust saves_per_action to make sure speed detection is good. 0.1 is safe step to capture shock speeds
    saves_per_action = max(saves_per_action, Int(env.dt ÷ 0.1))
    sol = solve_step(env; saves_per_action)

    if saves_per_action > 0 && length(sol.t) != saves_per_action + 1
        @debug "length(sol.t) ($(length(sol.t))) != saves_per_action + 1 ($(saves_per_action + 1)), at tspan=$(t_span)"
    end

    _act_postprocessing!(env, sol)
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
function _reset!(env::RDEEnv{T, A, O, RW, CS, V, OBS, M, RS, C}) where {T, A, O, RW, CS, V, OBS, M, RS, C}
    N = env.prob.params.N
    env.t = 0 #reset current time
    RDE.reset_state_and_pressure!(env.prob, env.prob.reset_strategy)
    #update state to initial conditions
    env.state[1:N] = env.prob.u0
    env.state[(N + 1):end] = env.prob.λ0
    set_termination_reward!(env, T(0))
    #reset other environment state
    env.steps_taken = 0
    env.done = false
    env.truncated = false
    env.terminated = false

    #TODO: is this necessary?
    set_reward!(env, env.reward_strat)

    # update observation to initial conditions
    compute_observation!(env.observation, env, env.observation_strat, env.cache.observation_cache, env.cache.context)
    env.info = Dict{String, Any}()

    #reset method cache
    RDE.reset_cache!(env.prob.method.cache, τ_smooth = env.τ_smooth, params = env.prob.params)
    #reset caches
    reset_cache!(env.cache)
    on_reset!(env.cache.context, env.context_strat, env)
    env.prob.sol = nothing
    # Initialize previous state
    N = env.prob.params.N
    env.cache.prev_u .= @view env.state[1:N]
    env.cache.prev_λ .= @view env.state[(N + 1):end]

    RDE.RDE_RHS!(zeros(T, 2 * env.prob.params.N), env.state, env.prob, T(0.0)) #to update caches
    #update observation to initial conditions
    compute_observation!(env.observation, env, env.observation_strat, env.cache.observation_cache, env.cache.context)

    return nothing
end

#TODO: purpose? candidate for removal
function set_termination_reward!(env::RDEEnv{T, A, O, RW, CS, V, OBS, M, RS, C}, value::Real) where {T, A, O, RW, CS, V <: Vector, OBS, M, RS, C}
    return fill!(env.reward, T(value))
end
function set_termination_reward!(env::RDEEnv{T, A, O, RW, CS, V, OBS, M, RS, C}, value::Real) where {T, A, O, RW, CS, V <: Real, OBS, M, RS, C}
    return env.reward = T(value)
end
