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
        dt = 1.0f0,
        smax = 4.0f0,
        u_pmax = 1.2f0,
        params::RDEParam{T} = RDEParam(),
        momentum = 0.0f0,
        τ_smooth = 0.1f0,
        action_type::A = ScalarPressureAction(),
        observation_strategy::O = SectionedStateObservation(),
        reward_type::RW = PeriodMinimumReward(),
        verbose::Bool = false,
        kwargs...
    ) where {T <: AbstractFloat, A <: AbstractActionType, O <: AbstractObservationStrategy, RW <: AbstractRDEReward}

    if τ_smooth > dt
        @warn "τ_smooth > dt, this will cause discontinuities in the control signal"
        @info "Setting τ_smooth = $(dt / 8)"
        τ_smooth = dt / 8
    end

    prob = RDEProblem(params; kwargs...)
    RDE.reset_state_and_pressure!(prob, prob.reset_strategy)
    reset_cache!(prob.method.cache, τ_smooth = τ_smooth, params = params)

    # Set N in action_type
    set_N!(action_type, params.N)

    initial_state = vcat(prob.u0, prob.λ0)
    init_observation = get_init_observation(observation_strategy, params.N, T)
    cache = RDEEnvCache{T}(params.N)
    ode_problem = ODEProblem{true, SciMLBase.FullSpecialize}(RDE_RHS!, initial_state, (zero(T), dt), prob)

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


    M = typeof(prob.method)
    RS = typeof(prob.reset_strategy)
    CS = typeof(prob.control_shift_strategy)

    return RDEEnv{T, A, O, RW, V, OBS, M, RS, CS}(
        prob, initial_state, init_observation,
        dt, T(0.0), false, false, false, initial_reward, smax, u_pmax,
        momentum, τ_smooth, cache,
        action_type, observation_strategy,
        reward_type, verbose, Dict{String, Any}(), 0, ode_problem
    )
end

RDEEnv(params::RDEParam{T}; kwargs...) where {T <: AbstractFloat} = RDEEnv(; params = params, kwargs...)

_observe(env::RDEEnv) = copy(env.observation)

# Scalar-action overloads (avoid building action vectors)
@inline function control_target(a::T, c_prev::T, c_max::T) where {T <: AbstractFloat}
    target = ifelse(a < 0, c_prev * (a + one(T)), c_prev + (c_max - c_prev) * a)::T
    return target
end

@inline function action_to_control(a::T, c_prev::T, c_max::T, α::T) where {T <: AbstractFloat}
    target::T = control_target(a, c_prev, c_max)
    return α * c_prev + (one(T) - α) * target
end

@inline get_saveat(env::RDEEnv{T}, saves_per_action::Int) where {T <: AbstractFloat} = saves_per_action == 0 ? nothing : (env.dt / saves_per_action)

function step_env!(env::RDEEnv{T, A, O, R, V, OBS}; saves_per_action::Int = 10) where {T <: AbstractFloat, A, O, R, V, OBS}
    return _step!(env, env.prob.method; saves_per_action = saves_per_action)
end

function _step!(env::RDEEnv{T, A, O, R, V, OBS}, ::RDE.PseudospectralMethod{T}; saves_per_action::Int = 10) where {T <: AbstractFloat, A, O, R, V, OBS}
    if saves_per_action == 0
        return OrdinaryDiffEq.solve(
            env.ode_problem, OrdinaryDiffEq.Tsit5(); save_on = false,
            isoutofdomain = RDE.outofdomain, verbose = env.verbose,
        )
    else
        saveat = get_saveat(env, saves_per_action)
        return OrdinaryDiffEq.solve(
            env.ode_problem, OrdinaryDiffEq.Tsit5(); saveat = saveat,
            isoutofdomain = RDE.outofdomain, verbose = env.verbose,
        )
    end
end

function _step!(env::RDEEnv{T, A, O, R, V, OBS, M, RS, C}, ::RDE.FiniteVolumeMethod{T}; saves_per_action::Int = 10) where {T <: AbstractFloat, A, O, R, V, OBS, M, RS, C}
    # SSPRK33 (fixed-step, no error adaptivity). Use CFL to set proposed dt and exact saveat spacing.
    params = env.prob.params
    ode_problem = env.ode_problem
    prob = env.prob::RDEProblem{T, M, RS, C}
    cache = prob.method.cache::RDE.FVCache{T}
    ode_u0 = ode_problem.u0
    u0_view = @view ode_u0[1:params.N]
    dtmax0::T = RDE.cfl_dtmax(params, u0_view, cache)

    function cfl_affect!(integrator)
        u = @view integrator.u[1:params.N]
        umax = RDE.turbo_maximum_abs(u)
        if !isfinite(umax) || umax <= 0
            SciMLBase.terminate!(integrator, SciMLBase.ReturnCode.Failure)
            return
        end
        dt_cfl::T = RDE.cfl_dtmax(params, u, cache)
        if !isfinite(dt_cfl) || dt_cfl <= 0
            SciMLBase.terminate!(integrator, SciMLBase.ReturnCode.Failure)
            return
        end
        SciMLBase.set_proposed_dt!(integrator, dt_cfl)
        return SciMLBase.u_modified!(integrator, false)
    end
    cfl_condition(u, t, integrator) = true
    cfl_cb = SciMLBase.DiscreteCallback(cfl_condition, cfl_affect!; save_positions = (false, false))

    if saves_per_action == 0
        sol = OrdinaryDiffEq.solve(
            ode_problem, OrdinaryDiffEq.SSPRK33(); adaptive = false, dt = dtmax0,
            save_on = false, isoutofdomain = RDE.outofdomain, callback = cfl_cb,
            verbose = env.verbose,
        )
    else
        t0, t1 = ode_problem.tspan
        save_times = collect(range(t0, t1; length = saves_per_action + 1))
        sol = OrdinaryDiffEq.solve(
            ode_problem, OrdinaryDiffEq.SSPRK33(); dt = dtmax0,
            saveat = save_times, isoutofdomain = RDE.outofdomain, callback = cfl_cb,
            verbose = env.verbose,
        )
    end
    prob.sol = sol
    return sol
end

function _step!(env::RDEEnv{T, A, O, R, V, OBS}, ::RDE.UpwindMethod{T}; saves_per_action::Int = 10) where {T <: AbstractFloat, A, O, R, V, OBS}
    params = env.prob.params
    ode_problem = env.ode_problem
    ode_u0 = ode_problem.u0
    u0_view = @view ode_u0[1:params.N]
    dtmax0::T = RDE.cfl_dtmax(params, u0_view, env.prob.method.cache)

    function cfl_affect!(integrator)
        u = @view integrator.u[1:params.N]
        umax = RDE.turbo_maximum_abs(u)
        if !isfinite(umax) || umax <= 0
            SciMLBase.terminate!(integrator; retcode = SciMLBase.ReturnCode.Failure)
            return
        end
        dt_cfl::T = RDE.cfl_dtmax(params, u, env.prob.method.cache)
        if !isfinite(dt_cfl) || dt_cfl <= 0
            SciMLBase.terminate!(integrator; retcode = SciMLBase.ReturnCode.Failure)
            return
        end
        SciMLBase.set_proposed_dt!(integrator, dt_cfl)
        return SciMLBase.u_modified!(integrator, false)
    end
    cfl_condition(u, t, integrator) = true
    cfl_cb = SciMLBase.DiscreteCallback(cfl_condition, cfl_affect!; save_positions = (false, false))

    if saves_per_action == 0
        return OrdinaryDiffEq.solve(
            ode_problem, OrdinaryDiffEq.SSPRK33(); adaptive = false, dt = dtmax0,
            save_on = false, isoutofdomain = RDE.outofdomain, callback = cfl_cb,
            verbose = env.verbose,
        )
    else
        t0, t1 = ode_problem.tspan
        save_times = collect(range(t0, t1; length = saves_per_action + 1))
        return OrdinaryDiffEq.solve(
            ode_problem, OrdinaryDiffEq.SSPRK33(); adaptive = false, dt = dtmax0,
            saveat = save_times, save_on = false, isoutofdomain = RDE.outofdomain, callback = cfl_cb,
            verbose = env.verbose,
        )
    end
end

function _step!(env::RDEEnv{T, A, O, R, V, OBS}, ::RDE.AbstractMethod; saves_per_action::Int = 10) where {T <: AbstractFloat, A, O, R, V, OBS}
    # Fallback: Tsit5 with/without saveat
    ode_problem = env.ode_problem
    if saves_per_action == 0
        return OrdinaryDiffEq.solve(
            ode_problem, OrdinaryDiffEq.Tsit5(); save_on = false,
            isoutofdomain = RDE.outofdomain, verbose = env.verbose,
        )
    else
        saveat = get_saveat(env, saves_per_action)
        return OrdinaryDiffEq.solve(
            ode_problem, OrdinaryDiffEq.Tsit5(); saveat = saveat,
            isoutofdomain = RDE.outofdomain, verbose = env.verbose,
        )
    end
end

function apply_action!(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, action::AbstractVector{T}) where {T <: AbstractFloat, A <: VectorPressureAction, O, RW, V, OBS, M, RS, C}
    action_type = env.action_type
    a = compute_standard_actions(action_type, action, env)
    env.cache.action[:, 1] = a[1]
    env.cache.action[:, 2] = a[2]

    cache = env.prob.method.cache

    if any(abs.(a[1]) .> one(T)) || any(abs.(a[2]) .> one(T))
        @warn "action out of bounds [-1,1]"
    end

    copyto!(cache.u_p_previous, cache.u_p_current)

    @assert action_type.N > 0 "Action type N not set"
    @assert length(action) == action_type.n_sections "Action length ($(length(action))) must match n_sections ($(action_type.n_sections))"
    @assert action_type.N % action_type.n_sections == 0 "N ($(action_type.N)) must be divisible by n_sections ($(action_type.n_sections))"

    # Calculate how many points per section
    points_per_section = action_type.N ÷ action_type.n_sections

    current_section_controls = @view cache.u_p_current[1:points_per_section:end]
    section_controls = action_to_control.(action, current_section_controls, env.u_pmax, env.α)
    # Initialize pressure actions array
    new_pressures = zeros(T, action_type.N)

    # Fill each section with its corresponding action value
    for i in 1:action_type.n_sections
        start_idx = (i - 1) * points_per_section + 1
        end_idx = i * points_per_section
        new_pressures[start_idx:end_idx] .= section_controls[i]
        cache.action[start_idx:end_idx, 2] .= action[i]
    end

    cache.u_p_current .= new_pressures
    return nothing
end

# Specialization: ScalarPressureAction — scalar action updates only u_p channel
function apply_action!(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, action::T) where {T <: AbstractFloat, A <: ScalarPressureAction, O, RW, V, OBS, M, RS, C}
    cache = env.prob.method.cache
    if abs(action) > one(T)
        @warn "action (u_p) out of bounds [-1,1]"
    end
    # Keep cache.action updated without allocating
    env.cache.action[:, 1] .= zero(T)
    env.cache.action[:, 2] .= action

    copyto!(cache.u_p_previous, cache.u_p_current)
    cache.u_p_current .= action_to_control(action, cache.u_p_current[1], env.u_pmax, env.α)

    copyto!(cache.s_previous, cache.s_current)
    return nothing
end

# Convenience: accept array-like scalar for ScalarPressureAction
function apply_action!(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, action::AbstractArray{T}) where {T <: AbstractFloat, A <: ScalarPressureAction, O, RW, V, OBS, M, RS, C}
    return apply_action!(env, action[1])
end

# Specialization: ScalarAreaScalarPressureAction — two scalar actions (s, u_p)
function apply_action!(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, action::AbstractVector{T}) where {T <: AbstractFloat, A <: ScalarAreaScalarPressureAction, O, RW, V, OBS, M, RS, C}
    @assert length(action) == 2
    a_s, a_up = action[1], action[2]
    cache = env.prob.method.cache

    if abs(a_s) > one(T) || abs(a_up) > one(T)
        @warn "action (s/u_p) out of bounds [-1,1]"
    end

    env.cache.action[:, 1] .= a_s
    env.cache.action[:, 2] .= a_up

    copyto!(cache.s_previous, cache.s_current)
    cache.s_current .= action_to_control(a_s, cache.s_current[1], env.smax, env.α)

    copyto!(cache.u_p_previous, cache.u_p_current)
    cache.u_p_current .= action_to_control(a_up, cache.u_p_current[1], env.u_pmax, env.α)
    return nothing
end

# function apply_action!(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, action::NTuple{2, T}) where {T <: AbstractFloat, A <: ScalarAreaScalarPressureAction, O, RW, V, OBS, M, RS, C}
#     return apply_action!(env, collect(action))
# end

# Specialization: PIDAction — gains => scalar u_p set via PID, s unchanged
function apply_action!(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, gains::AbstractVector{T}) where {T <: AbstractFloat, A <: PIDAction, O, RW, V, OBS, M, RS, C}
    @assert length(gains) == 3 "PIDAction expects [Kp, Ki, Kd]"
    Kp, Ki, Kd = gains

    cache = env.prob.method.cache
    u_p_mean::T = mean(cache.u_p_current)

    # PID terms (keep in T)
    err::T = env.action_type.target - u_p_mean
    env.action_type.integral += err * env.dt
    deriv::T = (err - env.action_type.previous_error) / env.dt
    u_p_action::T = clamp(Kp * err + Ki * env.action_type.integral + Kd * deriv, -one(T), one(T))
    env.action_type.previous_error = err

    # Keep cache.action updated (no allocations)
    env.cache.action[:, 1] .= zero(T)
    env.cache.action[:, 2] .= u_p_action

    # Apply control (only u_p)
    copyto!(cache.u_p_previous, cache.u_p_current)
    cache.u_p_current .= action_to_control(u_p_action, cache.u_p_current[1], env.u_pmax, env.α)

    # s channel unchanged, still keep previous in sync
    copyto!(cache.s_previous, cache.s_current)
    return nothing
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
function _act!(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, action; saves_per_action::Int = 10) where {T <: AbstractFloat, A, O, RW, V, OBS, M, RS, C}
    t_start = time()
    params = env.prob.params::RDEParam{T}
    prob = env.prob::RDEProblem{T}

    tmax = params.tmax
    t = env.t
    dt = env.dt
    method_cache = prob.method.cache
    env_cache = env.cache
    if t > tmax
        @warn "t > tmax! ($(t) > $(tmax))"
    end
    # Store current state before taking action
    @logmsg LogLevel(-10000) "Starting act! for environment on thread $(Threads.threadid())"
    N::Int = params.N
    env_cache.prev_u .= @view env.state[1:N]
    env_cache.prev_λ .= @view env.state[(N + 1):end]
    @logmsg LogLevel(-10000) "Stored previous state" prev_u = env_cache.prev_u prev_λ = env_cache.prev_λ

    t_span = (t, t + dt)::Tuple{T, T}
    @logmsg LogLevel(-10000) "Set time span" t_span = t_span
    method_cache.control_time::T = t
    @logmsg LogLevel(-10000) "Set control time" control_time = method_cache.control_time

    @logmsg LogLevel(-10000) "Initial controls" prev_s = method_cache.s_current prev_up = method_cache.u_p_current

    prev_means = mean.([method_cache.s_current, method_cache.u_p_current])
    apply_action!(env, action)
    curr_means = mean.([method_cache.s_current, method_cache.u_p_current])

    @logmsg LogLevel(-10000) "taking action $action at time $(env.t), controls: $(prev_means) to $(curr_means)"

    # Construct a fully-specialized ODEProblem to keep types concrete
    env.ode_problem = ODEProblem{true, SciMLBase.FullSpecialize}(RDE_RHS!, env.state, t_span, env.prob)
    #env.ode_problem = remake(env.ode_problem; u0 = env.state, tspan = t_span)::SciMLBase.ODEProblem

    sol = step_env!(env; saves_per_action)

    tvec::Vector{T} = sol.t
    sol_u::Vector{Vector{T}} = sol.u
    last_u::Vector{T} = sol_u[end]
    if saves_per_action > 0 && length(tvec) != saves_per_action + 1
        @debug "length(sol.t) ($(length(tvec))) != saves_per_action + 1 ($(saves_per_action + 1)), at tspan=$(t_span)"
    end

    #TODO: factor out this
    #Check termination caused by ODE solver
    if !SciMLBase.successful_retcode(sol) || any(isnan, last_u)
        if any(isnan, last_u)
            @warn "NaN state detected"
        end
        @logmsg LogLevel(-10000) "ODE solver failed, controls: $(prev_means) to $(curr_means)"
        env.terminated = true
        env.done = true
        set_termination_reward!(env, -100.0)
        env.info["Termination.Reason"] = "ODE solver failed"
        env.info["Termination.ReturnCode"] = sol.retcode
        env.info["Termination.env_t"] = env.t
        @logmsg LogLevel(-500) "ODE solver failed, t=$(env.t), terminating"
    else #advance environment
        prob.sol = sol
        env.t = tvec[end]::T
        env.state = last_u

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
    @debug "act! took $(round(t_elapsed * 1000, digits = 3)) ms"
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
function _reset!(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}) where {T, A, O, RW, V, OBS, M, RS, C}
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
    env.info = Dict{String, Any}()

    reset_cache!(env.prob.method.cache, τ_smooth = env.τ_smooth, params = env.prob.params)
    _reset_action!(env.action_type, env)
    env.prob.sol = nothing
    # Initialize previous state
    N = env.prob.params.N
    env.cache.prev_u .= @view env.state[1:N]
    env.cache.prev_λ .= @view env.state[(N + 1):end]

    return nothing
end

function set_termination_reward!(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, value::Number) where {T, A, O, RW, V <: Vector, OBS, M, RS, C}
    return fill!(env.reward, T(value))
end
function set_termination_reward!(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, value::Number) where {T, A, O, RW, V <: Number, OBS, M, RS, C}
    return env.reward = T(value)
end
