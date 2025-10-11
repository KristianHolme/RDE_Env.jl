@kwdef mutable struct ScalarPressureAction <: AbstractActionType
    N::Int = 512 #number of grid points
end


@kwdef mutable struct ScalarAreaScalarPressureAction <: AbstractActionType
    N::Int = 512 #number of grid points
end


@kwdef mutable struct VectorPressureAction <: AbstractActionType
    n_sections::Int = 1 #number of sections
    N::Int = 512 #number of grid points
end

"""
    PIDAction <: AbstractActionType

Action type where the agent supplies PID gains `[Kp, Ki, Kd]` and the
environment produces a scalar pressure action in `[-1, 1]` using a
built-in PID computation against a target value for the injection pressure.

State for the PID accumulators is stored on the action type and is reset
on environment reset via `_reset_action!`.
"""
@kwdef mutable struct PIDAction{T <: AbstractFloat} <: AbstractActionType
    N::Int = 512
    target::T = zero(T)
    integral::T = zero(T)
    previous_error::T = zero(T)
end

function action_dim(at::ScalarPressureAction)
    return 1
end

function action_dim(at::ScalarAreaScalarPressureAction)
    return 2
end

function action_dim(at::VectorPressureAction)
    return at.n_sections
end

function action_dim(at::PIDAction)
    return 3
end

function set_N!(action_type::AbstractActionType, N::Int)
    if action_type isa VectorPressureAction
        @assert N % action_type.n_sections == 0 "N ($N) must be divisible by n_sections ($(action_type.n_sections))"
    end
    setfield!(action_type, :N, N)
    return action_type
end

# Reset hook: default no-op and PID-specific state reset
_reset_action!(::AbstractActionType, ::RDEEnv) = nothing
function _reset_action!(action_type::PIDAction{T}, ::RDEEnv{T}) where {T}
    action_type.integral = zero(T)
    action_type.previous_error = zero(T)
    return nothing
end
function apply_action!(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, action::AbstractVector{T}) where {T <: AbstractFloat, A <: VectorPressureAction, O, RW, V, OBS, M, RS, C}
    action_type = env.action_type
    N = env.prob.params.N
    @assert N > 0 "Action type N not set"
    @assert length(action) == action_type.n_sections "Action length ($(length(action))) must match n_sections ($(action_type.n_sections))"
    @assert N % action_type.n_sections == 0 "N ($(N)) must be divisible by n_sections ($(action_type.n_sections))"

    # full_domain_action_vector = fill_standardized_vector_actions!(env.cache.actions[:, 2], env, action)
    # env.cache.action[:, 1] = a[1]
    # env.cache.action[:, 2] = full_action_vector #done in place above

    method_cache = env.prob.method.cache
    env_cache = env.cache

    if any(abs.(action) .> one(T))
        @warn "action out of bounds [-1,1]"
    end

    copyto!(method_cache.u_p_previous, method_cache.u_p_current)


    # Calculate how many points per section
    points_per_section = N ÷ action_type.n_sections

    current_section_controls = @view method_cache.u_p_current[1:points_per_section:end]
    section_controls = action_to_control.(action, current_section_controls, env.u_pmax, env.α)
    # Fill each section with its corresponding action value, directly writing into method_cache.u_p_current (no new allocation)
    for i in 1:action_type.n_sections
        start_idx = (i - 1) * points_per_section + 1
        end_idx = i * points_per_section
        method_cache.u_p_current[start_idx:end_idx] .= section_controls[i]
        env_cache.action[start_idx:end_idx, 2] .= action[i]
    end
    return nothing
end

# Specialization: ScalarPressureAction — scalar action updates only u_p channel
function apply_action!(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, action::T) where {T <: AbstractFloat, A <: ScalarPressureAction, O, RW, V, OBS, M, RS, C}
    env_cache = env.cache
    method_cache = env.prob.method.cache
    if abs(action) > one(T)
        @warn "action (u_p) out of bounds [-1,1]"
    end
    # Keep cache.action updated without allocating
    env_cache.action[:, 1] .= zero(T)
    env_cache.action[:, 2] .= action

    copyto!(method_cache.u_p_previous, method_cache.u_p_current)
    method_cache.u_p_current .= action_to_control(action, method_cache.u_p_current[1], env.u_pmax, env.α)

    copyto!(method_cache.s_previous, method_cache.s_current)
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
    method_cache = env.prob.method.cache
    env_cache = env.cache

    if abs(a_s) > one(T) || abs(a_up) > one(T)
        @warn "action (s/u_p) out of bounds [-1,1]"
    end

    env_cache.action[:, 1] .= a_s
    env_cache.action[:, 2] .= a_up

    copyto!(method_cache.s_previous, method_cache.s_current)
    method_cache.s_current .= action_to_control(a_s, method_cache.s_current[1], env.smax, env.α)

    copyto!(method_cache.u_p_previous, method_cache.u_p_current)
    method_cache.u_p_current .= action_to_control(a_up, method_cache.u_p_current[1], env.u_pmax, env.α)
    return nothing
end

# function apply_action!(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, action::NTuple{2, T}) where {T <: AbstractFloat, A <: ScalarAreaScalarPressureAction, O, RW, V, OBS, M, RS, C}
#     return apply_action!(env, collect(action))
# end

# Specialization: PIDAction — gains => scalar u_p set via PID, s unchanged
function apply_action!(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, gains::AbstractVector{T}) where {T <: AbstractFloat, A <: PIDAction, O, RW, V, OBS, M, RS, C}
    @assert length(gains) == 3 "PIDAction expects [Kp, Ki, Kd]"
    Kp, Ki, Kd = gains

    method_cache = env.prob.method.cache
    env_cache = env.cache
    u_p_mean::T = mean(cache.u_p_current)

    # PID terms (keep in T)
    err::T = env.action_type.target - u_p_mean
    env.action_type.integral += err * env.dt
    deriv::T = (err - env.action_type.previous_error) / env.dt
    u_p_action::T = clamp(Kp * err + Ki * env.action_type.integral + Kd * deriv, -one(T), one(T))
    env.action_type.previous_error = err

    # Keep cache.action updated (no allocations)
    env_cache.action[:, 1] .= zero(T)
    env_cache.action[:, 2] .= u_p_action

    # Apply control (only u_p)
    copyto!(method_cache.u_p_previous, method_cache.u_p_current)
    method_cache.u_p_current .= action_to_control(u_p_action, method_cache.u_p_current[1], env.u_pmax, env.α)

    # s channel unchanged, still keep previous in sync
    copyto!(method_cache.s_previous, method_cache.s_current)
    return nothing
end
