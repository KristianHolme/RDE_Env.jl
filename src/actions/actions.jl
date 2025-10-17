@kwdef mutable struct ScalarPressureAction{T <: AbstractFloat} <: AbstractActionType
    α::T = 0.0f0
end


@kwdef mutable struct ScalarAreaScalarPressureAction{T <: AbstractFloat} <: AbstractActionType
    α::T = 0.0f0
end


@kwdef mutable struct VectorPressureAction{T <: AbstractFloat} <: AbstractActionType
    n_sections::Int = 1 #number of sections
    α::T = 0.0f0
end

"""
    LinearScalarPressureAction <: AbstractActionType

Action type where a single scalar action in `[-1, 1]` directly maps linearly  
to pressure control in `[0, u_pmax]` via: `control = 0.5 * u_pmax * (action + 1)`.
Unlike `ScalarPressureAction`, the mapping is constant and independent of 
the previous control state.
"""
@kwdef struct LinearScalarPressureAction{T <: AbstractFloat} <: AbstractActionType
    α::T = 0.0f0
end

"""
    LinearVectorPressureAction <: AbstractActionType

Vector action type where each section's action in `[-1, 1]` directly maps linearly  
to pressure control in `[0, u_pmax]` via: `control = 0.5 * u_pmax * (action + 1)`.
Unlike `VectorPressureAction`, the mapping is constant and independent of 
the previous control state.

# Fields
- `n_sections::Int`: Number of sections to control independently (default: 1)
"""
@kwdef struct LinearVectorPressureAction{T <: AbstractFloat} <: AbstractActionType
    n_sections::Int = 1
    α::T = 0.0f0
end

@kwdef struct DirectScalarPressureAction{T <: AbstractFloat} <: AbstractActionType
    α::T = 0.0f0
end

@kwdef struct DirectVectorPressureAction{T <: AbstractFloat} <: AbstractActionType
    n_sections::Int = 1
    α::T = 0.0f0
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
    target::T = zero(T)
    integral::T = zero(T)
    previous_error::T = zero(T)
    α::T = 0.0f0
end
# Momentum API
momentum(at::AbstractActionType) = error("momentum not used by $(typeof(at))")
momentum(at::ScalarPressureAction{T}) where {T} = at.α
momentum(at::ScalarAreaScalarPressureAction{T}) where {T} = at.α
momentum(at::VectorPressureAction{T}) where {T} = at.α
momentum(at::PIDAction{T}) where {T} = at.α
momentum(at::LinearScalarPressureAction{T}) where {T} = at.α
momentum(at::LinearVectorPressureAction{T}) where {T} = at.α
momentum(at::DirectScalarPressureAction{T}) where {T} = at.α
momentum(at::DirectVectorPressureAction{T}) where {T} = at.α

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

function action_dim(::LinearScalarPressureAction)
    return 1
end

function action_dim(at::LinearVectorPressureAction)
    return at.n_sections
end

function action_dim(at::DirectScalarPressureAction)
    return 1
end

function action_dim(at::DirectVectorPressureAction)
    return at.n_sections
end

# Scalar-action overloads (avoid building action vectors)
@inline function control_target(a::T, c_prev::T, c_max::T) where {T <: AbstractFloat}
    target = ifelse(a < 0, c_prev * (a + one(T)), c_prev + (c_max - c_prev) * a)::T
    return target
end

#linear function from [-1,1] to [0, u_p_max]
@inline function linear_control_target(a::T, c_max::T) where {T <: AbstractFloat}
    target = T(0.5) * c_max * (a + T(1))
    return target
end

function momentum_target(control_target::T, previous_target::T, momentum::T) where {T <: AbstractFloat}
    return momentum * previous_target + (one(T) - momentum) * control_target
end

# translate action ∈ [-1,1] to a target injection pressure, using a piecewise linear function, where 0 maps to the current control, -1 to 0 and 1 to the max allowed control
@inline function action_to_control(a::T, c_prev::T, c_max::T, α::T) where {T <: AbstractFloat}
    target::T = control_target(a, c_prev, c_max)
    return α * c_prev + (one(T) - α) * target
end

function linear_action_to_control(a::T, c_prev::T, c_max::T, α::T) where {T <: AbstractFloat}
    control_target = linear_control_target(a, c_max)
    return momentum_target(control_target, c_prev, α)
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
    section_controls = action_to_control.(action, current_section_controls, env.u_pmax, momentum(env.action_type))
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
    method_cache.u_p_current .= action_to_control(action, method_cache.u_p_current[1], env.u_pmax, momentum(env.action_type))

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
    method_cache.s_current .= action_to_control(a_s, method_cache.s_current[1], env.smax, momentum(env.action_type))

    copyto!(method_cache.u_p_previous, method_cache.u_p_current)
    method_cache.u_p_current .= action_to_control(a_up, method_cache.u_p_current[1], env.u_pmax, momentum(env.action_type))
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
    u_p_mean::T = mean(method_cache.u_p_current)

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
    method_cache.u_p_current .= action_to_control(u_p_action, method_cache.u_p_current[1], env.u_pmax, momentum(env.action_type))

    # s channel unchanged, still keep previous in sync
    copyto!(method_cache.s_previous, method_cache.s_current)
    return nothing
end

function apply_action!(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, action::Vector{T}) where {T <: AbstractFloat, A <: LinearScalarPressureAction, O, RW, V, OBS, M, RS, C}
    @assert length(action) == 1 "LinearScalarPressureAction expects a single action"
    apply_action!(env, action[1])
    return nothing
end

function apply_action!(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, action::T) where {T <: AbstractFloat, A <: LinearScalarPressureAction, O, RW, V, OBS, M, RS, C}
    env_cache = env.cache
    method_cache = env.prob.method.cache
    if abs(action) > one(T)
        @warn "action (u_p) out of bounds [-1,1]"
    end
    # Keep cache.action updated without allocating
    env_cache.action[:, 1] .= zero(T)
    env_cache.action[:, 2] .= action

    copyto!(method_cache.u_p_previous, method_cache.u_p_current)
    method_cache.u_p_current .= linear_action_to_control(action, method_cache.u_p_current[1], env.u_pmax, momentum(env.action_type))

    copyto!(method_cache.s_previous, method_cache.s_current)
    return nothing
end

function apply_action!(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, action::AbstractVector{T}) where {T <: AbstractFloat, A <: LinearVectorPressureAction, O, RW, V, OBS, M, RS, C}
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
    section_controls = linear_action_to_control.(action, current_section_controls, env.u_pmax, momentum(env.action_type))
    # Fill each section with its corresponding action value, directly writing into method_cache.u_p_current (no new allocation)
    for i in 1:action_type.n_sections
        start_idx = (i - 1) * points_per_section + 1
        end_idx = i * points_per_section
        method_cache.u_p_current[start_idx:end_idx] .= section_controls[i]
        env_cache.action[start_idx:end_idx, 2] .= action[i]
    end
    return nothing
end

function apply_action!(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, action::Vector{T}) where {T <: AbstractFloat, A <: DirectScalarPressureAction, O, RW, V, OBS, M, RS, C}
    @assert length(action) == 1 "DirectScalarPressureAction expects a single action"
    apply_action!(env, action[1])
    return nothing
end

function apply_action!(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, action::T) where {T <: AbstractFloat, A <: DirectScalarPressureAction, O, RW, V, OBS, M, RS, C}
    env_cache = env.cache
    method_cache = env.prob.method.cache
    # Keep cache.action updated without allocating
    env_cache.action[:, 1] .= zero(T)
    env_cache.action[:, 2] .= action

    copyto!(method_cache.u_p_previous, method_cache.u_p_current)

    method_cache.u_p_current .= momentum_target(action, method_cache.u_p_current[1], momentum(env.action_type))
    copyto!(method_cache.s_previous, method_cache.s_current)
    return nothing
end

function apply_action!(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, action::AbstractVector{T}) where {T <: AbstractFloat, A <: DirectVectorPressureAction, O, RW, V, OBS, M, RS, C}
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

    copyto!(method_cache.u_p_previous, method_cache.u_p_current)


    # Calculate how many points per section
    points_per_section = N ÷ action_type.n_sections

    current_section_controls = @view method_cache.u_p_current[1:points_per_section:end]
    section_controls = momentum_target.(action, current_section_controls, momentum(env.action_type))
    # Fill each section with its corresponding action value, directly writing into method_cache.u_p_current (no new allocation)
    for i in 1:action_type.n_sections
        start_idx = (i - 1) * points_per_section + 1
        end_idx = i * points_per_section
        method_cache.u_p_current[start_idx:end_idx] .= section_controls[i]
        env_cache.action[start_idx:end_idx, 2] .= action[i]
    end
    return nothing
end
