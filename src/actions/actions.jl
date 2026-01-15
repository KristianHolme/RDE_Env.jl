@kwdef mutable struct ScalarPressureAction{T <: AbstractFloat} <: AbstractScalarActionStrategy
    momentum::T = 0.0f0
end


@kwdef mutable struct ScalarAreaScalarPressureAction{T <: AbstractFloat} <: AbstractVectorActionStrategy
    momentum::T = 0.0f0
end


@kwdef mutable struct VectorPressureAction{T <: AbstractFloat} <: AbstractVectorActionStrategy
    n_sections::Int = 1 #number of sections
    momentum::T = 0.0f0
end
struct VectorActionCache{T <: AbstractFloat} <: AbstractCache
    section_controls::Vector{T}
end

mutable struct PIDActionCache{T <: AbstractFloat} <: AbstractCache
    integral::T
    previous_error::T
end

function reset_cache!(cache::PIDActionCache{T}) where {T}
    cache.integral = zero(T)
    cache.previous_error = zero(T)
    return nothing
end

"""
    LinearScalarPressureAction <: AbstractActionStrategy

Action type where a single scalar action in `[-1, 1]` directly maps linearly  
to pressure control in `[0, u_pmax]` via: `control = 0.5 * u_pmax * (action + 1)`.
Unlike `ScalarPressureAction`, the mapping is constant and independent of 
the previous control state.
"""
@kwdef struct LinearScalarPressureAction{T <: AbstractFloat} <: AbstractScalarActionStrategy
    momentum::T = 0.0f0
end

"""
    LinearVectorPressureAction <: AbstractActionStrategy

Vector action type where each section's action in `[-1, 1]` directly maps linearly  
to pressure control in `[0, u_pmax]` via: `control = 0.5 * u_pmax * (action + 1)`.
Unlike `VectorPressureAction`, the mapping is constant and independent of 
the previous control state.

# Fields
- `n_sections::Int`: Number of sections to control independently (default: 1)
"""
@kwdef struct LinearVectorPressureAction{T <: AbstractFloat} <: AbstractVectorActionStrategy
    n_sections::Int = 1
    momentum::T = 0.0f0
end

@kwdef struct DirectScalarPressureAction{T <: AbstractFloat} <: AbstractScalarActionStrategy
    momentum::T = 0.0f0
end

@kwdef struct DirectVectorPressureAction{T <: AbstractFloat} <: AbstractVectorActionStrategy
    n_sections::Int = 1
    momentum::T = 0.0f0
end

"""
    LinearScalarPressureWithDtAction{T} <: AbstractVectorActionStrategy

Action type that combines linear pressure control with timestep (dt) control.
Two action elements: [pressure_action, dt_action] ∈ [-1,1]²

- pressure_action: Maps linearly to pressure control in [0, u_pmax]
- dt_action: Maps linearly to dt in [dt_min, dt_max]

# Fields
- `dt_min::T`: Minimum allowed timestep (default: 0.1)
- `dt_max::T`: Maximum allowed timestep (default: 20.0)
- `momentum::T`: Optional momentum parameter for pressure control (default: 0.0)
"""
@kwdef struct LinearScalarPressureWithDtAction{T <: AbstractFloat} <: AbstractVectorActionStrategy
    dt_min::T = 0.1f0
    dt_max::T = 20.0f0
    momentum::T = 0.0f0
end

"""
    PIDAction <: AbstractScalarActionStrategy

Action type where the agent supplies PID gains `[Kp, Ki, Kd]` and the
environment produces a scalar pressure action in `[-1, 1]` using a
built-in PID computation against a target value for the injection pressure.

State for PID accumulators (integral, previous_error) is stored in PIDActionCache.
Target value is derived from RDE.SHOCK_PRESSURES[target_shock_count] where 
target_shock_count comes from env.cache.goal.

# Fields
- `momentum::T`: Optional momentum parameter (default: 0.0), can be used for smoothing
"""
@kwdef struct PIDAction{T <: AbstractFloat} <: AbstractVectorActionStrategy
    momentum::T = 0.0f0
end
# Momentum API
momentum(at::AbstractActionStrategy) = getfield(at, :momentum)

function action_dim(at::AbstractScalarActionStrategy)
    return 1
end

function action_dim(at::ScalarAreaScalarPressureAction)
    return 2
end

function action_dim(at::VectorPressureAction)
    return at.n_sections
end

function action_dim(::PIDAction)
    return 3
end

function action_dim(at::LinearVectorPressureAction)
    return at.n_sections
end

function action_dim(at::DirectVectorPressureAction)
    return at.n_sections
end

function action_dim(::LinearScalarPressureWithDtAction)
    return 2
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
@inline function action_to_control(a::T, c_prev::T, c_max::T, momentum::T) where {T <: AbstractFloat}
    target::T = control_target(a, c_prev, c_max)
    return momentum * c_prev + (one(T) - momentum) * target
end

function linear_action_to_control(a::T, c_prev::T, c_max::T, momentum::T) where {T <: AbstractFloat}
    control_target = linear_control_target(a, c_max)
    return momentum_target(control_target, c_prev, momentum)
end

function apply_action!(env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C}, action::AbstractVector{T}) where {T <: AbstractFloat, A <: VectorPressureAction, O, RW, G, V, OBS, M, RS, C}
    action_strat = env.action_strat
    N = env.prob.params.N
    @assert N > 0 "Action type N not set"
    @assert length(action) == action_strat.n_sections "Action length ($(length(action))) must match n_sections ($(action_strat.n_sections))"
    @assert N % action_strat.n_sections == 0 "N ($(N)) must be divisible by n_sections ($(action_strat.n_sections))"

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
    points_per_section = N ÷ action_strat.n_sections

    current_section_controls = @view method_cache.u_p_current[1:points_per_section:end]
    # Use action-cache buffer to avoid per-step allocation
    @assert env.cache.action_cache isa VectorActionCache{T}
    section_controls = env.cache.action_cache.section_controls
    section_controls .= action_to_control.(action, current_section_controls, env.u_pmax, momentum(env.action_strat))
    # Fill each section with its corresponding action value, directly writing into method_cache.u_p_current (no new allocation)
    for i in 1:action_strat.n_sections
        start_idx = (i - 1) * points_per_section + 1
        end_idx = i * points_per_section
        method_cache.u_p_current[start_idx:end_idx] .= section_controls[i]
    end
    return nothing
end

# Specialization: ScalarPressureAction — scalar action updates only u_p channel
function apply_action!(env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C}, action::T) where {T <: AbstractFloat, A <: ScalarPressureAction, O, RW, G, V, OBS, M, RS, C}
    env_cache = env.cache
    method_cache = env.prob.method.cache
    if abs(action) > one(T)
        @warn "action (u_p) out of bounds [-1,1]"
    end

    copyto!(method_cache.u_p_previous, method_cache.u_p_current)
    method_cache.u_p_current .= action_to_control(action, method_cache.u_p_current[1], env.u_pmax, momentum(env.action_strat))

    copyto!(method_cache.s_previous, method_cache.s_current)
    return nothing
end

# Convenience: accept array-like scalar for ScalarPressureAction
function apply_action!(env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C}, action::AbstractArray{T}) where {T <: AbstractFloat, A <: ScalarPressureAction, O, RW, G, V, OBS, M, RS, C}
    return apply_action!(env, action[1])
end

# Specialization: ScalarAreaScalarPressureAction — two scalar actions (s, u_p)
function apply_action!(env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C}, action::AbstractVector{T}) where {T <: AbstractFloat, A <: ScalarAreaScalarPressureAction, O, RW, G, V, OBS, M, RS, C}
    @assert length(action) == 2
    a_s, a_up = action[1], action[2]
    method_cache = env.prob.method.cache
    env_cache = env.cache

    if abs(a_s) > one(T) || abs(a_up) > one(T)
        @warn "action (s/u_p) out of bounds [-1,1]"
    end

    copyto!(method_cache.s_previous, method_cache.s_current)
    method_cache.s_current .= action_to_control(a_s, method_cache.s_current[1], env.smax, momentum(env.action_strat))

    copyto!(method_cache.u_p_previous, method_cache.u_p_current)
    method_cache.u_p_current .= action_to_control(a_up, method_cache.u_p_current[1], env.u_pmax, momentum(env.action_strat))
    return nothing
end

# function apply_action!(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, action::NTuple{2, T}) where {T <: AbstractFloat, A <: ScalarAreaScalarPressureAction, O, RW, V, OBS, M, RS, C}
#     return apply_action!(env, collect(action))
# end

# Specialization: PIDAction — gains => scalar u_p set via PID, s unchanged
function apply_action!(env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C}, gains::AbstractVector{T}) where {T <: AbstractFloat, A <: PIDAction, O, RW, G, V, OBS, M, RS, C}
    @assert length(gains) == 3 "PIDAction expects [Kp, Ki, Kd]"
    Kp, Ki, Kd = gains

    method_cache = env.prob.method.cache
    env_cache = env.cache
    u_p_mean::T = mean(method_cache.u_p_current)

    # Get cache and target from goal
    @assert env_cache.action_cache isa PIDActionCache{T}
    cache = env_cache.action_cache::PIDActionCache{T}
    target::T = RDE.SHOCK_PRESSURES[get_target_shock_count(env)]

    # PID terms (keep in T)
    err::T = target - u_p_mean
    cache.integral += err * env.dt
    deriv::T = (err - cache.previous_error) / env.dt
    u_p_action::T = clamp(Kp * err + Ki * cache.integral + Kd * deriv, -one(T), one(T))
    cache.previous_error = err

    # Apply control (only u_p)
    copyto!(method_cache.u_p_previous, method_cache.u_p_current)
    method_cache.u_p_current .= action_to_control(u_p_action, method_cache.u_p_current[1], env.u_pmax, momentum(env.action_strat))

    # s channel unchanged, still keep previous in sync
    copyto!(method_cache.s_previous, method_cache.s_current)
    return nothing
end

function apply_action!(env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C}, action::Vector{T}) where {T <: AbstractFloat, A <: LinearScalarPressureAction, O, RW, G, V, OBS, M, RS, C}
    @assert length(action) == 1 "LinearScalarPressureAction expects a single action"
    apply_action!(env, action[1])
    return nothing
end

function apply_action!(env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C}, action::T) where {T <: AbstractFloat, A <: LinearScalarPressureAction, O, RW, G, V, OBS, M, RS, C}
    env_cache = env.cache
    method_cache = env.prob.method.cache
    if abs(action) > one(T)
        @warn "action (u_p) out of bounds [-1,1]"
    end

    copyto!(method_cache.u_p_previous, method_cache.u_p_current)
    method_cache.u_p_current .= linear_action_to_control(action, method_cache.u_p_current[1], env.u_pmax, momentum(env.action_strat))

    copyto!(method_cache.s_previous, method_cache.s_current)
    return nothing
end

function apply_action!(env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C}, action::AbstractVector{T}) where {T <: AbstractFloat, A <: LinearVectorPressureAction, O, RW, G, V, OBS, M, RS, C}
    action_strat = env.action_strat
    N = env.prob.params.N
    @assert N > 0 "Action type N not set"
    @assert length(action) == action_strat.n_sections "Action length ($(length(action))) must match n_sections ($(action_strat.n_sections))"
    @assert N % action_strat.n_sections == 0 "N ($(N)) must be divisible by n_sections ($(action_strat.n_sections))"

    method_cache = env.prob.method.cache
    env_cache = env.cache

    if any(abs.(action) .> one(T))
        @warn "action out of bounds [-1,1]"
    end

    copyto!(method_cache.u_p_previous, method_cache.u_p_current)


    # Calculate how many points per section
    points_per_section = N ÷ action_strat.n_sections

    current_section_controls = @view method_cache.u_p_current[1:points_per_section:end]
    @assert env_cache.action_cache isa VectorActionCache{T}
    section_controls = (env_cache.action_cache::VectorActionCache{T}).section_controls
    section_controls .= linear_action_to_control.(action, current_section_controls, env.u_pmax, momentum(env.action_strat))
    # Fill each section with its corresponding action value, directly writing into method_cache.u_p_current (no new allocation)
    for i in 1:action_strat.n_sections
        start_idx = (i - 1) * points_per_section + 1
        end_idx = i * points_per_section
        method_cache.u_p_current[start_idx:end_idx] .= section_controls[i]
    end
    return nothing
end

function apply_action!(env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C}, action::Vector{T}) where {T <: AbstractFloat, A <: DirectScalarPressureAction, O, RW, G, V, OBS, M, RS, C}
    @assert length(action) == 1 "DirectScalarPressureAction expects a single action"
    apply_action!(env, action[1])
    return nothing
end

function apply_action!(env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C}, action::T) where {T <: AbstractFloat, A <: DirectScalarPressureAction, O, RW, G, V, OBS, M, RS, C}
    method_cache = env.prob.method.cache

    copyto!(method_cache.u_p_previous, method_cache.u_p_current)

    if action < zero(T) || action > env.u_pmax
        @warn "direct action (u_p) out of bounds [0, u_pmax]"
    end
    clamped_action = clamp(action, zero(T), env.u_pmax)
    method_cache.u_p_current .= momentum_target(clamped_action, method_cache.u_p_current[1], momentum(env.action_strat))
    copyto!(method_cache.s_previous, method_cache.s_current)
    return nothing
end

function apply_action!(env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C}, action::AbstractVector{T}) where {T <: AbstractFloat, A <: LinearScalarPressureWithDtAction, O, RW, G, V, OBS, M, RS, C}
    @assert length(action) == 2 "LinearScalarPressureWithDtAction expects two actions: [pressure_action, dt_action]"
    a_pressure, a_dt = action[1], action[2]

    env_cache = env.cache
    method_cache = env.prob.method.cache

    if abs(a_pressure) > one(T)
        @warn "pressure action out of bounds [-1,1]"
    end
    if abs(a_dt) > one(T)
        @warn "dt action out of bounds [-1,1]"
    end

    # Apply pressure control (same as LinearScalarPressureAction)
    copyto!(method_cache.u_p_previous, method_cache.u_p_current)
    method_cache.u_p_current .= linear_action_to_control(a_pressure, method_cache.u_p_current[1], env.u_pmax, momentum(env.action_strat))

    # Apply dt control: map [-1,1] to [dt_min, dt_max]
    dt_range = env.action_strat.dt_max - env.action_strat.dt_min
    env.dt = env.action_strat.dt_min + T(0.5) * dt_range * (a_dt + one(T))

    copyto!(method_cache.s_previous, method_cache.s_current)
    return nothing
end

function apply_action!(env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C}, action::AbstractVector{T}) where {T <: AbstractFloat, A <: DirectVectorPressureAction, O, RW, G, V, OBS, M, RS, C}
    action_strat = env.action_strat
    N = env.prob.params.N
    @assert N > 0 "Action type N not set"
    @assert length(action) == action_strat.n_sections "Action length ($(length(action))) must match n_sections ($(action_strat.n_sections))"
    @assert N % action_strat.n_sections == 0 "N ($(N)) must be divisible by n_sections ($(action_strat.n_sections))"

    method_cache = env.prob.method.cache
    env_cache = env.cache

    if any(action .< zero(T)) || any(action .> env.u_pmax)
        @warn "direct action out of bounds [0, u_pmax]"
    end

    copyto!(method_cache.u_p_previous, method_cache.u_p_current)


    # Calculate how many points per section
    points_per_section = N ÷ action_strat.n_sections

    current_section_controls = @view method_cache.u_p_current[1:points_per_section:end]
    @assert env_cache.action_cache isa VectorActionCache{T}
    section_controls = env_cache.action_cache.section_controls
    clamped_action = clamp.(action, zero(T), env.u_pmax)
    section_controls .= momentum_target.(clamped_action, current_section_controls, momentum(env.action_strat))
    # Fill each section with its corresponding action value, directly writing into method_cache.u_p_current (no new allocation)
    for i in 1:action_strat.n_sections
        start_idx = (i - 1) * points_per_section + 1
        end_idx = i * points_per_section
        method_cache.u_p_current[start_idx:end_idx] .= section_controls[i]
    end
    return nothing
end

# Action cache initializers for vector action types
initialize_cache(at::VectorPressureAction{T}, N::Int, ::Type{T}) where {T <: AbstractFloat} =
    VectorActionCache{T}(Vector{T}(undef, at.n_sections))

initialize_cache(at::LinearVectorPressureAction{T}, N::Int, ::Type{T}) where {T <: AbstractFloat} =
    VectorActionCache{T}(Vector{T}(undef, at.n_sections))

initialize_cache(at::DirectVectorPressureAction{T}, N::Int, ::Type{T}) where {T <: AbstractFloat} =
    VectorActionCache{T}(Vector{T}(undef, at.n_sections))

initialize_cache(::PIDAction{T}, N::Int, ::Type{T}) where {T <: AbstractFloat} =
    PIDActionCache{T}(zero(T), zero(T))
