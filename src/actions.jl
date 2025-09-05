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

"""
    get_standardized_actions(action_type::AbstractActionType, action) -> Vector{Vector{T}}

Convert normalized actions to standard form [area_actions pressure_actions].

# Arguments
- `action_type::AbstractActionType`: Type of action
- `action`: Raw action values

# Returns
- `Vector{Vector{T}}`: Standard form [area_actions, pressure_actions]

# Throws
- `AssertionError`: If action_type.N is not set or action dimensions don't match

# Note
This is a fallback method that issues a warning. Each action type should implement
its own specific version.
"""
function get_standardized_actions(action_type::AbstractActionType, action)
    @assert action_type.N > 0 "Action type N not set"
    return @warn "get_standardized_actions is not implemented for action_type $action_type"
end


function get_standardized_actions(action_type::ScalarAreaScalarPressureAction, action::Vector)
    @assert action_type.N > 0 "Action type N not set"
    @assert length(action) == 2
    return [fill(action[1], action_type.N), fill(action[2], action_type.N)]
end


function get_standardized_actions(action_type::ScalarPressureAction, action::AbstractArray)
    return get_standardized_actions(action_type, action[1])
end

function get_standardized_actions(action_type::ScalarPressureAction, action::Number)
    @assert action_type.N > 0 "Action type N not set"
    return [zeros(action_type.N), ones(action_type.N) .* action]
end


function get_standardized_actions(action_type::VectorPressureAction, action::Vector)
    @assert action_type.N > 0 "Action type N not set"
    @assert length(action) == action_type.n_sections "Action length ($(length(action))) must match n_sections ($(action_type.n_sections))"
    @assert action_type.N % action_type.n_sections == 0 "N ($N) must be divisible by n_sections ($(action_type.n_sections))"

    # Calculate how many points per section
    points_per_section = action_type.N ÷ action_type.n_sections

    # Initialize pressure actions array
    pressure_actions = zeros(action_type.N)

    # Fill each section with its corresponding action value
    for i in 1:action_type.n_sections
        start_idx = (i - 1) * points_per_section + 1
        end_idx = i * points_per_section
        pressure_actions[start_idx:end_idx] .= action[i]
    end
    return [zeros(action_type.N), pressure_actions]
end

# Default compute_standard_actions: defer to get_standardized_actions
compute_standard_actions(action_type::AbstractActionType, action, env::RDEEnv) = get_standardized_actions(action_type, action)

# PIDAction: the agent supplies [Kp, Ki, Kd] in [-1, 1]; we compute a scalar pressure action in [-1, 1]
function compute_standard_actions(action_type::PIDAction, gains::AbstractVector, env::RDEEnv)
    @assert length(gains) == 3 "PIDAction expects 3 coefficients [Kp, Ki, Kd]"
    @assert action_type.N > 0 "Action type N not set"
    Kp, Ki, Kd = gains
    # Use current observation for u_p feedback; assume first element encodes normalized mean injection pressure in [0, 1]
    # If observation strategy differs, users should adapt this mapping.
    # obs = env.observation
    u_p = mean(env.prob.method.cache.u_p_current)
    error = action_type.target - u_p
    action_type.integral += error * env.dt
    derivative = (error - action_type.previous_error) / env.dt
    u_p_action = Kp * error + Ki * action_type.integral + Kd * derivative
    action_type.previous_error = error
    # Clamp to normalized action range
    u_p_action = clamp(u_p_action, -1.0f0, 1.0f0)
    return [zeros(action_type.N), fill(u_p_action, action_type.N)]
end

# For PIDAction, prevent accidental use of get_standardized_actions without env context
function get_standardized_actions(action_type::PIDAction, action)
    throw(ArgumentError("PIDAction requires environment context; use compute_standard_actions(action_type, gains, env) via _act!"))
end

# Reset hook: default no-op and PID-specific state reset
_reset_action!(::AbstractActionType, ::RDEEnv) = nothing
function _reset_action!(action_type::PIDAction, ::RDEEnv)
    action_type.integral = 0.0f0
    action_type.previous_error = 0.0f0
    return nothing
end
