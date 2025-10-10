# PIDAction: the agent supplies [Kp, Ki, Kd] in [-1, 1]; we compute a scalar pressure action in [-1, 1]
function compute_standard_actions(action_type::PIDAction{T}, gains::AbstractVector, env::RDEEnv{T}) where {T}
    @assert length(gains) == 3 "PIDAction expects 3 coefficients [Kp, Ki, Kd]"
    @assert action_type.N > 0 "Action type N not set"
    Kp, Ki, Kd = gains
    u_p = mean(env.prob.method.cache.u_p_current)
    error = action_type.target - u_p
    action_type.integral += error * env.dt
    derivative = (error - action_type.previous_error) / env.dt
    u_p_action = Kp * error + Ki * action_type.integral + Kd * derivative
    action_type.previous_error = error
    u_p_action = clamp(u_p_action, T(-1), T(1))
    return [zeros(T, action_type.N), fill(T(u_p_action), action_type.N)]
end

# For PIDAction, prevent accidental use of get_standardized_actions without env context
function get_standardized_actions(action_type::PIDAction, action)
    throw(ArgumentError("PIDAction requires environment context; use compute_standard_actions(action_type, gains, env) via _act!"))
end

function _reset_action!(action_type::PIDAction{T}, ::RDEEnv{T}) where {T}
    action_type.integral = zero(T)
    action_type.previous_error = zero(T)
    return nothing
end
