@kwdef mutable struct MovingFrameControlShift{T <: AbstractFloat} <: RDE.AbstractControlShift
    position::T = 0.0f0 #postition at t_last
    velocity::T = 1.8f0 # current average wave speed for the step ending at t_last
    t_last::T = 0.0f0
end

function get_control_shift(control_shift_strategy::MovingFrameControlShift, u::AbstractVector, t::Real)
    t_last = control_shift_strategy.t_last
    position = control_shift_strategy.position
    velocity = control_shift_strategy.velocity
    return position + (t - t_last) * velocity
end
