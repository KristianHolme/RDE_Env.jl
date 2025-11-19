@kwdef mutable struct MovingFrameControlShift{T <: AbstractFloat} <: RDE.AbstractControlShift
    position::T = 0.0f0 #postition at t_last
    velocity::T = 0.0f0 # current average wave speed for the step ending at t_last
    t_last::T = 0.0f0
end

function RDE.get_control_shift(control_shift_strategy::MovingFrameControlShift, ::AbstractVector, t::Real)
    @assert isfinite(t) "t is not finite"
    @assert isfinite(control_shift_strategy.t_last) "control_shift_strategy.t_last is not finite"
    @assert isfinite(control_shift_strategy.position) "control_shift_strategy.position is not finite"
    @assert isfinite(control_shift_strategy.velocity) "control_shift_strategy.velocity is not finite"
    t_last = control_shift_strategy.t_last
    position = control_shift_strategy.position
    velocity = control_shift_strategy.velocity
    shift_position = position + (t - t_last) * velocity
    return shift_position
end
