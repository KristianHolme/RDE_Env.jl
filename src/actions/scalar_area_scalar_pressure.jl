function get_standardized_actions(action_type::ScalarAreaScalarPressureAction, action::Vector{T})::Vector{Vector{T}} where {T <: AbstractFloat}
    @assert action_type.N > 0 "Action type N not set"
    @assert length(action) == 2
    return [fill(action[1], action_type.N), fill(action[2], action_type.N)]
end
