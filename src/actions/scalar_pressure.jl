function get_standardized_actions(action_type::ScalarPressureAction, action::AbstractArray{T})::Vector{Vector{T}} where {T <: AbstractFloat}
    return get_standardized_actions(action_type, action[1])
end

function get_standardized_actions(action_type::ScalarPressureAction, action::T)::Vector{Vector{T}} where {T <: AbstractFloat}
    @assert action_type.N > 0 "Action type N not set"
    return [zeros(T, action_type.N), ones(T, action_type.N) .* action]
end
