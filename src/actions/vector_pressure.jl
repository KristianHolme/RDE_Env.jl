function get_standardized_actions(action_type::VectorPressureAction, action::Vector{T})::Vector{Vector{T}} where {T <: AbstractFloat}
    @assert action_type.N > 0 "Action type N not set"
    @assert length(action) == action_type.n_sections "Action length ($(length(action))) must match n_sections ($(action_type.n_sections))"
    @assert action_type.N % action_type.n_sections == 0 "N ($N) must be divisible by n_sections ($(action_type.n_sections))"

    points_per_section = action_type.N ÷ action_type.n_sections
    pressure_actions = zeros(T, action_type.N)

    for i in 1:action_type.n_sections
        start_idx = (i - 1) * points_per_section + 1
        end_idx = i * points_per_section
        pressure_actions[start_idx:end_idx] .= action[i]
    end
    return [zeros(T, action_type.N), pressure_actions]
end
