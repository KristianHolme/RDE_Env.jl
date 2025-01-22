function action_dim(at::ScalarPressureAction) 
    return 1
end

function action_dim(at::ScalarAreaScalarPressureAction)
    return 2
end

function action_dim(at::VectorPressureAction)
    return at.n_sections
end


function set_N!(action_type::AbstractActionType, N::Int)
    if action_type isa VectorPressureAction
        @assert N % action_type.n_sections == 0 "N ($N) must be divisible by n_sections ($(action_type.n_sections))"
    end
    setfield!(action_type, :N, N)
    return action_type
end

"""
    get_standard_normalized_actions(action_type::AbstractActionType, action) -> Vector{Vector{T}}

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
function get_standard_normalized_actions(action_type::AbstractActionType, action)
    @assert action_type.N > 0 "Action type N not set"
    @warn "get_standard_normalized_actions is not implemented for action_type $action_type"
end


function get_standard_normalized_actions(action_type::ScalarAreaScalarPressureAction, action)
    @assert action_type.N > 0 "Action type N not set"
    @assert length(action) == 2
    return [fill(action[1], action_type.N), fill(action[2], action_type.N)]
end


function get_standard_normalized_actions(action_type::ScalarPressureAction, action)
    @assert action_type.N > 0 "Action type N not set"
    if isa(action, AbstractArray)
        return [zeros(action_type.N), ones(action_type.N) .* action[1]]
    else
        @assert length(action) == 1
        return [zeros(action_type.N), ones(action_type.N) .* action]
    end
end


function get_standard_normalized_actions(action_type::VectorPressureAction, action)
    @assert action_type.N > 0 "Action type N not set"
    @assert length(action) == action_type.n_sections
    @assert action_type.N % action_type.n_sections == 0
    
    # Calculate how many points per section
    points_per_section = action_type.N ÷ action_type.n_sections
    
    # Initialize pressure actions array
    pressure_actions = zeros(action_type.N)
    
    # Fill each section with its corresponding action value
    for i in 1:action_type.n_sections
        start_idx = (i-1) * points_per_section + 1
        end_idx = i * points_per_section
        pressure_actions[start_idx:end_idx] .= action[i]
    end
    return [zeros(action_type.N), pressure_actions]
end