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

Note: This is a fallback method. Each action type should implement its own specific version.
"""
function get_standardized_actions(action_type::AbstractActionType, action)
    @assert action_type.N > 0 "Action type N not set"
    return @warn "get_standardized_actions is not implemented for action_type $action_type"
end

# Default compute_standard_actions: defer to get_standardized_actions
compute_standard_actions(action_type::AbstractActionType, action, env::RDEEnv) = get_standardized_actions(action_type, action)

# Reset hook: default no-op
_reset_action!(::AbstractActionType, ::RDEEnv) = nothing
