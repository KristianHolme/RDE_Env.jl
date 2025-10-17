abstract type _AbstractEnv end
abstract type AbstractRDEEnv <: _AbstractEnv end

# Actions
abstract type AbstractActionType end
"""
    action_dim(action_type::AbstractActionType)

Return the dimension of the action space for the given action type.
"""
function action_dim end

"""
    _reset_action!(action_type::AbstractActionType)

Reset the action type to its initial state. Mainly used for resetting cache. Called when environment is reset.
"""
function _reset_action! end

# Constructors
# PIDAction(; N::Int = 512, target::Float32 = 0.0f0) = PIDAction{Float32}(N = N, target = target)

#Observations
abstract type AbstractObservationStrategy end
abstract type AbstractMultiAgentObservationStrategy <: AbstractObservationStrategy end

"""
    compute_observation!(obs, env::RDEEnv, observation_strategy::AbstractObservationStrategy)

Compute the observation from the environment and store it in obs.
"""
function compute_observation! end

"""
    get_init_observation(observation_strategy::AbstractObservationStrategy, N::Int, T::Type{<:AbstractFloat})

Returns an array of the correct size and type for the observation. Called at env construction.
"""
function get_init_observation end

#Rewards

abstract type AbstractRDEReward end

"""
    set_reward!(env::AbstractRDEEnv, rt::AbstractRDEReward)

Set the reward for the environment in env.reward.
"""
function set_reward! end

abstract type CachedCompositeReward <: AbstractRDEReward end

abstract type MultiAgentCachedCompositeReward <: CachedCompositeReward end

## env
"""
    RDEEnvCache{T<:AbstractFloat}

Cache for RDE environment computations and state tracking.

# Fields
- `circ_u::CircularVector{T}`: Circular buffer for velocity field
- `circ_λ::CircularVector{T}`: Circular buffer for reaction progress
- `prev_u::Vector{T}`: Previous velocity field
- `prev_λ::Vector{T}`: Previous reaction progress
"""
mutable struct RDEEnvCache{T <: AbstractFloat} #TODO remove circ
    circ_u::CircularVector{T, Vector{T}}
    circ_λ::CircularVector{T, Vector{T}}
    prev_u::Vector{T}  # Previous step's u values
    prev_λ::Vector{T}  # Previous step's λ values
    action::Matrix{T} # column 1 = s action, column 2 = u_p action
    function RDEEnvCache{T}(N::Int) where {T <: AbstractFloat}
        # Initialize all arrays with zeros instead of undefined values
        circ_u = CircularArray(zeros(T, N))
        circ_λ = CircularArray(zeros(T, N))
        prev_u = zeros(T, N)
        prev_λ = zeros(T, N)
        action = zeros(T, N, 2)
        return new{T}(circ_u, circ_λ, prev_u, prev_λ, action)
    end
end

#TODO_is it necessary to parametrize by A, O, R?
mutable struct RDEEnv{T, A, O, RW, V, OBS, M, RS, C} <: AbstractRDEEnv where {
        T <: AbstractFloat,
        A <: AbstractActionType,
        O <: AbstractObservationStrategy,
        RW <: AbstractRDEReward,
        V <: Union{T, Vector{T}},
        OBS <: AbstractArray{T},
        M <: AbstractMethod,
        RS <: AbstractReset,
        C <: AbstractControlShift,
    }
    prob::RDEProblem{T, M, RS, C}                  # RDE problem
    state::Vector{T}
    observation::OBS
    dt::T                       # time step
    t::T                        # Current time
    done::Bool                        # Termination flag
    truncated::Bool
    terminated::Bool
    reward::V
    smax::T
    u_pmax::T
    τ_smooth::T #smoothing time constant
    cache::RDEEnvCache{T}
    action_type::A
    observation_strategy::O
    reward_type::RW
    verbose::Bool               # Control solver output
    info::Dict{String, Any}
    steps_taken::Int
    ode_problem::SciMLBase.ODEProblem
end

# Helper functions to determine observation array type
#TODO remove
observation_array_type(::Type{T}, ::AbstractObservationStrategy) where {T} = Vector{T}  # Default: Vector
observation_array_type(::Type{T}, ::AbstractMultiAgentObservationStrategy) where {T} = Matrix{T}  # Multi-agent: Matrix
