abstract type _AbstractEnv end
abstract type AbstractRDEEnv <: _AbstractEnv end

# Actions
abstract type AbstractActionType end
abstract type AbstractVectorActionType <: AbstractActionType end
abstract type AbstractScalarActionType <: AbstractActionType end
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

# Cache API
abstract type AbstractCache end
struct NoCache <: AbstractCache end

mutable struct GoalCache <: AbstractCache
    target_shock_count::Int
end

initialize_cache(::Any, ::Int, ::Type{T}) where {T} = NoCache()
reset_cache!(::AbstractCache) = nothing

get_target_shock_count(env::AbstractRDEEnv) = env.cache.goal.target_shock_count
set_target_shock_count!(env::AbstractRDEEnv, v::Int) = env.cache.goal.target_shock_count = v

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
mutable struct RDEEnvCache{T <: AbstractFloat, RC <: AbstractCache, AC <: AbstractCache, OC <: AbstractCache, GC <: GoalCache} <: AbstractCache
    prev_u::Vector{T}  # Previous step's u values
    prev_λ::Vector{T}  # Previous step's λ values
    # New subcaches
    reward_cache::RC
    action_cache::AC
    observation_cache::OC
    goal::GC
    function RDEEnvCache{T, RC, AC, OC, GC}(N::Int; reward_cache, action_cache, observation_cache, goal) where {T <: AbstractFloat, RC <: AbstractCache, AC <: AbstractCache, OC <: AbstractCache, GC <: GoalCache}
        # Initialize all arrays with zeros instead of undefined values
        prev_u = zeros(T, N)
        prev_λ = zeros(T, N)
        # Default subcaches are NoCache(), and GoalCache defaults to 3 shocks
        return new{T, RC, AC, OC, GC}(prev_u, prev_λ, reward_cache, action_cache, observation_cache, goal)
    end
end

function reset_cache!(cache::RDEEnvCache{T}) where {T}
    reset_cache!(cache.reward_cache)
    reset_cache!(cache.action_cache)
    reset_cache!(cache.observation_cache)
    cache.prev_u .= zero(T)
    cache.prev_λ .= zero(T)
    return nothing
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
