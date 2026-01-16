abstract type _AbstractEnv end
abstract type AbstractRDEEnv <: DRiL.AbstractEnv end
# Actions
abstract type AbstractActionStrategy end
abstract type AbstractVectorActionStrategy <: AbstractActionStrategy end
abstract type AbstractScalarActionStrategy <: AbstractActionStrategy end
# Constructors
#Observations
abstract type AbstractObservationStrategy end
abstract type AbstractMultiAgentObservationStrategy <: AbstractObservationStrategy end

"""
    compute_observation!(obs, env::RDEEnv, observation_strat::AbstractObservationStrategy, context::AbstractCache)

Compute the observation from the environment and store it in obs.
"""
function compute_observation! end

"""
    get_init_observation(observation_strat::AbstractObservationStrategy, N::Int, T::Type{<:AbstractFloat})

Returns an array of the correct size and type for the observation. Called at env construction.
"""
function get_init_observation end

# Rewards

abstract type AbstractRewardStrategy end
abstract type AbstractVectorRewardStrategy <: AbstractRewardStrategy end
abstract type AbstractScalarRewardStrategy <: AbstractRewardStrategy end

"""
    set_reward!(env::AbstractRDEEnv, rt::AbstractRewardStrategy, context::AbstractCache)

Set the reward for the environment in env.reward.
"""
function set_reward! end

"""
    initialize_cache(rew_strat::AbstractRewardStrategy, N::Int, ::Type{T})

Initialize the cache for the given reward type.
"""
function initialize_cache(::AbstractRewardStrategy, ::Int, ::Type{T}) where {T}
    return NoCache()
end


# Cache API
abstract type AbstractCache end
struct NoCache <: AbstractCache end
"""
    reset_cache!(cache::AbstractCache)

Reset the cache for the given cache type. Called when environment is reset.
"""
reset_cache!(::AbstractCache) = nothing

"""
    initialize_cache(::Any, ::Int, ::Type{T})

Initialize the cache for the given cache type.
"""
initialize_cache(::Any, ::Int, ::Type{T}) where {T} = NoCache()

# Context
abstract type AbstractContextStrategy end

"""
    on_reset!(cache::AbstractCache, context::AbstractContextStrategy, env::AbstractRDEEnv)

Update the context cache. Called when environment is reset.
"""
function on_reset! end

"""
    on_step!(cache::AbstractCache, context::AbstractContextStrategy, env::AbstractRDEEnv)

Update the context cache after each environment step.
"""
function on_step! end

on_step!(::AbstractCache, ::AbstractContextStrategy, ::AbstractRDEEnv) = nothing

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
struct RDEEnvCache{T <: AbstractFloat, RC <: AbstractCache, AC <: AbstractCache, OC <: AbstractCache, CC <: AbstractCache} <: AbstractCache
    prev_u::Vector{T}  # Previous step's u values
    prev_λ::Vector{T}  # Previous step's λ values
    # New subcaches
    reward_cache::RC
    action_cache::AC
    observation_cache::OC
    context::CC
    function RDEEnvCache{T, RC, AC, OC, CC}(N::Int; reward_cache, action_cache, observation_cache, context) where {T <: AbstractFloat, RC <: AbstractCache, AC <: AbstractCache, OC <: AbstractCache, CC <: AbstractCache}
        # Initialize all arrays with zeros instead of undefined values
        prev_u = zeros(T, N)
        prev_λ = zeros(T, N)
        # Default subcaches are NoCache()
        return new{T, RC, AC, OC, CC}(prev_u, prev_λ, reward_cache, action_cache, observation_cache, context)
    end
end

function reset_cache!(cache::RDEEnvCache{T}) where {T}
    reset_cache!(cache.reward_cache)
    reset_cache!(cache.action_cache)
    reset_cache!(cache.observation_cache)
    reset_cache!(cache.context)
    cache.prev_u .= zero(T)
    cache.prev_λ .= zero(T)
    return nothing
end

#TODO:is it necessary to parametrize by all these?
mutable struct RDEEnv{T, A, O, RW, CS, V, OBS, M, RS, C} <: AbstractRDEEnv where {
        T <: AbstractFloat,
        A <: AbstractActionStrategy,
        O <: AbstractObservationStrategy,
        RW <: Union{AbstractScalarRewardStrategy, AbstractVectorRewardStrategy},
        CS <: AbstractContextStrategy,
        V <: Union{T, Vector{T}},
        OBS <: AbstractArray{T},
        M <: AbstractMethod,
        RS <: AbstractReset,
        C <: AbstractControlShift,
    }
    prob::RDEProblem{T, M, RS, C}                  # RDE problem
    state::Vector{T} #TODO:move this to cache?
    observation::OBS #TODO:move this to cache?
    dt::T                       # time step
    t::T                        # Current time
    done::Bool                        # Termination flag
    truncated::Bool
    terminated::Bool #TODO:move these to cache?
    reward::V #TODO:move this to cache?
    smax::T
    u_pmax::T
    τ_smooth::T #smoothing time constant #TODO:move this to action_type?
    cache::RDEEnvCache{T}
    action_strat::A
    observation_strat::O
    reward_strat::RW
    context_strat::CS
    verbose::Bool               # Control solver output
    info::Dict{String, Any} #TODO:move this to cache?
    steps_taken::Int #TODO:move this to cache?
    ode_problem::SciMLBase.ODEProblem #TODO:move this to cache?
end

# Currently nothing to randomize. Maybe random sampling of target requires having internal rng in RDEEnv?
Random.seed!(env::RDEEnv, seed::Int) = nothing
