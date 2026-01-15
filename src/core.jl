abstract type _AbstractEnv end
abstract type AbstractRDEEnv <: DRiL.AbstractEnv end
# Actions
abstract type AbstractActionStrategy end
abstract type AbstractVectorActionStrategy <: AbstractActionStrategy end
abstract type AbstractScalarActionStrategy <: AbstractActionStrategy end
"""
    action_dim(action_strat::AbstractActionStrategy)

Return the dimension of the action space for the given action type.
"""
function action_dim end

"""
    _reset_action!(action_strat::AbstractActionStrategy)

Reset the action type to its initial state. Mainly used for resetting cache. Called when environment is reset.
"""
function _reset_action! end

# Constructors
# PIDAction(; N::Int = 512, target::Float32 = 0.0f0) = PIDAction{Float32}(N = N, target = target)

#Observations
abstract type AbstractObservationStrategy end
abstract type AbstractMultiAgentObservationStrategy <: AbstractObservationStrategy end

"""
    compute_observation!(obs, env::RDEEnv, observation_strat::AbstractObservationStrategy)

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
    reward_value_type(T::Type{<:AbstractFloat}, rt::AbstractRewardStrategy)

Return the type of the reward value for the given reward type.

# Examples
```julia
reward_value_type(Float32, ShockSpanReward())  # Returns Float32
reward_value_type(Float32, MultiSectionReward())  # Returns Vector{Float32}
```
"""
function reward_value_type end

reward_value_type(::Type{T}, ::AbstractScalarRewardStrategy) where {T} = T
reward_value_type(::Type{T}, ::AbstractVectorRewardStrategy) where {T} = Vector{T}

"""
    set_reward!(env::AbstractRDEEnv, rt::AbstractRewardStrategy)

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

#TODO: deprecate, remove
abstract type CachedCompositeReward <: AbstractRewardStrategy end
abstract type MultiAgentCachedCompositeReward <: CachedCompositeReward end

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

# goals
abstract type AbstractGoalStrategy end
abstract type AbstractGoalCache <: AbstractCache end

mutable struct GoalCache <: AbstractGoalCache
    target_shock_count::Int
end

"""
    update_goal!(cache::AbstractGoalCache, goal::AbstractGoalStrategy, env::AbstractRDEEnv)

Update the goal cache. Called when environment is reset.
"""
function update_goal! end

"""
    get_target_shock_count(goal_strat::AbstractGoalStrategy, env::AbstractRDEEnv)

Get the target shock count from the goal strategy.
"""
function get_target_shock_count end

"""
    set_target_shock_count!(goal_strat::AbstractGoalStrategy, env::AbstractRDEEnv, v::Int)
Set the target shock count from the goal strategy.
"""
function set_target_shock_count! end

"""
    get_target_shock_count(env::AbstractRDEEnv) -> Int
Get the target shock count from the goal strategy.
"""
get_target_shock_count(env::AbstractRDEEnv) = get_target_shock_count(env.goal_strat, env)
set_target_shock_count!(env::AbstractRDEEnv, v::Int) = set_target_shock_count!(env.goal_strat, env, v)

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
struct RDEEnvCache{T <: AbstractFloat, RC <: AbstractCache, AC <: AbstractCache, OC <: AbstractCache, GC <: AbstractCache} <: AbstractCache
    prev_u::Vector{T}  # Previous step's u values
    prev_λ::Vector{T}  # Previous step's λ values
    # New subcaches
    reward_cache::RC
    action_cache::AC
    observation_cache::OC
    goal_cache::GC
    function RDEEnvCache{T, RC, AC, OC, GC}(N::Int; reward_cache, action_cache, observation_cache, goal_cache) where {T <: AbstractFloat, RC <: AbstractCache, AC <: AbstractCache, OC <: AbstractCache, GC <: AbstractCache}
        # Initialize all arrays with zeros instead of undefined values
        prev_u = zeros(T, N)
        prev_λ = zeros(T, N)
        # Default subcaches are NoCache()
        return new{T, RC, AC, OC, GC}(prev_u, prev_λ, reward_cache, action_cache, observation_cache, goal_cache)
    end
end

function reset_cache!(cache::RDEEnvCache{T}) where {T}
    reset_cache!(cache.reward_cache)
    reset_cache!(cache.action_cache)
    reset_cache!(cache.observation_cache)
    reset_cache!(cache.goal_cache)
    cache.prev_u .= zero(T)
    cache.prev_λ .= zero(T)
    return nothing
end

#TODO:is it necessary to parametrize by all these?
mutable struct RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C} <: AbstractRDEEnv where {
        T <: AbstractFloat,
        A <: AbstractActionStrategy,
        O <: AbstractObservationStrategy,
        RW <: AbstractRewardStrategy,
        G <: AbstractGoalStrategy,
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
    goal_strat::G
    verbose::Bool               # Control solver output
    info::Dict{String, Any} #TODO:move this to cache?
    steps_taken::Int #TODO:move this to cache?
    ode_problem::SciMLBase.ODEProblem #TODO:move this to cache?
end

# Currently nothing to randomize. Maybe random sampling of target requires having internal rng in RDEEnv?
Random.seed!(env::RDEEnv, seed::Int) = nothing
