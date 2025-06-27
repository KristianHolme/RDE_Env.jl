abstract type _AbstractEnv end
abstract type AbstractRDEEnv <: _AbstractEnv end

abstract type AbstractActionType end

@kwdef mutable struct ScalarPressureAction <: AbstractActionType
    N::Int = 512 #number of grid points
end



@kwdef mutable struct ScalarAreaScalarPressureAction <: AbstractActionType
    N::Int = 512 #number of grid points
end



@kwdef mutable struct VectorPressureAction <: AbstractActionType
    n_sections::Int = 1 #number of sections 
    N::Int = 512 #number of grid points
end



function get_standardized_actions end
function action_dim end


#Observations
abstract type AbstractObservationStrategy end
abstract type AbstractMultiAgentObservationStrategy <: AbstractObservationStrategy end
function compute_observation end

function get_init_observation end

struct FourierObservation <: AbstractObservationStrategy
    fft_terms::Int
end



struct StateObservation <: AbstractObservationStrategy end



@kwdef struct SectionedStateObservation <: AbstractObservationStrategy
    minisections::Int = 32
    target_shock_count::Int = 3
end



struct SampledStateObservation <: AbstractObservationStrategy
    n_samples::Int
end






#Rewards

abstract type AbstractRDEReward end

abstract type CachedCompositeReward <: AbstractRDEReward end

abstract type MultiAgentCachedCompositeReward <: CachedCompositeReward end

@kwdef struct ShockSpanReward <: AbstractRDEReward
    target_shock_count::Int = 3
    span_scale::Float32 = 4.0f0
    shock_weight::Float32 = 0.8f0
end


@kwdef mutable struct ShockPreservingReward <: AbstractRDEReward
    target_shock_count::Int = 3
    span_scale::Float32 = 4.0f0
    shock_weight::Float32 = 0.8f0
    abscence_limit::Float32 = 5.0f0
    abscence_start::Union{Float32,Nothing} = nothing
end



mutable struct ShockPreservingSymmetryReward <: AbstractRDEReward
    target_shock_count::Int
    cache::Vector{Float32}
    function ShockPreservingSymmetryReward(; target_shock_count::Int=4,
        N::Int=512)
        return new(target_shock_count, zeros(Float32, N))
    end
end


mutable struct PeriodicityReward <: AbstractRDEReward
    cache::Vector{Float32}
    function PeriodicityReward(; N::Int=512)
        return new(zeros(Float32, N))
    end
end

function set_reward! end


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
mutable struct RDEEnvCache{T<:AbstractFloat}#TODO remove circ
    circ_u::CircularVector{T,Vector{T}}
    circ_λ::CircularVector{T,Vector{T}}
    prev_u::Vector{T}  # Previous step's u values
    prev_λ::Vector{T}  # Previous step's λ values
    action::Matrix{T} # column 1 = s action, column 2 = u_p action
    function RDEEnvCache{T}(N::Int) where {T<:AbstractFloat}
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
mutable struct RDEEnv{T,A,O,R,V,OBS} <: AbstractRDEEnv where {T<:AbstractFloat,A<:AbstractActionType,O<:AbstractObservationStrategy,R<:AbstractRDEReward,V<:Union{T,Vector{T}},OBS<:AbstractArray{T}}
    prob::RDEProblem                  # RDE problem
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
    α::T #action momentum
    τ_smooth::T #smoothing time constant
    cache::RDEEnvCache{T}
    action_type::A
    observation_strategy::O
    reward_type::R
    verbose::Bool               # Control solver output
    info::Dict{String,Any}
    steps_taken::Int
    ode_problem::Union{Nothing,ODEProblem}
end

# Helper functions to determine observation array type
observation_array_type(::Type{T}, ::AbstractObservationStrategy) where {T} = Vector{T}  # Default: Vector
observation_array_type(::Type{T}, ::AbstractMultiAgentObservationStrategy) where {T} = Matrix{T}  # Multi-agent: Matrix