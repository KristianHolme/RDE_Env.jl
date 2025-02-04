abstract type AbstractRDEEnv{T} <: AbstractEnv where T <: AbstractFloat end

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

function get_standard_normalized_actions end
function action_dim end


#Observations
abstract type AbstractObservationStrategy end

struct FourierObservation <: AbstractObservationStrategy
    fft_terms::Int
end

struct StateObservation <: AbstractObservationStrategy end

struct SampledStateObservation <: AbstractObservationStrategy 
    n_samples::Int
end


function compute_observation end

function get_init_observation end


#Rewards

abstract type AbstractRDEReward end 

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
    abscence_start::Union{Float32, Nothing} = nothing
end

mutable struct ShockPreservingSymmetryReward <: AbstractRDEReward 
    target_shock_count::Int
    cache::Vector{Float32}  
    function ShockPreservingSymmetryReward(;target_shock_count::Int=4,
                                    N::Int = 512)
        return new(target_shock_count, zeros(Float32, N))
    end
end

mutable struct PeriodicityReward <: AbstractRDEReward
    cache::Vector{Float32}
    function PeriodicityReward(;N::Int = 512)
        return new(zeros(Float32, N))
    end
end

function set_reward! end