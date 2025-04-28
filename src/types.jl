abstract type AbstractRDEEnv{T} <: AbstractEnv where T <: AbstractFloat end

abstract type AbstractActionType end

@kwdef mutable struct ScalarPressureAction <: AbstractActionType 
    N::Int = 512 #number of grid points
end

function Base.show(io::IO, a::ScalarPressureAction)
    print(io, "ScalarPressureAction()")
end

function Base.show(io::IO, ::MIME"text/plain", a::ScalarPressureAction)
    println(io, "ScalarPressureAction:")
    println(io, "  N: $(a.N)")
end

@kwdef mutable struct ScalarAreaScalarPressureAction <: AbstractActionType 
    N::Int = 512 #number of grid points
end

function Base.show(io::IO, a::ScalarAreaScalarPressureAction)
    print(io, "ScalarAreaScalarPressureAction()")
end

function Base.show(io::IO, ::MIME"text/plain", a::ScalarAreaScalarPressureAction)
    println(io, "ScalarAreaScalarPressureAction:")
    println(io, "  N: $(a.N)")
end

@kwdef mutable struct VectorPressureAction <: AbstractActionType 
    n_sections::Int = 1 #number of sections 
    N::Int = 512 #number of grid points
end

function Base.show(io::IO, a::VectorPressureAction)
    print(io, "VectorPressureAction(n_sections=$(a.n_sections))")
end

function Base.show(io::IO, ::MIME"text/plain", a::VectorPressureAction)
    println(io, "VectorPressureAction:")
    println(io, "  n_sections: $(a.n_sections)")
    println(io, "  N: $(a.N)")
end

function get_standard_normalized_actions end
function action_dim end


#Observations
abstract type AbstractObservationStrategy end
abstract type AbstractMultiAgentObservationStrategy <: AbstractObservationStrategy end

struct FourierObservation <: AbstractObservationStrategy
    fft_terms::Int
end

function Base.show(io::IO, obs::FourierObservation)
    print(io, "FourierObservation(fft_terms=$(obs.fft_terms))")
end

function Base.show(io::IO, ::MIME"text/plain", obs::FourierObservation)
    println(io, "FourierObservation:")
    println(io, "  fft_terms: $(obs.fft_terms)")
end

struct StateObservation <: AbstractObservationStrategy end

function Base.show(io::IO, ::StateObservation)
    print(io, "StateObservation()")
end

function Base.show(io::IO, ::MIME"text/plain", ::StateObservation)
    println(io, "StateObservation: returns full state")
end

@kwdef struct SectionedStateObservation <: AbstractObservationStrategy
    minisections::Int=32
    target_shock_count::Int=3
end

function Base.show(io::IO, obs::SectionedStateObservation)
    print(io, "SectionedStateObservation(minisections=$(obs.minisections))")
end

function Base.show(io::IO, ::MIME"text/plain", obs::SectionedStateObservation)
    println(io, "SectionedStateObservation:")
    println(io, "  minisections: $(obs.minisections)")
    println(io, "  target_shock_count: $(obs.target_shock_count)")
end

struct SampledStateObservation <: AbstractObservationStrategy 
    n_samples::Int
end

function Base.show(io::IO, obs::SampledStateObservation)
    print(io, "SampledStateObservation(n_samples=$(obs.n_samples))")
end

function Base.show(io::IO, ::MIME"text/plain", obs::SampledStateObservation)
    println(io, "SampledStateObservation:")
    println(io, "  n_samples: $(obs.n_samples)")
end


function compute_observation end

function get_init_observation end


#Rewards

abstract type AbstractRDEReward end 

abstract type CachedCompositeReward <: AbstractRDEReward end

abstract type MultiAgentCachedCompositeReward <: CachedCompositeReward end

@kwdef struct ShockSpanReward <: AbstractRDEReward 
    target_shock_count::Int = 3
    span_scale::Float32 = 4.0f0
    shock_weight::Float32 = 0.8f0
end

function Base.show(io::IO, rt::ShockSpanReward)
    print(io, "ShockSpanReward(target_shock_count=$(rt.target_shock_count))")
end

function Base.show(io::IO, ::MIME"text/plain", rt::ShockSpanReward)
    println(io, "ShockSpanReward:")
    println(io, "  target_shock_count: $(rt.target_shock_count)")
    println(io, "  span_scale: $(rt.span_scale)")
    println(io, "  shock_weight: $(rt.shock_weight)")
end

@kwdef mutable struct ShockPreservingReward <: AbstractRDEReward 
    target_shock_count::Int = 3
    span_scale::Float32 = 4.0f0
    shock_weight::Float32 = 0.8f0
    abscence_limit::Float32 = 5.0f0
    abscence_start::Union{Float32, Nothing} = nothing
end

function Base.show(io::IO, rt::ShockPreservingReward)
    print(io, "ShockPreservingReward(target_shock_count=$(rt.target_shock_count))")
end

function Base.show(io::IO, ::MIME"text/plain", rt::ShockPreservingReward)
    println(io, "ShockPreservingReward:")
    println(io, "  target_shock_count: $(rt.target_shock_count)")
    println(io, "  span_scale: $(rt.span_scale)")
    println(io, "  shock_weight: $(rt.shock_weight)")
    println(io, "  abscence_limit: $(rt.abscence_limit)")
end

mutable struct ShockPreservingSymmetryReward <: AbstractRDEReward 
    target_shock_count::Int
    cache::Vector{Float32}  
    function ShockPreservingSymmetryReward(;target_shock_count::Int=4,
                                    N::Int = 512)
        return new(target_shock_count, zeros(Float32, N))
    end
end

function Base.show(io::IO, rt::ShockPreservingSymmetryReward)
    print(io, "ShockPreservingSymmetryReward(target_shock_count=$(rt.target_shock_count))")
end

function Base.show(io::IO, ::MIME"text/plain", rt::ShockPreservingSymmetryReward)
    println(io, "ShockPreservingSymmetryReward:")
    println(io, "  target_shock_count: $(rt.target_shock_count)")
    println(io, "  cache size: $(length(rt.cache))")
end

mutable struct PeriodicityReward <: AbstractRDEReward
    cache::Vector{Float32}
    function PeriodicityReward(;N::Int = 512)
        return new(zeros(Float32, N))
    end
end

function set_reward! end