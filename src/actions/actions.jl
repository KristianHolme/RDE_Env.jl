struct VectorActionCache{T <: AbstractFloat} <: AbstractCache
    section_controls::Vector{T}
end

@kwdef struct DirectScalarPressureAction{T <: AbstractFloat} <: AbstractScalarActionStrategy
    momentum::T = 0.0f0
end

@kwdef struct DirectVectorPressureAction{T <: AbstractFloat} <: AbstractVectorActionStrategy
    n_sections::Int = 1
    momentum::T = 0.0f0
end

function momentum(at::AbstractActionStrategy)
    return getfield(at, :momentum)
end

function action_dim(::AbstractScalarActionStrategy)
    return 1
end

function action_dim(at::DirectVectorPressureAction)
    return at.n_sections
end

function momentum_target(control_target::T, previous_target::T, momentum::T) where {T <: AbstractFloat}
    return momentum * previous_target + (one(T) - momentum) * control_target
end

function apply_action!(env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C}, action::Vector{T}) where {T <: AbstractFloat, A <: DirectScalarPressureAction, O, RW, G, V, OBS, M, RS, C}
    @assert length(action) == 1 "DirectScalarPressureAction expects a single action"
    apply_action!(env, action[1])
    return nothing
end

function apply_action!(env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C}, action::T) where {T <: AbstractFloat, A <: DirectScalarPressureAction, O, RW, G, V, OBS, M, RS, C}
    method_cache = env.prob.method.cache

    copyto!(method_cache.u_p_previous, method_cache.u_p_current)

    if action < zero(T) || action > env.u_pmax
        @warn "direct action (u_p) out of bounds [0, u_pmax]"
    end
    clamped_action = clamp(action, zero(T), env.u_pmax)
    method_cache.u_p_current .= momentum_target(
        clamped_action,
        method_cache.u_p_current[1],
        momentum(env.action_strat)
    )
    copyto!(method_cache.s_previous, method_cache.s_current)
    return nothing
end

function apply_action!(env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C}, action::AbstractVector{T}) where {T <: AbstractFloat, A <: DirectVectorPressureAction, O, RW, G, V, OBS, M, RS, C}
    action_strat = env.action_strat
    N = env.prob.params.N
    @assert N > 0 "Action type N not set"
    @assert length(action) == action_strat.n_sections "Action length ($(length(action))) must match n_sections ($(action_strat.n_sections))"
    @assert N % action_strat.n_sections == 0 "N ($(N)) must be divisible by n_sections ($(action_strat.n_sections))"

    method_cache = env.prob.method.cache

    if any(action .< zero(T)) || any(action .> env.u_pmax)
        @warn "direct action out of bounds [0, u_pmax]"
    end

    copyto!(method_cache.u_p_previous, method_cache.u_p_current)

    points_per_section = N ÷ action_strat.n_sections
    current_section_controls = @view method_cache.u_p_current[1:points_per_section:end]
    @assert env.cache.action_cache isa VectorActionCache{T}
    section_controls = env.cache.action_cache.section_controls
    clamped_action = clamp.(action, zero(T), env.u_pmax)
    section_controls .= momentum_target.(clamped_action, current_section_controls, momentum(env.action_strat))
    for i in 1:action_strat.n_sections
        start_idx = (i - 1) * points_per_section + 1
        end_idx = i * points_per_section
        method_cache.u_p_current[start_idx:end_idx] .= section_controls[i]
    end
    return nothing
end

initialize_cache(at::DirectVectorPressureAction{T}, N::Int, ::Type{T}) where {T <: AbstractFloat} =
    VectorActionCache{T}(Vector{T}(undef, at.n_sections))
