struct VectorActionCache{T <: AbstractFloat} <: AbstractCache
    section_controls::Vector{T}
    field::Vector{T}
end

struct MultiStepActionCache{T <: AbstractFloat} <: AbstractCache
    section_controls::Vector{T}
    fields::Vector{Vector{T}}
    times::Vector{T}
end

@kwdef struct DirectScalarPressureAction{T <: AbstractFloat} <: AbstractScalarActionStrategy
    momentum::T = 0.0f0
end

@kwdef struct DirectVectorPressureAction{T <: AbstractFloat} <: AbstractVectorActionStrategy
    n_sections::Int = 1
    momentum::T = 0.0f0
end

@kwdef struct MultiStepPressureAction{T <: AbstractFloat} <: AbstractVectorActionStrategy
    n_sections::Int = 1
    n_steps::Int = 1
    momentum::T = 0.0f0
end

function momentum(at::AbstractActionStrategy)
    return getfield(at, :momentum)
end

function momentum_target(control_target::T, previous_target::T, momentum::T) where {T <: AbstractFloat}
    return momentum * previous_target + (one(T) - momentum) * control_target
end

function fill_sections!(dest::AbstractVector{T}, section_values::AbstractVector{T}) where {T}
    n = length(section_values)
    N = length(dest)
    points_per_section = N ÷ n
    for i in 1:n
        start_idx = (i - 1) * points_per_section + 1
        dest[start_idx:(i * points_per_section)] .= section_values[i]
    end
    return dest
end

function commit_uniform_u_p!(env::RDEEnv{T}, target::T; s = nothing) where {T <: AbstractFloat}
    t0 = env.t
    commit_schedule!(env.prob.injection, t0, t0 + env.dt, target; s = s)
    return nothing
end

function commit_section_u_p!(
        env::RDEEnv{T},
        section_targets::AbstractVector{T},
        dest::Vector{T};
        s = nothing,
    ) where {T <: AbstractFloat}
    fill_sections!(dest, section_targets)
    t0 = env.t
    commit_schedule!(env.prob.injection, t0, t0 + env.dt, dest; s = s)
    return nothing
end

function apply_action!(
        env::RDEEnv{T, A, O, RW, CS, V, OBS, M, RS, C},
        action::Vector{T},
        action_strat::DirectScalarPressureAction,
        action_cache::AbstractCache,
        context::AbstractCache,
    ) where {T <: AbstractFloat, A <: DirectScalarPressureAction, O, RW, CS, V, OBS, M, RS, C}
    @assert length(action) == 1 "DirectScalarPressureAction expects a single action"
    apply_action!(env, action[1], action_strat, action_cache, context)
    return nothing
end

function apply_action!(
        env::RDEEnv{T, A, O, RW, CS, V, OBS, M, RS, C},
        action::T,
        action_strat::DirectScalarPressureAction,
        ::AbstractCache,
        ::AbstractCache,
    ) where {T <: AbstractFloat, A <: DirectScalarPressureAction, O, RW, CS, V, OBS, M, RS, C}
    if action < zero(T) || action > env.u_pmax
        @warn "direct action (u_p) out of bounds [0, u_pmax]"
    end
    clamped_action = clamp(action, zero(T), env.u_pmax)
    prev = current_u_p(env.prob.injection)
    prev_s = prev isa AbstractVector ? prev[1] : prev
    target = momentum_target(clamped_action, prev_s, momentum(action_strat))
    commit_uniform_u_p!(env, target)
    return nothing
end

function apply_action!(
        env::RDEEnv{T, A, O, RW, CS, V, OBS, M, RS, C},
        action::AbstractVector{T},
        action_strat::DirectVectorPressureAction,
        action_cache::VectorActionCache{T},
        ::AbstractCache,
    ) where {T <: AbstractFloat, A <: DirectVectorPressureAction, O, RW, CS, V, OBS, M, RS, C}
    N = env.prob.params.N
    @assert N > 0 "Action type N not set"
    @assert length(action) == action_strat.n_sections "Action length ($(length(action))) must match n_sections ($(action_strat.n_sections))"
    @assert N % action_strat.n_sections == 0 "N ($(N)) must be divisible by n_sections ($(action_strat.n_sections))"

    if any(action .< zero(T)) || any(action .> env.u_pmax)
        @warn "direct action out of bounds [0, u_pmax]"
    end

    current = current_u_p(env.prob.injection)
    if current isa AbstractVector
        current_section_controls = section_midpoint_values(current, action_strat.n_sections)
    else
        current_section_controls = fill(current, action_strat.n_sections)
    end
    clamped_action = clamp.(action, zero(T), env.u_pmax)
    section_controls = action_cache.section_controls
    section_controls .= momentum_target.(clamped_action, current_section_controls, momentum(env.action_strat))
    if action_strat.n_sections == 1
        commit_uniform_u_p!(env, section_controls[1])
    else
        commit_section_u_p!(env, section_controls, action_cache.field)
    end
    return nothing
end

function apply_action!(
        env::RDEEnv{T, A, O, RW, CS, V, OBS, M, RS, C},
        action::AbstractVector{T},
        action_strat::MultiStepPressureAction,
        action_cache::MultiStepActionCache{T},
        ::AbstractCache,
    ) where {T <: AbstractFloat, A <: MultiStepPressureAction, O, RW, CS, V, OBS, M, RS, C}
    n_sec = action_strat.n_sections
    n_steps = action_strat.n_steps
    expected = n_sec * n_steps
    @assert length(action) == expected "MultiStepPressureAction expects length $expected"
    t0 = env.t
    dt = env.dt
    step_dt = dt / T(n_steps)
    for k in 1:n_steps
        action_cache.times[k] = t0 + T(k) * step_dt
    end
    if n_sec == 1
        prev = current_u_p(env.prob.injection)
        prev_s = prev isa AbstractVector ? prev[1] : prev
        targets = action_cache.section_controls
        resize!(targets, n_steps)
        for k in 1:n_steps
            a = clamp(action[k], zero(T), env.u_pmax)
            targets[k] = momentum_target(a, prev_s, momentum(action_strat))
            prev_s = targets[k]
        end
        commit_schedule!(env.prob.injection, t0, action_cache.times, targets)
    else
        N = env.prob.params.N
        @assert N % n_sec == 0
        current = current_u_p(env.prob.injection)
        if current isa AbstractVector
            section_prev = section_midpoint_values(current, n_sec)
        else
            section_prev = fill(current, n_sec)
        end
        targets = action_cache.fields
        for k in 1:n_steps
            lo = (k - 1) * n_sec + 1
            chunk = view(action, lo:(k * n_sec))
            clamped = clamp.(chunk, zero(T), env.u_pmax)
            section_controls = action_cache.section_controls
            section_controls .= momentum_target.(clamped, section_prev, momentum(action_strat))
            copyto!(section_prev, section_controls)
            fill_sections!(targets[k], section_controls)
        end
        commit_schedule!(env.prob.injection, t0, action_cache.times, targets)
    end
    return nothing
end

initialize_cache(at::DirectVectorPressureAction{T}, N::Int, ::Type{T}) where {T <: AbstractFloat} =
    VectorActionCache{T}(Vector{T}(undef, at.n_sections), Vector{T}(undef, N))

function initialize_cache(at::MultiStepPressureAction{T}, N::Int, ::Type{T}) where {T <: AbstractFloat}
    n_sec = at.n_sections
    n_steps = at.n_steps
    fields = [zeros(T, N) for _ in 1:n_steps]
    return MultiStepActionCache{T}(
        Vector{T}(undef, max(n_sec, n_steps)),
        fields,
        Vector{T}(undef, n_steps),
    )
end
