function set_reward!(env::AbstractRDEEnv, rt::AbstractRDEReward)
    @error "No reward set for type $(typeof(rt))"
end
function set_reward!(env::AbstractRDEEnv{T}, rt::ShockSpanReward) where T
    target_shock_count = rt.target_shock_count
    max_span = rt.span_scale
    λ = rt.shock_weight

    u, = RDE.split_sol_view(env.state)
    dx = env.prob.params.L/env.prob.params.N
    shocks = Float32(RDE.count_shocks(u, dx))
    span = maximum(u) - minimum(u)
    span_reward = span/max_span
    if shocks >= target_shock_count
        shock_reward = one(T)
    elseif shocks > 0
        shock_reward = shocks/(2*target_shock_count)
    else
        shock_reward = T(-1.0)
    end
    
    env.reward = λ*shock_reward + (1-λ)*span_reward
    nothing
end

"""
    set_reward!(env::AbstractRDEEnv, rt::ShockPreservingReward)

Reward for preserving a given number of shocks. terminated/truncated if the number of shocks is not preserved.
    penalize reward if shocks are not evenly spaced, reward for large span.
"""
function set_reward!(env::AbstractRDEEnv{T}, rt::ShockPreservingReward) where T
    target_shock_count = rt.target_shock_count
    max_span = rt.span_scale
    λ = rt.shock_weight

    u, = RDE.split_sol_view(env.state)
    dx = env.prob.x[2] - env.prob.x[1]
    N = env.prob.params.N
    L = env.prob.params.L
    shock_inds = RDE.shock_indices(u, dx)

    span = maximum(u) - minimum(u)
    span_reward = span/max_span

    if length(shock_inds) != target_shock_count
        if isnothing(rt.abscence_start)
            rt.abscence_start = env.t
        elseif env.t - rt.abscence_start > rt.abscence_limit
            env.terminated = true
            env.reward = T(-2.0)
            return nothing
        end
        shock_reward = T(-1.0)
    else
        optimal_spacing = L/target_shock_count
        shock_spacing = mod.(RDE.periodic_diff(shock_inds), N) .* dx
        shock_reward = T(1.0) - mean(abs.((shock_spacing .- optimal_spacing)./optimal_spacing))
    end
    env.reward = λ*shock_reward + (1-λ)*span_reward
    nothing
end

function set_reward!(env::AbstractRDEEnv, rt::ShockPreservingSymmetryReward)
    target_shock_count = rt.target_shock_count
    N = env.prob.params.N
    u = env.state[1:N]
    
    errs = zeros(target_shock_count-1)
    cache = rt.cache
    shift_steps = N ÷ target_shock_count
    for i in 1:(target_shock_count-1)
        cache .= u
        RDE.apply_periodic_shift!(cache, u, shift_steps * i)
        errs[i] = norm(u - cache)/sqrt(N)
    end
    maxerr = maximum(errs)
    env.reward = 1f0 - (maxerr-0.1f0)/0.5f0
    nothing
end

function set_reward!(env::AbstractRDEEnv, rt::PeriodicityReward)
    u, = RDE.split_sol_view(env.state)
    N = env.prob.params.N
    dx = env.prob.x[2] - env.prob.x[1]
    L = env.prob.params.L
    shock_inds = RDE.shock_indices(u, dx)
    shocks = length(shock_inds)

    cache = rt.cache
    if shocks > 1
        shift_steps = N ÷ shocks
        errs = zeros(shocks-1)
        for i in 1:(shocks-1)
            cache .= u
            RDE.apply_periodic_shift!(cache, u, shift_steps * i)
            errs[i] = norm(u - cache)/sqrt(N)
        end
        maxerr = maximum(errs)
        periodicity_reward = 1f0 - (max(maxerr-0.08f0, 0f0)/sqrt(3f0))
    else
        periodicity_reward = 1f0
    end


    if shocks > 1
        optimal_spacing = L/shocks
        shock_spacing = mod.(RDE.periodic_diff(shock_inds), N) .* dx
        shock_spacing_reward = 1f0 - maximum(abs.((shock_spacing .- optimal_spacing)./optimal_spacing))
    else
        shock_spacing_reward = 1f0
    end


    env.reward = (Float32(periodicity_reward + shock_spacing_reward))/2f0
    nothing
end

function Base.show(io::IO, rt::PeriodicityReward)
    print(io, "PeriodicityReward(N=$(length(rt.cache)))")
end